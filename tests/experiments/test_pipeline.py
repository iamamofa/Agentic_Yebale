import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowing_where.context import ContextDocument
from knowing_where.evidence.features import KAPExtractor
from knowing_where.experiments.data import download_dataset, load_dataset, parse_example, split_dataset
from knowing_where.experiments.decisions import MondrianCalibrator, decide
from knowing_where.experiments.demo import write_demo
from knowing_where.experiments.generation import GenerationRequest, ReplayGenerator, generate_paths
from knowing_where.experiments.metrics import evaluate
from knowing_where.experiments.runner import run_experiment
from knowing_where.experiments.training import fit_head
from knowing_where.routing import ModelProfile, RoutingPolicy
from knowing_where.types import OUTSIDE, Source


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset, self.models = write_demo(self.temp.name)
        self.rows = load_dataset(self.dataset)

    def test_seeded_split_is_order_invariant_and_disjoint(self):
        a, b = split_dataset(self.rows, 7), split_dataset(list(reversed(self.rows)), 7)
        self.assertEqual(a, b)
        self.assertEqual([len(a[k]) for k in ('train', 'calibration', 'test')], [40, 20, 40])
        sets = [set(r.id for r in a[k]) for k in a]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertNotEqual(a, split_dataset(self.rows, 8))

    def test_related_examples_stay_in_one_partition(self):
        raw = [json.loads(line) for line in self.dataset.read_text().splitlines()]
        for i, row in enumerate(raw):
            row['split_group'] = str(i // 2)
        parts = split_dataset([parse_example(r) for r in raw])
        memberships = {}
        for name, rows in parts.items():
            for row in rows:
                self.assertEqual(memberships.setdefault(row.split_group, name), name)

    def test_labels_are_strict_and_duplicate_questions_rejected(self):
        row = json.loads(self.dataset.read_text().splitlines()[0])
        row['corpus_present']['P'] = 1
        with self.assertRaises(ValueError):
            parse_example(row)
        raw = [json.loads(line) for line in self.dataset.read_text().splitlines()]
        raw[1]['question'] = raw[0]['question']
        self.dataset.write_text('\n'.join(json.dumps(r) for r in raw))
        with self.assertRaises(ValueError):
            load_dataset(self.dataset)

    def test_checked_download_is_atomic(self):
        content = self.dataset.read_bytes()
        destination = Path(self.temp.name) / 'downloaded.jsonl'
        destination.write_text('existing')
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def geturl(self): return 'https://example.org/data.jsonl'
            def read(self, size): return content[:size]
        with patch('urllib.request.urlopen', return_value=Response()):
            with self.assertRaises(ValueError):
                download_dataset('https://example.org/data.jsonl', destination, '0' * 64)
            self.assertEqual(destination.read_text(), 'existing')
            download_dataset('https://example.org/data.jsonl', destination, hashlib.sha256(content).hexdigest())
        self.assertEqual(destination.read_bytes(), content)
        with self.assertRaises(ValueError):
            download_dataset('http://example.org/data.jsonl', destination, '0' * 64)


class TrainingAndGenerationTests(unittest.TestCase):
    def test_fitted_logistic_head_learns_separation_and_single_class_is_smoothed(self):
        head, audit = fit_head([{'x': -2.0}, {'x': -1.0}, {'x': 1.0}, {'x': 2.0}],
                               [False, False, True, True])
        self.assertLess(head.predict({'x': -1}), 0.2)
        self.assertGreater(head.predict({'x': 1}), 0.8)
        self.assertFalse(audit['single_class'])
        head, audit = fit_head([{'x': 100}], [True])
        self.assertAlmostEqual(head.predict({'x': -100}), 2 / 3)
        self.assertTrue(audit['single_class'])

    def test_ten_paths_are_one_source_distribution_and_ties_are_stable(self):
        request = GenerationRequest('x', 'Question?', Source.P, 'model', (), ())
        backend = ReplayGenerator({'x': {Source.P: ('b',) * 5 + ('a',) * 5}})
        confidence = generate_paths(backend, request)
        self.assertEqual(confidence.answer, 'a')
        self.assertEqual(confidence.probabilities, {'a': 0.5, 'b': 0.5})
        self.assertEqual(confidence.agreement, 0.5)
        self.assertEqual(len(confidence.paths), 10)
        self.assertFalse({'gold', 'corpus_present', 'metadata'} & {f.name for f in fields(request)})
        with self.assertRaises(ValueError):
            generate_paths(backend, request, paths=11)

    def test_bad_generator_outputs_fail(self):
        request = GenerationRequest('x', 'Question?', Source.P, 'model', (), ())
        with self.assertRaises(ValueError):
            generate_paths(ReplayGenerator({'x': {Source.P: (OUTSIDE,) * 10}}), request)


class DecisionTests(unittest.TestCase):
    def test_exact_finite_sample_quantile_and_group_isolation(self):
        observations = [('a', {'yes': 0.9, OUTSIDE: 0.1}, 'yes')] * 9
        observations += [('b', {'yes': 0.2, OUTSIDE: 0.8}, 'yes')] * 9
        cal = MondrianCalibrator.fit(observations, 0.1)
        self.assertAlmostEqual(cal.threshold('a'), 0.1)
        self.assertAlmostEqual(cal.threshold('b'), 0.8)
        self.assertEqual(cal.predict({'yes': 0.9, OUTSIDE: 0.1}, 'a'), ('yes',))
        self.assertEqual(set(cal.predict({'yes': 0.9, OUTSIDE: 0.1}, 'unseen')), {'yes', OUTSIDE})

    def test_sparse_groups_return_full_frame(self):
        cal = MondrianCalibrator.fit([('small', {'yes': 1.0, OUTSIDE: 0.0}, 'yes')], 0.1)
        self.assertEqual(cal.threshold('small'), float('inf'))
        self.assertEqual(len(cal.predict({'yes': 1.0, OUTSIDE: 0.0}, 'small')), 2)
        self.assertIsNone(cal.summary()['small']['threshold'])

    def test_abstention_reasons(self):
        self.assertEqual(decide(('yes',))['answer'], 'yes')
        for values, reason in [((OUTSIDE,), 'outside_in_set'), ((), 'empty_set'),
                               (('a', 'b'), 'ambiguous_set')]:
            self.assertEqual(decide(values)['reason'], reason)
        self.assertEqual(decide(('yes',), total_conflict=True)['reason'], 'total_conflict')
        self.assertEqual(decide(('yes',), routed=False)['reason'], 'no_feasible_model')

    def test_invalid_alpha_and_unknown_calibration_truth(self):
        for alpha in (0, 1, -1, float('nan')):
            with self.assertRaises(ValueError):
                MondrianCalibrator.fit([], alpha)
        with self.assertRaises(ValueError):
            MondrianCalibrator.fit([('a', {'yes': 1}, 'no')])

    def test_metrics_distinguish_outside_coverage_from_answer_coverage(self):
        rows = [{'gold': 'missing', 'target': OUTSIDE, 'probabilities': {'wrong': 0, OUTSIDE: 1},
                 'prediction_set': [OUTSIDE], 'decision': decide((OUTSIDE,))}]
        metrics = evaluate(rows)
        self.assertEqual(metrics['mapped_set_coverage'], 1)
        self.assertEqual(metrics['canonical_answer_set_coverage'], 0)
        self.assertEqual(metrics['candidate_recall'], 0)
        self.assertIsNone(metrics['selective_risk'])
        self.assertEqual(metrics['multiclass_brier'], 0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        dataset, model_path = write_demo(cls.temp.name)
        cls.rows = load_dataset(dataset)
        cls.models = [ModelProfile(**m) for m in json.loads(model_path.read_text())]
        cls.report = run_experiment(cls.rows, cls.models, joint_ghost='independent')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_end_to_end_artifacts_and_reproducibility(self):
        again = run_experiment(self.rows, self.models, joint_ghost='independent')
        self.assertEqual(self.report, again)
        self.assertEqual(len(self.report['comparisons']), 5)
        self.assertEqual(self.report['training']['training_ids'], self.report['splits']['train']['ids'])
        self.assertEqual(sum(g['calibration_count'] for g in self.report['calibration']['bayesian'].values()), 20)
        self.assertEqual(len(self.report['test_predictions']['bayesian']), 40)
        json.dumps(self.report, allow_nan=False)

    def test_heldout_labels_never_change_fitted_parameters_or_scores(self):
        train_ids = set(self.report['splits']['train']['ids'])
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row.id not in train_ids:
                object.__setattr__(row, 'gold', 'changed-heldout-truth')
                row.corpus_present.update({s: False for s in Source})
        again = run_experiment(changed, self.models, joint_ghost='independent')
        self.assertEqual(self.report['training'], again['training'])
        self.assertEqual(self.report['test_audit'], again['test_audit'])
        for method in self.report['test_predictions']:
            before = [r['probabilities'] for r in self.report['test_predictions'][method]]
            after = [r['probabilities'] for r in again['test_predictions'][method]]
            self.assertEqual(before, after)

    def test_generation_receives_only_source_context(self):
        requests = []
        replay = ReplayGenerator({r.id: r.paths for r in self.rows})
        class Spy:
            def generate(self, request, path_index, seed):
                requests.append(request)
                return replay.generate(request, path_index, seed)
        run_experiment(self.rows, self.models, joint_ghost='independent', generator=Spy())
        self.assertEqual(len(requests), 100 * 4 * 10)
        for request in requests:
            self.assertEqual(len(request.excerpts), 1)
            self.assertIn(f'Synthetic {request.source.value} context', request.excerpts[0])

    def test_no_feasible_training_model_fails_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, 'no routed training examples'):
            run_experiment(self.rows, self.models, joint_ghost='independent',
                           policy=RoutingPolicy(max_cost_usd=0))

    def test_joint_assumption_must_be_explicit(self):
        with self.assertRaises(ValueError):
            run_experiment(self.rows, self.models, joint_ghost='implicit')

    def test_rejected_test_request_abstains_without_generation(self):
        changed = copy.deepcopy(self.rows)
        target_id = self.report['splits']['test']['ids'][0]
        target = next(row for row in changed if row.id == target_id)
        object.__setattr__(target, 'documents', (ContextDocument('oversized', Source.K, 'x' * 20000),))
        object.__setattr__(target, 'required_document_ids', ('oversized',))
        replay = ReplayGenerator({r.id: r.paths for r in changed if r.id != target_id})
        report = run_experiment(changed, self.models, joint_ghost='independent', generator=replay)
        for predictions in report['test_predictions'].values():
            prediction = next(p for p in predictions if p['id'] == target_id)
            self.assertEqual(prediction['decision']['reason'], 'no_feasible_model')
        self.assertEqual(report['test_audit'][target_id]['path_confidence'], {})

    def test_no_presence_supervision_fails_instead_of_inventing_a_head(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row.corpus_present[Source.K] = False
        with self.assertRaisesRegex(ValueError, 'no corpus-present training examples for source K'):
            run_experiment(changed, self.models, joint_ghost='independent')
