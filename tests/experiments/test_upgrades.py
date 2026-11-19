"""Regression checks for inference artifacts, preflight controls, and strict input."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from knowing_where.experiments.data import load_dataset, load_queries, parse_example, parse_query, split_dataset
from knowing_where.experiments.decisions import FrozenCalibrator
from knowing_where.experiments.demo import write_demo
from knowing_where.experiments.generation import GenerationBudgetExceeded, GenerationRequest, ManagedGenerator, ReplayGenerator
from knowing_where.experiments.io import loads, read_json, write_json
from knowing_where.experiments.pipeline import prepare_route
from knowing_where.experiments.pipeline import score_sources
from knowing_where.experiments.generation import PathConfidence
from knowing_where.evidence.features import KAPExtractor
from knowing_where.evidence.prediction import SourceEstimate
from knowing_where.experiments.runner import run_experiment
from knowing_where.experiments.system import FittedSystem
from knowing_where.experiments.training import fit_head
from knowing_where.routing import ModelProfile, RoutingPolicy
from knowing_where.types import Source


class StrictInputTests(unittest.TestCase):
    def setUp(self):
        self.raw = {'id': 'a', 'question': 'Which answer?', 'group': 'lookup',
                    'gold': 'yes', 'corpus_present': {s.value: True for s in Source}}

    def test_wrong_container_types_and_unknown_fields_are_rejected(self):
        mutations = [{'instructions': 'follow me'}, {'required_document_ids': 'a'},
                     {'documents': {}}, {'paths': []}, {'metadata': []},
                     {'metadata': {'quality': None}}, {'unexpected': True},
                     {'documents': [{'id': 'a', 'source': 'K', 'text': 'x', 'depends_on': 'b'}]}]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                parse_example({**self.raw, **mutation})

    def test_fractional_and_colliding_hop_keys_are_not_coerced(self):
        for probabilities in ({'1.5': 1}, {'0': 0.5, '00': 0.5}, {1.5: 1}, {'-1': 1}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                parse_example({**self.raw, 'metadata': {'hops': {'probabilities': probabilities}}})
        parsed = parse_example({**self.raw, 'metadata': {'hops': {'probabilities': {'2': 1}}}})
        self.assertEqual(parsed.metadata.hops.mean, 2)

    def test_json_rejects_duplicate_keys_and_nonfinite_literals(self):
        for data in ('{"id":"a","id":"b"}', '{"x":NaN}', '{"x":Infinity}',
                     '{"x":1e400}', '{"x":' + '9' * 400 + '}'):
            with self.assertRaises(ValueError):
                loads(data)

    def test_labeled_inputs_cannot_enter_prediction_parser(self):
        with self.assertRaisesRegex(ValueError, 'unknown query fields'):
            parse_query(self.raw)
        query = parse_query({k: v for k, v in self.raw.items() if k not in ('gold', 'corpus_present')})
        self.assertFalse(hasattr(query, 'gold'))

    def test_programmatic_splits_cannot_bypass_duplicate_question_check(self):
        rows = [parse_example({**self.raw, 'id': str(i), 'question': f'Question {i}?'}) for i in range(5)]
        object.__setattr__(rows[1], 'question', rows[0].question)
        with self.assertRaisesRegex(ValueError, 'duplicate questions'):
            split_dataset(rows)

    def test_dataset_errors_include_line_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.jsonl'
            path.write_text(json.dumps(self.raw) + '\n{"id": NaN}\n')
            with self.assertRaisesRegex(ValueError, r'bad.jsonl:2:'):
                load_dataset(path)

    def test_failed_serialization_preserves_existing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'output.json'
            path.write_text('existing')
            with self.assertRaises(ValueError):
                write_json(path, {'broken': float('nan')})
            self.assertEqual(path.read_text(), 'existing')
            write_json(path, {'valid': True})
            self.assertEqual(read_json(path), {'valid': True})
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])


class GenerationControlTests(unittest.TestCase):
    def setUp(self):
        self.request = GenerationRequest('x', 'Question?', Source.P, 'model', (), ())

    def test_transient_retry_preserves_request_seed_and_counts_attempts(self):
        calls = []
        class Backend:
            def generate(self, request, path_index, seed):
                calls.append((request, path_index, seed))
                if len(calls) == 1:
                    raise TimeoutError('transient')
                return 'answer'
        managed = ManagedGenerator(Backend(), max_calls=2, max_retries=1)
        with patch('knowing_where.experiments.generation.time.sleep') as sleep:
            self.assertEqual(managed.generate(self.request, 0, 99), 'answer')
            sleep.assert_called_once_with(0.1)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(managed.summary(1)['retry_calls'], 1)
        self.assertEqual(managed.calls, 2)

    def test_retries_never_exceed_global_call_limit(self):
        class Backend:
            def generate(self, *args):
                raise TimeoutError('still down')
        managed = ManagedGenerator(Backend(), max_calls=1, max_retries=5)
        with self.assertRaises(GenerationBudgetExceeded):
            managed.generate(self.request, 0, 1)
        self.assertEqual(managed.calls, 1)

    def test_validation_failures_are_not_retried(self):
        class Backend:
            def generate(self, *args):
                raise ValueError('invalid request')
        managed = ManagedGenerator(Backend(), max_retries=5)
        with self.assertRaises(ValueError):
            managed.generate(self.request, 0, 1)
        self.assertEqual(managed.calls, 1)
        self.assertEqual(managed.retries, 0)

    def test_source_handoff_preserves_own_claims_and_offsets(self):
        raw = {'id': 'q', 'question': 'What is the latency?', 'group': 'lookup',
               'documents': [
                   {'id': 'web', 'source': 'W', 'text': 'Latency is 1000 ms.', 'claims': {'latency': '1000'}},
                   {'id': 'memory', 'source': 'M', 'text': 'Latency is 2000 ms.', 'claims': {'latency': '2000'}}]}
        query = parse_query(raw)
        model = ModelProfile('m', 'demo', {}, 5000, 500)
        _, requests = prepare_route(query, [model], RoutingPolicy())
        web = next(r for r in requests if r.source == Source.W)
        self.assertEqual(web.source_claims, ({'key': 'latency', 'value': '1000', 'document_ids': ['web']},))
        self.assertTrue(all(d['source'] == 'W' for d in web.source_documents))
        self.assertEqual(web.excerpt_records[0]['document_id'], 'web')
        self.assertEqual(web.excerpt_records[0]['start'], 0)
        self.assertIn('untrusted data', web.evidence_policy)


class NumericalAndCalibrationTests(unittest.TestCase):
    def test_impossible_bayesian_evidence_is_reported_without_crashing(self):
        class Predictor:
            def predict(self, features):
                return {source: SourceEstimate(1.0, 0.0) for source in Source}
        confidence = {source: PathConfidence('a' if source == Source.P else 'b',
                        {'a' if source == Source.P else 'b': 1.0}, 1.0, 0.0, ()) for source in Source}
        result = score_sources(KAPExtractor().extract('Which answer?'), confidence, Predictor())
        self.assertEqual(result['diagnostics']['bayesian_status'], 'zero_evidence')
        self.assertEqual(result['diagnostics']['ds_status'], 'total_conflict')
        self.assertIsNone(result['diagnostics']['log_evidence_ratio'])
        self.assertEqual(result['scores']['bayesian']['Ø'], 1)
        self.assertFalse(result['uncertainty']['bayesian']['available'])
        self.assertIsNone(result['uncertainty']['bayesian']['normalized_entropy'])

    def test_large_finite_features_do_not_overflow_training(self):
        head, audit = fit_head([{'x': -1e308}, {'x': -5e307}, {'x': 5e307}, {'x': 1e308}],
                               [False, False, True, True])
        self.assertGreater(head.predict({'x': 1e308}), 0.9)
        self.assertLess(head.predict({'x': -1e308}), 0.1)
        self.assertGreater(audit['epochs_run'], 0)
        self.assertLess(audit['training_log_loss'], 0.2)

    def test_constant_fractional_features_get_zero_coefficients(self):
        head, _ = fit_head([{'x': i, 'constant': 0.9} for i in range(40)],
                           [i > 20 for i in range(40)])
        self.assertEqual(head.weights['constant'], 0)

    def test_inconsistent_saved_calibration_is_rejected(self):
        for data in ({'calibration_count': 2, 'threshold': 0.2, 'full_frame': False},
                     {'calibration_count': 10, 'threshold': None, 'full_frame': True},
                     {'calibration_count': 10, 'threshold': float('nan'), 'full_frame': False}):
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                FrozenCalibrator(0.1, {'group': data})


class FittedSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        dataset, models = write_demo(cls.temp.name)
        cls.rows = load_dataset(dataset)
        cls.models = [ModelProfile(**m) for m in read_json(models)]
        cls.report = run_experiment(cls.rows, cls.models, joint_ghost='independent')
        test_ids = set(cls.report['splits']['test']['ids'])
        cls.query_rows = [{k: v for k, v in json.loads(line).items()
                           if k not in ('gold', 'split_group', 'corpus_present')}
                          for line in dataset.read_text().splitlines()
                          if json.loads(line)['id'] in test_ids]
        cls.queries = [parse_query(row) for row in cls.query_rows]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_system_reproduces_every_heldout_method_without_labels(self):
        path = Path(self.temp.name) / 'system.json'
        FittedSystem.from_report(self.report).save(path)
        system = FittedSystem.load(path)
        result = system.predict(self.queries)
        for record in result['predictions']:
            for method, prediction in record['methods'].items():
                expected = next(r for r in self.report['test_predictions'][method] if r['id'] == record['id'])
                for field in ('probabilities', 'prediction_set', 'decision', 'uncertainty'):
                    self.assertEqual(prediction[field], expected[field])
        self.assertEqual(result['generation']['attempted_calls'], 40 * 4 * 10)

    def test_artifact_fingerprint_detects_modified_coefficients(self):
        artifact = FittedSystem.from_report(self.report).artifact
        artifact['predictor']['heads']['P']['reliability']['intercept'] += 1
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            FittedSystem(artifact)

    def test_export_does_not_include_reference_answers_or_recorded_paths(self):
        artifact = FittedSystem.from_report(self.report).artifact
        text = json.dumps(artifact)
        for key in ('"gold"', '"corpus_present"', '"training_ids"', '"test_predictions"'):
            self.assertNotIn(key, text)
        self.assertIsInstance(artifact['configuration']['paths'], int)

    def test_labeled_examples_and_changed_backend_are_rejected(self):
        system = FittedSystem.from_report(self.report)
        with self.assertRaisesRegex(ValueError, 'label-free'):
            system.predict(self.rows[:1])
        class DifferentBackend:
            def generate(self, *args):
                raise AssertionError('must fail before generation')
        with self.assertRaisesRegex(ValueError, 'backend changed'):
            system.predict(self.queries[:1], generator=DifferentBackend())

    def test_call_budget_fails_before_any_backend_work(self):
        calls = []
        class Spy:
            def generate(self, *args):
                calls.append(args)
                return 'answer'
        with self.assertRaisesRegex(ValueError, 'planned calls'):
            run_experiment(self.rows, self.models, joint_ghost='independent', generator=Spy(), max_calls=3999)
        self.assertEqual(calls, [])

    def test_inference_budget_and_sparse_groups_are_conservative(self):
        system = FittedSystem.from_report(self.report)
        with self.assertRaisesRegex(ValueError, 'planned calls'):
            system.predict(self.queries[:1], max_calls=39)
        query = copy.deepcopy(self.queries[0])
        object.__setattr__(query, 'group', 'unseen-domain')
        result = system.predict([query], max_calls=40)
        for prediction in result['predictions'][0]['methods'].values():
            self.assertTrue(prediction['calibration']['full_frame'])
            self.assertTrue(prediction['decision']['abstained'])

    def test_cli_export_and_prediction(self):
        folder = Path(self.temp.name)
        report, artifact, queries, output = [folder / name for name in
                                             ('report.json', 'cli-system.json', 'queries.jsonl', 'predictions.json')]
        write_json(report, self.report)
        queries.write_text('\n'.join(json.dumps(q) for q in self.query_rows[:2]))
        subprocess.run([sys.executable, '-m', 'knowing_where.cli.experiment', 'export', str(report),
                        '--output', str(artifact)], check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, '-m', 'knowing_where.cli.experiment', 'predict', str(artifact),
                        str(queries), '--output', str(output), '--max-calls', '80'],
                       check=True, capture_output=True, text=True)
        self.assertEqual(len(read_json(output)['predictions']), 2)
        self.assertEqual(len(load_queries(queries)), 2)

    def test_cli_rejects_input_overwrite(self):
        report = Path(self.temp.name) / 'do-not-overwrite.json'
        write_json(report, self.report)
        original = report.read_bytes()
        result = subprocess.run([sys.executable, '-m', 'knowing_where.cli.experiment', 'export', str(report),
                                 '--output', str(report)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('cannot overwrite input', result.stderr)
        self.assertEqual(report.read_bytes(), original)

    def test_incomplete_replay_is_detected_before_generation(self):
        rows = copy.deepcopy(self.rows)
        rows[-1].paths[Source.M] = ()
        with patch.object(ReplayGenerator, 'generate') as generate:
            with self.assertRaisesRegex(ValueError, 'insufficient replay'):
                run_experiment(rows, self.models, joint_ghost='independent')
            generate.assert_not_called()
