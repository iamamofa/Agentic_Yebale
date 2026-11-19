"""One connected, auditable train/calibrate/test experiment."""

from dataclasses import asdict
import hashlib
import json

from ..evidence.features import KAPExtractor
from ..routing import RoutingPolicy
from ..types import OUTSIDE, Source, probability
from .data import split_dataset
from .decisions import MondrianCalibrator, decide
from .generation import ManagedGenerator, ReplayGenerator
from .pipeline import METHODS, collect_confidence, prepare_route, score_sources
from .metrics import evaluate
from .training import train_predictor



def run_experiment(rows, models, *, joint_ghost, seed=42, alpha=0.1, paths=10,
                   policy=None, generator=None, compiler=None, max_calls=None, max_retries=0):
    """Fit only train; calibrate only calibration; report only test performance.

    joint_ghost must explicitly select the independence approximation. Source
    probabilities describe path votes, not calibrated correctness confidence.
    """
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    if joint_ghost != 'independent':
        raise ValueError('explicitly select joint_ghost="independent" or supply a future joint model')
    if not 0 < probability(alpha, 'alpha') < 1:
        raise ValueError('alpha must be strictly between zero and one')
    if type(paths) is not int or paths < 1:
        raise ValueError('paths must be a positive integer')
    partitions = split_dataset(rows, seed)
    generator = generator if generator is not None else ReplayGenerator({r.id: r.paths for r in rows})
    extractor = KAPExtractor()
    features, confidence, routing, requests = {}, {}, {}, {}
    ordered = [r for name in ('train', 'calibration', 'test') for r in partitions[name]]
    for row in ordered:
        features[row.id] = extractor.extract(row.question, row.metadata)
        routing[row.id], requests[row.id] = prepare_route(row, models, policy or RoutingPolicy(), compiler)
    train_rows = [r for r in partitions['train'] if requests[r.id]]
    if not train_rows:
        raise ValueError('no routed training examples; relax the routing policy or add an eligible model')
    for source in Source:
        if not any(r.corpus_present[source] for r in train_rows):
            raise ValueError(f'no corpus-present training examples for source {source.value}')
    planned_calls = sum(len(value) * paths for value in requests.values())
    managed = ManagedGenerator(generator, max_calls=max_calls, max_retries=max_retries)
    managed.preflight(planned_calls)
    # Validate replay completeness before any generation or partial experiment work.
    if type(generator) is ReplayGenerator:
        generator.validate((request for group in requests.values() for request in group), paths)
    # Fit and freeze before generating calibration or test outputs.
    for row in partitions['train']:
        confidence[row.id] = collect_confidence(requests[row.id], managed, paths=paths, seed=seed)
    predictor, artifact = train_predictor(train_rows, features, confidence)
    for row in (*partitions['calibration'], *partitions['test']):
        confidence[row.id] = collect_confidence(requests[row.id], managed, paths=paths, seed=seed)
    observations = {row.id: score_sources(features[row.id], confidence[row.id], predictor) for row in ordered}
    calibration, comparisons, predictions = {}, {}, {}
    for method in METHODS:
        calibration_rows = []
        for row in partitions['calibration']:
            scores = observations[row.id]['scores'][method]
            target = row.gold if row.gold in scores else OUTSIDE
            calibration_rows.append((row.group, scores, target))
        calibrator = MondrianCalibrator.fit(calibration_rows, alpha)
        calibration[method] = calibrator.summary()
        records = []
        for row in partitions['test']:
            observation = observations[row.id]
            scores = observation['scores'][method]
            prediction_set = calibrator.predict(scores, row.group)
            decision = decide(prediction_set,
                              total_conflict=method in ('bayesian', 'dempster_shafer') and
                              observation['diagnostics'].get('ds_status') == 'total_conflict',
                              routed=routing[row.id]['selected_model_id'] is not None)
            records.append({'id': row.id, 'group': row.group, 'gold': row.gold,
                            'target': row.gold if row.gold in scores else OUTSIDE,
                            'probabilities': scores, 'prediction_set': list(prediction_set), 'decision': decision,
                            'routed': routing[row.id]['selected_model_id'] is not None})
        predictions[method] = records
        comparisons[method] = {'overall': evaluate(records), 'groups': {
            g: evaluate([r for r in records if r['group'] == g]) for g in sorted({r['group'] for r in records})}}
    canonical_data = [{**asdict(r), 'paths': {s.value: p for s, p in r.paths.items()},
                       'corpus_present': {s.value: v for s, v in r.corpus_present.items()}} for r in sorted(rows, key=lambda r: r.id)]
    return {
        'schema_version': 2,
        'configuration': {'seed': seed, 'alpha': alpha, 'paths_per_source': paths,
                          'joint_ghost': 'product under explicit corpus-absence independence assumption',
                          'routing_policy': asdict(policy or RoutingPolicy()),
                          'generation_backend_id': f'{type(generator).__module__}.{type(generator).__qualname__}',
                          'generation_backend': type(generator).__name__},
        'dataset_sha256': hashlib.sha256(json.dumps(canonical_data, sort_keys=True).encode()).hexdigest(),
        'models': [m.summary() for m in models],
        'splits': {name: {'count': len(part), 'ids': [r.id for r in part],
                          'split_groups': sorted({r.split_group for r in part})} for name, part in partitions.items()},
        'training': artifact,
        'generation': managed.summary(planned_calls),
        'calibration': calibration,
        'quality_summary': {
            'excluded_training_ids': [r.id for r in partitions['train'] if confidence[r.id] is None],
            'unseen_test_groups': sorted({r.group for r in partitions['test']} -
                                        {r.group for r in partitions['calibration']}),
            'full_frame_calibration_groups': {name: [g for g, c in groups.items() if c['full_frame']]
                                               for name, groups in calibration.items()},
            'methods_with_no_test_answers': [name for name, c in comparisons.items()
                                             if c['overall']['answered_count'] == 0],
        },
        'comparisons': comparisons,
        'test_predictions': predictions,
        'test_audit': {r.id: {'routing': routing[r.id], **{k: v for k, v in observations[r.id].items() if k != 'scores'},
                            'path_confidence': {s.value: asdict(c) for s, c in (confidence[r.id] or {}).items()}}
                       for r in partitions['test']},
        'limitations': [
            'Replay experiments measure recorded paths; they do not call or benchmark a live model.',
            'Path agreement is a sampling statistic, not calibrated correctness probability.',
            'Repeated paths are aggregated within a source, never treated as independent sources.',
            'Conformal coverage concerns labels mapped into the generated frame, with missing gold mapped to Ø.',
            'Group coverage requires exchangeability within fixed groups and a frozen scoring pipeline.',
            'Source independence and corpus-absence independence are modeling assumptions.',
            'Sparse or unseen calibration groups return the entire frame and can force abstention.',
            'No benchmark or deployment coverage guarantee follows from synthetic results.',
        ],
    }
