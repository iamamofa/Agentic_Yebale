"""Portable fitted systems that predict new queries without ground-truth labels."""

from dataclasses import fields
import hashlib
import json
from math import isfinite

from ..evidence.features import KAPExtractor
from ..evidence.prediction import LinearKAPPredictor, LogisticHead, SourceHeads
from ..routing import ModelProfile, RoutingPolicy
from ..types import Source
from .data import Example, Query
from .decisions import FrozenCalibrator, decide
from .generation import ManagedGenerator, ReplayGenerator
from .io import read_json, write_json
from .pipeline import METHODS, collect_confidence, prepare_route, score_sources


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


class FittedSystem:
    """Load a frozen scoring pipeline and calibration thresholds; never refit on queries."""

    def __init__(self, artifact):
        self._artifact = json.loads(json.dumps(artifact, allow_nan=False))
        artifact = self._artifact
        if artifact.get('kind') != 'knowing_where_fitted_system' or artifact.get('schema_version') != 1:
            raise ValueError('unsupported fitted system artifact')
        payload = {k: v for k, v in artifact.items() if k != 'sha256'}
        if artifact.get('sha256') != _digest(payload):
            raise ValueError('fitted system fingerprint mismatch')
        config = artifact['configuration']
        if config['joint_ghost'] != 'independent':
            raise ValueError('unsupported joint ghost policy')
        if type(config['paths']) is not int or config['paths'] < 1 or type(config['seed']) is not int:
            raise ValueError('invalid generation configuration')
        if not isinstance(config['generation_backend'], str) or not config['generation_backend']:
            raise ValueError('missing generation backend identity')
        self.models = [ModelProfile(**m) for m in artifact['models']]
        if not self.models or len({m.id for m in self.models}) != len(self.models):
            raise ValueError('artifact requires unique model profiles')
        self.policy = RoutingPolicy(**artifact['routing_policy'])
        training = artifact['predictor']
        if set(training['heads']) != {s.value for s in Source}:
            raise ValueError('artifact must contain predictor heads for all sources')
        if any(not isfinite(v) for v in training['missing_values'].values()):
            raise ValueError('nonfinite imputation')
        heads = {Source(s): SourceHeads(LogisticHead(**h['reliability']), LogisticHead(**h['ghost_mass']))
                 for s, h in training['heads'].items()}
        self.predictor = LinearKAPPredictor(heads, training['missing_values'])
        expected_features = set(KAPExtractor().extract('Validate artifact feature schema').vector())
        if set(training['missing_values']) != expected_features:
            raise ValueError('artifact imputation feature schema differs from this pipeline')
        self.predictor.predict(KAPExtractor().extract('Validate artifact feature schema'))
        if set(artifact['calibration']) != set(METHODS):
            raise ValueError('artifact must contain calibration for every scoring method')
        self.calibrators = {name: FrozenCalibrator(config['alpha'], groups)
                            for name, groups in artifact['calibration'].items()}

    @property
    def artifact(self):
        return json.loads(json.dumps(self._artifact))

    @classmethod
    def from_report(cls, report):
        if report.get('schema_version') != 2:
            raise ValueError('export requires a version 2 experiment report; rerun the experiment')
        config = report['configuration']
        allowed = {f.name for f in fields(ModelProfile)}
        payload = {
            'schema_version': 1, 'kind': 'knowing_where_fitted_system',
            'configuration': {'seed': config['seed'], 'alpha': config['alpha'],
                              'paths': config['paths_per_source'], 'joint_ghost': 'independent',
                              'generation_backend': config['generation_backend'],
                              'generation_backend_id': config['generation_backend_id']},
            'models': [{k: v for k, v in m.items() if k in allowed} for m in report['models']],
            'routing_policy': config['routing_policy'],
            'predictor': {k: report['training'][k] for k in ('heads', 'missing_values')},
            'calibration': report['calibration'],
            'provenance': {'training_dataset_sha256': report['dataset_sha256'],
                           'training_count': len(report['training']['training_ids'])},
        }
        return cls({**payload, 'sha256': _digest(payload)})

    @classmethod
    def load(cls, path):
        return cls(read_json(path))

    def save(self, path):
        write_json(path, self._artifact)

    def predict(self, queries, *, generator=None, compiler=None, max_calls=None, max_retries=0):
        """Predict a batch of Query objects using saved coefficients and thresholds.

        A different backend is rejected because it changes the calibrated scoring
        pipeline. Recalibrate a new experiment when changing generation backends.
        """
        queries = tuple(queries)
        if not queries or any(not isinstance(q, Query) or isinstance(q, Example) for q in queries):
            raise ValueError('predict requires label-free Query inputs, not labeled Examples')
        if len({q.id for q in queries}) != len(queries):
            raise ValueError('prediction IDs must be unique')
        generator = generator if generator is not None else ReplayGenerator({q.id: q.paths for q in queries})
        config = self._artifact['configuration']
        if f'{type(generator).__module__}.{type(generator).__qualname__}' != config['generation_backend_id']:
            raise ValueError('generation backend changed; run a new calibration experiment')
        managed = ManagedGenerator(generator, max_calls=max_calls, max_retries=max_retries)
        prepared = [(q, *prepare_route(q, self.models, self.policy, compiler)) for q in queries]
        planned = sum(len(requests) * config['paths'] for _, _, requests in prepared)
        managed.preflight(planned)
        if type(generator) is ReplayGenerator:
            generator.validate((r for _, _, requests in prepared for r in requests), config['paths'])
        records = []
        for query, routing, requests in prepared:
            features = KAPExtractor().extract(query.question, query.metadata)
            confidence = collect_confidence(requests, managed, paths=config['paths'], seed=config['seed'])
            observation = score_sources(features, confidence, self.predictor)
            methods = {}
            for method in METHODS:
                probabilities = observation['scores'][method]
                calibrator = self.calibrators[method]
                prediction_set = calibrator.predict(probabilities, query.group)
                methods[method] = {
                    'probabilities': probabilities, 'prediction_set': list(prediction_set),
                    'decision': decide(prediction_set,
                                       total_conflict=method in ('bayesian', 'dempster_shafer') and
                                       observation['diagnostics'].get('ds_status') == 'total_conflict',
                                       routed=routing['selected_model_id'] is not None),
                    'calibration': {'group': query.group, 'seen_group': query.group in calibrator.groups,
                                    'full_frame': calibrator.threshold(query.group) == float('inf')},
                }
            records.append({'id': query.id, 'group': query.group, 'methods': methods,
                            'audit': {'routing': routing, **{k: v for k, v in observation.items() if k != 'scores'}}})
        return {'schema_version': 1, 'artifact_sha256': self._artifact['sha256'],
                'generation': managed.summary(planned), 'predictions': records}
