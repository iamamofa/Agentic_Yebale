"""Group-conditional split conformal sets and explicit abstention decisions."""

from dataclasses import dataclass
from math import ceil

from ..types import OUTSIDE, distribution, probability


@dataclass(frozen=True)
class MondrianCalibrator:
    alpha: float
    scores: dict[str, tuple[float, ...]]

    @classmethod
    def fit(cls, observations, alpha=0.1):
        if not 0 < probability(alpha, 'alpha') < 1:
            raise ValueError('alpha must be strictly between zero and one')
        grouped = {}
        for group, probabilities, label in observations:
            probabilities = distribution(probabilities, 'calibration probabilities')
            if label not in probabilities:
                raise ValueError('calibration truth must be represented in the candidate frame')
            grouped.setdefault(group, []).append(1 - probabilities[label])
        return cls(alpha, {g: tuple(sorted(scores)) for g, scores in grouped.items()})

    def threshold(self, group):
        scores = self.scores.get(group, ())
        rank = ceil((len(scores) + 1) * (1 - self.alpha))
        # Missing/sparse groups retain the entire frame, never a pooled fallback.
        return scores[rank - 1] if rank <= len(scores) else float('inf')

    def predict(self, probabilities, group):
        probabilities = distribution(probabilities, 'prediction probabilities')
        threshold = self.threshold(group)
        return tuple(sorted(label for label, p in probabilities.items() if 1 - p <= threshold))

    def summary(self):
        return {group: {'calibration_count': len(scores),
                        'threshold': None if self.threshold(group) == float('inf') else self.threshold(group),
                        'full_frame': self.threshold(group) == float('inf')}
                for group, scores in sorted(self.scores.items())}


def decide(prediction_set, *, total_conflict=False, routed=True):
    """Answer only a singleton canonical candidate; Ø always means abstain."""
    if not routed:
        return {'answer': None, 'abstained': True, 'reason': 'no_feasible_model'}
    if total_conflict:
        return {'answer': None, 'abstained': True, 'reason': 'total_conflict'}
    if len(prediction_set) == 1 and prediction_set[0] != OUTSIDE:
        return {'answer': prediction_set[0], 'abstained': False, 'reason': 'singleton_set'}
    reason = 'outside_in_set' if OUTSIDE in prediction_set else 'empty_set' if not prediction_set else 'ambiguous_set'
    return {'answer': None, 'abstained': True, 'reason': reason}


@dataclass(frozen=True)
class FrozenCalibrator:
    """Validated, portable group thresholds for prediction without calibration labels."""

    alpha: float
    groups: dict

    def __post_init__(self):
        if not 0 < probability(self.alpha, 'alpha') < 1:
            raise ValueError('alpha must be strictly between zero and one')
        if not isinstance(self.groups, dict):
            raise ValueError('calibration groups must be an object')
        for group, data in self.groups.items():
            if not isinstance(group, str) or not group.strip() or not isinstance(data, dict):
                raise ValueError('invalid calibration group')
            if set(data) != {'calibration_count', 'threshold', 'full_frame'}:
                raise ValueError('invalid calibration summary fields')
            n = data['calibration_count']
            if type(n) is not int or n < 1:
                raise ValueError('calibration count must be a positive integer')
            full = ceil((n + 1) * (1 - self.alpha)) > n
            if type(data['full_frame']) is not bool or data['full_frame'] != full:
                raise ValueError('full-frame flag is inconsistent with calibration count and alpha')
            if full:
                if data['threshold'] is not None:
                    raise ValueError('full-frame threshold must be null')
            else:
                probability(data['threshold'], 'calibration threshold')

    def threshold(self, group):
        data = self.groups.get(group)
        return float('inf') if data is None or data['full_frame'] else data['threshold']

    def predict(self, probabilities, group):
        probabilities = distribution(probabilities, 'prediction probabilities')
        threshold = self.threshold(group)
        return tuple(sorted(label for label, p in probabilities.items() if 1 - p <= threshold))
