"""Held-out prediction, abstention, probability, and set metrics."""

from collections import Counter
from math import log

from ..types import OUTSIDE
from ..evidence.uncertainty import predictive_uncertainty


def evaluate(records):
    if not records:
        raise ValueError('metrics require held-out records')
    n = len(records)
    answered = [r for r in records if not r['decision']['abstained']]
    correct = sum(r['decision']['answer'] == r['gold'] for r in answered)
    bins = [[] for _ in range(10)]
    for r in records:
        p = r['probabilities']
        prediction = min(p, key=lambda label: (-p[label], label))
        confidence = p[prediction]
        bins[min(9, int(confidence * 10))].append((confidence, prediction == r['target']))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b))
              for b in bins if b)
    routed = [r for r in records if r.get('routed', True)]
    uncertainty = [r.get('uncertainty') or predictive_uncertainty(r['probabilities']) for r in records]
    available = [u for u in uncertainty if u['available']]
    return {
        'uncertainty_available_count': len(available),
        'mean_normalized_entropy': sum(u['normalized_entropy'] for u in available) / len(available) if available else None,
        'mean_top_two_margin': sum(u['top_two_margin'] for u in available) / len(available) if available else None,
        'answered_count': len(answered),
        'correct_answer_count': correct,
        'routed_count': len(routed),
        'unrouted_count': n - len(routed),
        'routing_success_rate': len(routed) / n,
        'mapped_point_accuracy': sum(min(r['probabilities'], key=lambda a: (-r['probabilities'][a], a)) == r['target']
                                     for r in records) / n,
        'routed_mapped_set_coverage': sum(r['target'] in r['prediction_set'] for r in routed) / len(routed) if routed else None,
        'routed_canonical_answer_set_coverage': sum(r['gold'] in r['prediction_set'] for r in routed) / len(routed) if routed else None,
        'decision_reasons': dict(sorted(Counter(r['decision']['reason'] for r in records).items())),
        'set_size_histogram': dict(sorted(Counter(str(len(r['prediction_set'])) for r in records).items())),
        'count': n,
        'point_accuracy': sum(min(r['probabilities'], key=lambda a: (-r['probabilities'][a], a)) == r['gold']
                              for r in records) / n,
        'candidate_recall': sum(r['gold'] in r['probabilities'] for r in records) / n,
        'answer_rate': len(answered) / n,
        'abstention_rate': 1 - len(answered) / n,
        'selective_accuracy': correct / len(answered) if answered else None,
        'selective_risk': 1 - correct / len(answered) if answered else None,
        'correct_answer_rate': correct / n,
        'mapped_set_coverage': sum(r['target'] in r['prediction_set'] for r in records) / n,
        'canonical_answer_set_coverage': sum(r['gold'] in r['prediction_set'] for r in records) / n,
        'mean_set_size': sum(len(r['prediction_set']) for r in records) / n,
        'outside_in_set_rate': sum(OUTSIDE in r['prediction_set'] for r in records) / n,
        'negative_log_likelihood': -sum(log(max(r['probabilities'][r['target']], 1e-15)) for r in records) / n,
        'multiclass_brier': sum(sum((p - float(label == r['target'])) ** 2
                                    for label, p in r['probabilities'].items()) for r in records) / n,
        'ece_10_bins': ece,
    }
