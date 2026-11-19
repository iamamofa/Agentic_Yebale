"""Descriptive uncertainty diagnostics; none is an error probability estimate."""

from collections import Counter
from math import fsum, log2, prod

from ..types import OUTSIDE, distribution
from .fusion import ImpossibleEvidenceError, bayesian_fusion


def entropy_bits(probabilities):
    values = distribution(probabilities, 'uncertainty distribution')
    return -fsum(p * log2(p) for p in values.values() if p > 0)


def predictive_uncertainty(probabilities, *, unavailable_reason=None):
    """Entropy over the entire declared frame, including zero-mass candidates.

    Zero entropy describes concentration, not correctness or sufficient evidence.
    Synthetic abstention scores from failed routing/fusion are unavailable.
    """
    values = distribution(probabilities, 'prediction probabilities')
    fields = ('entropy_bits', 'normalized_entropy', 'effective_candidates', 'top_probability',
              'runner_up_probability', 'top_two_margin', 'outside_probability', 'top_candidate', 'variation_ratio')
    if unavailable_reason:
        return {'available': False, 'reason': unavailable_reason, 'candidate_count': len(values),
                **{name: None for name in fields}}
    ranked = sorted(values, key=lambda name: (-values[name], name))
    top = values[ranked[0]]
    second = values[ranked[1]] if len(ranked) > 1 else 0.0
    entropy = max(0.0, entropy_bits(values))
    return {'available': True, 'reason': None, 'candidate_count': len(values),
            'entropy_bits': entropy,
            'normalized_entropy': min(1.0, entropy / log2(len(values))) if len(values) > 1 else 0.0,
            'effective_candidates': 2 ** entropy,
            'top_probability': top, 'runner_up_probability': second,
            'top_two_margin': top - second, 'outside_probability': values.get(OUTSIDE, 0.0),
            'top_candidate': ranked[0], 'variation_ratio': 1 - top}


def source_uncertainty(distributions, *, path_counts=None):
    """Equal-source mixture entropy = mean source entropy + Jensen-Shannon gap.

    Sources are evidence channels, not posterior samples of model parameters.
    Call the terms within-source ambiguity and between-source disagreement,
    rather than claiming an identified aleatoric/epistemic decomposition.
    """
    if not distributions:
        return {'available': False, 'reason': 'no_source_paths'}
    values = {str(getattr(s, 'value', s)): distribution(p, 'source probabilities')
              for s, p in distributions.items()}
    candidates = sorted({a for p in values.values() for a in p})
    counts = {str(getattr(s, 'value', s)): count for s, count in (path_counts or {}).items()}
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError('path counts must be nonnegative integers')
    n = len(values)
    mixture = {a: fsum(p.get(a, 0) / n for p in values.values()) for a in candidates}
    within = fsum(entropy_bits(p) / n for p in values.values())
    total = entropy_bits(mixture)
    disagreement = max(0.0, total - within)
    bound = log2(min(n, len(candidates))) if min(n, len(candidates)) > 1 else 0.0
    tops = [min(p, key=lambda a: (-p[a], a)) for p in values.values()]
    return {'available': True, 'source_count': n, 'candidate_count': len(candidates),
            'mixture_probabilities': mixture, 'mixture_entropy_bits': max(0.0, total),
            'within_source_entropy_bits': max(0.0, within),
            'between_source_disagreement_bits': disagreement,
            'normalized_disagreement': min(1.0, disagreement / bound) if bound else 0.0,
            'majority_vote_disagreement': 1 - max(Counter(tops).values()) / n,
            'sources': {s: {**predictive_uncertainty({a: p.get(a, 0) for a in candidates}),
                            'probabilities': p, 'path_count': counts.get(s)} for s, p in values.items()}}


def source_sensitivity(evidence, posterior):
    """Bayesian leave-one-source-out sensitivity using a fixed candidate frame.

    The independence-product ghost prior is recomputed for each subset. This
    measures sensitivity under that model; it is not causal source importance.
    """
    if posterior is None:
        return {'available': False, 'reason': 'invalid_bayesian_posterior', 'sources': {}}
    candidates = tuple(a for a in posterior if a != OUTSIDE)
    top = min(posterior, key=lambda a: (-posterior[a], a))
    rows = {}
    for removed in evidence:
        remaining = [e for e in evidence if e.source != removed.source]
        if not remaining:
            rows[removed.source.value] = {'available': False, 'reason': 'no_remaining_sources'}
            continue
        try:
            result = bayesian_fusion(remaining, candidates=candidates,
                                     joint_ghost_mass=prod(e.estimate.ghost_mass for e in remaining))
        except ImpossibleEvidenceError:
            rows[removed.source.value] = {'available': False, 'reason': 'zero_evidence'}
            continue
        p = result.posterior
        new_top = min(p, key=lambda a: (-p[a], a))
        rows[removed.source.value] = {
            'available': True, 'total_variation': min(1.0, fsum(abs(p[a] - posterior[a]) for a in p) / 2),
            'top_candidate_without_source': new_top, 'top_candidate_changed': new_top != top,
            'original_top_probability_change': p[top] - posterior[top],
            'outside_probability_change': p[OUTSIDE] - posterior[OUTSIDE],
        }
    return {'available': True, 'method': 'bayesian', 'sources': rows}


def risk_coverage_curve(records):
    """Held-out point-answer review curve, grouped at tied entropy thresholds.

    Does not change conformal decisions. Excludes unrouted/invalid scores and
    outside-option top predictions. Coverage uses all test examples as denominator.
    """
    eligible = []
    for row in records:
        uncertainty = row.get('uncertainty') or predictive_uncertainty(row['probabilities'])
        if not row.get('routed', True) or not uncertainty['available'] or uncertainty['top_candidate'] == OUTSIDE:
            continue
        eligible.append((uncertainty['normalized_entropy'], uncertainty['top_candidate'] == row['gold']))
    eligible.sort(key=lambda item: item[0])
    points, count, errors = [], 0, 0
    index = 0
    while index < len(eligible):
        threshold = eligible[index][0]
        while index < len(eligible) and eligible[index][0] == threshold:
            count += 1
            errors += not eligible[index][1]
            index += 1
        points.append({'entropy_threshold': threshold, 'coverage': count / len(records),
                       'error_rate': errors / count, 'reviewed_count': count})
    return points
