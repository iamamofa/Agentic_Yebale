"""Shared label-free routing, generation handoffs, and source scoring."""

from dataclasses import asdict
import hashlib
import json
from math import prod

from ..evidence.fusion import ImpossibleEvidenceError, bayesian_fusion, dempster_shafer_fusion, joint_ghost_bounds
from ..evidence.uncertainty import predictive_uncertainty, source_sensitivity, source_uncertainty
from ..routing import route_context
from ..types import OUTSIDE, Source
from .generation import GenerationRequest, generate_paths


METHODS = ('bayesian', 'dempster_shafer', 'equal_sources', 'majority', 'parametric_only')


def _normalize(values):
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def prepare_route(row, models, policy, compiler=None):
    route = route_context(row.context_request(), models, policy=policy, compiler=compiler)
    selected = next((c for c in route.candidates if c.model_id == route.selected_model_id), None)
    audit = {'selected_model_id': route.selected_model_id, 'rejected': dict(route.rejected),
             'reason': route.decision_reason}
    if selected is None:
        return audit, ()
    packet = json.loads(selected.packet.payload)
    sources = {d['id']: d['source'] for d in packet['documents']}
    audit['context_audit'] = {
        'used_units': selected.packet.used_units, 'budget_units': selected.packet.budget_units,
        'fingerprint': hashlib.sha256(selected.packet.payload.encode()).hexdigest(),
        'query_term_coverage': selected.packet.query_term_coverage,
        'omissions': list(selected.packet.omissions),
    }
    requests = []
    for source in Source:
        excerpts = tuple(e for e in packet['excerpts'] if sources[e['document_id']] == source.value)
        documents = tuple(d for d in packet['documents'] if d['source'] == source.value)
        claims = []
        for conflict in packet['conflicts']:
            for alternative in conflict['alternatives']:
                ids = [id_ for id_ in alternative['document_ids'] if sources[id_] == source.value]
                if ids:
                    claims.append({'key': conflict['key'], 'value': alternative['value'], 'document_ids': ids})
        requests.append(GenerationRequest(row.id, row.question, source, selected.model_id,
                                         row.instructions, tuple(e['text'] for e in excerpts),
                                         excerpts, documents, tuple(claims), packet['evidence_policy']))
    return audit, tuple(requests)


def collect_confidence(requests, generator, *, paths, seed):
    result = {}
    for request in requests:
        path_seed = int.from_bytes(hashlib.sha256(
            f'{seed}:{request.example_id}:{request.source.value}'.encode()).digest()[:8], 'big')
        result[request.source] = generate_paths(generator, request, paths=paths, seed=path_seed)
    return result or None


def score_sources(features, confidence, predictor):
    if confidence is None:
        return {'scores': {name: {OUTSIDE: 1.0} for name in METHODS},
                'diagnostics': {'no_feasible_model': True},
                'uncertainty': {name: predictive_uncertainty({OUTSIDE: 1.0},
                                unavailable_reason='no_feasible_model') for name in METHODS},
                'source_uncertainty': {'available': False, 'reason': 'no_source_paths'},
                'source_sensitivity': {'available': False, 'reason': 'no_feasible_model', 'sources': {}}}
    estimates = predictor.predict(features)
    evidence = [confidence[s].evidence(s, estimates[s]) for s in Source]
    gamma = prod(e.estimate.ghost_mass for e in evidence)
    ds = dempster_shafer_fusion(evidence)
    frame = tuple(dict.fromkeys(answer for e in evidence for answer in (e.answer, *(e.probabilities or {})))) + (OUTSIDE,)
    try:
        bayes = bayesian_fusion(evidence, joint_ghost_mass=gamma)
    except ImpossibleEvidenceError:
        bayes = None
    bayes_scores = dict(bayes.posterior) if bayes is not None else {a: float(a == OUTSIDE) for a in frame}
    def endpoint_outside(value):
        try:
            return bayesian_fusion(evidence, joint_ghost_mass=value).outside_probability
        except ImpossibleEvidenceError:
            return None
    equal = {a: sum(e.probabilities.get(a, 0) for e in evidence) / len(evidence) for a in frame}
    majority = {a: sum(e.answer == a for e in evidence) / len(evidence) for a in frame}
    parametric = {a: confidence[Source.P].probabilities.get(a, 0) for a in frame}
    # Total conflict has no DS probability summary: expose an outside-only abstention score.
    ds_scores = dict(ds.pignistic) if ds.status == 'ok' else {a: float(a == OUTSIDE) for a in frame}
    lower, upper = joint_ghost_bounds(evidence)
    scores = {'bayesian': bayes_scores, 'dempster_shafer': ds_scores,
              'equal_sources': _normalize(equal), 'majority': _normalize(majority),
              'parametric_only': _normalize(parametric)}
    return {
        'scores': scores,
        'uncertainty': {name: predictive_uncertainty(p, unavailable_reason=(
            'zero_evidence' if name == 'bayesian' and bayes is None else
            'total_conflict' if name == 'dempster_shafer' and ds.status != 'ok' else None))
                        for name, p in scores.items()},
        'source_uncertainty': source_uncertainty({s: c.probabilities for s, c in confidence.items()},
                                                 path_counts={s: len(c.paths) for s, c in confidence.items()}),
        'source_sensitivity': source_sensitivity(evidence, bayes.posterior if bayes is not None else None),
        'diagnostics': {
            'joint_ghost_prior': gamma, 'joint_ghost_bounds': [lower, upper],
            'outside_posterior_at_ghost_bounds': [
                endpoint_outside(value) for value in (lower, upper)],
            'bayesian_status': 'ok' if bayes is not None else 'zero_evidence',
            'log_evidence_ratio': bayes.log_evidence_ratio if bayes is not None else None,
            'ds_conflict': ds.conflict, 'ds_status': ds.status,
            'pairwise_conflict': {f'{a.value}/{b.value}': value for (a, b), value in ds.pairwise_conflict.items()},
        },
        'source_estimates': {s.value: asdict(e) for s, e in estimates.items()},
    }
