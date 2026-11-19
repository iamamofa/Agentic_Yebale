"""Bayesian channel and Dempster-Shafer fusion on canonical answer IDs.

Both rules assume independent source evidence conditional on the answer.
Source correlation needs a separate model or evidence deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, fsum, isfinite, log
from typing import Mapping, Sequence

from .prediction import SourceEstimate
from ..types import OUTSIDE, Source, distribution, probability


class ImpossibleEvidenceError(ValueError):
    """All prior-supported answers have zero likelihood under the source model."""


@dataclass(frozen=True)
class SourceEvidence:
    source: Source
    answer: str
    estimate: SourceEstimate
    probabilities: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, Source):
            raise ValueError("source must be a Source enum member")
        if not isinstance(self.answer, str) or not self.answer.strip() or self.answer == OUTSIDE:
            raise ValueError("answer must be a nonempty canonical ID other than Ø")
        if self.probabilities is not None:
            object.__setattr__(self, "probabilities", distribution(self.probabilities, "source probabilities"))


def _frame(evidence: Sequence[SourceEvidence], candidates: Sequence[str] | None) -> tuple[str, ...]:
    if not evidence:
        raise ValueError("at least one source is required")
    if len({item.source for item in evidence}) != len(evidence):
        raise ValueError("duplicate sources would double count evidence")
    answers = list(candidates) if candidates is not None else list(dict.fromkeys(
        answer for item in evidence for answer in (item.answer, *(item.probabilities or {}))
        if answer != OUTSIDE
    ))
    if not answers or any(not isinstance(a, str) or not a.strip() or a == OUTSIDE for a in answers):
        raise ValueError("candidates must be nonempty canonical answer IDs excluding Ø")
    if len(set(answers)) != len(answers):
        raise ValueError("duplicate candidate IDs")
    frame = tuple([*answers, OUTSIDE])
    for item in evidence:
        if item.answer not in answers or (item.probabilities is not None and set(item.probabilities) - set(frame)):
            raise ValueError("source answer/distribution falls outside the frame")
    return frame


def joint_ghost_bounds(evidence: Sequence[SourceEvidence]) -> tuple[float, float]:
    """Frechet bounds on all consulted corpora missing the answer."""
    if not evidence:
        raise ValueError("at least one source is required")
    ghosts = [item.estimate.ghost_mass for item in evidence]
    return max(0.0, fsum(ghosts) - len(ghosts) + 1), min(ghosts)


def _logsumexp(values: Sequence[float]) -> float:
    peak = max(values)
    return peak + log(fsum(exp(value - peak) for value in values)) if peak != float("-inf") else peak


@dataclass(frozen=True)
class BayesianResult:
    posterior: Mapping[str, float]
    prior: Mapping[str, float]
    channels: Mapping[Source, Mapping[str, float]]
    log_evidence: float
    log_evidence_ratio: float
    joint_ghost_mass: float

    @property
    def outside_probability(self) -> float:
        return self.posterior[OUTSIDE]


def bayesian_fusion(
    evidence: Sequence[SourceEvidence],
    *,
    joint_ghost_mass: float,
    candidates: Sequence[str] | None = None,
    answer_prior: Mapping[str, float] | None = None,
    noise: Mapping[str, float] | None = None,
) -> BayesianResult:
    """Implement the reference channel product, normalized in log space.

    L_s(y) = (1-g_s)[r_s p_s(y) + (1-r_s)u(y)] + g_s u(y).
    p0(Ø)=gamma; remaining prior is uniform or answer_prior (summing to one).
    The joint ghost mass must be estimated explicitly; independence is never
    silently assumed. Gamma is a PRIOR, not a fixed posterior mass on Ø.
    Ø represents both unwritten truth and truth outside the supplied candidates;
    this initial model equates their prior masses. Candidate incompleteness
    beyond ghosts needs an expanded outside prior in a future model.

    A missing predictive distribution means a point mass at the source answer.
    Log evidence ratio is log Z_joint - sum_s log Z_s, a tension diagnostic.
    """
    frame = _frame(evidence, candidates)
    gamma = probability(joint_ghost_mass, "joint_ghost_mass")
    lower, upper = joint_ghost_bounds(evidence)
    if gamma < lower - 1e-12 or gamma > upper + 1e-12:
        raise ValueError(f"joint ghost mass must lie within [{lower}, {upper}]")
    answers = frame[:-1]
    base = distribution(answer_prior, "answer_prior") if answer_prior is not None else {
        answer: 1 / len(answers) for answer in answers
    }
    if set(base) != set(answers):
        raise ValueError("answer_prior must cover exactly the candidates")
    prior = {answer: (1 - gamma) * base[answer] for answer in answers}
    prior[OUTSIDE] = gamma
    u = distribution(noise, "noise") if noise is not None else {answer: 1 / len(frame) for answer in frame}
    if set(u) != set(frame):
        raise ValueError("noise must cover exactly the candidates plus Ø")
    channels = {}
    scores = {answer: log(mass) if mass else float("-inf") for answer, mass in prior.items()}
    individual_log_z = []
    for item in evidence:
        r, g = item.estimate.reliability, item.estimate.ghost_mass
        p = item.probabilities if item.probabilities is not None else {item.answer: 1.0}
        channel = {
            answer: (1 - g) * r * p.get(answer, 0.0) + ((1 - g) * (1 - r) + g) * u[answer]
            for answer in frame
        }
        channels[item.source] = channel
        single_scores = []
        for answer, mass in channel.items():
            log_mass = log(mass) if mass else float("-inf")
            scores[answer] += log_mass
            single_scores.append((log(prior[answer]) if prior[answer] else float("-inf")) + log_mass)
        individual_log_z.append(_logsumexp(single_scores))
    log_z = _logsumexp(list(scores.values()))
    if not isfinite(log_z):
        raise ImpossibleEvidenceError("evidence has zero probability under every prior-supported answer")
    posterior = {answer: exp(score - log_z) for answer, score in scores.items()}
    return BayesianResult(posterior, prior, channels, log_z, log_z - fsum(individual_log_z), gamma)


FocalSet = frozenset[str]


@dataclass(frozen=True)
class DSResult:
    masses: Mapping[FocalSet, float]
    belief: Mapping[str, float]
    plausibility: Mapping[str, float]
    pignistic: Mapping[str, float]
    conflict: float
    pairwise_conflict: Mapping[tuple[Source, Source], float]
    status: str


def dempster_shafer_fusion(
    evidence: Sequence[SourceEvidence], *, candidates: Sequence[str] | None = None,
) -> DSResult:
    """Simple support m_s({answer})=r_s, m_s(Theta)=1-r_s.

    Ghost mass is intentionally NOT used to discount r_s in this reference rule.
    This is an experimental comparator, not a ghost-aware posterior. Theta
    includes Ø, so a one-candidate frame still distinguishes belief/ignorance.
    Report pairwise conflict and total conjunctive conflict before normalization.
    Total conflict returns empty summaries with status='total_conflict'; never
    interpret that result as a probability distribution or an answer.
    """
    frame = _frame(evidence, candidates)
    theta = frozenset(frame)
    pairwise = {
        (a.source, b.source): a.estimate.reliability * b.estimate.reliability if a.answer != b.answer else 0.0
        for a, b in combinations(evidence, 2)
    }
    # Keep the empty set until the end: avoids hiding conflict by repeated
    # normalization and handles total conflict without division by zero.
    masses = {theta: 1.0}
    for item in evidence:
        r = item.estimate.reliability
        support = {frozenset([item.answer]): r, theta: 1 - r}
        updated: dict[FocalSet, float] = {}
        for left, left_mass in masses.items():
            for right, right_mass in support.items():
                intersection = left & right
                updated[intersection] = updated.get(intersection, 0.0) + left_mass * right_mass
        masses = updated
    conflict = min(1.0, max(0.0, masses.get(frozenset(), 0.0)))
    surviving = {focal: mass for focal, mass in masses.items() if focal and mass > 0}
    norm = fsum(surviving.values())
    if norm == 0:
        return DSResult({}, {}, {}, {}, conflict, pairwise, "total_conflict")
    combined = {focal: mass / norm for focal, mass in surviving.items()}
    belief = {answer: combined.get(frozenset([answer]), 0.0) for answer in frame}
    plausibility = {answer: fsum(mass for focal, mass in combined.items() if answer in focal) for answer in frame}
    pignistic = {answer: fsum(mass / len(focal) for focal, mass in combined.items() if answer in focal) for answer in frame}
    return DSResult(combined, belief, plausibility, pignistic, conflict, pairwise, "ok")
