"""Question-side extraction only: no answer, labels, or full-document inputs.

Adapters must return only cheap metadata. The extractor enforces one web-probe
invocation, but the adapter owner is responsible for its network-call budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
import re
from typing import Callable, Mapping, Sequence

from ..types import distribution, probability


@dataclass(frozen=True)
class QuestionQuality:
    scope_fit: float | None = None
    well_posedness: float | None = None
    guardrail_exposure: float | None = None

    def __post_init__(self) -> None:
        for name in ("scope_fit", "well_posedness", "guardrail_exposure"):
            value = getattr(self, name)
            if value is not None:
                probability(value, name)


@dataclass(frozen=True)
class ParametricMetadata:
    popularity_decile: int | None = None  # 1 = least popular, 10 = most
    entity_perplexity: float | None = None

    def __post_init__(self) -> None:
        if self.popularity_decile is not None and (
            type(self.popularity_decile) is not int or not 1 <= self.popularity_decile <= 10
        ):
            raise ValueError("popularity_decile must be an integer from 1 to 10")
        if self.entity_perplexity is not None and (
            not isfinite(self.entity_perplexity) or self.entity_perplexity < 1
        ):
            raise ValueError("entity_perplexity must be finite and >= 1")


@dataclass(frozen=True)
class WebMetadata:
    total_hits: int
    snippets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _count(self.total_hits, "total_hits")
        if len(self.snippets) > self.total_hits:
            raise ValueError("snippet count cannot exceed total_hits")


@dataclass(frozen=True)
class KnowledgeMetadata:
    matched: int
    checked: int

    def __post_init__(self) -> None:
        _count(self.matched, "matched")
        _count(self.checked, "checked")
        if self.matched > self.checked:
            raise ValueError("matched cannot exceed checked")


@dataclass(frozen=True)
class MemoryMetadata:
    hits: int
    checked: int
    successes: int
    evaluated: int

    def __post_init__(self) -> None:
        for name in ("hits", "checked", "successes", "evaluated"):
            _count(getattr(self, name), name)
        if self.hits > self.checked or self.successes > self.evaluated:
            raise ValueError("memory numerator exceeds denominator")


def _count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class HopEstimate:
    probabilities: Mapping[int, float]
    subquestions: tuple[str, ...] = ()
    method: str = "metadata"

    def __post_init__(self) -> None:
        for hop in self.probabilities:
            _count(hop, "hop")
        object.__setattr__(self, "probabilities", distribution(self.probabilities, "p(d|q)"))

    @property
    def mean(self) -> float:
        return sum(hop * mass for hop, mass in self.probabilities.items())


@dataclass(frozen=True)
class QueryMetadata:
    """Allowlisted, available-at-query-time metadata; never benchmark outcomes.

    Historical memory outcomes must predate the query and exclude held-out runs.
    Hop metadata must be a query-side estimate, not a gold benchmark hop label.
    """

    quality: QuestionQuality | None = None
    parametric: ParametricMetadata | None = None
    web: WebMetadata | None = None
    knowledge: KnowledgeMetadata | None = None
    memory: MemoryMetadata | None = None
    hops: HopEstimate | None = None


@dataclass(frozen=True)
class KAPFeatures:
    question: str
    omega: QuestionQuality
    rho_p: ParametricMetadata | None
    rho_w: WebMetadata | None
    snippet_agreement: float | None
    rho_k: KnowledgeMetadata | None
    rho_m: MemoryMetadata | None
    distance: HopEstimate
    provenance: Mapping[str, str] = field(default_factory=dict)

    def vector(self) -> dict[str, float | None]:
        """Stable numeric feature names; missing values stay explicitly missing."""
        p, w, k, m = self.rho_p, self.rho_w, self.rho_k, self.rho_m
        return {
            "scope_fit": self.omega.scope_fit,
            "well_posedness": self.omega.well_posedness,
            "guardrail_exposure": self.omega.guardrail_exposure,
            "popularity": (p.popularity_decile - 1) / 9 if p and p.popularity_decile is not None else None,
            "inverse_perplexity": 1 / p.entity_perplexity if p and p.entity_perplexity is not None else None,
            "web_hits": float(w.total_hits) if w is not None else None,
            "snippet_agreement": self.snippet_agreement,
            "knowledge_coverage": k.matched / k.checked if k and k.checked else None,
            "memory_hit_rate": m.hits / m.checked if m and m.checked else None,
            "memory_success_rate": m.successes / m.evaluated if m and m.evaluated else None,
            "distance_mean": self.distance.mean,
            "distance_zero": self.distance.probabilities.get(0, 0.0),
        }


def lexical_snippet_agreement(snippets: Sequence[str]) -> float | None:
    """Mean pairwise token Jaccard, a lexical proxy, NOT factual agreement.

    Fewer than two nonempty snippets cannot establish agreement. A semantic
    agreement scorer can be injected when one has been independently validated.
    """
    tokens = [set(re.findall(r"\w+", snippet.casefold())) for snippet in snippets]
    tokens = [item for item in tokens if item]
    pairs = [len(a & b) / len(a | b) for i, a in enumerate(tokens) for b in tokens[i + 1:]]
    return sum(pairs) / len(pairs) if pairs else None


def heuristic_decomposition(question: str) -> HopEstimate:
    """Untrained fallback: split explicit clauses, with uncertainty of +/-1 hop.

    A direct lookup is d=0; connecting n lookup clauses requires n-1 links.
    This is only a reproducible placeholder for a query-only decomposer.
    """
    clauses = tuple(part.strip(" ?.") for part in re.split(
        r"\s+(?:and then|then|and)\s+|;", question, flags=re.IGNORECASE
    ) if part.strip(" ?."))
    hop = max(0, len(clauses) - 1)
    masses = {hop: 0.6, hop + 1: 0.2}
    lower = max(0, hop - 1)
    masses[lower] = masses.get(lower, 0) + 0.2
    return HopEstimate(masses, clauses, "untrained_clause_heuristic")


@dataclass
class KAPExtractor:
    quality_probe: Callable[[str], QuestionQuality] | None = None
    parametric_probe: Callable[[str], ParametricMetadata] | None = None
    web_probe: Callable[[str], WebMetadata] | None = None
    knowledge_probe: Callable[[str], KnowledgeMetadata] | None = None
    memory_probe: Callable[[str], MemoryMetadata] | None = None
    decomposer: Callable[[str], HopEstimate] = heuristic_decomposition
    agreement_scorer: Callable[[Sequence[str]], float | None] = lexical_snippet_agreement

    def extract(self, question: str, metadata: QueryMetadata | None = None) -> KAPFeatures:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        metadata = metadata or QueryMetadata()
        provenance = {}

        def resolve(name: str, probe: Callable | None):
            value = getattr(metadata, name)
            if value is not None:
                provenance[name] = "metadata"
                return value
            provenance[name] = "probe" if probe else "missing"
            return probe(question) if probe else None

        quality = resolve("quality", self.quality_probe)
        p = resolve("parametric", self.parametric_probe)
        w = resolve("web", self.web_probe)
        k = resolve("knowledge", self.knowledge_probe)
        m = resolve("memory", self.memory_probe)
        hops = resolve("hops", self.decomposer)
        agreement = self.agreement_scorer(w.snippets) if w else None
        if agreement is not None:
            probability(agreement, "snippet_agreement")
        provenance["snippet_agreement"] = getattr(self.agreement_scorer, "__name__", "custom") if w else "missing"
        provenance["distance_method"] = hops.method
        return KAPFeatures(question, quality or QuestionQuality(), p, w, agreement, k, m, hops, provenance)
