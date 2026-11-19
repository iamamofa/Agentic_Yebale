"""Declared model cards and transparent, uncalibrated routing heuristics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Mapping, Sequence

from ..context.compiler import ContextCompiler, ContextPacket, ContextRequest, BudgetExceeded, _integer, _text
from ..types import probability


@dataclass(frozen=True)
class ModelProfile:
    id: str
    description: str
    capabilities: Mapping[str, float]
    context_capacity: int
    output_reserve: int
    unit_name: str = "utf8_bytes"
    estimated_cost_usd: float | None = None
    estimated_latency_ms: float | None = None
    limitations: tuple[str, ...] = ()
    provenance: str = "user_declared"

    def __post_init__(self) -> None:
        for name in ("id", "description", "unit_name", "provenance"):
            _text(getattr(self, name), name)
        _integer(self.context_capacity, "context_capacity", 1)
        _integer(self.output_reserve, "output_reserve")
        if self.output_reserve >= self.context_capacity:
            raise ValueError("output reserve must leave room for input")
        for key, value in self.capabilities.items():
            _text(key, "capability")
            probability(value, "capability score")
        for value in (self.estimated_cost_usd, self.estimated_latency_ms):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("cost and latency estimates must be finite and nonnegative")

    def summary(self) -> dict:
        """Portable model card; capability numbers are not accuracy probabilities."""
        result = asdict(self)
        result["input_budget"] = self.context_capacity - self.output_reserve
        result["score_semantics"] = "declared suitability, not calibrated accuracy"
        return result


@dataclass(frozen=True)
class RoutingPolicy:
    capability_weights: Mapping[str, float] = field(default_factory=dict)
    required_capabilities: Mapping[str, float] = field(default_factory=dict)
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    selection_mode: str = "quality"
    min_suitability: float = 0.0
    min_query_term_coverage: float = 0.0

    def __post_init__(self) -> None:
        if self.selection_mode not in {"quality", "economy"}:
            raise ValueError("selection_mode must be quality or economy")
        probability(self.min_suitability, "min_suitability")
        probability(self.min_query_term_coverage, "min_query_term_coverage")
        for key, weight in self.capability_weights.items():
            _text(key, "capability")
            if not isfinite(weight) or weight < 0:
                raise ValueError("capability weights must be finite and nonnegative")
        for key, value in self.required_capabilities.items():
            _text(key, "required capability")
            probability(value, "required capability")
        for value in (self.max_cost_usd, self.max_latency_ms):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("routing limits must be finite and nonnegative")


@dataclass(frozen=True)
class RouteCandidate:
    model_id: str
    suitability: float
    packet: ContextPacket


@dataclass(frozen=True)
class RoutingResult:
    selected_model_id: str | None
    candidates: tuple[RouteCandidate, ...]
    rejected: Mapping[str, str]
    ranking_rule: str = "suitability, query term coverage, cost, latency, model id"
    decision_reason: str = ""


def route_context(request: ContextRequest, models: Sequence[ModelProfile], *,
                  policy: RoutingPolicy | None = None,
                  compiler: ContextCompiler | None = None) -> RoutingResult:
    """Build a packet for every feasible model, then rank with an explicit rule.

    Estimates are per-request caller inputs. Unknown cost/latency fails a hard
    limit; without limits, known estimates win ties over unknown estimates.
    Nothing here invokes a model or converts suitability to confidence.
    """
    if len({model.id for model in models}) != len(models):
        raise ValueError("duplicate model ids")
    policy, compiler = policy or RoutingPolicy(), compiler or ContextCompiler()
    rejected = {}
    candidates = []
    profiles = {model.id: model for model in models}
    for model in models:
        reason = None
        if model.unit_name != compiler.unit_name:
            reason = "model and compiler budget units differ"
        elif any(key not in model.capabilities or model.capabilities[key] < minimum
                 for key, minimum in policy.required_capabilities.items()):
            reason = "required capability missing or below minimum"
        elif policy.max_cost_usd is not None and (model.estimated_cost_usd is None or
                                                 model.estimated_cost_usd > policy.max_cost_usd):
            reason = "cost estimate unknown or exceeds limit"
        elif policy.max_latency_ms is not None and (model.estimated_latency_ms is None or
                                                   model.estimated_latency_ms > policy.max_latency_ms):
            reason = "latency estimate unknown or exceeds limit"
        if reason:
            rejected[model.id] = reason
            continue
        try:
            packet = compiler.compile(request, model_id=model.id,
                                      budget_units=model.context_capacity - model.output_reserve)
        except BudgetExceeded as error:
            rejected[model.id] = str(error)
            continue
        scale = max(policy.capability_weights.values(), default=0.0)
        weights = {key: weight / scale for key, weight in policy.capability_weights.items()} if scale else {}
        total_weight = sum(weights.values())
        fit = sum(weight * model.capabilities.get(key, 0.0)
                  for key, weight in weights.items()) / total_weight if total_weight else 0.0
        if fit < policy.min_suitability:
            rejected[model.id] = "declared suitability below minimum"
            continue
        if packet.query_term_coverage < policy.min_query_term_coverage:
            rejected[model.id] = "retained query term coverage below minimum"
            continue
        if policy.selection_mode == "economy" and model.estimated_cost_usd is None:
            rejected[model.id] = "economy routing requires a known cost estimate"
            continue
        candidates.append(RouteCandidate(model.id, fit, packet))

    def rank(candidate: RouteCandidate) -> tuple:
        model = profiles[candidate.model_id]
        cost = model.estimated_cost_usd if model.estimated_cost_usd is not None else float("inf")
        latency = model.estimated_latency_ms if model.estimated_latency_ms is not None else float("inf")
        quality = (-candidate.suitability, -candidate.packet.query_term_coverage)
        return (cost, *quality, latency, model.id) if policy.selection_mode == "economy" else (
            *quality, cost, latency, model.id)

    candidates.sort(key=rank)
    rule = ("cost, suitability, query term coverage, latency, model id" if policy.selection_mode == "economy"
            else "suitability, query term coverage, cost, latency, model id")
    reason = (f"Selected {candidates[0].model_id} using {policy.selection_mode} ranking after capability, "
              "budget, cost, latency, and coverage gates." if candidates else
              "No model satisfies the routing policy; inspect rejection reasons.")
    return RoutingResult(candidates[0].model_id if candidates else None, tuple(candidates), rejected, rule, reason)
