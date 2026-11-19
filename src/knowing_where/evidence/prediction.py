"""Configurable query-only heads for conditional reliability and corpus absence."""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Mapping

from .features import KAPFeatures
from ..types import Source, probability


@dataclass(frozen=True)
class SourceEstimate:
    reliability: float  # P(correct | answer in corpus, engaged, features)
    ghost_mass: float   # P(answer absent from corpus | engaged, features)

    def __post_init__(self) -> None:
        probability(self.reliability, "reliability")
        probability(self.ghost_mass, "ghost_mass")


@dataclass(frozen=True)
class LogisticHead:
    weights: Mapping[str, float]
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in [self.intercept, *self.weights.values()]):
            raise ValueError("head coefficients must be finite")

    def predict(self, vector: Mapping[str, float]) -> float:
        unknown = self.weights.keys() - vector.keys()
        if unknown:
            raise ValueError(f"unknown model features: {sorted(unknown)}")
        z = self.intercept + sum(weight * vector[name] for name, weight in self.weights.items())
        if not isfinite(z):
            raise ValueError("nonfinite model score")
        return 1 / (1 + exp(-z)) if z >= 0 else exp(z) / (1 + exp(z))


@dataclass(frozen=True)
class SourceHeads:
    reliability: LogisticHead
    ghost_mass: LogisticHead


@dataclass
class LinearKAPPredictor:
    """Inference only; callers supply coefficients and training-set imputations.

    No default coefficients are claimed to be calibrated. Missing indicators
    named '<feature>__missing' accompany imputed values. Fit reliability on
    engaged, corpus-present examples; ghost heads need corpus-absence labels.
    Fit heads on a training partition, never conformal calibration or test data.
    """

    heads: Mapping[Source, SourceHeads]
    missing_values: Mapping[str, float]

    def predict(self, features: KAPFeatures) -> dict[Source, SourceEstimate]:
        if set(self.heads) != set(Source):
            raise ValueError("predictor requires heads for all four sources")
        raw = features.vector()
        vector = {}
        for name, value in raw.items():
            vector[f"{name}__missing"] = float(value is None)
            if value is None:
                if name not in self.missing_values:
                    raise ValueError(f"missing feature {name!r}: provide a training-set imputation")
                value = self.missing_values[name]
            if not isfinite(value):
                raise ValueError(f"nonfinite feature {name}")
            vector[name] = value
        return {
            source: SourceEstimate(head.reliability.predict(vector), head.ghost_mass.predict(vector))
            for source, head in self.heads.items()
        }
