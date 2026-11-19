"""Shared identifiers and strict probability validation."""

from enum import Enum
from math import isclose, isfinite
from typing import Mapping, TypeVar


class Source(str, Enum):
    P = "P"
    W = "W"
    K = "K"
    M = "M"


OUTSIDE = "Ø"
T = TypeVar("T")


def probability(value: float, name: str) -> float:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def distribution(values: Mapping[T, float], name: str) -> dict[T, float]:
    result = dict(values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    for value in result.values():
        probability(value, name)
    if not isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to one")
    return result
