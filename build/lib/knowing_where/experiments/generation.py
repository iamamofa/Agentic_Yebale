"""Label-free generation adapters and empirical ten-path confidence."""

from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Protocol
import time

from ..evidence.fusion import SourceEvidence
from ..types import OUTSIDE, Source


@dataclass(frozen=True)
class GenerationRequest:
    example_id: str
    question: str
    source: Source
    model_id: str
    instructions: tuple[str, ...]
    excerpts: tuple[str, ...]
    excerpt_records: tuple[dict, ...] = ()
    source_documents: tuple[dict, ...] = ()
    source_claims: tuple[dict, ...] = ()
    evidence_policy: str = ""


class Generator(Protocol):
    def generate(self, request: GenerationRequest, path_index: int, seed: int) -> str:
        """Return a canonical answer ID; labels are not part of this interface."""
        ...


class ReplayGenerator:
    """Replay recorded canonical answers, without access to gold/presence labels."""

    def __init__(self, traces):
        self.traces = traces

    def validate(self, requests, paths):
        for request in requests:
            answers = self.traces.get(request.example_id, {}).get(request.source, ())
            if len(answers) < paths:
                raise ValueError(f'insufficient replay paths for {request.example_id}/{request.source.value}')
            if any(not isinstance(a, str) or not a.strip() or a == OUTSIDE for a in answers[:paths]):
                raise ValueError(f'invalid replay answers for {request.example_id}/{request.source.value}')

    def generate(self, request, path_index, seed):
        try:
            return self.traces[request.example_id][request.source][path_index]
        except (KeyError, IndexError) as error:
            raise ValueError('replay requires enough recorded paths for every source') from error


@dataclass(frozen=True)
class PathConfidence:
    answer: str
    probabilities: dict[str, float]
    agreement: float
    entropy: float
    paths: tuple[str, ...]

    def evidence(self, source, estimate):
        return SourceEvidence(source, self.answer, estimate, self.probabilities)


def generate_paths(generator, request, *, paths=10, seed=42):
    if type(paths) is not int or paths < 1:
        raise ValueError('paths must be a positive integer')
    answers = []
    for index in range(paths):
        try:
            answer = generator.generate(request, index, seed + index)
            if not isinstance(answer, str) or not answer.strip() or answer == OUTSIDE:
                raise ValueError('generator must return canonical answer IDs excluding Ø')
        except (ValueError, TimeoutError, ConnectionError) as error:
            raise ValueError(f'generation failed for {request.example_id}/{request.source.value} '
                             f'path {index}: {error}') from error
        answers.append(answer)
    answers = tuple(answers)
    counts = Counter(answers)
    probabilities = {answer: count / paths for answer, count in sorted(counts.items())}
    answer = min(counts, key=lambda a: (-counts[a], a))
    return PathConfidence(answer, probabilities, probabilities[answer],
                          -sum(p * log(p) for p in probabilities.values()), answers)


class GenerationBudgetExceeded(ValueError):
    """The run cannot issue another generator call within its configured limit."""


class ManagedGenerator:
    """Bound total backend attempts and retry only transient connection failures."""

    def __init__(self, backend, *, max_calls=None, max_retries=0):
        if max_calls is not None and (type(max_calls) is not int or max_calls < 0):
            raise ValueError('max_calls must be a nonnegative integer or None')
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        self.backend, self.max_calls, self.max_retries = backend, max_calls, max_retries
        self.calls = self.retries = 0

    def preflight(self, planned_calls):
        if self.max_calls is not None and planned_calls > self.max_calls - self.calls:
            raise GenerationBudgetExceeded(f'{planned_calls} planned calls exceed the remaining generation budget')

    def generate(self, request, path_index, seed):
        for attempt in range(self.max_retries + 1):
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise GenerationBudgetExceeded('generation call budget exhausted, including retry attempts')
            self.calls += 1
            if attempt:
                self.retries += 1
                time.sleep(min(0.1 * 2 ** min(attempt - 1, 5), 2.0))
            try:
                return self.backend.generate(request, path_index, seed)
            except (TimeoutError, ConnectionError):
                if attempt == self.max_retries:
                    raise

    def summary(self, planned_calls):
        return {'planned_calls': planned_calls, 'attempted_calls': self.calls,
                'retry_calls': self.retries, 'max_calls': self.max_calls,
                'max_retries': self.max_retries}
