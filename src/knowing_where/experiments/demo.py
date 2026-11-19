"""Clearly synthetic fixtures for exercising the complete experiment pipeline."""

from pathlib import Path
import random

from ..types import Source
from .io import write_json, write_jsonl


def write_demo(directory, *, count=100, seed=17):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        difficulty = rng.random()
        gold, distractor = f'answer-{i}', f'distractor-{i}'
        present = {s.value: rng.random() > difficulty * 0.65 for s in Source}
        paths = {}
        for source in Source:
            accuracy = {'P': 0.72, 'W': 0.78, 'K': 0.86, 'M': 0.6}[source.value] - difficulty * 0.3
            paths[source.value] = [gold if present[source.value] and rng.random() < accuracy else distractor
                                   for _ in range(10)]
        rows.append({
            'id': f'synthetic-{i:03}', 'question': f'Which canonical answer matches synthetic item {i}?',
            'gold': gold, 'group': 'lookup' if i % 2 == 0 else 'reasoning',
            'corpus_present': present, 'paths': paths,
            'metadata': {'quality': {'scope_fit': 1 - difficulty, 'well_posedness': 0.9,
                                     'guardrail_exposure': 0.0},
                         'parametric': {'popularity_decile': max(1, round(10 - difficulty * 9))}},
            'documents': [{'id': f'context-{s.value}', 'source': s.value,
                           'text': f'Synthetic {s.value} context for item {i}; recorded paths are provided separately.'}
                          for s in Source],
        })
    dataset = directory / 'dataset.jsonl'
    write_jsonl(dataset, rows)
    write_jsonl(directory / 'queries.jsonl', [{k: v for k, v in row.items()
                  if k not in ('gold', 'corpus_present', 'split_group')} for row in rows[:3]])
    models = directory / 'models.json'
    write_json(models, [{
        'id': 'synthetic-replay-model', 'description': 'Synthetic profile for offline replay; no measured model quality.',
        'capabilities': {'reasoning': 0.8}, 'context_capacity': 16000, 'output_reserve': 2000,
        'estimated_cost_usd': 0.001, 'estimated_latency_ms': 100,
        'provenance': 'synthetic_demo',
    }])
    return dataset, models
