"""Validated UI actions using the same pipeline as the command-line tools."""

import json
from pathlib import Path

from ..cli.context import build_report
from ..experiments.data import parse_example, parse_query, validate_dataset
from ..experiments.io import loads, read_json
from ..experiments.runner import run_experiment
from ..experiments.system import FittedSystem
from ..routing import ModelProfile, RoutingPolicy

ROOT = Path(__file__).resolve().parents[3]
MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 1000


def decode(value):
    if isinstance(value, bytes):
        if len(value) > MAX_BYTES:
            raise ValueError('File exceeds the 10 MB limit.')
        value = value.decode('utf-8-sig')
    if not isinstance(value, str) or len(value.encode('utf-8')) > MAX_BYTES:
        raise ValueError('Input must be UTF-8 text under 10 MB.')
    return value


def json_input(value):
    return loads(decode(value))


def jsonl_input(value, *, labeled):
    rows = []
    for line_number, line in enumerate(decode(value).splitlines(), 1):
        if not line.strip():
            continue
        if len(rows) >= MAX_ROWS:
            raise ValueError(f'Use at most {MAX_ROWS} examples per interactive run.')
        try:
            rows.append((parse_example if labeled else parse_query)(loads(line)))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'Line {line_number}: {error}') from error
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError('Provide at least one example with unique IDs.')
    if labeled:
        validate_dataset(rows)
    return rows


def sample_text(name):
    paths = {'context': 'examples/context/request.json',
             'dataset': 'examples/experiments/demo/dataset.jsonl',
             'models': 'examples/experiments/demo/models.json',
             'system': 'examples/experiments/demo/system.json',
             'queries': 'examples/experiments/demo/queries.jsonl'}
    return (ROOT / paths[name]).read_text(encoding='utf-8')


def sample_report():
    return read_json(ROOT / 'examples/experiments/demo/report.json')


def compile_context(text, *, question=None, selection_mode=None, max_cost=None, max_latency=None):
    data = json_input(text)
    if not isinstance(data, dict):
        raise ValueError('A context request must be a JSON object.')
    if question is not None:
        data['question'] = question
    policy = dict(data.get('policy', {}))
    for key, value in [('selection_mode', selection_mode), ('max_cost_usd', max_cost),
                       ('max_latency_ms', max_latency)]:
        if value is not None:
            policy[key] = value
    data['policy'] = policy
    return build_report(data, sweep_budgets=(1800, 2200, 2500, 3200, 5000))


def experiment(dataset, models, *, seed, alpha, paths, max_calls, policy):
    profiles = json_input(models)
    if not isinstance(profiles, list) or not profiles:
        raise ValueError('Model profiles must be a nonempty JSON array.')
    report = run_experiment(jsonl_input(dataset, labeled=True),
                            [ModelProfile(**model) for model in profiles],
                            joint_ghost='independent', seed=seed, alpha=alpha, paths=paths,
                            max_calls=max_calls, policy=RoutingPolicy(**policy))
    return report, FittedSystem.from_report(report).artifact


def predict(artifact, queries, *, max_calls):
    data = json_input(artifact)
    if not isinstance(data, dict):
        raise ValueError('A fitted system must be a JSON object.')
    return FittedSystem(data).predict(jsonl_input(queries, labeled=False), max_calls=max_calls)


def download_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def comparison_rows(report):
    return [{'Method': name.replace('_', ' ').title(), **result['overall']}
            for name, result in report['comparisons'].items()]
