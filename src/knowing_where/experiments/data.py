"""Validated JSONL datasets, checked downloads, and group-isolated partitions."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import urllib.request

from ..context import ContextDocument, ContextRequest
from ..evidence.features import (
    HopEstimate, KnowledgeMetadata, MemoryMetadata, ParametricMetadata,
    QueryMetadata, QuestionQuality, WebMetadata,
)
from ..types import OUTSIDE, Source
from .io import loads


@dataclass(frozen=True)
class Query:
    """Inference input: contains no answer or supervision fields."""

    id: str
    question: str
    group: str
    metadata: QueryMetadata
    documents: tuple[ContextDocument, ...]
    instructions: tuple[str, ...]
    required_document_ids: tuple[str, ...]
    paths: dict[Source, tuple[str, ...]]

    def context_request(self):
        return ContextRequest(self.question, self.documents, self.instructions,
                              self.required_document_ids)


@dataclass(frozen=True)
class Example(Query):
    gold: str
    split_group: str
    corpus_present: dict[Source, bool]


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a JSON object')
    return value


def _list(value, name):
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a JSON array')
    return value


def parse_query(row, *, labeled=False):
    _mapping(row, 'query')
    allowed = {'id', 'question', 'group', 'metadata', 'documents', 'instructions',
               'required_document_ids', 'paths'}
    if labeled:
        allowed |= {'gold', 'split_group', 'corpus_present'}
    if row.keys() - allowed:
        raise ValueError(f'unknown query fields: {sorted(row.keys() - allowed)}')
    for key in ('id', 'question', 'group'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'{key} must be nonempty text')
    metadata = dict(_mapping(row.get('metadata', {}), 'metadata'))
    constructors = {'quality': QuestionQuality, 'parametric': ParametricMetadata,
                    'web': WebMetadata, 'knowledge': KnowledgeMetadata,
                    'memory': MemoryMetadata, 'hops': HopEstimate}
    if metadata.keys() - constructors.keys():
        raise ValueError('unknown query metadata fields')
    for key, value in metadata.items():
        _mapping(value, f'metadata.{key}')
    if 'hops' in metadata:
        hops = dict(metadata['hops'])
        probabilities = _mapping(hops.get('probabilities'), 'hop probabilities')
        # Never truncate fractional hops or overwrite colliding normalized keys.
        normalized = {}
        for key, value in probabilities.items():
            if not isinstance(key, str) or not key.isascii() or not key.isdecimal():
                raise ValueError('hop keys must be nonnegative integer strings')
            hop = int(key)
            if hop in normalized:
                raise ValueError('duplicate normalized hop key')
            normalized[hop] = value
        hops['probabilities'] = normalized
        metadata['hops'] = hops
    query_metadata = QueryMetadata(**{k: constructors[k](**v) for k, v in metadata.items()})
    paths = {}
    for source, answers in _mapping(row.get('paths', {}), 'paths').items():
        _list(answers, f'paths.{source}')
        if any(not isinstance(a, str) or not a.strip() or a == OUTSIDE for a in answers):
            raise ValueError('paths must contain canonical nonempty answer IDs excluding Ø')
        paths[Source(source)] = tuple(answers)
    documents = []
    for doc in _list(row.get('documents', []), 'documents'):
        _mapping(doc, 'document')
        if 'depends_on' in doc:
            _list(doc['depends_on'], 'depends_on')
        if 'claims' in doc:
            _mapping(doc['claims'], 'claims')
        documents.append(ContextDocument(**{**doc, 'source': Source(doc['source'])}))
    instructions = tuple(_list(row.get('instructions', []), 'instructions'))
    required = tuple(_list(row.get('required_document_ids', []), 'required_document_ids'))
    query = Query(row['id'], row['question'], row['group'], query_metadata,
                  tuple(documents), instructions, required, paths)
    query.context_request()
    return query


def parse_example(row):
    query = parse_query(row, labeled=True)
    gold = row.get('gold')
    if not isinstance(gold, str) or not gold.strip() or gold == OUTSIDE:
        raise ValueError('gold must be a nonempty canonical answer other than Ø')
    split_group = row.get('split_group', row['id'])
    if not isinstance(split_group, str) or not split_group.strip():
        raise ValueError('split_group must be nonempty text')
    labels = _mapping(row.get('corpus_present'), 'corpus_present')
    if set(labels) != {s.value for s in Source} or any(type(v) is not bool for v in labels.values()):
        raise ValueError('corpus_present requires a boolean for every source')
    return Example(**query.__dict__, gold=gold, split_group=split_group,
                   corpus_present={Source(k): v for k, v in labels.items()})


def validate_dataset(rows):
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError('dataset must be nonempty with unique IDs')
    question_groups = {}
    for row in rows:
        key = ' '.join(row.question.casefold().split())
        if key in question_groups and question_groups[key] != row.split_group:
            raise ValueError('duplicate questions must share split_group')
        question_groups[key] = row.split_group


def _read_lines(path, parser):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(parser(loads(line)))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'{path}:{line_number}: {error}') from error
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError('input must be nonempty with unique IDs')
    return rows


def load_dataset(path):
    rows = _read_lines(path, parse_example)
    validate_dataset(rows)
    return rows


def load_queries(path):
    return _read_lines(path, parse_query)


def split_dataset(rows, seed=42):
    """Shuffle stable group IDs; allocate 40/20/40 percent of groups, rounded down.

    Related examples must share split_group. Row fractions can differ when groups
    have different sizes. Input order never changes the partition membership.
    """
    validate_dataset(rows)
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    groups = sorted({row.split_group for row in rows})
    if len(groups) < 5:
        raise ValueError('at least five independent split groups are required')
    random.Random(seed).shuffle(groups)
    train_end, cal_end = 2 * len(groups) // 5, 3 * len(groups) // 5
    assignments = {g: 'train' if i < train_end else 'calibration' if i < cal_end else 'test'
                   for i, g in enumerate(groups)}
    return {name: tuple(sorted((r for r in rows if assignments[r.split_group] == name), key=lambda r: r.id))
            for name in ('train', 'calibration', 'test')}


def download_dataset(url, destination, sha256, max_bytes=100_000_000):
    """Download a normalized JSONL dataset only after checksum and schema checks."""
    if not url.startswith('https://'):
        raise ValueError('dataset downloads require HTTPS')
    if len(sha256) != 64 or any(c not in '0123456789abcdef' for c in sha256):
        raise ValueError('supply the published lowercase SHA-256 digest')
    with urllib.request.urlopen(url, timeout=30) as response:
        if not response.geturl().startswith('https://'):
            raise ValueError('dataset redirected outside HTTPS')
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError('dataset exceeds download limit')
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError('dataset checksum mismatch')
    # Validate without leaving a partially downloaded destination behind.
    import tempfile
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix='.jsonl') as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        load_dataset(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
