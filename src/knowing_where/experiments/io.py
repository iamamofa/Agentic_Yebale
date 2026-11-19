"""Strict JSON parsing and atomic artifact writes."""

import json
from math import isfinite
import os
from pathlib import Path
import tempfile


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def loads(text):
    def reject(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    def finite_float(value):
        result = float(value)
        if not isfinite(result):
            raise ValueError('JSON number exceeds supported finite range')
        return result
    def finite_int(value):
        result = int(value)
        try:
            finite = isfinite(result)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('JSON integer exceeds supported finite range')
        return result
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject,
                      parse_float=finite_float, parse_int=finite_int)


def read_json(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Serialize before touching the destination; replace only a complete file."""
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    _write_text(path, rendered)


def write_jsonl(path, values):
    rendered = ''.join(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n' for value in values)
    _write_text(path, rendered)


def _write_text(path, rendered):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False, suffix='.tmp') as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
