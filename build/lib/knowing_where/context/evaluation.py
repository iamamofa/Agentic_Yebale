"""Offline compression checks. Evidence annotations never enter the compiler."""

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from .compiler import ContextPacket, ContextRequest, _integer, _text


@dataclass(frozen=True)
class EvidenceSpan:
    document_id: str
    start: int
    end: int
    label: str = ""

    def __post_init__(self) -> None:
        _text(self.document_id, "evidence document id")
        _integer(self.start, "evidence start")
        _integer(self.end, "evidence end", self.start + 1)


def evaluate_packet(request: ContextRequest, packet: ContextPacket,
                    evidence: Sequence[EvidenceSpan] = ()) -> dict:
    """Validate provenance and measure annotated evidence retention, not accuracy.

    Character recall weights spans by their length. Overlapping annotations are
    independent items; each is counted once in span recall and by length in
    character recall. No evidence annotations means recall is unknown (None).
    """
    body = json.loads(packet.payload)
    docs = {doc.id: doc for doc in request.documents}
    ranges = {id_: [] for id_ in docs}
    references = {ref["id"]: ref for ref in body["documents"]}
    for id_, ref in references.items():
        if id_ not in docs or ref["source"] != docs[id_].source.value or ref["sha256"] != hashlib.sha256(
                docs[id_].text.encode("utf-8")).hexdigest():
            raise ValueError("packet document provenance does not match input")
    for excerpt in body["excerpts"]:
        id_, start, end = excerpt["document_id"], excerpt["start"], excerpt["end"]
        if id_ not in docs or id_ not in references:
            raise ValueError("excerpt has no input document reference")
        _integer(start, "excerpt start")
        _integer(end, "excerpt end", start + 1)
        if end > len(docs[id_].text) or docs[id_].text[start:end] != excerpt["text"]:
            raise ValueError("excerpt does not match source offsets")
        ranges[id_].append((start, end))
    # Merge ranges so repeated or overlapping excerpts cannot inflate recall.
    for id_, intervals in ranges.items():
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        ranges[id_] = merged
    claims = {}
    for doc in request.documents:
        for key, value in doc.claims.items():
            claims.setdefault(key, {}).setdefault(value, set()).add(doc.id)
    expected_conflicts = {(key, value, id_) for key, values in claims.items() if len(values) > 1
                          for value, ids in values.items() for id_ in ids}
    actual_conflicts = {(conflict["key"], alt["value"], id_) for conflict in body["conflicts"]
                        for alt in conflict["alternatives"] for id_ in alt["document_ids"]}
    dependencies_retained = all(
        ranges[dependency] == [(0, len(docs[dependency].text))]
        for id_, intervals in ranges.items() if intervals for dependency in docs[id_].depends_on)
    details = []
    for span in evidence:
        if span.document_id not in docs or span.end > len(docs[span.document_id].text):
            raise ValueError("evidence annotation falls outside input document")
        retained = sum(max(0, min(end, span.end) - max(start, span.start))
                       for start, end in ranges[span.document_id])
        details.append({"document_id": span.document_id, "label": span.label,
                        "start": span.start, "end": span.end,
                        "retained_characters": retained,
                        "fully_retained": retained == span.end - span.start})
    expected_characters = sum(span.end - span.start for span in evidence)
    return {
        "provenance_valid": True,
        "question_retained": body["question"] == request.question,
        "instructions_retained": body["instructions"] == list(request.instructions),
        "conflicts_retained": expected_conflicts == actual_conflicts,
        "dependencies_retained": dependencies_retained,
        "required_documents_retained": all(ranges[id_] == [(0, len(docs[id_].text))]
                                           for id_ in request.required_document_ids),
        "budget_utilization": packet.used_units / packet.budget_units,
        "retained_character_fraction": packet.coverage,
        "query_term_coverage": packet.query_term_coverage,
        "evidence_span_recall": sum(item["fully_retained"] for item in details) / len(details) if details else None,
        "evidence_character_recall": sum(item["retained_characters"] for item in details) / expected_characters
                                     if expected_characters else None,
        "evidence": details,
    }
