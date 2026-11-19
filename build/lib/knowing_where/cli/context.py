"""Compile JSON context and model cards offline: python -m knowing_where.cli.context."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..context import BudgetExceeded, ContextCompiler, ContextDocument, ContextRequest
from ..context.evaluation import EvidenceSpan, evaluate_packet
from ..routing import ModelProfile, RoutingPolicy, route_context
from ..types import Source


def build_report(data: dict, sweep_budgets=()) -> dict:
    documents = tuple(ContextDocument(**{**doc, "source": Source(doc["source"])})
                      for doc in data.get("documents", []))
    request = ContextRequest(
        data["question"], documents, tuple(data.get("instructions", [])),
        tuple(data.get("required_document_ids", [])),
    )
    models = [ModelProfile(**model) for model in data["models"]]
    result = route_context(request, models, policy=RoutingPolicy(**data.get("policy", {})))
    # Evaluation labels are deliberately kept out of ContextRequest and routing.
    evidence = tuple(EvidenceSpan(**span) for span in data.get("evaluation_spans", []))
    by_id = {doc.id: doc for doc in documents}
    for span in evidence:
        if span.document_id not in by_id or span.end > len(by_id[span.document_id].text):
            raise ValueError("evidence annotation falls outside input document")
    report = {
        "schema_version": 2,
        "method": "offline extractive context selection and declared capability routing",
        "selected_model_id": result.selected_model_id,
        "ranking_rule": result.ranking_rule,
        "decision_reason": result.decision_reason,
        "model_summaries": [model.summary() for model in models],
        "rejected": dict(result.rejected),
        "candidates": [{"model_id": item.model_id, "suitability": item.suitability,
                        "evaluation": evaluate_packet(request, item.packet, evidence),
                        "audit": {key: value for key, value in asdict(item.packet).items() if key != "payload"},
                        "context": json.loads(item.packet.payload),
                        "payload": item.packet.payload} for item in result.candidates],
    }
    if sweep_budgets:
        report["budget_sweep"] = []
        compiler = ContextCompiler()
        for budget in sweep_budgets:
            try:
                packet = compiler.compile(request, model_id="budget-sweep", budget_units=budget)
                report["budget_sweep"].append({
                    "budget_units": budget, "used_units": packet.used_units, "status": "compiled",
                    "evaluation": evaluate_packet(request, packet, evidence),
                    "omissions": list(packet.omissions),
                })
            except BudgetExceeded as error:
                report["budget_sweep"].append({"budget_units": budget, "status": "mandatory_overflow",
                                               "reason": str(error)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON request with documents, model profiles, and policy")
    parser.add_argument("--output", type=Path, help="Write report to this path instead of stdout")
    parser.add_argument("--sweep-budgets", nargs="+", type=int, default=(),
                        help="Also evaluate these input budgets in UTF-8 bytes")
    args = parser.parse_args()
    try:
        report = build_report(json.loads(args.input.read_text(encoding="utf-8")), args.sweep_budgets)
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"context compiler: {error}\n")


if __name__ == "__main__":
    main()
