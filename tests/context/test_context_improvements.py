from dataclasses import replace
import json
import unittest

from knowing_where.context import (
    BudgetExceeded, ContextCompiler, ContextDocument, ContextRequest,
)
from knowing_where.cli.context import build_report
from knowing_where.context.evaluation import EvidenceSpan, evaluate_packet
from knowing_where.routing import ModelProfile, RoutingPolicy, route_context
from knowing_where.types import Source


class CompressionTests(unittest.TestCase):
    def test_repeated_evidence_does_not_consume_remaining_budget(self):
        repeated = "Latency target is 1500 ms."
        request = ContextRequest("latency target", (
            ContextDocument("a", Source.W, " ".join([repeated] * 50)),
            ContextDocument("b", Source.M, repeated),
            ContextDocument("c", Source.K, "Latency limit is not yet verified."),
        ))
        packet = ContextCompiler().compile(request, model_id="m", budget_units=3000)
        excerpts = json.loads(packet.payload)["excerpts"]
        self.assertEqual(sum(item["text"] == repeated for item in excerpts), 1)
        self.assertTrue(any(item["document_id"] == "c" for item in excerpts))
        self.assertEqual(len(packet.omissions), 50)
        self.assertTrue(all(item["reason"] == "duplicate" for item in packet.omissions))

    def test_case_sensitive_evidence_is_distinct(self):
        request = ContextRequest("US us", (ContextDocument("a", Source.W, "US. us."),))
        packet = ContextCompiler().compile(request, model_id="m", budget_units=2000)
        self.assertEqual(packet.selected_excerpts, 2)

    def test_dependencies_preserved_transitively_and_verbatim(self):
        definitions = "  Code:\n    return False\n"
        request = ContextRequest("apply recommendation", (
            ContextDocument("answer", Source.P, "Apply recommendation.", depends_on=("assumption",)),
            ContextDocument("assumption", Source.M, "Assumes this definition.", depends_on=("definition",)),
            ContextDocument("definition", Source.K, definitions),
        ), required_document_ids=("answer",))
        compiler = ContextCompiler()
        packet = compiler.compile(request, model_id="m", budget_units=5000)
        data = json.loads(packet.payload)
        self.assertEqual(set(packet.selected_document_ids), {"answer", "assumption", "definition"})
        self.assertEqual(next(item["text"] for item in data["excerpts"] if item["document_id"] == "definition"), definitions)
        self.assertTrue(evaluate_packet(request, packet)["dependencies_retained"])
        with self.assertRaises(BudgetExceeded):
            compiler.compile(request, model_id="m", budget_units=packet.used_units - 1)
        optional = replace(request, required_document_ids=())
        smaller = compiler.compile(optional, model_id="m", budget_units=packet.used_units - 1)
        self.assertNotIn("answer", smaller.selected_document_ids)
        self.assertTrue(evaluate_packet(optional, smaller)["dependencies_retained"])

    def test_invalid_dependency_graphs(self):
        with self.assertRaisesRegex(ValueError, "unknown dependency"):
            ContextRequest("q", (ContextDocument("a", Source.W, "x", depends_on=("missing",)),))
        with self.assertRaisesRegex(ValueError, "cyclic"):
            ContextRequest("q", (
                ContextDocument("a", Source.W, "x", depends_on=("b",)),
                ContextDocument("b", Source.W, "y", depends_on=("a",)),
            ))

    def test_density_retains_two_short_relevant_facts_under_tight_budget(self):
        request = ContextRequest("latency budget", (
            ContextDocument("a", Source.W, "Latency budget " + "filler " * 200 + "."),
            ContextDocument("b", Source.K, "Latency is 1500 ms. Budget is 0.02 USD."),
        ))
        packet = ContextCompiler().compile(request, model_id="m", budget_units=1000)
        self.assertEqual(packet.selected_document_ids, ("b",))
        self.assertEqual(packet.query_term_coverage, 1.0)
        self.assertEqual(packet.missing_query_terms, ())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.request = ContextRequest("latency", (
            ContextDocument("a", Source.K, "Latency is low. Budget is small."),
        ), ("Cite evidence.",))
        self.packet = ContextCompiler().compile(self.request, model_id="m", budget_units=3000)

    def test_partial_recall_and_unknown_without_annotations(self):
        body = json.loads(self.packet.payload)
        body["excerpts"] = body["excerpts"][:1]
        packet = replace(self.packet, payload=json.dumps(body))
        spans = [EvidenceSpan("a", 0, 15, "latency"), EvidenceSpan("a", 16, 32, "budget")]
        result = evaluate_packet(self.request, packet, spans)
        self.assertEqual(result["evidence_span_recall"], 0.5)
        self.assertAlmostEqual(result["evidence_character_recall"], 15 / 31)
        self.assertTrue(result["instructions_retained"])
        self.assertIsNone(evaluate_packet(self.request, self.packet)["evidence_span_recall"])

    def test_overlaps_cannot_inflate_recall(self):
        body = json.loads(self.packet.payload)
        body["excerpts"] *= 2
        packet = replace(self.packet, payload=json.dumps(body))
        result = evaluate_packet(self.request, packet, [EvidenceSpan("a", 0, 15)])
        self.assertEqual(result["evidence_character_recall"], 1.0)

    def test_bad_source_offsets_and_invalid_annotations_fail(self):
        body = json.loads(self.packet.payload)
        body["excerpts"][0]["text"] = "Forged evidence."
        with self.assertRaisesRegex(ValueError, "source offsets"):
            evaluate_packet(self.request, replace(self.packet, payload=json.dumps(body)))
        with self.assertRaises(ValueError):
            evaluate_packet(self.request, self.packet, [EvidenceSpan("a", 0, 500)])

    def test_annotations_cannot_change_selection(self):
        data = {"question": "latency", "documents": [{"id": "a", "source": "K", "text": "Latency is low."}],
                "models": [{"id": "m", "description": "test", "capabilities": {},
                            "context_capacity": 3000, "output_reserve": 1000}]}
        original = build_report(data)
        annotated = build_report({**data, "evaluation_spans": [{"document_id": "a", "start": 0, "end": 15}]})
        self.assertEqual(original["candidates"][0]["payload"], annotated["candidates"][0]["payload"])

    def test_budget_sweep_reports_overflow_and_measured_recall(self):
        data = {"question": "latency", "documents": [{"id": "a", "source": "K", "text": "Latency is low."}],
                "models": [], "evaluation_spans": [{"document_id": "a", "start": 0, "end": 15}]}
        result = build_report(data, sweep_budgets=(1, 3000))["budget_sweep"]
        self.assertEqual(result[0]["status"], "mandatory_overflow")
        self.assertEqual(result[1]["evaluation"]["evidence_span_recall"], 1.0)
        self.assertLessEqual(result[1]["used_units"], 3000)


class RoutingImprovementsTests(unittest.TestCase):
    def model(self, id_, score, cost):
        return ModelProfile(id_, "test", {"coding": score}, 3000, 500,
                            estimated_cost_usd=cost, estimated_latency_ms=100)

    def test_economy_selects_cheapest_model_above_suitability_floor(self):
        models = [self.model("weak", 0.2, 0.001), self.model("good", 0.8, 0.01),
                  self.model("best", 0.95, 0.03), self.model("unknown", 1.0, None)]
        request = ContextRequest("coding", ())
        economy = route_context(request, models, policy=RoutingPolicy(
            {"coding": 1}, selection_mode="economy", min_suitability=0.75))
        self.assertEqual(economy.selected_model_id, "good")
        self.assertEqual(set(economy.rejected), {"weak", "unknown"})
        self.assertIn("economy", economy.decision_reason)
        quality = route_context(request, models[:3], policy=RoutingPolicy({"coding": 1}))
        self.assertEqual(quality.selected_model_id, "best")

    def test_missing_query_evidence_can_block_routing(self):
        request = ContextRequest("coding latency", (ContextDocument("a", Source.W, "Coffee is ready."),))
        result = route_context(request, [self.model("m", 1, 0.01)],
                               policy=RoutingPolicy(min_query_term_coverage=0.5))
        self.assertIsNone(result.selected_model_id)
        self.assertIn("coverage", result.rejected["m"])

    def test_invalid_policy(self):
        for args in ({"selection_mode": "magic"}, {"min_suitability": 1.1}, {"min_query_term_coverage": -1}):
            with self.assertRaises(ValueError):
                RoutingPolicy(**args)

    def test_large_finite_weights_do_not_produce_nan(self):
        result = route_context(ContextRequest("q", ()), [self.model("m", 0.8, 0.01)],
                               policy=RoutingPolicy({"coding": 1e308, "unknown": 1e308}))
        self.assertAlmostEqual(result.candidates[0].suitability, 0.4)


if __name__ == "__main__":
    unittest.main()
