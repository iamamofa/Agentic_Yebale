import math
import unittest

from knowing_where.evidence.features import (
    HopEstimate, KAPExtractor, KnowledgeMetadata, MemoryMetadata,
    ParametricMetadata, QueryMetadata, QuestionQuality, WebMetadata,
    lexical_snippet_agreement,
)
from knowing_where.evidence.prediction import LinearKAPPredictor, LogisticHead, SourceEstimate, SourceHeads
from knowing_where.types import Source


class FeatureTests(unittest.TestCase):
    def test_single_web_probe_and_metadata_override(self):
        calls = []

        def web(question):
            calls.append(question)
            return WebMetadata(2, ("same snippet", "same snippet"))

        extractor = KAPExtractor(web_probe=web)
        features = extractor.extract("Question?")
        self.assertEqual(calls, ["Question?"])
        self.assertEqual(features.snippet_agreement, 1)
        self.assertEqual(features.provenance["web"], "probe")
        overridden = extractor.extract("Question?", QueryMetadata(web=WebMetadata(0)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(overridden.provenance["web"], "metadata")
        self.assertEqual(overridden.vector()["web_hits"], 0)

    def test_missing_is_distinct_from_zero(self):
        features = KAPExtractor().extract("Who?")
        self.assertIsNone(features.vector()["web_hits"])
        self.assertIsNone(features.omega.scope_fit)
        self.assertEqual(features.provenance["quality"], "missing")
        self.assertAlmostEqual(sum(features.distance.probabilities.values()), 1)
        self.assertEqual(features.distance.probabilities[0], 0.8)

    def test_metadata_features_and_distance_distribution(self):
        features = KAPExtractor().extract("Who?", QueryMetadata(
            quality=QuestionQuality(0.8, 0.7, 0.1),
            parametric=ParametricMetadata(10, 4),
            knowledge=KnowledgeMetadata(2, 4),
            memory=MemoryMetadata(2, 8, 3, 5),
            hops=HopEstimate({0: 0.25, 2: 0.75}),
        ))
        vector = features.vector()
        self.assertEqual(vector["popularity"], 1)
        self.assertEqual(vector["inverse_perplexity"], 0.25)
        self.assertEqual(vector["knowledge_coverage"], 0.5)
        self.assertEqual(vector["memory_hit_rate"], 0.25)
        self.assertEqual(vector["memory_success_rate"], 0.6)
        self.assertEqual(vector["distance_mean"], 1.5)

    def test_no_observations_do_not_mean_failure(self):
        features = KAPExtractor().extract("Who?", QueryMetadata(
            knowledge=KnowledgeMetadata(0, 0), memory=MemoryMetadata(0, 0, 0, 0)))
        self.assertIsNone(features.vector()["knowledge_coverage"])
        self.assertIsNone(features.vector()["memory_success_rate"])

    def test_agreement_is_only_a_proxy(self):
        self.assertEqual(lexical_snippet_agreement(["a b", "b c"]), 1 / 3)
        self.assertEqual(lexical_snippet_agreement(["red", "blue"]), 0)
        self.assertIsNone(lexical_snippet_agreement(["one snippet"]))
        self.assertIsNone(lexical_snippet_agreement(["", ""]))

    def test_validation(self):
        factories = [lambda: QuestionQuality(scope_fit=math.nan),
                     lambda: ParametricMetadata(11), lambda: ParametricMetadata(1.5),
                     lambda: ParametricMetadata(entity_perplexity=0),
                     lambda: WebMetadata(-1), lambda: WebMetadata(0, ("hit",)),
                     lambda: KnowledgeMetadata(2, 1), lambda: MemoryMetadata(1, 1, 2, 1),
                     lambda: HopEstimate({1: 0.5}), lambda: HopEstimate({-1: 1}),
                     lambda: KAPExtractor().extract(" ")]
        for factory in factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()
        with self.assertRaises(TypeError):
            QueryMetadata(gold_answer="leak")


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.features = KAPExtractor().extract("Who?")
        self.imputation = {name: 0 for name, value in self.features.vector().items() if value is None}
        self.heads = {s: SourceHeads(LogisticHead({"popularity__missing": 1}), LogisticHead({}, -2)) for s in Source}

    def test_all_sources_and_missing_indicators(self):
        predictions = LinearKAPPredictor(self.heads, self.imputation).predict(self.features)
        self.assertEqual(set(predictions), set(Source))
        for estimate in predictions.values():
            self.assertAlmostEqual(estimate.reliability, 1 / (1 + math.exp(-1)))
            self.assertAlmostEqual(estimate.ghost_mass, 1 / (1 + math.exp(2)))

    def test_missing_requires_explicit_imputation(self):
        with self.assertRaises(ValueError):
            LinearKAPPredictor(self.heads, {}).predict(self.features)

    def test_head_validation_and_extreme_logits(self):
        with self.assertRaises(ValueError):
            LogisticHead({"unknown": 1}).predict({})
        with self.assertRaises(ValueError):
            LogisticHead({}, math.inf)
        with self.assertRaises(ValueError):
            SourceEstimate(0.5, -0.1)
        with self.assertRaises(ValueError):
            LinearKAPPredictor({}, self.imputation).predict(self.features)
        self.assertEqual(LogisticHead({}, 1000).predict({}), 1)
        self.assertEqual(LogisticHead({}, -1000).predict({}), 0)


if __name__ == "__main__":
    unittest.main()
