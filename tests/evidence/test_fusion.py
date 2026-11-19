import itertools
import math
import unittest

from knowing_where.evidence.fusion import (
    SourceEvidence, bayesian_fusion, dempster_shafer_fusion, joint_ghost_bounds,
)
from knowing_where.evidence.prediction import SourceEstimate
from knowing_where.types import OUTSIDE, Source


def evidence(source, answer="a", r=0.8, g=0.2, probabilities=None):
    return SourceEvidence(source, answer, SourceEstimate(r, g), probabilities)


class BayesianTests(unittest.TestCase):
    def test_matches_channel_equation_with_nonuniform_noise(self):
        items = [evidence(Source.P, probabilities={"a": 0.7, "b": 0.3}),
                 evidence(Source.K, "b", r=0.9, g=0.1)]
        noise = {"a": 0.2, "b": 0.5, OUTSIDE: 0.3}
        result = bayesian_fusion(items, joint_ghost_mass=0.05, noise=noise, answer_prior={"a": 0.6, "b": 0.4})
        raw = {}
        for answer in result.prior:
            value = result.prior[answer]
            for item in items:
                r, g = item.estimate.reliability, item.estimate.ghost_mass
                p = item.probabilities if item.probabilities is not None else {item.answer: 1}
                channel = (1 - g) * (r * p.get(answer, 0) + (1 - r) * noise[answer]) + g * noise[answer]
                value *= channel
            raw[answer] = value
        z = sum(raw.values())
        for answer in raw:
            self.assertAlmostEqual(result.posterior[answer], raw[answer] / z)
        self.assertAlmostEqual(result.log_evidence, math.log(z))
        single_z = [sum(result.prior[a] * result.channels[item.source][a] for a in result.prior) for item in items]
        self.assertAlmostEqual(result.log_evidence_ratio, math.log(z / math.prod(single_z)))

    def test_uninformative_channels_preserve_prior(self):
        result = bayesian_fusion([evidence(Source.P, r=0, g=0.4)], joint_ghost_mass=0.4, candidates=["a", "b"])
        for answer in result.prior:
            self.assertAlmostEqual(result.posterior[answer], result.prior[answer])
        self.assertAlmostEqual(result.log_evidence_ratio, 0)

    def test_all_ghost_means_outside(self):
        items = [evidence(s, g=1) for s in Source]
        result = bayesian_fusion(items, joint_ghost_mass=1)
        self.assertEqual(result.outside_probability, 1)

    def test_high_ghost_weakens_channel(self):
        low = bayesian_fusion([evidence(Source.P, g=0.1)], joint_ghost_mass=0.1, candidates=["a", "b"])
        high = bayesian_fusion([evidence(Source.P, g=0.9)], joint_ghost_mass=0.9, candidates=["a", "b"])
        self.assertLess(high.channels[Source.P]["a"], low.channels[Source.P]["a"])
        self.assertGreater(high.outside_probability, low.outside_probability)

    def test_ghost_bounds_do_not_assume_independence(self):
        items = [evidence(Source.P, g=0.8), evidence(Source.W, g=0.7)]
        lower, upper = joint_ghost_bounds(items)
        self.assertAlmostEqual(lower, 0.5)
        self.assertEqual(upper, 0.7)
        with self.assertRaises(ValueError):
            bayesian_fusion(items, joint_ghost_mass=0.4)
        with self.assertRaises(ValueError):
            bayesian_fusion(items, joint_ghost_mass=0.9)

    def test_tiny_ghost_noise_survives_and_product_does_not_underflow(self):
        items = [evidence(Source.P, r=1, g=1e-200), evidence(Source.W, "b", r=1, g=1e-200)]
        result = bayesian_fusion(items, joint_ghost_mass=0)
        self.assertGreater(result.channels[Source.P]["b"], 0)
        self.assertAlmostEqual(result.posterior["a"], 0.5)
        self.assertAlmostEqual(result.posterior["b"], 0.5)
        self.assertTrue(math.isfinite(result.log_evidence))

    def test_order_invariance(self):
        items = [evidence(s, "a" if s != Source.M else "b") for s in Source]
        expected = bayesian_fusion(items, joint_ghost_mass=0.01)
        for order in itertools.permutations(items):
            actual = bayesian_fusion(order, joint_ghost_mass=0.01)
            for answer in expected.posterior:
                self.assertAlmostEqual(actual.posterior[answer], expected.posterior[answer])

    def test_impossible_evidence_and_invalid_inputs(self):
        with self.assertRaises(ValueError):
            bayesian_fusion([evidence(Source.P, r=1, g=0), evidence(Source.W, "b", r=1, g=0)], joint_ghost_mass=0)
        factories = [lambda: bayesian_fusion([], joint_ghost_mass=0),
                     lambda: bayesian_fusion([evidence(Source.P)] * 2, joint_ghost_mass=0.1),
                     lambda: bayesian_fusion([evidence(Source.P)], joint_ghost_mass=math.nan),
                     lambda: bayesian_fusion([evidence(Source.P)], joint_ghost_mass=0.2, candidates=["b"]),
                     lambda: bayesian_fusion([evidence(Source.P)], joint_ghost_mass=0.2, noise={"a": 1}),
                     lambda: evidence(Source.P, probabilities={"a": 0.2}),
                     lambda: evidence(Source.P, OUTSIDE)]
        for factory in factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()


class DSTests(unittest.TestCase):
    def test_disagreement_closed_form_and_reported_conflict(self):
        result = dempster_shafer_fusion([evidence(Source.P, r=0.8), evidence(Source.W, "b", r=0.6)])
        self.assertAlmostEqual(result.conflict, 0.48)
        self.assertAlmostEqual(result.pairwise_conflict[(Source.P, Source.W)], 0.48)
        self.assertAlmostEqual(result.belief["a"], 0.32 / 0.52)
        self.assertAlmostEqual(result.belief["b"], 0.12 / 0.52)
        self.assertAlmostEqual(result.plausibility["a"], 0.40 / 0.52)
        self.assertAlmostEqual(sum(result.masses.values()), 1)
        self.assertAlmostEqual(sum(result.pignistic.values()), 1)

    def test_agreement_and_single_candidate_keep_ignorance(self):
        result = dempster_shafer_fusion([evidence(Source.P, r=0.8), evidence(Source.K, r=0.6)])
        self.assertEqual(result.conflict, 0)
        self.assertAlmostEqual(result.belief["a"], 0.92)
        self.assertAlmostEqual(result.masses[frozenset(["a", OUTSIDE])], 0.08)
        self.assertAlmostEqual(result.plausibility[OUTSIDE], 0.08)

    def test_total_conflict_is_explicit(self):
        result = dempster_shafer_fusion([evidence(Source.P, r=1), evidence(Source.K, "b", r=1)])
        self.assertEqual(result.status, "total_conflict")
        self.assertEqual(result.conflict, 1)
        self.assertEqual(result.masses, {})
        self.assertEqual(result.pignistic, {})

    def test_ignorance_is_identity(self):
        source = evidence(Source.P)
        original = dempster_shafer_fusion([source], candidates=["a", "b"])
        combined = dempster_shafer_fusion([source, evidence(Source.W, "b", r=0)])
        self.assertEqual(original.masses, combined.masses)

    def test_reference_ds_does_not_silently_discount_ghosts(self):
        a = dempster_shafer_fusion([evidence(Source.P, g=0)])
        b = dempster_shafer_fusion([evidence(Source.P, g=1)])
        self.assertEqual(a.masses, b.masses)

    def test_order_invariance_and_total_conflict_vs_pairwise(self):
        items = [evidence(Source.P, r=0.9), evidence(Source.W, "b", r=0.8), evidence(Source.K, "c", r=0.7)]
        expected = dempster_shafer_fusion(items)
        # Independent enumeration of all 2^3 commit/ignorance combinations.
        conflict = 0
        for committed in itertools.product([False, True], repeat=3):
            mass = math.prod(item.estimate.reliability if flag else 1 - item.estimate.reliability for item, flag in zip(items, committed))
            if len({item.answer for item, flag in zip(items, committed) if flag}) > 1:
                conflict += mass
        self.assertAlmostEqual(expected.conflict, conflict)
        for order in itertools.permutations(items):
            actual = dempster_shafer_fusion(order)
            self.assertAlmostEqual(actual.conflict, expected.conflict)
            for answer in expected.belief:
                self.assertAlmostEqual(actual.belief[answer], expected.belief[answer])


if __name__ == "__main__":
    unittest.main()
