"""Analytical cases for descriptive uncertainty and source sensitivity."""

import json
import math
import unittest

from knowing_where.evidence.fusion import SourceEvidence, bayesian_fusion
from knowing_where.evidence.prediction import SourceEstimate
from knowing_where.evidence.uncertainty import (
    entropy_bits, predictive_uncertainty, risk_coverage_curve, source_sensitivity, source_uncertainty,
)
from knowing_where.types import OUTSIDE, Source


class PredictiveUncertaintyTests(unittest.TestCase):
    def test_point_mass_and_uniform_reference_cases(self):
        certain = predictive_uncertainty({'a': 1, 'b': 0, OUTSIDE: 0})
        self.assertEqual(certain['entropy_bits'], 0)
        self.assertEqual(certain['normalized_entropy'], 0)
        self.assertEqual(certain['effective_candidates'], 1)
        self.assertEqual(certain['top_two_margin'], 1)
        uniform = predictive_uncertainty({str(i): 0.25 for i in range(4)})
        self.assertEqual(uniform['entropy_bits'], 2)
        self.assertEqual(uniform['normalized_entropy'], 1)
        self.assertEqual(uniform['effective_candidates'], 4)
        self.assertEqual(uniform['top_two_margin'], 0)

    def test_normalization_includes_declared_zero_mass_candidates(self):
        result = predictive_uncertainty({'a': 0.5, 'b': 0.5, OUTSIDE: 0})
        self.assertAlmostEqual(result['normalized_entropy'], 1 / math.log2(3))
        self.assertEqual(result['candidate_count'], 3)
        self.assertEqual(result['top_candidate'], 'a')

    def test_outside_and_unavailable_are_not_false_certainty(self):
        outside = predictive_uncertainty({OUTSIDE: 1})
        self.assertEqual(outside['outside_probability'], 1)
        invalid = predictive_uncertainty({OUTSIDE: 1}, unavailable_reason='no_feasible_model')
        self.assertFalse(invalid['available'])
        self.assertIsNone(invalid['normalized_entropy'])
        self.assertIsNone(invalid['top_probability'])
        json.dumps(invalid, allow_nan=False)

    def test_invalid_distributions_fail(self):
        for probabilities in ({}, {'a': 0.2}, {'a': math.nan}, {'a': -1, 'b': 2}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                entropy_bits(probabilities)


class SourceUncertaintyTests(unittest.TestCase):
    def test_agreement_has_no_ambiguity_or_disagreement(self):
        result = source_uncertainty({s: {'a': 1, 'b': 0} for s in Source})
        self.assertEqual(result['mixture_entropy_bits'], 0)
        self.assertEqual(result['within_source_entropy_bits'], 0)
        self.assertEqual(result['between_source_disagreement_bits'], 0)
        self.assertEqual(result['normalized_disagreement'], 0)

    def test_same_mixture_entropy_can_have_different_causes(self):
        ambiguous = source_uncertainty({s: {'a': 0.5, 'b': 0.5} for s in Source})
        disagree = source_uncertainty({s: {'a': 1} if s in (Source.P, Source.W) else {'b': 1} for s in Source})
        self.assertEqual(ambiguous['mixture_entropy_bits'], 1)
        self.assertEqual(disagree['mixture_entropy_bits'], 1)
        self.assertEqual(ambiguous['within_source_entropy_bits'], 1)
        self.assertEqual(ambiguous['between_source_disagreement_bits'], 0)
        self.assertEqual(disagree['within_source_entropy_bits'], 0)
        self.assertEqual(disagree['between_source_disagreement_bits'], 1)
        self.assertEqual(disagree['normalized_disagreement'], 1)

    def test_decomposition_identity_and_bounds(self):
        distributions = {'P': {'a': 0.8, 'b': 0.2}, 'W': {'b': 0.9, 'c': 0.1},
                         'K': {'a': 0.2, 'c': 0.8}, 'M': {'a': 0.1, 'b': 0.4, 'c': 0.5}}
        result = source_uncertainty(distributions)
        self.assertAlmostEqual(result['mixture_entropy_bits'], result['within_source_entropy_bits'] +
                               result['between_source_disagreement_bits'])
        self.assertGreaterEqual(result['normalized_disagreement'], 0)
        self.assertLessEqual(result['normalized_disagreement'], 1)
        self.assertAlmostEqual(sum(result['mixture_probabilities'].values()), 1)
        self.assertEqual(result, source_uncertainty(dict(reversed(list(distributions.items())))))

    def test_empty_and_single_source(self):
        self.assertFalse(source_uncertainty({})['available'])
        single = source_uncertainty({'P': {'a': 0.7, 'b': 0.3}})
        self.assertEqual(single['between_source_disagreement_bits'], 0)
        self.assertEqual(single['normalized_disagreement'], 0)

    def test_leave_one_out_keeps_frame_and_detects_top_flip(self):
        evidence = [SourceEvidence(s, 'a' if s == Source.P else 'b',
                                  SourceEstimate(0.99 if s == Source.P else 0.1, 0.1)) for s in Source]
        posterior = bayesian_fusion(evidence, joint_ghost_mass=0.1 ** 4).posterior
        self.assertEqual(max(posterior, key=posterior.get), 'a')
        result = source_sensitivity(evidence, posterior)
        self.assertTrue(result['sources']['P']['top_candidate_changed'])
        self.assertEqual(result['sources']['P']['top_candidate_without_source'], 'b')
        for row in result['sources'].values():
            self.assertTrue(row['available'])
            self.assertGreaterEqual(row['total_variation'], 0)
            self.assertLessEqual(row['total_variation'], 1)
        self.assertFalse(source_sensitivity(evidence, None)['available'])


class RiskCoverageTests(unittest.TestCase):
    def row(self, id_, probabilities, gold, routed=True):
        return {'id': id_, 'probabilities': probabilities, 'gold': gold, 'routed': routed,
                'uncertainty': predictive_uncertainty(probabilities)}

    def test_tied_entropies_enter_together_and_do_not_exploit_labels(self):
        records = [self.row('a', {'a': 0.8, 'b': 0.2}, 'a'),
                   self.row('b', {'a': 0.8, 'b': 0.2}, 'b')]
        points = risk_coverage_curve(records)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['coverage'], 1)
        self.assertEqual(points[0]['error_rate'], 0.5)
        self.assertEqual(points, risk_coverage_curve(list(reversed(records))))

    def test_outside_and_unrouted_cases_do_not_become_answers(self):
        records = [self.row('a', {'a': 0.9, 'b': 0.1}, 'a'),
                   self.row('b', {OUTSIDE: 1}, 'missing'),
                   self.row('c', {'a': 1}, 'a', routed=False)]
        points = risk_coverage_curve(records)
        self.assertEqual(points[0]['coverage'], 1 / 3)
        self.assertEqual(points[0]['error_rate'], 0)
        self.assertEqual(risk_coverage_curve([]), [])
