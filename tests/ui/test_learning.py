"""Invariants of the educational source policy, separate from fitted inference."""

import importlib.util
import unittest

HAS_UI = importlib.util.find_spec('streamlit') is not None
if HAS_UI:
    from knowing_where.ui.learning import outcome_masses, route_illustration


@unittest.skipUnless(HAS_UI, 'Install UI dependencies for the learning workspaces')
class SourcePolicyTests(unittest.TestCase):
    def test_higher_reliability_moves_only_present_questions(self):
        before = outcome_masses(.33, .2)
        after = outcome_masses(.33, .6)
        self.assertEqual(before['Absent'], after['Absent'])
        self.assertGreater(after['Found'], before['Found'])
        self.assertLess(after['Present, missed'], before['Present, missed'])
        self.assertAlmostEqual(sum(after.values()), 1)

    def test_absent_source_never_produces_useful_retrieval(self):
        self.assertEqual(outcome_masses(1, 1)['Found'], 0)
        selected, rows = route_illustration({'W': (1, 1)}, {'W': 0}, 0)
        self.assertIsNone(selected)
        self.assertEqual(rows[0]['Useful retrieval'], 0)

    def test_cost_can_change_destination_before_sampling(self):
        estimates = {'W': (.1, .9), 'K': (.2, .9)}
        costs = {'W': 1, 'K': .1}
        self.assertEqual(route_illustration(estimates, costs, 0)[0], 'W')
        self.assertEqual(route_illustration(estimates, costs, .5)[0], 'K')

    def test_invalid_estimates_fail_explicitly(self):
        for value in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                route_illustration({'K': (value, .9)}, {'K': .1}, .2)
