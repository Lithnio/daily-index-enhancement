import unittest

import pandas as pd
from verify_results import nav_statistics


class NavStatistics(unittest.TestCase):
    def test_carried_opening_close_and_relative_wealth(self):
        strategy = pd.Series([80.0, 72.0, 88.0])
        benchmark = pd.Series([100.0, 99.0, 105.0])
        stats = nav_statistics(strategy, benchmark)

        self.assertAlmostEqual(stats["strategy_cumulative_return"], 0.10)
        self.assertAlmostEqual(stats["benchmark_cumulative_return"], 0.05)
        self.assertAlmostEqual(stats["cumulative_return_difference"], 0.05)
        self.assertAlmostEqual(stats["cumulative_relative_return"], 1.10 / 1.05 - 1)
        self.assertAlmostEqual(
            stats["annualized_relative_return"], (1.10 / 1.05) ** 126 - 1
        )
        self.assertAlmostEqual(stats["strategy_max_drawdown"], -0.10)
        self.assertAlmostEqual(stats["benchmark_max_drawdown"], -0.01)
        self.assertAlmostEqual(stats["relative_max_drawdown"], 0.90 / 0.99 - 1)

    def test_undefined_ir_and_one_day_tracking_error(self):
        path = pd.Series([1.0, 0.98, 1.02])
        self.assertIsNone(
            nav_statistics(path.iloc[:2], path.iloc[:2])["tracking_error"]
        )
        stats = nav_statistics(path, path)
        self.assertEqual(stats["tracking_error"], 0.0)
        self.assertIsNone(stats["information_ratio"])
