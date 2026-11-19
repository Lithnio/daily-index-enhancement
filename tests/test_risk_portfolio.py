import unittest
from dataclasses import fields, replace

import numpy as np
import pandas as pd
from demo import example
from index_enhancement.backtest import Execution, simulate
from index_enhancement.risk_portfolio import RiskAwarePortfolio


class RiskAwarePortfolioTests(unittest.TestCase):
    def setUp(self):
        self.market, _ = example()
        self.forecast = self.market.weights * 0.0
        self.execution = Execution(adv_fraction=0.1)
        self.nav = self.execution.initial_cash
        self.i = 70
        self.base = self.market.weights.iloc[self.i].to_numpy(copy=True)

    def policy(self, *, risk=0.002, fees=1.0, risk_tilt=1.0):
        return RiskAwarePortfolio(
            self.market, self.forecast, risk, self.execution, fees, risk_tilt=risk_tilt
        )

    def set_capacity(self, weights):
        self.market.adv.iloc[self.i] = (
            np.asarray(weights) * self.nav / self.execution.adv_fraction
        )

    def test_asymmetric_costs_offset_the_return_of_a_paired_trade(self):
        self.set_capacity([0.03, 0.03, 0, 0, 0, 0, 0, 0])
        held = self.base.copy()
        held[:2] += [-0.002, 0.002]
        self.forecast.iloc[self.i, :2] = [0.0012, -0.0012]
        risk, fees = 0.002, 1.25
        cost = (self.execution.buy_bps + self.execution.sell_bps) * fees / 10_000
        paired = (0.0024 - cost) * self.base[0] / (2 * risk)
        expected = self.base.copy()
        expected[:2] += [paired, -paired]
        policy = self.policy(risk=risk, fees=fees, risk_tilt=0)
        np.testing.assert_allclose(policy(self.i, held, self.nav), expected, atol=1e-12)

    def test_asymmetric_no_trade_interval(self):
        self.forecast.iloc[self.i] = [0.0004, -0.0011, 0, 0, 0, 0, 0, 0]
        sticky = self.policy()
        np.testing.assert_allclose(sticky(self.i, self.base, self.nav), self.base)
        self.assertAlmostEqual(sticky.audit[-1]["planned_one_way"], 0)

    def test_adjusted_close_curvature_matches_quadratic_optimum(self):
        n_dates, n_stocks = self.market.weights.shape
        amplitude = np.array([0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
        returns = (-1) ** np.arange(n_dates)[:, None] * amplitude
        returns[:10] *= 3
        price = 20 * (1 + returns).cumprod(axis=0)
        adjustment = np.ones_like(price)
        adjustment[45:, ::2] = 2
        self.market.adjustment.iloc[:, :] = adjustment
        self.market.close.iloc[:, :] = price / adjustment
        self.market.close.iloc[:43, -1] = np.nan
        self.market.industry.iloc[:, :] = 0
        mu = np.linspace(-0.0004, 0.0004, n_stocks)
        self.forecast.iloc[:, :] = mu
        risk = 0.02
        policy = self.policy(risk=risk, fees=0)
        uniform = self.policy(risk=risk, fees=0, risk_tilt=0)

        np.testing.assert_array_equal(
            policy(29, self.base, self.nav), uniform(29, self.base, self.nav)
        )
        variance = returns[self.i - 59 : self.i + 1, :-1].var(axis=0, ddof=1)
        ratio = np.clip(variance / np.median(variance), 0.25, 4.0)
        # The final name has fewer than 30 valid return observations.
        curvature = risk * np.r_[0.5 + 0.5 * ratio, 1.0]
        scale = self.base / curvature
        expected = self.base + scale * (mu - np.average(mu, weights=scale))
        target = policy(self.i, self.base, self.nav)
        np.testing.assert_allclose(target, expected, atol=1e-12, rtol=0)
        self.assertGreater(
            np.abs(target - uniform(self.i, self.base, self.nav)).max(), 1e-4
        )

    def test_trade_bounds_bind_both_sides_with_feasible_industry_budgets(self):
        capacity = np.array([0.0007, 0.0006, 0.02, 0.02, 0, 0, 0, 0])
        self.set_capacity(capacity)
        self.forecast.iloc[self.i, :2] = [0.05, -0.05]
        policy = self.policy(fees=0)
        target = policy(self.i, self.base, self.nav)
        np.testing.assert_allclose(
            target[:2] - self.base[:2], [capacity[0], -capacity[1]], atol=1e-12
        )
        self.assertTrue((np.abs(target - self.base) <= capacity).all())
        self.assertAlmostEqual(target[:4].sum(), 0.5)
        self.assertAlmostEqual(target[4:].sum(), 0.5)
        audit = policy.audit[-1]
        self.assertAlmostEqual(audit["cash_target"], 0)
        self.assertAlmostEqual(audit["max_industry_gap"], 0)
        self.assertEqual(audit["max_capacity_excess"], 0)
        self.assertAlmostEqual(audit["name_bound_gap"], 0)
        self.assertAlmostEqual(
            audit["planned_one_way"], np.abs(target - self.base).sum() / 2
        )

    def test_wider_name_bounds_preserve_industry_and_trade_limits(self):
        base = np.array([0.002, 0.025, 0.03, 0.443, 0.1, 0.1, 0.15, 0.15])
        self.market.weights.iloc[self.i] = base
        self.forecast.iloc[self.i] = [0.25, 0.30, -0.35, 0, 0.30, -0.30, 0, 0]
        capacity = np.array([1, 1, 1, 1, 0.003, 0.003, 1, 1])
        self.set_capacity(capacity)
        policy = RiskAwarePortfolio(
            self.market,
            self.forecast,
            0.002,
            self.execution,
            0.0,
            risk_tilt=0,
            name_floor=0,
            max_relative_overweight=5,
            max_absolute_overweight=0.02,
        )
        target = policy(self.i, base, self.nav)
        np.testing.assert_allclose(
            target,
            [0.012, 0.045, 0, 0.443, 0.103, 0.097, 0.15, 0.15],
            atol=1e-12,
            rtol=0,
        )
        self.assertTrue((np.abs(target - base) <= capacity).all())
        self.assertAlmostEqual(policy.audit[-1]["max_industry_gap"], 0)
        self.assertAlmostEqual(policy.audit[-1]["name_bound_gap"], 0)

    def test_unavailable_capacity_locks_positions_and_preserves_cash(self):
        self.set_capacity([np.nan, np.inf, -0.01, 0, 0.02, 0, 0, 0])
        held = np.array([0.08, 0.05, 0, 0, 0, 0, 0, 0])
        policy = self.policy()
        target = policy(self.i, held, self.nav)
        np.testing.assert_allclose(target, [0.08, 0.05, 0, 0, 0.02, 0, 0, 0])
        audit = policy.audit[-1]
        self.assertAlmostEqual(audit["cash_target"], 0.85)
        self.assertAlmostEqual(audit["max_industry_gap"], 0.48)
        self.assertAlmostEqual(audit["name_bound_gap"], 0.0125)
        self.assertEqual(audit["max_capacity_excess"], 0)
        self.assertAlmostEqual(policy(self.i, held, 2 * self.nav)[4], 0.01)

    def test_industry_projection_when_one_budget_is_infeasible(self):
        self.market.industry.iloc[:, :] = np.repeat(np.arange(4), 2)
        self.set_capacity([0.11, 0.11, 1, 1, 1, 1, 1, 1])
        policy = self.policy()
        target = policy(self.i, np.zeros_like(self.base), self.nav)
        np.testing.assert_allclose(
            target.reshape(4, 2).sum(axis=1), [0.22, 0.26, 0.26, 0.26], atol=1e-12
        )
        self.assertAlmostEqual(policy.audit[-1]["max_industry_gap"], 0.03)
        self.assertAlmostEqual(policy.audit[-1]["cash_target"], 0)
        self.assertAlmostEqual(policy.audit[-1]["name_bound_gap"], 0)

    def test_cash_roundoff_adjusts_an_existing_trade(self):
        held = np.array([30, 28, 13, 10, 5, 6, 2, 6]) / 100
        self.market.weights.iloc[self.i] = held
        self.forecast.iloc[self.i, -2:] = [0.05, -0.05]
        # Grouped sums satisfy the industry budgets, while the overall weight
        # needs a floating-point correction. The largest name is already held.
        target = self.policy(risk_tilt=0)(self.i, held, self.nav)
        self.assertLessEqual(target.sum(), 1.0)
        self.assertEqual(target[0], held[0])
        self.assertAlmostEqual(target[-2] - held[-2], 0.01)

    def test_partial_forced_exit_counts_retained_weight_in_its_industry(self):
        self.market.weights.iloc[:, 0] = 0
        self.market.weights.iloc[:, 1] = 0.25
        self.set_capacity([0.025, 1, 1, 1, 1, 1, 1, 1])
        policy = self.policy()
        target = policy(self.i, self.base, self.nav)
        self.assertAlmostEqual(target[0], 0.1)
        self.assertAlmostEqual(target[1:4].sum(), 0.4)
        self.assertAlmostEqual(target[4:].sum(), 0.5)
        self.assertAlmostEqual(policy.audit[-1]["max_industry_gap"], 0)
        self.assertAlmostEqual(policy.audit[-1]["outside_index_weight"], 0.1)
        self.assertAlmostEqual(policy.audit[-1]["name_bound_gap"], 0.1)
        self.assertEqual(policy(self.i + 1, target, self.nav)[0], 0)

    def test_relaxes_name_floors_when_unsellable_positions_tie_up_cash(self):
        self.market.weights.iloc[:, 0] = 0
        self.market.weights.iloc[:, 1] = 0.25
        self.set_capacity([0, 1, 1, 1, 1, 1, 1, 1])
        held = np.zeros_like(self.base)
        held[0] = 0.98
        policy = self.policy()
        target = policy(self.i, held, self.nav)
        self.assertEqual(target[0], held[0])
        np.testing.assert_allclose(
            target[1:], 0.02 * self.market.weights.iloc[self.i, 1:], atol=1e-12
        )
        self.assertLessEqual(target.sum(), 1.0)
        self.assertAlmostEqual(policy.audit[-1]["max_industry_gap"], 0.49)
        self.assertAlmostEqual(policy.audit[-1]["name_bound_gap"], 0.98)
        self.assertEqual(policy.audit[-1]["max_capacity_excess"], 0)

    def test_simulation_matches_exactly_with_future_market_data_removed(self):
        last = self.i
        dates = self.market.weights.index
        self.forecast.iloc[:, :] = np.linspace(-0.002, 0.002, len(self.base))
        prefix = replace(
            self.market,
            **{
                field.name: getattr(self.market, field.name).iloc[: last + 1].copy()
                for field in fields(self.market)
                if field.name != "exchanges"
            },
        )
        prefix_forecast = self.forecast.iloc[: last + 1].copy()
        self.market.close.iloc[last + 1 :] *= 1000
        self.market.adjustment.iloc[last + 1 :] = 0.01
        self.market.adv.iloc[last + 1 :] = 1
        self.market.weights.iloc[last + 1 :, :2] = [0, 0.25]
        self.forecast.iloc[last + 1 :] = np.nan
        full_policy = self.policy()
        prefix_policy = RiskAwarePortfolio(
            prefix, prefix_forecast, 0.002, self.execution, 1.0
        )
        full, full_orders = simulate(
            self.market,
            full_policy,
            dates[26],
            dates[last],
            self.execution,
            detail=True,
        )
        short, short_orders = simulate(
            prefix, prefix_policy, dates[26], dates[last], self.execution, detail=True
        )
        pd.testing.assert_frame_equal(full, short, check_exact=True)
        pd.testing.assert_frame_equal(full_orders, short_orders, check_exact=True)
        self.assertEqual(full_policy.audit, prefix_policy.audit)
        self.assertLess(full.capacity_cut_value.max(), 1e-7)
