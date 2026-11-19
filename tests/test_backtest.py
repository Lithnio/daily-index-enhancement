import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from index_enhancement.backtest import (
    Execution,
    Market,
    ShareExchange,
    simulate,
)


def fixture(n=6):
    dates = pd.bdate_range("2021-01-04", periods=n)

    def frame(value):
        return pd.DataFrame(value, index=dates, columns=["A", "B"], dtype=float)

    m = Market(
        frame(0.5),
        frame(10),
        frame(10),
        frame(1),
        frame(20),
        frame(1),
        frame(0),
        frame(1e7),
        frame(0),
        pd.Series(100.0, index=dates),
    )
    return m, frame


FREE = Execution(initial_cash=1000, buy_bps=0, sell_bps=0, adv_fraction=1)


class Accounting(unittest.TestCase):
    def test_previous_close_order_and_overnight_owner(self):
        m, f = fixture()
        dates = m.close.index
        target = f(0)
        target.iloc[0, 0] = 1
        m.close.iloc[1, 0] = 12
        m.open.iloc[2, 0] = 15
        daily, orders = simulate(m, target, dates[1], dates[2], FREE, detail=True)
        np.testing.assert_allclose(daily.nav, [1200, 1500])
        np.testing.assert_allclose(daily.cash, [0, 1500])
        self.assertEqual(orders.iloc[0].decision_date, dates[0])
        self.assertEqual(orders.iloc[-1].signed_fill_value, -1500)
        self.assertEqual(daily.overnight_pnl.iloc[-1], 300)

    def test_unchanged_holdings_do_not_generate_fractional_dust_fills(self):
        market, _ = fixture()
        market.open.loc[:, "A"] = market.close.loc[:, "A"] = 5.271
        market.close.iloc[1:, 0] = 13.1

        def policy(i, held, nav):
            return np.array([0.351, 0.0]) if i == 0 else held

        dates = market.weights.index
        daily, detail = simulate(market, policy, dates[1], dates[2], FREE, detail=True)
        self.assertEqual(daily.one_way_turnover.iloc[1], 0)
        self.assertTrue(detail[detail.date == dates[2]].fill_price.isna().all())

    def test_limits_halt_capacity_and_fees(self):
        m, f = fixture(7)
        dates = m.close.index
        target = f(0)
        target.iloc[:3, 0] = 1
        m.limit_up.iloc[1, 0] = 10
        m.suspended.iloc[2, 0] = 1
        m.adv.iloc[2, 0] = 400
        m.limit_down.iloc[4, 0] = 10
        m.suspended.iloc[5, 0] = 1
        config = replace(FREE, buy_bps=10, sell_bps=20)
        daily, _ = simulate(m, target, dates[1], dates[6], config)
        np.testing.assert_allclose(daily.buy_value, [0, 0, 400, 0, 0, 0])
        np.testing.assert_allclose(daily.sell_value, [0, 0, 0, 0, 0, 400])
        self.assertAlmostEqual(daily.cash.iloc[-1], 998.8)
        self.assertEqual(daily.capacity_cut_value.iloc[2], 600)

    def test_price_adjustments_and_cash_constraint(self):
        m, f = fixture()
        dates = m.close.index
        target = f(0)
        target["A"] = 1
        # Two-for-one split before open. Then model a dividend via total-return reinvestment.
        m.adjustment.iloc[2:, 0] = 2
        m.open.iloc[2:, 0] = m.close.iloc[2:, 0] = 5
        m.adjustment.iloc[3:, 0] = 2 * 5 / 4.5
        m.open.iloc[3:, 0] = m.close.iloc[3:, 0] = 4.5
        daily, _ = simulate(m, target, dates[1], dates[-1], replace(FREE, buy_bps=10))
        np.testing.assert_allclose(daily.nav, 1000 / 1.001)
        self.assertTrue((daily.cash >= 0).all())
        self.assertTrue((daily.buy_value.iloc[1:] < 1e-8).all())

    def test_merger_delivers_successor_and_preserves_pnl(self):
        m, f = fixture()
        dates = m.close.index
        target = f(0)
        target.iloc[:2, 0] = 1
        m.exchanges = (ShareExchange(dates[2], "A", "B", 2),)
        m.open.loc[dates[2] :, "A"] = np.nan
        m.close.loc[dates[2] :, "A"] = np.nan
        m.open.loc[dates[2] :, "B"] = m.close.loc[dates[2] :, "B"] = 6
        daily, records = simulate(m, target, dates[1], dates[3], FREE, detail=True)
        np.testing.assert_allclose(daily.nav, [1000, 1200, 1200])
        self.assertEqual(records.query("symbol == 'B'").exchange_units.sum(), 200)
        self.assertEqual(daily.cash.iloc[-1], 1200)

    def test_stale_asset_keeps_recovery_entitlement(self):
        m, f = fixture(65)
        dates = m.close.index
        t = f(0)
        t["A"] = 1
        m.open.iloc[2:63, 0] = m.close.iloc[2:63, 0] = np.nan
        # A cash reserve keeps the account solvent even when the asset is marked to zero.
        t["A"] = 0.8
        daily, _ = simulate(m, t, dates[1], dates[-1], FREE)
        self.assertEqual(daily.writeoff.sum(), 800)
        self.assertEqual(daily.nav.min(), 200)
        self.assertEqual(daily.nav.iloc[-1], 1000)

    def test_future_inputs_do_not_change_prior_book(self):
        m, f = fixture()
        dates = m.close.index
        target = f(0.5)

        def run():
            return simulate(m, target, dates[1], dates[-1], FREE, detail=True)

        previous, orders = run()
        m.open.iloc[-2:] *= 1.7
        m.close.iloc[-2:] *= 1.4
        target.iloc[-2:] = [1.0, 0.0]
        changed, new_orders = run()
        pd.testing.assert_frame_equal(previous.iloc[:-2], changed.iloc[:-2])
        pd.testing.assert_frame_equal(
            orders[orders.date < dates[-2]], new_orders[new_orders.date < dates[-2]]
        )
        # Same submitted orders for first changed opening, even though fills can vary.
        actual = new_orders[new_orders.date == dates[-2]].order_units
        pd.testing.assert_series_equal(
            orders[orders.date == dates[-2]].order_units, actual
        )

    def test_policy_receives_realized_holdings_after_unfilled_orders(self):
        m, _ = fixture()
        dates = m.weights.index
        m.suspended.iloc[1] = 1
        m.adv.iloc[1] = 400
        seen = []

        def decide(i, held, nav):
            seen.append(held.copy())
            return m.weights.iloc[i].to_numpy()

        simulate(m, decide, dates[1], dates[3], FREE)
        np.testing.assert_array_equal(seen, [[0, 0], [0, 0], [0.4, 0.4]])


if __name__ == "__main__":
    unittest.main()
