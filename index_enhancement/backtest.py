"""Daily long-only, next-open simulation with a cash and holdings ledger."""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShareExchange:
    trade_date: pd.Timestamp
    old_symbol: str
    new_symbol: str
    new_shares_per_old: float


@dataclass
class Market:
    weights: (
        pd.DataFrame
    )  # Available at decision close; provider applies publication lag.
    open: pd.DataFrame  # Raw prices and raw price limits.
    close: pd.DataFrame
    adjustment: (
        pd.DataFrame
    )  # Total-return units: adjusted price = raw price * adjustment.
    limit_up: pd.DataFrame
    limit_down: pd.DataFrame
    suspended: pd.DataFrame
    adv: pd.DataFrame  # Trailing traded value available at decision close.
    industry: pd.DataFrame
    benchmark: pd.Series  # Total-return index, close.
    exchanges: tuple[ShareExchange, ...] = ()

    def __post_init__(self):
        dates, names = self.weights.axes
        if (
            not isinstance(dates, pd.DatetimeIndex)
            or not dates.is_unique
            or not dates.is_monotonic_increasing
        ):
            raise ValueError("A unique ascending trading calendar is required")
        if not names.is_unique:
            raise ValueError("Duplicate security identifiers")
        for field in fields(self):
            if field.name == "exchanges":
                continue
            frame = getattr(self, field.name)
            if not frame.index.equals(dates):
                raise ValueError(f"Trading calendar mismatch: {field.name}")
            if field.name != "benchmark" and not frame.columns.equals(names):
                raise ValueError(f"Security axis mismatch: {field.name}")
        weights = self.weights.to_numpy()
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.allclose(weights.sum(axis=1), 1)
        ):
            raise ValueError(
                "Decision weights must be finite, long only and sum to one"
            )
        if not (np.isfinite(self.benchmark) & (self.benchmark > 0)).all():
            raise ValueError("Invalid benchmark close")
        for event in self.exchanges:
            if (
                event.old_symbol not in names
                or event.new_symbol not in names
                or event.old_symbol == event.new_symbol
                or not np.isfinite(event.new_shares_per_old)
                or event.new_shares_per_old <= 0
                or event.trade_date not in dates
            ):
                raise ValueError("Invalid share exchange")


@dataclass(frozen=True)
class Execution:
    initial_cash: float = 10_000_000
    buy_bps: float = 5
    sell_bps: float = 12
    adv_fraction: float = 0.01

    def __post_init__(self):
        values = [self.initial_cash, self.buy_bps, self.sell_bps, self.adv_fraction]
        if not np.isfinite(values).all() or self.initial_cash <= 0:
            raise ValueError("Invalid cash or execution parameters")
        if not (
            0 <= self.buy_bps < 10000
            and 0 <= self.sell_bps < 10000
            and 0 < self.adv_fraction <= 1
        ):
            raise ValueError("Invalid fees or participation fraction")


def simulate(market, target, start, end, execution=None, *, detail=False):
    """Return daily accounts and optional per-stock orders, fills and end-of-day positions.

    Orders are measured in total-return units at the previous close, with fractional
    shares and implicit reinvestment at ex-date. There is one trading event per day.
    """
    if execution is None:
        execution = Execution()
    dates, symbols = market.weights.axes
    policy = target if callable(target) else None
    if policy is None:
        if not target.index.equals(dates) or not target.columns.equals(symbols):
            raise ValueError("Target axes differ from market axes")
        w = target.to_numpy()
        if (
            not np.isfinite(w).all()
            or np.any(w < 0)
            or np.any(w.sum(axis=1) > 1 + 1e-9)
        ):
            raise ValueError(
                "Targets must be finite long-only weights without leverage"
            )
    rows = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    if len(rows) == 0 or rows[0] < 1:
        raise ValueError("Need a trading date and at least one prior decision close")

    adj = market.adjustment.where(
        market.adjustment.gt(0) & np.isfinite(market.adjustment)
    )
    conversion_adjustment = adj.ffill().to_numpy()
    by_day = {}
    for event in market.exchanges:
        by_day.setdefault(event.trade_date, []).append(event)
    raw_open = market.open.to_numpy()
    open_px = (market.open * adj).to_numpy()
    close_px = (market.close * adj).to_numpy()
    limits_up, limits_down = market.limit_up.to_numpy(), market.limit_down.to_numpy()
    suspended, adv = market.suspended.to_numpy(), market.adv.to_numpy()
    bench = market.benchmark.to_numpy()
    units = np.zeros(len(symbols))
    stale = np.zeros(len(symbols), dtype=int)
    mark = close_px[rows[0] - 1].copy()
    mark = np.where(np.isfinite(mark) & (mark > 0), mark, 0)
    cash = nav = execution.initial_cash
    buy_fee, sell_fee = execution.buy_bps / 10000, execution.sell_bps / 10000
    records, details = [], []

    for i in rows:
        previous_nav = nav
        if policy is None:
            decision = w[i - 1]
        else:
            decision = np.asarray(policy(i - 1, units * mark / nav, nav), dtype=float)
            if (
                decision.shape != units.shape
                or not np.isfinite(decision).all()
                or np.any(decision < 0)
                or decision.sum() > 1 + 1e-9
            ):
                raise ValueError("Policy returned invalid portfolio weights")
        # Commit to quantities before seeing any execution-day fields.
        desired = (
            np.divide(nav * decision, mark, out=units.copy(), where=mark > 0) - units
        )
        # Converting an unchanged weight back into units can leave a few ULPs
        # of dust. Keep a no-trade instruction from creating a spurious fill.
        roundoff = 8 * np.finfo(float).eps * np.abs(units)
        desired[np.abs(desired) <= roundoff] = 0
        capacity = (
            np.where(np.isfinite(adv[i - 1]), np.maximum(adv[i - 1], 0), 0)
            * execution.adv_fraction
        )
        max_units = np.divide(capacity, mark, out=np.zeros_like(mark), where=mark > 0)
        order = np.clip(desired, -max_units, max_units)
        capacity_cut = np.abs(desired - order) @ mark
        valid_open = np.isfinite(open_px[i]) & (open_px[i] > 0) & (suspended[i] == 0)
        can_buy = (
            valid_open
            & np.isfinite(limits_up[i])
            & (raw_open[i] < limits_up[i] - 0.001)
        )
        can_sell = (
            valid_open & (limits_down[i] > 0) & (raw_open[i] > limits_down[i] + 0.001)
        )
        at_open = np.where(valid_open, open_px[i], mark)
        prior_value = units @ mark
        converted = np.zeros_like(units)
        for event in by_day.get(dates[i], ()):
            old, new = symbols.get_indexer([event.old_symbol, event.new_symbol])
            if units[old] > 0:
                ratio = (
                    conversion_adjustment[i - 1, old]
                    * event.new_shares_per_old
                    / adj.iat[i, new]
                )
                if not np.isfinite(ratio) or not valid_open[new]:
                    raise ValueError(
                        f"Missing effective-date valuation for {event.new_symbol}"
                    )
                converted[new] += units[old] * ratio
                converted[old] -= units[old]
            order[old] = 0
        units += converted
        overnight = units @ at_open - prior_value
        sold = np.where(can_sell, np.minimum(units, np.maximum(-order, 0)), 0)
        sell_value = sold @ at_open
        cash += sell_value * (1 - sell_fee)
        buy_order = np.where(can_buy, np.maximum(order, 0), 0)
        needed = (buy_order @ at_open) * (1 + buy_fee)
        fraction = min(1.0, cash / needed) if needed > 0 else 1.0
        bought = buy_order * fraction
        buy_value = bought @ at_open
        cash -= buy_value * (1 + buy_fee)
        units += bought - sold
        cost = buy_value * buy_fee + sell_value * sell_fee
        if cash < -1e-6 or np.min(units) < -1e-10:
            raise ArithmeticError("Negative cash or short position")
        cash = max(cash, 0.0)
        valid_close = np.isfinite(close_px[i]) & (close_px[i] > 0)
        mark = np.where(valid_close, close_px[i], at_open)
        stale = np.where(valid_close | (units == 0), 0, stale + 1)
        intraday = units @ (mark - at_open)
        # Mark a prolonged unpriced entitlement to zero but retain units for any recovery.
        expired = (stale >= 60) & (units > 0)
        writeoff = units[expired] @ mark[expired]
        mark[expired] = 0
        nav = cash + units @ mark
        if nav <= 0:
            raise ArithmeticError(f"Account exhausted on {dates[i].date()}")
        if not np.isclose(
            nav,
            previous_nav + overnight + intraday - cost - writeoff,
            rtol=1e-11,
            atol=1e-6,
        ):
            raise ArithmeticError("Daily P&L did not reconcile")
        filled = bought - sold
        blocked = ((order > 0) & ~can_buy) | ((order < 0) & ~can_sell)
        records.append(
            (
                dates[i],
                nav,
                cash,
                nav / previous_nav - 1,
                bench[i] / bench[i - 1] - 1,
                buy_value,
                sell_value,
                cost,
                (buy_value + sell_value) / (2 * previous_nav),
                overnight,
                intraday,
                capacity_cut,
                np.abs(order[blocked]) @ at_open[blocked],
                (buy_order - bought) @ at_open,
                int((stale[units > 0]).max(initial=0)),
                writeoff,
            )
        )
        if detail:
            keep = (decision > 0) | (units > 0) | (order != 0) | (converted != 0)
            details.append(
                pd.DataFrame(
                    {
                        "date": dates[i],
                        "decision_date": dates[i - 1],
                        "symbol": symbols[keep],
                        "target_weight": decision[keep],
                        "order_units": order[keep],
                        "filled_units": filled[keep],
                        "signed_fill_value": (filled * at_open)[keep],
                        "blocked_at_open": blocked[keep],
                        "filled_shares": np.where(
                            filled != 0, filled * adj.iloc[i].to_numpy(), 0
                        )[keep],
                        "fill_price": np.where(filled != 0, raw_open[i], np.nan)[keep],
                        "cost": ((bought * buy_fee + sold * sell_fee) * at_open)[keep],
                        "position_value": (units * mark)[keep],
                        "position_units": units[keep],
                        "exchange_units": converted[keep],
                        "stale_mark": expired[keep],
                    }
                )
            )
    daily = pd.DataFrame(
        records,
        columns=[
            "date",
            "nav",
            "cash",
            "return",
            "benchmark_return",
            "buy_value",
            "sell_value",
            "cost",
            "one_way_turnover",
            "overnight_pnl",
            "intraday_pnl",
            "capacity_cut_value",
            "blocked_value",
            "cash_cut_value",
            "max_stale_closes",
            "writeoff",
        ],
    ).set_index("date")
    return daily, pd.concat(details, ignore_index=True) if details else None


def performance(daily):
    returns = daily["return"]
    active = returns - daily["benchmark_return"]
    te = active.std() * np.sqrt(252)
    relative = (1 + returns).cumprod() / (1 + daily["benchmark_return"]).cumprod()
    relative_peak = relative.cummax().clip(lower=1)
    return {
        "days": len(daily),
        "annual_active": float(active.mean() * 252),
        "tracking_error": float(te) if np.isfinite(te) else None,
        "information_ratio": float(active.mean() * 252 / te) if te > 0 else None,
        "relative_max_drawdown": float((relative / relative_peak - 1).min()),
        "mean_one_way_turnover": float(daily.one_way_turnover.mean()),
        "cost": float(daily.cost.sum()),
        "max_stale_closes": int(daily.max_stale_closes.max()),
        "writeoff": float(daily.writeoff.sum()),
    }
