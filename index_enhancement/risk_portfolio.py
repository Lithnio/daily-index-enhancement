"""Benchmark-relative portfolio with daily diagonal risk and trade capacity."""

import numpy as np
import pandas as pd

_BISECTION_STEPS = 52


def _project_industries(reference, lower, upper, total):
    """Fit the feasible industry totals closest to the benchmark in squared error."""
    if total <= lower.sum():
        return lower
    if total >= upper.sum():
        return upper
    clipped = np.clip(reference, lower, upper)
    if clipped.sum() == total:
        return clipped

    left = (lower - reference).min()
    right = (upper - reference).max()
    for _ in range(_BISECTION_STEPS):
        mid = (left + right) / 2
        if np.clip(reference + mid, lower, upper).sum() > total:
            right = mid
        else:
            left = mid
    return np.clip(reference + left, lower, upper)


class RiskAwarePortfolio:
    """Allocate from the actual holdings at the decision close.

    Minimize diagonal active-weight risk and asymmetric trade costs less predicted
    return, holding each industry's projected budget constant. Trade capacity and
    the gross investment limit take priority over name and industry targets. The
    execution ledger funds fees and handles opening prices and tradability.

    The name floor is a fraction of benchmark weight. The name ceiling is the
    benchmark weight plus the lesser of the relative and absolute overweight limits.
    """

    def __init__(
        self,
        market,
        forecast,
        risk,
        execution,
        fee_multiplier,
        *,
        risk_tilt=1.0,
        name_floor=0.1,
        max_relative_overweight=3.0,
        max_absolute_overweight=0.01,
    ):
        if (
            not np.isfinite(risk)
            or risk <= 0
            or not np.isfinite(fee_multiplier)
            or fee_multiplier < 0
            or not np.isfinite(risk_tilt)
            or risk_tilt < 0
            or not market.weights.index.equals(forecast.index)
            or not market.weights.columns.equals(forecast.columns)
        ):
            raise ValueError(
                "Invalid forecast axes, risk, risk tilt or cost multiplier"
            )
        if not (
            0 <= name_floor <= 1
            and np.isfinite(max_relative_overweight)
            and max_relative_overweight >= 0
            and 0 <= max_absolute_overweight <= 1
        ):
            raise ValueError("Invalid name bounds")
        self.dates = market.weights.index
        self.base = market.weights.to_numpy()
        self.groups = market.industry.to_numpy()
        self.forecast = forecast.to_numpy()
        self.adv = market.adv.to_numpy()
        self.adv_fraction = execution.adv_fraction
        self.risk = risk
        self.risk_tilt = risk_tilt
        self.name_floor = name_floor
        self.max_relative_overweight = max_relative_overweight
        self.max_absolute_overweight = max_absolute_overweight
        self.buy = (execution.buy_bps / 10000) * fee_multiplier
        self.sell = (execution.sell_bps / 10000) * fee_multiplier
        self.audit = []

        self.variance = None
        if risk_tilt:
            adjustment = market.adjustment.where(market.adjustment.gt(0))
            with np.errstate(over="ignore", invalid="ignore"):
                price = market.close * adjustment
                price = price.where(price.gt(0) & np.isfinite(price))
                returns = price.pct_change(fill_method=None)
            self.variance = (
                returns.where(np.isfinite(returns))
                .rolling(60, min_periods=30)
                .var()
                .to_numpy()
            )

    def _risk_factor(self, i, inside):
        if self.variance is None:
            return 1.0
        variance = self.variance[i, inside]
        valid = np.isfinite(variance)
        reference = variance[valid & (variance > 0)]
        if reference.size == 0:
            return 1.0
        with np.errstate(over="ignore"):
            relative = np.clip(variance / np.median(reference), 0.25, 4.0)
            shrunk = np.where(valid, 0.5 + 0.5 * relative, 1.0)
            return np.clip(1 + self.risk_tilt * (shrunk - 1), 0.25, 4.0)

    def _industry_tilt(
        self, base, forecast, held, curvature, group, lower, upper, reference, available
    ):
        n_groups = len(reference)
        lower_total = np.bincount(group, weights=lower, minlength=n_groups)
        upper_total = np.bincount(group, weights=upper, minlength=n_groups)
        total = min(available, upper_total.sum())
        budget = _project_industries(reference, lower_total, upper_total, total)

        result = lower.copy()
        capped = (budget == upper_total)[group]
        result[capped] = upper[capped]
        free = (lower < upper) & ~capped & (budget > lower_total)[group]
        if not free.any():
            return result
        remaining = budget - np.bincount(
            group[~free], weights=result[~free], minlength=n_groups
        )
        scaling = (base / curvature)[free]
        b, mu, previous = base[free], forecast[free], held[free]
        sector, lo, hi = group[free], lower[free], upper[free]

        # Bracket each sector shadow price by the marginal values at its name bounds.
        left = np.full(n_groups, np.inf)
        right = np.full(n_groups, -np.inf)
        np.minimum.at(left, sector, mu - (hi - b) / scaling - self.buy)
        np.maximum.at(right, sector, mu - (lo - b) / scaling + self.sell)
        participating = np.isfinite(left)
        left[~participating] = right[~participating] = 0.0

        def weights_at(shadow_price):
            frictionless = b + scaling * (mu - shadow_price[sector])
            # The optimum stays at the current weight within the no-trade interval.
            weight = np.clip(
                previous,
                frictionless - scaling * self.buy,
                frictionless + scaling * self.sell,
            )
            return np.clip(weight, lo, hi)

        for _ in range(_BISECTION_STEPS):
            mid = (left + right) / 2
            over = np.bincount(sector, weights=weights_at(mid), minlength=n_groups)
            over = over > remaining
            left = np.where(over, mid, left)
            right = np.where(over, right, mid)
        result[free] = weights_at(right)
        return result

    def __call__(self, i, held, nav):
        benchmark = self.base[i]
        held = np.asarray(held, dtype=float)
        if (
            held.shape != benchmark.shape
            or not np.isfinite(held).all()
            or (held < 0).any()
            or held.sum() > 1 + 1e-12
            or not np.isfinite(nav)
            or nav <= 0
        ):
            raise ValueError("Expected long-only holdings and positive NAV")
        inside = benchmark > 0
        b = benchmark[inside]
        mu = self.forecast[i, inside]
        if not np.isfinite(mu).all():
            raise ValueError("Expected a finite forecast for every constituent")

        adv = self.adv[i]
        with np.errstate(over="ignore"):
            capacity = np.where(np.isfinite(adv) & (adv > 0), adv, 0.0)
            capacity = capacity * self.adv_fraction / nav
        trade_limit = np.minimum(capacity, 1.0)
        hard_lower = np.maximum(held - trade_limit, 0.0)
        hard_upper = held + trade_limit
        # Keep rounded trade bounds on the feasible side of the liquidity interval.
        hard_lower = np.where(
            held - hard_lower > trade_limit, np.nextafter(hard_lower, held), hard_lower
        )
        hard_upper = np.where(
            hard_upper - held > trade_limit, np.nextafter(hard_upper, held), hard_upper
        )

        outside = np.where(inside, 0.0, hard_lower)
        available = max(0.0, 1.0 - outside.sum())
        floor, ceiling = hard_lower[inside], hard_upper[inside]
        name_lower = self.name_floor * b
        name_upper = b + np.minimum(
            self.max_relative_overweight * b, self.max_absolute_overweight
        )
        lower = np.clip(name_lower, floor, ceiling)
        upper = np.clip(name_upper, floor, ceiling)
        if lower.sum() > available:
            # Reduce discretionary name floors proportionally if retaining them
            # would require borrowing against holdings that cannot be sold.
            uplift = lower - floor
            if uplift.sum() > 0:
                lower = (
                    floor + uplift * max(0.0, available - floor.sum()) / uplift.sum()
                )

        relevant = inside | (outside > 0)
        sectors, labels = pd.factorize(self.groups[i, relevant], use_na_sentinel=False)
        group = sectors[inside[relevant]]
        reference = np.bincount(group, weights=b, minlength=len(labels))
        retained = np.bincount(
            sectors, weights=outside[relevant], minlength=len(labels)
        )
        target = outside.copy()
        target[inside] = self._industry_tilt(
            b,
            mu,
            held[inside],
            self.risk * self._risk_factor(i, inside),
            group,
            lower,
            upper,
            reference - retained,
            available,
        )
        # Group and portfolio sums can round differently after the dual solves.
        excess = target.sum() - 1.0
        if excess > 0:
            slack = target - hard_lower
            roundoff = 8 * np.finfo(float).eps * held
            trading = (np.abs(target - held) > roundoff) & (slack >= excess)
            # Put the correction on an existing trade when it has room.
            j = np.argmax(np.where(trading, slack, -np.inf) if trading.any() else slack)
            target[j] = max(hard_lower[j], np.nextafter(target[j] - excess, 0.0))

        exposure = np.bincount(sectors, weights=target[relevant], minlength=len(labels))
        trades = np.abs(target - held)
        actual = target[inside]
        self.audit.append(
            {
                "decision_date": self.dates[i],
                "nav": float(nav),
                "cash_target": max(0.0, 1.0 - float(target.sum())),
                "max_industry_gap": float(np.max(np.abs(exposure - reference))),
                "planned_one_way": float(trades.sum() / 2),
                "max_capacity_excess": float(np.max(trades - capacity, initial=0.0)),
                "name_bound_gap": float(
                    max(
                        outside.max(initial=0.0),
                        (name_lower - actual).max(initial=0.0),
                        (actual - name_upper).max(initial=0.0),
                    )
                ),
                "outside_index_weight": float(outside.sum()),
            }
        )
        return target
