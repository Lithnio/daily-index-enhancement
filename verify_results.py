"""Verify reported return and risk statistics from the aggregate daily NAV."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SESSIONS_PER_YEAR = 252
STRATEGIES = ("nominal", "passive", "sector_cost", "double_fees", "capacity")


def nav_statistics(strategy, benchmark):
    """The first observation of each aligned series is the opening close."""
    if len(strategy) < 2 or not strategy.index.equals(benchmark.index):
        raise ValueError("NAV series need aligned opening and trading closes")
    for series in (strategy, benchmark):
        if not (np.isfinite(series) & series.gt(0)).all():
            raise ValueError("NAV series must be finite and positive")

    strategy = strategy / strategy.iloc[0]
    benchmark = benchmark / benchmark.iloc[0]
    relative = strategy / benchmark
    active = strategy.pct_change().iloc[1:] - benchmark.pct_change().iloc[1:]
    annual_active = float(active.mean() * SESSIONS_PER_YEAR)
    te = float(active.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR))
    years = len(active) / SESSIONS_PER_YEAR

    return {
        "strategy_cumulative_return": float(strategy.iloc[-1] - 1),
        "benchmark_cumulative_return": float(benchmark.iloc[-1] - 1),
        "cumulative_return_difference": float(strategy.iloc[-1] - benchmark.iloc[-1]),
        "cumulative_relative_return": float(relative.iloc[-1] - 1),
        "annualized_strategy_return": float(strategy.iloc[-1] ** (1 / years) - 1),
        "annualized_benchmark_return": float(benchmark.iloc[-1] ** (1 / years) - 1),
        "annualized_relative_return": float(relative.iloc[-1] ** (1 / years) - 1),
        "annual_active": annual_active,
        "tracking_error": te if math.isfinite(te) else None,
        "information_ratio": annual_active / te if te > 0 else None,
        "strategy_max_drawdown": float((strategy / strategy.cummax() - 1).min()),
        "benchmark_max_drawdown": float((benchmark / benchmark.cummax() - 1).min()),
        "relative_max_drawdown": float((relative / relative.cummax() - 1).min()),
    }


def verify(results):
    metrics = json.loads((results / "full-metrics.json").read_text())
    nav = pd.read_csv(results / "full-nav.csv", index_col="date", parse_dates=["date"])
    if (
        not isinstance(nav.index, pd.DatetimeIndex)
        or nav.index.hasnans
        or not nav.index.is_unique
        or not nav.index.is_monotonic_increasing
        or len(nav) < 2
        or not nav.iloc[0].eq(1).all()
    ):
        raise ValueError("Expected an ascending NAV calendar with an opening row at 1")
    if metrics["sessions_per_year"] != SESSIONS_PER_YEAR:
        raise ValueError("Unexpected annualization convention")
    full = metrics["periods"]["full"]
    if nav.index[0] != pd.Timestamp(full["opening_close"]) or nav.index[
        -1
    ] != pd.Timestamp(full["last_trade"]):
        raise ValueError("Reported full period does not match the NAV calendar")

    for name in STRATEGIES:
        if not np.allclose(
            nav[f"{name}_relative_nav"],
            nav[f"{name}_nav"] / nav["benchmark_nav"],
            rtol=0,
            atol=2e-11,
        ):
            raise ValueError(f"Relative NAV does not reconcile for {name}")

    for section in ("periods", "annual", "half_year"):
        for label, period in metrics[section].items():
            opening, first, last = map(
                pd.Timestamp,
                (period["opening_close"], period["first_trade"], period["last_trade"]),
            )
            span = nav.loc[opening:last]
            if (
                len(span) < 2
                or len(span) != period["days"] + 1
                or (span.index[0], span.index[1], span.index[-1])
                != (opening, first, last)
            ):
                raise ValueError(
                    f"Reported dates do not match the NAV calendar for {label}"
                )
            for name in STRATEGIES:
                published = period["accounts"][name]
                calculated = nav_statistics(span[f"{name}_nav"], span["benchmark_nav"])
                for key, actual in calculated.items():
                    expected = published[key]
                    same = (
                        actual is expected
                        if actual is None or expected is None
                        else math.isclose(actual, expected, rel_tol=2e-9, abs_tol=2e-9)
                    )
                    if not same:
                        raise ValueError(
                            f"NAV statistic differs for {label}/{name}/{key}"
                        )
    return full["days"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    try:
        days = verify(args.results)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified NAV-based return and risk statistics for {days:,} sessions.")
