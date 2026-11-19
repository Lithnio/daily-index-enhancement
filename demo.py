"""Run a long-only portfolio on fictional prices and externally supplied forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from index_enhancement.backtest import Execution, Market, performance, simulate
from index_enhancement.risk_portfolio import RiskAwarePortfolio


def example(periods=100, stock_count=8):
    rng = np.random.default_rng(31)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    names = [f"sample_{i}" for i in range(stock_count)]

    def frame(values):
        return pd.DataFrame(values, index=dates, columns=names)

    close = frame(
        20 * np.exp(rng.normal(0, 0.012, (periods, stock_count)).cumsum(axis=0))
    )
    opening = close.shift(1).fillna(20) * frame(
        np.exp(rng.normal(0, 0.003, close.shape))
    )
    weights = frame(1 / stock_count)
    benchmark = (
        1 + close.pct_change(fill_method=None).mean(axis=1).fillna(0)
    ).cumprod()
    industry = frame(np.tile(np.arange(stock_count) // 4, (periods, 1)))
    market = Market(
        weights=weights,
        open=opening,
        close=close,
        adjustment=frame(1.0),
        limit_up=opening * 1.1,
        limit_down=opening * 0.9,
        suspended=frame(0.0),
        adv=frame(1e8),
        industry=industry,
        benchmark=benchmark,
    )
    # Independent random forecasts stand in for numbers supplied at the decision close.
    forecast = frame(np.random.default_rng(91).normal(0, 0.0015, close.shape))
    return market, forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, help="Save the synthetic daily account, fills and targets"
    )
    args = parser.parse_args()
    market, forecast = example()
    execution = Execution()
    # The risk scale is fixed in this example. The historical strategy updates it at each fit.
    policy = RiskAwarePortfolio(
        market,
        forecast,
        risk=0.5 * 0.002,
        execution=execution,
        fee_multiplier=1.0,
        name_floor=0.0,
        max_relative_overweight=5.0,
        max_absolute_overweight=0.02,
    )
    dates = market.weights.index
    daily, positions = simulate(
        market, policy, dates[1], dates[-1], execution, detail=bool(args.out)
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.out / "daily.csv")
        positions.to_csv(args.out / "positions.csv", index=False)
        pd.DataFrame(policy.audit).to_csv(args.out / "optimizer.csv", index=False)
    print(json.dumps(performance(daily), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
