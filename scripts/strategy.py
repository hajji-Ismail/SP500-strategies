"""Backtest the daily top-k long/short portfolio against the available benchmark."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def calculate_max_drawdown(cumulative_value):
    """Largest peak-to-trough loss of a cumulative wealth series."""
    return (cumulative_value / cumulative_value.cummax() - 1).min()


def _load_benchmark(benchmark_path, dates, universe_returns):
    path = Path(benchmark_path)
    if path.exists():
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip()
        if {"Date", "Close"}.issubset(raw.columns):
            benchmark = raw.assign(date=pd.to_datetime(raw["Date"])).sort_values("date").set_index("date")["Close"].pct_change().shift(-1)
            benchmark = benchmark.reindex(dates).dropna()
            if not benchmark.empty:
                return benchmark.rename("benchmark_daily_return"), "S&P 500 Index"
    # The supplied HistoricalPrices.csv does not overlap the 2013-2018 constituent file.
    # This deterministic fallback keeps the script runnable but must not be presented as the index.
    return universe_returns.reindex(dates).dropna().rename("benchmark_daily_return"), "Equal-weight constituent proxy"


def run_backtest(signal_path=PROJECT_ROOT / "results/selected-model/ml_signal.csv", data_path=PROJECT_ROOT / "data/processed_data.csv", benchmark_path=PROJECT_ROOT / "data/HistoricalData.csv", output_dir=PROJECT_ROOT / "results/strategy", k=10):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(signal_path, parse_dates=["date"]).set_index(["date", "ticker"]).sort_index()
    data = pd.read_csv(data_path, parse_dates=["date"]).set_index(["date", "ticker"]).sort_index()
    merged = signals.join(data[["forward_return"]], how="inner").dropna()
    if merged.empty:
        raise ValueError("No overlapping signals and forward returns.")

    def daily_return(group):
        if len(group) < 2 * k:
            return np.nan
        ranked = group.sort_values("signal")
        # Gross capital is $1/day: +$0.50 long and -$0.50 short.
        return ranked.tail(k)["forward_return"].mean() * .5 - ranked.head(k)["forward_return"].mean() * .5

    strategy = merged.groupby(level="date", group_keys=False).apply(daily_return).dropna().rename("strategy_daily_return")
    universe = data["forward_return"].groupby(level="date").mean()
    if not Path(benchmark_path).exists():
        alternate = PROJECT_ROOT / "data/HistoricalPrices.csv"
        benchmark_path = alternate if alternate.exists() else benchmark_path
    benchmark, benchmark_label = _load_benchmark(benchmark_path, strategy.index, universe)
    returns = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    wealth = (1 + returns).cumprod()
    split = pd.Timestamp("2017-01-01")
    rows = []
    for period, mask in {"train": returns.index < split, "test": returns.index >= split}.items():
        period_returns = returns.loc[mask]
        for column in returns:
            value = (1 + period_returns[column]).cumprod()
            rows.extend([{"period": period, "series": column, "metric": "total_return", "value": (1 + period_returns[column]).prod() - 1}, {"period": period, "series": column, "metric": "max_drawdown", "value": calculate_max_drawdown(value)}])
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wealth.index, wealth["strategy_daily_return"], label="ML top-10 long/short strategy", color="navy")
    ax.plot(wealth.index, wealth["benchmark_daily_return"], label=benchmark_label, color="darkorange", linestyle="--")
    ax.axvline(split, color="crimson", linestyle=":", label="Train/test split")
    ax.set(xlabel="Date", ylabel="Cumulative wealth ($1 base)", title="Strategy and benchmark cumulative PnL")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "strategy.png", dpi=200); plt.close(fig)

    report = f"""# Quantitative ML Strategy Report

## Data and leakage controls
The data set contains daily OHLCV observations for S&P 500 constituents. Features are calculated separately per ticker and use only data available at the close of date D. `forward_return` is `(close[D+2] / close[D+1]) - 1`; the binary target is one when that return is positive. The train period ends on 2016-12-31 and the test period begins on 2017-01-01.

## Features and model
Features are RSI(14), MACD(12, 26, 9), and 20-day Bollinger upper, middle, and lower bands. The pipeline is `StandardScaler` followed by `XGBClassifier`; it contains no imputer or dimensionality-reduction stage because rows with incomplete indicator lookback are excluded. Hyperparameters and fold results are saved with the model artifacts.

## Validation and signals
The model uses ten expanding, date-level time-series folds. Its first training window is more than two years; every validation window is later in time and the final validation window ends before the test period. Training signals are strictly out-of-fold: each validation prediction comes from a model fitted only on preceding dates. The test signal comes from a model fitted on all training data.

## Strategy
For each date, the strategy buys the ten highest-probability names and shorts the ten lowest-probability names. It assigns $0.50 to each side, equally divided among its ten names, so gross daily capital is $1. Each position is multiplied by the same `forward_return` that the signal predicts: return(D+1, D+2).

## Results
See [results.csv](results.csv) for total return and maximum drawdown by period, and [strategy.png](strategy.png) for cumulative PnL. Benchmark used: **{benchmark_label}**. Replace the supplied non-overlapping historical-price file with `data/HistoricalData.csv` covering the backtest period to compare directly against the S&P 500 index.
"""
    (output_dir / "report.md").write_text(report)
    print(f"Saved strategy results to {output_dir}")


if __name__ == "__main__":
    run_backtest()
