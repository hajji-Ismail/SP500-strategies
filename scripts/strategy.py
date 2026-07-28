"""Create the strategy backtest and its three final output files."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features_engineering import feature_engineering, load_data


SIGNAL_FILE = "results/selected-model/ml_signal.csv"
STOCK_FILE = "data/all_stocks_5yr.csv"
BENCHMARK_FILE = "data/HistoricalData.csv"
OUTPUT_FOLDER = "results/strategy"
SPLIT_DATE = pd.Timestamp("2017-01-01")


def max_drawdown(values):
    return (values / values.cummax() - 1).min()


def get_benchmark(dates, stock_returns):
    for filename in [BENCHMARK_FILE, "data/HistoricalPrices.csv"]:
        if not os.path.exists(filename):
            continue
        prices = pd.read_csv(filename)
        prices.columns = prices.columns.str.strip()
        if "Date" in prices and "Close" in prices:
            prices["date"] = pd.to_datetime(prices["Date"])
            returns = prices.sort_values("date").set_index("date")["Close"].pct_change().shift(-1)
            returns = returns.reindex(dates).dropna()
            if not returns.empty:
                return returns.rename("benchmark"), "S&P 500"
    return stock_returns.reindex(dates).rename("benchmark"), "Equal-weight stock proxy"


def save_strategy_plot(wealth, benchmark_name):
    """Save results/strategy/strategy.png."""
    plt.figure(figsize=(12, 6))
    plt.plot(wealth.index, wealth["strategy"], label="ML long/short strategy", color="navy")
    plt.plot(wealth.index, wealth["benchmark"], label=benchmark_name, color="darkorange")
    plt.axvline(SPLIT_DATE, color="red", linestyle="--", label="Train/test split")
    plt.title("Strategy PnL vs benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL ($1 base)")
    plt.ylim(min(wealth.min()) * 0.95, max(wealth.max()) * 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/strategy.png", dpi=200)
    plt.close()


def save_results_csv(returns):
    """Save results/strategy/results.csv."""
    rows = []
    for period, period_returns in {"train": returns[returns.index < SPLIT_DATE], "test": returns[returns.index >= SPLIT_DATE]}.items():
        for name in ["strategy", "benchmark"]:
            wealth = (1 + period_returns[name]).cumprod()
            rows.append({"period": period, "series": name, "total_return": (1 + period_returns[name]).prod() - 1, "max_drawdown": max_drawdown(wealth)})
    results = pd.DataFrame(rows)
    results.to_csv(f"{OUTPUT_FOLDER}/results.csv", index=False)
    return results


def save_report(results, benchmark_name):
    """Save results/strategy/report.md."""
    report = f"""# Strategy Report

## Model and features
The model uses RSI, MACD and Bollinger Bands. It is evaluated with 10 expanding time-series folds. Features on date D only use information available on D, and the target is the return from D+1 to D+2.

## Strategy
Each day, the strategy buys the 10 highest model signals and shorts the 10 lowest signals. It allocates $0.50 to longs and $0.50 to shorts, for $1 total daily capital. PnL uses the forward return from D+1 to D+2.

## Results
Benchmark: **{benchmark_name}**.

```
{results.to_string(index=False)}
```

See [strategy.png](strategy.png) for the PnL chart.
"""
    with open(f"{OUTPUT_FOLDER}/report.md", "w") as file:
        file.write(report)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    signals = pd.read_csv(SIGNAL_FILE, parse_dates=["date"]).set_index(["date", "Name"])
    data = feature_engineering(load_data(STOCK_FILE))
    positions = signals.join(data[["forward_return"]], how="inner").dropna()

    def daily_strategy(group):
        if len(group) < 20:
            return np.nan
        ranked = group.sort_values("signal")
        return 0.5 * ranked.tail(10)["forward_return"].mean() - 0.5 * ranked.head(10)["forward_return"].mean()

    strategy = positions.groupby(level="date").apply(daily_strategy).dropna().rename("strategy")
    stock_returns = data["forward_return"].groupby(level="date").mean()
    benchmark, benchmark_name = get_benchmark(strategy.index, stock_returns)
    returns = pd.concat([strategy, benchmark], axis=1).dropna()
    wealth = (1 + returns).cumprod()

    save_strategy_plot(wealth, benchmark_name)
    results = save_results_csv(returns)
    save_report(results, benchmark_name)
    print("Saved strategy.png, results.csv and report.md")


if __name__ == "__main__":
    main()
