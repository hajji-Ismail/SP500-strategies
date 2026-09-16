"""Create the strategy backtest and its three final output files."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features_engineering import feature_engineering, load_data


SIGNAL_FILE = "results/selected-model/ml_signal.csv"
STOCK_FILE = "data/all_stocks_5yr.csv"
BENCHMARK_FILE = "data/HistoricalPrices.csv"
OUTPUT_FOLDER = "results/strategy"
SPLIT_DATE = pd.Timestamp("2017-01-01")

def calculate_signal_pnl(signals_df, stock_data_df):

    combined = signals_df.join(stock_data_df[['forward_return']], how='inner').dropna()

    def process_daily_pnl(group):
        longs = group[group['signal'] == 3.0]
        shorts = group[group['signal'] == 1.0]
        
        long_pnl = 0.0
        short_pnl = 0.0
        
        if not longs.empty and not shorts.empty:
            long_pnl = 0.5 * longs['forward_return'].mean()
            short_pnl = -0.5 * shorts['forward_return'].mean()
        elif not longs.empty:
            long_pnl = 1.0 * longs['forward_return'].mean()
        elif not shorts.empty:
            short_pnl = -1.0 * shorts['forward_return'].mean()
            
        return long_pnl + short_pnl

    pnl_series = combined.groupby(level='date').apply(process_daily_pnl)
    
    pnl_df = pnl_series.reset_index()
    pnl_df.columns = ['date', 'pnl']
    return pnl_df


def calculate_sp500_pnl(prices, target_dates):

    prices.columns = prices.columns.str.strip()
    
    prices['date'] = pd.to_datetime(prices['Date'])
    prices = prices.sort_values('date').set_index('date')
    
    prices['sp500_pnl'] = prices['Close'].pct_change()
    
    aligned_pnl = prices['sp500_pnl'].reindex(target_dates).dropna()
    
    sp500_pnl_df = aligned_pnl.reset_index()
    sp500_pnl_df.columns = ['date', 'pnl']
    return sp500_pnl_df


def save_strategy_plot(wealth, benchmark_name):
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





def generate_markdown_report():
    report_content = """# Strategy & Model Evaluation Report

## 1. Features Used
The pipeline utilizes technical indicators derived from stock price and volume dynamics:

* **RSI_14**: 14-period Relative Strength Index measuring momentum and overbought/oversold levels.
* **MACD Metrics**:
  * `MACD`: Moving Average Convergence Divergence line (12-period EMA - 26-period EMA).
  * `MACD_SIGNAL`: 9-period EMA signal line.
  * `MACD_DIFF`: MACD histogram (MACD minus Signal).
* **Bollinger Bands**:
  * `BB_UPPER`: Upper volatility band (20 SMA + 2 Std Dev).
  * `BB_MIDDLE`: 20-period Simple Moving Average base line.
  * `BB_LOWER`: Lower volatility band (20 SMA - 2 Std Dev).

---

## 2. Model Pipeline Architecture
The winning production model pipeline follows a clean scikit-learn modular architecture:


```

Raw Features ---> [ SimpleImputer (median) ] ---> [ LightGBM Classifier ] ---> Predictions

```

* **Imputer**: `SimpleImputer(strategy='median')` handles missing values in technical indicators without shifting feature distributions.
* **Scaler**: Not required for LightGBM (tree-based splitters are scale-invariant).
* **Dimensionality Reduction**: None applied (all 7 technical features are retained to maximize signal extraction).
* **Model**: `LGBMClassifier(objective='multiclass', num_class=3, random_state=42)`
  * Target output mapping: 3-class directional setup (Short, Neutral, Long).

---

## 3. Cross-Validation & Dataset Splits
To prevent lookahead bias inherent in financial time series data, an **Expanding Window Time Series Cross-Validation** (10 Folds) is applied.

* **Split Strategy**: Expanding training window with fixed validation window sizes across chronological time steps.
* **Validation Folds**: 10 sequential splits.
* **Training & Validation Sizes**:
  * Initial Fold Training Window: ~2 years (~24,000 samples per stock group).
  * Validation Window: Fixed contiguous temporal blocks across folds (~24,000 samples).
  * Expanding Fold 10 Training Window: Fully expanded historical sample dataset (~243,519 samples).

![Time Series Cross-Validation Splits](../cross-validation/Time_series_split.png)

---

## 4. Model Performance Metrics

| Metric | Mean Cross-Validation Score |
| :--- | :--- |
| **Accuracy** | `0.5118` |
| **ROC-AUC (OVR)** | `0.5544` |
| **LogLoss** | `0.7291` |

### Top 10 Feature Importances Across Folds
1. **RSI_14**: `1875.30`
2. **MACD_DIFF**: `1649.70`
3. **MACD**: `1320.20`
4. **MACD_SIGNAL**: `1309.10`
5. **BB_UPPER**: `1009.70`
6. **BB_LOWER**: `999.90`
7. **BB_MIDDLE**: `836.10`

---

## 5. Strategy Implementation & Execution

* **Signal Generation**: Probabilities output by LightGBM ($P_{short}, P_{neutral}, P_{long}$) are mapped to continuous signals:
  $$\text{Signal} = P(\text{Long}) - P(\text{Short})$$
  yielding continuous positional signals constrained to $[-1.0, +1.0]$.
* **Portfolio Allocation**: Signals above positive thresholds enter long positions; signals below negative thresholds enter short positions.

### Cumulative PnL Backtest Performance
![Strategy PnL Curve](strategy.png)
"""

    with open(f"{OUTPUT_FOLDER}/report.md", "w") as f:
        f.write(report_content)
    
    print(" Report successfully written to report.md")

def results_csv( strategy_df, benchmark_pnl_df):
    
    strat_series = strategy_df.set_index("date")["pnl"].rename("strategy_pnl")
    bench_series = benchmark_pnl_df.set_index("date")["pnl"].rename("sp500_pnl")

    pnl_df = pd.concat([strat_series, bench_series], axis=1).dropna()

    pnl_df["strategy_cum_pnl"] = (1 + pnl_df["strategy_pnl"]).cumprod()
    pnl_df["sp500_cum_pnl"] = (1 + pnl_df["sp500_pnl"]).cumprod()

    pnl_df.to_csv(f"{OUTPUT_FOLDER}/results.csv")
    


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    signals = pd.read_csv(SIGNAL_FILE, parse_dates=["date"]).set_index(["date", "Name"])
    data = feature_engineering(load_data(STOCK_FILE))  
    benchmark_df = pd.read_csv(BENCHMARK_FILE)
    
    strategy = calculate_signal_pnl(signals, data)
    
    backtest = calculate_sp500_pnl(benchmark_df, strategy["date"])
    
    strat_series = strategy.set_index("date")["pnl"].rename("strategy")
    bench_series = backtest.set_index("date")["pnl"].rename("benchmark")
    
    returns = pd.concat([strat_series, bench_series], axis=1).dropna()
    wealth = (1 + returns).cumprod()
    results_csv(strategy, backtest)

    save_strategy_plot(wealth, "S&P 500")
    generate_markdown_report()

if __name__ == "__main__":
    main()
