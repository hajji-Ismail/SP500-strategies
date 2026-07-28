# S&P 500 Machine-Learning Long/Short Strategy

This project builds and backtests a daily, cross-sectional equity strategy using technical indicators and an XGBoost classifier. It follows a strict time-series workflow: observations are ordered by date, models are evaluated with expanding date-level folds, and training-period predictions are generated out of fold.

## Objective

Estimate whether a constituent's return from **D+1 to D+2** will be positive using information known at the close of **D**, then construct a dollar-neutral portfolio that buys the most positive predictions and shorts the least positive predictions.

## Methodology

Raw data is loaded from `data/all_stocks_5yr.csv`. All calculations are grouped by ticker before features or targets are produced.

| Component | Implementation |
| --- | --- |
| Features | RSI(14), MACD(12,26,9), MACD signal/difference, and 20-day Bollinger bands |
| Target | `sign(close[D+2] / close[D+1] - 1)` |
| Split | Train: dates before 2017-01-01; test: dates on/after 2017-01-01 |
| Validation | 10 expanding time-series folds, split by whole dates; first training window exceeds two years |
| Pipeline | `StandardScaler` → `XGBClassifier` |
| Portfolio | Long 10 highest signals / short 10 lowest signals; $0.50 per side, $1 gross capital per day |

The target alignment is deliberate. At D, indicators contain no data after D. The signal is used for the holding-period return from D+1 to D+2; using a contemporaneous or past return would misalign the strategy and can introduce leakage.

## Setup

Use Python 3.10+ and install the dependencies:

```bash
conda create --name <env> --file <this file>

```

## Run the workflow

Run commands from the project root, in this order:

```bash
python scripts/model_selection.py
python scripts/create_signal.py
python scripts/strategy.py
```

The engineered data is created in memory when needed; no `processed_data.csv` file is created.

`gridsearch.py` is optional if the default model settings are acceptable. When it runs, its selected parameters are saved and automatically consumed by `model_selection.py`.

## Outputs

| Location | Contents |
| --- | --- |
| `results/cross-validation/` | Fold diagram, metrics, metric plot, and per-fold top-ten feature importance |
| `results/selected-model/` | Serialized selected pipeline, readable settings/metrics, and ML signal |
| `results/strategy/` | Cumulative PnL plot, performance table, and backtest report |

For an index comparison, add `data/HistoricalData.csv` with `Date` and `Close` columns spanning the backtest dates. The repository's `HistoricalPrices.csv` begins in 2022 and therefore does not overlap its 2013–2018 constituent sample; the backtest explicitly labels its equal-weight-universe fallback rather than misrepresenting it as the S&P 500.

## Audit notes

- Features and the future return are computed independently within each ticker.
- All observations on a given date remain in the same CV partition.
- Validation signals are produced by a separate model fitted only on earlier fold dates.
- The final test prediction model is fitted only on pre-2017 observations.
- Backtest PnL always uses the forward return from D+1 to D+2.
