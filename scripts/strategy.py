import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_max_drawdown(cumulative_pnl):
    peak = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - peak) / peak.replace(0, 1)
    return drawdown.min()

def run_backtest(signal_path="results/selected-model/ml_signal.csv",
                 data_path="data/processed_data.csv",
                 benchmark_path="data/HistoricalData.csv",
                 output_dir="results/strategy"):
    
    print("--- [Task 5] Starting Strategy Backtest ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    signals = pd.read_csv(signal_path)
    signals['date'] = pd.to_datetime(signals['date'])
    signals.set_index(['date', 'ticker'], inplace=True)
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date', 'ticker'], inplace=True)
    
    merged = signals.join(df['forward_return'], how='inner').dropna()
    
    # 2. Strategy Rules: Long/Short Top-K Picking ($1 Budget Allocation per Day)
    # Long top 10 stocks (highest probability), Short bottom 10 stocks (lowest probability)
    k = 10
    
    def calculate_daily_pnl(group):
        if len(group) < 2 * k:
            return 0.0
        
        sorted_group = group.sort_values('signal', ascending=False)
        longs = sorted_group.head(k)
        shorts = sorted_group.tail(k)
        
        # $0.50 budget allocated to Longs, $0.50 budget allocated to Shorts
        long_return = (longs['forward_return'] * (0.5 / k)).sum()
        short_return = (-shorts['forward_return'] * (0.5 / k)).sum()
        
        return long_return + short_return

    daily_pnl = merged.groupby('date').apply(calculate_daily_pnl)
    daily_pnl.name = 'strategy_daily_return'
    
    # Cumulative PnL ($1 initial investment base)
    strat_cum = (1 + daily_pnl).cumprod()
    
    # 3. Load S&P 500 Benchmark Data
    sp500 = pd.read_csv(benchmark_path)
    sp500['date'] = pd.to_datetime(sp500['Date'])
    sp500.sort_values('date', inplace=True)
    sp500.set_index('date', inplace=True)
    
    # Compute daily percentage change
    sp500['sp500_daily_return'] = sp500['Close'].pct_change().shift(-1) # Forward return aligned
    sp500_aligned = sp500.loc[daily_pnl.index].dropna()
    sp500_cum = (1 + sp500_aligned['sp500_daily_return']).cumprod()
    
    # 4. Metrics Calculation (Train vs Test)
    split_date = pd.to_datetime('2017-01-01')
    
    train_strat = daily_pnl[daily_pnl.index < split_date]
    test_strat = daily_pnl[daily_pnl.index >= split_date]
    
    train_sp = sp500_aligned['sp500_daily_return'][sp500_aligned.index < split_date]
    test_sp = sp500_aligned['sp500_daily_return'][sp500_aligned.index >= split_date]
    
    metrics = {
        'Train PnL Total Return': (1 + train_strat).prod() - 1,
        'Test PnL Total Return': (1 + test_strat).prod() - 1,
        'Train S&P 500 Return': (1 + train_sp).prod() - 1,
        'Test S&P 500 Return': (1 + test_sp).prod() - 1,
        'Train Max Drawdown': calculate_max_drawdown((1 + train_strat).cumprod()),
        'Test Max Drawdown': calculate_max_drawdown((1 + test_strat).cumprod())
    }
    
    res_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    res_df.to_csv(os.path.join(output_dir, "results.csv"))
    
    # 5. Plot Cumulative PnL Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(strat_cum.index, strat_cum.values, label='ML Long-Short Strategy PnL', color='navy', lw=2)
    plt.plot(sp500_cum.index, sp500_cum.values, label='S&P 500 Index Benchmark', color='orange', lw=1.5, linestyle='--')
    plt.axvline(x=split_date, color='red', linestyle=':', label='Train/Test Split (2017-01-01)')
    
    plt.title('Cumulative Profit and Loss (PnL) Strategy vs S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return ($1 Base)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "strategy.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Generate Markdown Report
    report_content = f"""# Quantitative Machine Learning Strategy Report

## Strategy Overview
* **Model Pipeline:** Imputer, Scaler, Classifier Pipeline (Trained on S&P 500 Constituents).
* **Technical Indicators:** Bollinger Bands ($%B$), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD).
* **Target:** Binary direction of forward return $R(d+1, d+2) > 0$.
* **Allocation Scheme:** Long top 10 stocks, Short bottom 10 stocks, allocating strictly $1 per day.

## Performance Metrics
| Metric | Strategy | S&P 500 Benchmark |
|---|---|---|
| **Train Return (<2017)** | {metrics['Train PnL Total Return']:.2%} | {metrics['Train S&P 500 Return']:.2%} |
| **Test Return (≥2017)** | {metrics['Test PnL Total Return']:.2%} | {metrics['Test S&P 500 Return']:.2%} |
| **Train Max Drawdown** | {metrics['Train Max Drawdown']:.2%} | - |
| **Test Max Drawdown** | {metrics['Test Max Drawdown']:.2%} | - |

## PnL Chart
![Strategy PnL](strategy.png)
"""
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(report_content)
        
    print(f"--- [Task 5] Backtest finished. Artifacts saved in {output_dir} ---")

if __name__ == "__main__":
    run_backtest()