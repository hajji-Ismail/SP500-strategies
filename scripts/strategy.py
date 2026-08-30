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
import numpy as np
import pandas as pd

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
    
    # Format output DataFrame
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

    save_strategy_plot(wealth, "S&P 500")

if __name__ == "__main__":
    main()
