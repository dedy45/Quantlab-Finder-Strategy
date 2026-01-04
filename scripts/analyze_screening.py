"""
Analyze Screening Results - Detailed analysis of strategy performance.

Tujuan:
- Melihat distribusi Sharpe Ratio dari semua strategi
- Identifikasi kenapa tidak ada yang lolos filter
- Rekomendasi parameter adjustment
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """Load XAUUSD data."""
    from core.data_engine import DataManager
    
    dm = DataManager()
    available = dm.list_available()
    
    # Find 1H file
    for f in available:
        if f['file'] == 'XAUUSD_1H.parquet':
            start = f['start'].strftime('%Y-%m-%d')
            end = f['end'].strftime('%Y-%m-%d')
            break
    
    prices = dm.load('XAUUSD', start, end, timeframe='1H')
    
    if 'timestamp' in prices.columns:
        prices = prices.set_index('timestamp')
    
    return prices


def analyze_momentum(prices: pd.DataFrame):
    """Analyze all momentum parameter combinations."""
    from strategies import MomentumStrategy
    from strategies.momentum_strategy import MomentumConfig
    from backtest import VectorBTAdapter
    
    logger.info("\n" + "=" * 60)
    logger.info("MOMENTUM STRATEGY ANALYSIS")
    logger.info("=" * 60)
    
    adapter = VectorBTAdapter()
    
    # Extended parameter grid
    fast_periods = [5, 10, 15, 20, 30, 50]
    slow_periods = [30, 50, 100, 150, 200]
    
    results = []
    
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            
            try:
                config = MomentumConfig(fast_period=fast, slow_period=slow)
                strategy = MomentumStrategy(config)
                strategy.fit(prices)
                signals = strategy.predict(prices)
                
                result = adapter.run(prices, signals, 'XAUUSD', f'MOM_{fast}_{slow}')
                
                results.append({
                    'fast': fast,
                    'slow': slow,
                    'sharpe': result.metrics.sharpe_ratio,
                    'max_dd': result.metrics.max_drawdown,
                    'total_return': result.metrics.total_return,
                    'trades': result.metrics.total_trades
                })
            except Exception as e:
                logger.warning(f"Error with fast={fast}, slow={slow}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    logger.info(f"\nTotal combinations tested: {len(df)}")
    logger.info(f"\nSharpe Ratio Distribution:")
    logger.info(f"  Min: {df['sharpe'].min():.3f}")
    logger.info(f"  Max: {df['sharpe'].max():.3f}")
    logger.info(f"  Mean: {df['sharpe'].mean():.3f}")
    logger.info(f"  Median: {df['sharpe'].median():.3f}")
    logger.info(f"  Std: {df['sharpe'].std():.3f}")
    
    logger.info(f"\nMax Drawdown Distribution:")
    logger.info(f"  Min: {df['max_dd'].min():.1%}")
    logger.info(f"  Max: {df['max_dd'].max():.1%}")
    logger.info(f"  Mean: {df['max_dd'].mean():.1%}")
    
    # Count passing filter
    passing = df[(df['sharpe'] > 0.3) & (df['max_dd'] < 0.30)]
    logger.info(f"\nPassing filter (Sharpe>0.3, MaxDD<30%): {len(passing)}/{len(df)}")
    
    # Show top 10
    df_sorted = df.sort_values('sharpe', ascending=False)
    logger.info(f"\nTop 10 Momentum combinations:")
    for i, row in df_sorted.head(10).iterrows():
        status = "[OK]" if row['sharpe'] > 0.3 and row['max_dd'] < 0.30 else "[FAIL]"
        logger.info(f"  {status} fast={row['fast']}, slow={row['slow']}: "
                   f"Sharpe={row['sharpe']:.3f}, MaxDD={row['max_dd']:.1%}, "
                   f"Return={row['total_return']:.1%}")
    
    return df


def analyze_mean_reversion(prices: pd.DataFrame):
    """Analyze all mean reversion parameter combinations."""
    from strategies import MeanReversionStrategy
    from strategies.mean_reversion import MeanReversionConfig
    from backtest import VectorBTAdapter
    
    logger.info("\n" + "=" * 60)
    logger.info("MEAN REVERSION STRATEGY ANALYSIS")
    logger.info("=" * 60)
    
    adapter = VectorBTAdapter()
    
    # Extended parameter grid
    lookbacks = [5, 10, 15, 20, 30, 50]
    entry_zs = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    results = []
    
    for lookback in lookbacks:
        for entry_z in entry_zs:
            try:
                config = MeanReversionConfig(lookback=lookback, entry_z=entry_z)
                strategy = MeanReversionStrategy(config)
                strategy.fit(prices)
                signals = strategy.predict(prices)
                
                result = adapter.run(prices, signals, 'XAUUSD', f'MR_{lookback}_{entry_z}')
                
                results.append({
                    'lookback': lookback,
                    'entry_z': entry_z,
                    'sharpe': result.metrics.sharpe_ratio,
                    'max_dd': result.metrics.max_drawdown,
                    'total_return': result.metrics.total_return,
                    'trades': result.metrics.total_trades
                })
            except Exception as e:
                logger.warning(f"Error with lookback={lookback}, entry_z={entry_z}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    logger.info(f"\nTotal combinations tested: {len(df)}")
    logger.info(f"\nSharpe Ratio Distribution:")
    logger.info(f"  Min: {df['sharpe'].min():.3f}")
    logger.info(f"  Max: {df['sharpe'].max():.3f}")
    logger.info(f"  Mean: {df['sharpe'].mean():.3f}")
    logger.info(f"  Median: {df['sharpe'].median():.3f}")
    logger.info(f"  Std: {df['sharpe'].std():.3f}")
    
    logger.info(f"\nMax Drawdown Distribution:")
    logger.info(f"  Min: {df['max_dd'].min():.1%}")
    logger.info(f"  Max: {df['max_dd'].max():.1%}")
    logger.info(f"  Mean: {df['max_dd'].mean():.1%}")
    
    # Count passing filter
    passing = df[(df['sharpe'] > 0.3) & (df['max_dd'] < 0.30)]
    logger.info(f"\nPassing filter (Sharpe>0.3, MaxDD<30%): {len(passing)}/{len(df)}")
    
    # Show top 10
    df_sorted = df.sort_values('sharpe', ascending=False)
    logger.info(f"\nTop 10 Mean Reversion combinations:")
    for i, row in df_sorted.head(10).iterrows():
        status = "[OK]" if row['sharpe'] > 0.3 and row['max_dd'] < 0.30 else "[FAIL]"
        logger.info(f"  {status} lookback={row['lookback']}, entry_z={row['entry_z']}: "
                   f"Sharpe={row['sharpe']:.3f}, MaxDD={row['max_dd']:.1%}, "
                   f"Return={row['total_return']:.1%}")
    
    return df


def analyze_market_regime(prices: pd.DataFrame):
    """Analyze market regime to understand why strategies fail."""
    logger.info("\n" + "=" * 60)
    logger.info("MARKET REGIME ANALYSIS")
    logger.info("=" * 60)
    
    # Calculate returns
    returns = prices['close'].pct_change().dropna()
    
    # Basic statistics
    logger.info(f"\nReturn Statistics (10 years):")
    logger.info(f"  Mean daily return: {returns.mean():.4%}")
    logger.info(f"  Std daily return: {returns.std():.4%}")
    logger.info(f"  Annualized return: {returns.mean() * 252:.2%}")
    logger.info(f"  Annualized volatility: {returns.std() * np.sqrt(252):.2%}")
    logger.info(f"  Buy-and-hold Sharpe: {(returns.mean() / returns.std()) * np.sqrt(252):.3f}")
    
    # Trend analysis
    logger.info(f"\nTrend Analysis:")
    
    # Calculate rolling Hurst exponent proxy (autocorrelation)
    autocorr_1 = returns.autocorr(lag=1)
    autocorr_5 = returns.autocorr(lag=5)
    autocorr_20 = returns.autocorr(lag=20)
    
    logger.info(f"  Autocorrelation lag-1: {autocorr_1:.4f}")
    logger.info(f"  Autocorrelation lag-5: {autocorr_5:.4f}")
    logger.info(f"  Autocorrelation lag-20: {autocorr_20:.4f}")
    
    if autocorr_1 < 0:
        logger.info(f"  [INFO] Negative autocorr suggests mean-reverting behavior")
    else:
        logger.info(f"  [INFO] Positive autocorr suggests trending behavior")
    
    # Volatility regime
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    
    logger.info(f"\nVolatility Regime:")
    logger.info(f"  Current vol: {rolling_vol.iloc[-1]:.2%}")
    logger.info(f"  Mean vol: {rolling_vol.mean():.2%}")
    logger.info(f"  Min vol: {rolling_vol.min():.2%}")
    logger.info(f"  Max vol: {rolling_vol.max():.2%}")
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    
    logger.info(f"\nDrawdown Analysis:")
    logger.info(f"  Max drawdown: {drawdown.min():.2%}")
    logger.info(f"  Mean drawdown: {drawdown.mean():.2%}")
    
    return {
        'autocorr_1': autocorr_1,
        'mean_vol': rolling_vol.mean(),
        'max_dd': drawdown.min()
    }


def main():
    """Run detailed screening analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("QUANT LAB - SCREENING ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    logger.info("\nLoading data...")
    prices = load_data()
    logger.info(f"Loaded {len(prices)} bars")
    logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Analyze market regime first
    regime = analyze_market_regime(prices)
    
    # Analyze strategies
    mom_df = analyze_momentum(prices)
    mr_df = analyze_mean_reversion(prices)
    
    # Summary and recommendations
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY & RECOMMENDATIONS")
    logger.info("=" * 60)
    
    best_mom_sharpe = mom_df['sharpe'].max()
    best_mr_sharpe = mr_df['sharpe'].max()
    
    logger.info(f"\nBest Sharpe found:")
    logger.info(f"  Momentum: {best_mom_sharpe:.3f}")
    logger.info(f"  Mean Reversion: {best_mr_sharpe:.3f}")
    
    if best_mom_sharpe < 0.3 and best_mr_sharpe < 0.3:
        logger.info(f"\n[WARN] No strategy achieves Sharpe > 0.3")
        logger.info(f"\nPossible reasons:")
        logger.info(f"  1. Simple strategies don't capture alpha in XAUUSD")
        logger.info(f"  2. Market is efficient for basic technical signals")
        logger.info(f"  3. Need regime-adaptive or ML-based strategies")
        
        logger.info(f"\nRecommendations:")
        logger.info(f"  1. Try ML strategy (RandomForest, LightGBM)")
        logger.info(f"  2. Add regime filter (trade only in favorable conditions)")
        logger.info(f"  3. Try different assets (F_ES, F_GC futures)")
        logger.info(f"  4. Lower filter threshold for exploration (Sharpe > 0.1)")
    else:
        logger.info(f"\n[OK] Found strategies with Sharpe > 0.3")
        logger.info(f"  Proceed with validation phase")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
