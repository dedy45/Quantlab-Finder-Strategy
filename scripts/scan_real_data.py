"""
Strategy Scanner for REAL Data (ArcticDB).

Scans multiple strategies on real XAUUSD/EURUSD data from ArcticDB.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.data_engine import ArcticStore
from core.validation_engine import calculate_sharpe, calculate_psr, calculate_dsr
from strategies import MomentumStrategy, MeanReversionStrategy
from strategies.momentum_strategy import MomentumConfig
from strategies.mean_reversion import MeanReversionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def scan_strategy(
    df: pd.DataFrame,
    strategy,
    strategy_name: str
) -> Dict:
    """Scan single strategy on data."""
    
    # Ensure proper index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Split 70/30
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    # Fit and predict
    strategy.fit(train)
    signals = strategy.predict(test)
    
    # Calculate returns
    price_ret = test['close'].pct_change()
    strat_ret = signals.shift(1) * price_ret
    strat_ret = strat_ret.dropna()
    
    # Remove zeros and infinities
    strat_ret = strat_ret.replace([np.inf, -np.inf], np.nan).dropna()
    strat_ret = strat_ret[strat_ret != 0]
    
    if len(strat_ret) < 10:
        return {
            'name': strategy_name,
            'sharpe': 0.0,
            'psr': 0.5,
            'max_dd': 0.0,
            'total_return': 0.0,
            'n_obs': len(strat_ret)
        }
    
    # Metrics
    sharpe = calculate_sharpe(strat_ret)
    psr = calculate_psr(strat_ret)
    
    # Drawdown
    cum = (1 + strat_ret).cumprod()
    roll_max = cum.expanding().max()
    dd = (cum - roll_max) / roll_max
    max_dd = abs(dd.min()) if len(dd) > 0 else 0.0
    
    # Total return
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    
    return {
        'name': strategy_name,
        'sharpe': sharpe,
        'psr': psr,
        'max_dd': max_dd,
        'total_return': total_ret,
        'n_obs': len(strat_ret)
    }


def create_momentum_strategy(fast: int, slow: int, use_regime: bool = False) -> MomentumStrategy:
    """Create momentum strategy with config."""
    config = MomentumConfig(
        fast_period=fast, 
        slow_period=slow, 
        signal_type='ma_crossover',
        use_regime_filter=use_regime,
        hurst_threshold=0.55
    )
    return MomentumStrategy(config)


def create_dual_momentum_strategy(slow: int, lookback: int) -> MomentumStrategy:
    """Create dual momentum strategy."""
    config = MomentumConfig(
        slow_period=slow,
        momentum_lookback=lookback,
        signal_type='dual'
    )
    return MomentumStrategy(config)


def create_meanrev_strategy(signal_type: str, lookback: int = 20) -> MeanReversionStrategy:
    """Create mean reversion strategy with config."""
    config = MeanReversionConfig(signal_type=signal_type, lookback=lookback)
    return MeanReversionStrategy(config)


def main():
    """Run scanner on real data."""
    
    logger.info("=" * 60)
    logger.info("STRATEGY SCANNER - REAL DATA")
    logger.info("=" * 60)
    
    # Check available data (use single store instance)
    store = ArcticStore()
    available = store.list_symbols()
    logger.info(f"Available symbols: {available}")
    
    # Load XAUUSD 1H (real data) - use store directly to avoid multiple instances
    symbol = 'XAUUSD'
    timeframe = '1H'
    
    logger.info(f"\nLoading {symbol} {timeframe}...")
    df = store.read(symbol, timeframe, start='2015-01-01', end='2024-12-31')
    
    if df is None or len(df) == 0:
        logger.error(f"[FAIL] No data found for {symbol} {timeframe}")
        return False
    
    # Ensure proper DatetimeIndex
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    logger.info(f"Loaded {len(df)} bars")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Define strategies to scan
    strategies = [
        # Basic Momentum (MA Crossover)
        ('Mom_10_30', create_momentum_strategy(10, 30)),
        ('Mom_20_50', create_momentum_strategy(20, 50)),
        ('Mom_20_100', create_momentum_strategy(20, 100)),
        ('Mom_50_200', create_momentum_strategy(50, 200)),
        
        # Dual Momentum (Time-series + Cross-sectional)
        ('DualMom_50_126', create_dual_momentum_strategy(50, 126)),
        ('DualMom_100_252', create_dual_momentum_strategy(100, 252)),
        
        # Mean Reversion
        ('MeanRev_ZScore', create_meanrev_strategy('zscore', 20)),
        ('MeanRev_RSI', create_meanrev_strategy('rsi', 14)),
        ('MeanRev_Bollinger', create_meanrev_strategy('bollinger', 20)),
    ]
    
    logger.info(f"\nScanning {len(strategies)} strategies...")
    logger.info("-" * 60)
    
    results = []
    for name, strategy in strategies:
        try:
            result = scan_strategy(df, strategy, name)
            results.append(result)
            
            status = "[CANDIDATE]" if result['psr'] >= 0.95 else ""
            logger.info(
                f"{name:20s} | Sharpe={result['sharpe']:6.2f} | "
                f"PSR={result['psr']:6.1%} | MaxDD={result['max_dd']:6.1%} {status}"
            )
        except Exception as e:
            logger.error(f"{name}: Error - {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    candidates = [r for r in results if r['psr'] >= 0.95]
    logger.info(f"Total strategies: {len(results)}")
    logger.info(f"Candidates (PSR >= 95%): {len(candidates)}")
    
    if candidates:
        logger.info("\nCANDIDATES:")
        for c in candidates:
            logger.info(f"  {c['name']}: Sharpe={c['sharpe']:.2f}, PSR={c['psr']:.1%}")
    else:
        logger.info("\nNo candidates found with PSR >= 95%")
        logger.info("Top 3 by PSR:")
        sorted_results = sorted(results, key=lambda x: x['psr'], reverse=True)[:3]
        for r in sorted_results:
            logger.info(f"  {r['name']}: Sharpe={r['sharpe']:.2f}, PSR={r['psr']:.1%}")
    
    # Also scan EURUSD if available
    if 'EURUSD_1H' in available:
        logger.info("\n" + "=" * 60)
        logger.info("SCANNING EURUSD...")
        logger.info("=" * 60)
        
        df_eur = store.read('EURUSD', '1H', start='2020-01-01', end='2024-12-31')
        
        if df_eur is not None and len(df_eur) > 0:
            # Ensure proper DatetimeIndex
            if 'timestamp' in df_eur.columns:
                df_eur = df_eur.set_index('timestamp')
            if not isinstance(df_eur.index, pd.DatetimeIndex):
                df_eur.index = pd.to_datetime(df_eur.index)
            
            logger.info(f"Loaded {len(df_eur)} bars")
            
            eur_strategies = [
                ('Momentum_10_30', create_momentum_strategy(10, 30)),
                ('Momentum_20_50', create_momentum_strategy(20, 50)),
                ('MeanRev_ZScore', create_meanrev_strategy('zscore', 20)),
            ]
            
            for name, strategy in eur_strategies:
                try:
                    result = scan_strategy(df_eur, strategy, name)
                    logger.info(
                        f"{name:20s} | Sharpe={result['sharpe']:6.2f} | "
                        f"PSR={result['psr']:6.1%}"
                    )
                except Exception as e:
                    logger.error(f"{name}: Error - {e}")
    
    logger.info("\n[DONE]")
    return len(candidates) > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
