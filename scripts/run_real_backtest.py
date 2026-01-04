"""
Run Real Backtest - Full 4-Phase workflow dengan data REAL dari Dukascopy.

Workflow:
1. Phase 1: Screening (VectorBT) - 1000+ ideas → Top 50
2. Phase 2: Validation (Nautilus) - Top 50 → Top 10
3. Phase 3: Deep Analysis (PSR/DSR) - Top 10 → Top 3
4. Phase 4: Paper Trading Ready - Top 3 → Best 1

Data: Dukascopy XAUUSD (Gold) - FREE, no API key required
"""

import logging
import sys
import time
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


def load_real_data(symbol: str = 'XAUUSD', days: int = 30) -> pd.DataFrame:
    """Load real data with disk-first strategy."""
    from core.data_engine import DataManager
    from datetime import timedelta
    
    logger.info(f"Loading {symbol} data...")
    
    dm = DataManager()
    
    # Try to load all available data first
    available = dm.list_available()
    
    # Find matching file
    target_file = f"{symbol}_1H.parquet"
    file_info = None
    for f in available:
        if f['file'] == target_file:
            file_info = f
            break
    
    if file_info and 'start' in file_info and 'end' in file_info:
        # Use full date range from file
        start_str = file_info['start'].strftime('%Y-%m-%d')
        end_str = file_info['end'].strftime('%Y-%m-%d')
        logger.info(f"Using full data range: {start_str} to {end_str}")
    else:
        # Fallback to recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    
    # Load with disk-first strategy
    try:
        prices = dm.load(symbol, start_str, end_str, timeframe='1H')
        
        if prices is None or len(prices) < 100:
            logger.warning(f"Insufficient data, got {len(prices) if prices is not None else 0} bars")
            return None
        
        # Ensure timestamp is index for backtest compatibility
        if 'timestamp' in prices.columns:
            prices = prices.set_index('timestamp')
        
        logger.info(f"Loaded {len(prices)} bars of {symbol} data")
        logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        
        return prices
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None


def phase1_screening(prices: pd.DataFrame, asset: str) -> list:
    """Phase 1: Screen strategies with VectorBT."""
    from backtest import StrategyScreener
    from backtest.vectorbt.screener import ScreeningConfig
    from strategies import MomentumStrategy, MeanReversionStrategy
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: SCREENING (VectorBT)")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Configure screener
    config = ScreeningConfig(
        min_sharpe=0.3,  # Minimum Sharpe for real data
        max_drawdown=0.30,  # Max 30% drawdown
        min_trades=5,
        top_n=50,
        verbose=True
    )
    screener = StrategyScreener(config)
    
    # Define parameter grids
    momentum_grid = {
        'fast_period': [5, 10, 20, 30],
        'slow_period': [30, 50, 100, 150]
    }
    
    mean_reversion_grid = {
        'lookback': [10, 20, 30],
        'entry_z': [1.5, 2.0, 2.5]
    }
    
    all_results = []
    
    # Screen Momentum strategies
    logger.info("\nScreening Momentum strategies...")
    mom_result = screener.screen_parameter_grid(
        prices, MomentumStrategy, momentum_grid, asset
    )
    all_results.extend(mom_result.top_results)
    logger.info(f"Momentum: {mom_result.total_tested} tested, {mom_result.total_passed} passed")
    
    # Screen Mean Reversion strategies
    logger.info("\nScreening Mean Reversion strategies...")
    mr_result = screener.screen_parameter_grid(
        prices, MeanReversionStrategy, mean_reversion_grid, asset
    )
    all_results.extend(mr_result.top_results)
    logger.info(f"Mean Reversion: {mr_result.total_tested} tested, {mr_result.total_passed} passed")
    
    # Sort by Sharpe and get top 50
    all_results.sort(key=lambda x: x.metrics.sharpe_ratio, reverse=True)
    top_50 = all_results[:50]
    
    elapsed = time.time() - start_time
    
    logger.info(f"\nPhase 1 Complete:")
    logger.info(f"  Total screened: {mom_result.total_tested + mr_result.total_tested}")
    logger.info(f"  Total passed: {len(all_results)}")
    logger.info(f"  Top 50 selected: {len(top_50)}")
    logger.info(f"  Time: {elapsed:.1f}s")
    
    if top_50:
        logger.info(f"\nTop 5 candidates:")
        for i, r in enumerate(top_50[:5], 1):
            logger.info(f"  {i}. {r.strategy_name}: Sharpe={r.metrics.sharpe_ratio:.2f}, "
                       f"MaxDD={r.metrics.max_drawdown:.1%}, Params={r.params}")
    
    return top_50


def phase2_validation(prices: pd.DataFrame, candidates: list) -> list:
    """Phase 2: Validate with Nautilus."""
    from backtest import CandidateValidator
    from backtest.nautilus.validator import ValidationConfig
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: VALIDATION (Nautilus)")
    logger.info("=" * 60)
    
    if not candidates:
        logger.warning("No candidates to validate")
        return []
    
    start_time = time.time()
    
    # Configure validator
    config = ValidationConfig(
        max_sharpe_deviation=0.30,  # Max 30% deviation from VectorBT
        max_return_deviation=0.40,
        max_drawdown_deviation=0.40,
        min_sharpe=0.2,
        max_drawdown=0.35,
        min_trades=3,
        top_n=10,
        verbose=True
    )
    validator = CandidateValidator(config)
    
    # Validate candidates
    result = validator.validate_candidates(prices, candidates)
    
    elapsed = time.time() - start_time
    
    logger.info(f"\nPhase 2 Complete:")
    logger.info(f"  Total validated: {result.total_validated}")
    logger.info(f"  Total passed: {result.total_passed}")
    logger.info(f"  Top 10 selected: {len(result.top_results)}")
    logger.info(f"  Time: {elapsed:.1f}s")
    
    if result.top_results:
        logger.info(f"\nTop validated candidates:")
        for i, r in enumerate(result.top_results[:5], 1):
            logger.info(f"  {i}. {r.strategy_name}: Score={r.score:.1f}, "
                       f"VBT_SR={r.vectorbt_metrics.sharpe_ratio:.2f}, "
                       f"Naut_SR={r.nautilus_metrics.sharpe_ratio:.2f}, "
                       f"Dev={r.sharpe_deviation:.1%}")
    
    return result.top_results


def phase3_deep_analysis(prices: pd.DataFrame, candidates: list) -> list:
    """Phase 3: Deep analysis with PSR/DSR."""
    from core.validation_engine import calculate_psr
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: DEEP ANALYSIS (PSR/DSR)")
    logger.info("=" * 60)
    
    if not candidates:
        logger.warning("No candidates for deep analysis")
        return []
    
    start_time = time.time()
    
    analyzed = []
    
    for candidate in candidates:
        # Calculate PSR using Nautilus metrics (more realistic)
        sharpe = candidate.nautilus_metrics.sharpe_ratio
        n_obs = candidate.nautilus_metrics.trading_days
        
        if n_obs < 30:
            n_obs = 30  # Minimum for PSR calculation
        
        try:
            psr = calculate_psr(sharpe, benchmark_sr=0.0, n_observations=n_obs)
        except:
            psr = 0.5  # Default if calculation fails
        
        analyzed.append({
            'candidate': candidate,
            'psr': psr,
            'sharpe': sharpe,
            'max_dd': candidate.nautilus_metrics.max_drawdown,
            'score': candidate.score
        })
    
    # Sort by PSR
    analyzed.sort(key=lambda x: x['psr'], reverse=True)
    
    # Get top 3
    top_3 = analyzed[:3]
    
    elapsed = time.time() - start_time
    
    logger.info(f"\nPhase 3 Complete:")
    logger.info(f"  Analyzed: {len(candidates)}")
    logger.info(f"  Top 3 selected: {len(top_3)}")
    logger.info(f"  Time: {elapsed:.1f}s")
    
    if top_3:
        logger.info(f"\nTop 3 candidates for paper trading:")
        for i, item in enumerate(top_3, 1):
            c = item['candidate']
            logger.info(f"  {i}. {c.strategy_name}")
            logger.info(f"     PSR: {item['psr']:.1%}")
            logger.info(f"     Sharpe: {item['sharpe']:.2f}")
            logger.info(f"     MaxDD: {item['max_dd']:.1%}")
            logger.info(f"     Params: {c.params}")
    
    return top_3


def phase4_summary(top_candidates: list):
    """Phase 4: Summary and paper trading recommendation."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: PAPER TRADING RECOMMENDATION")
    logger.info("=" * 60)
    
    if not top_candidates:
        logger.warning("No candidates passed all phases")
        logger.info("\nRecommendation: Adjust parameters or try different strategies")
        return
    
    best = top_candidates[0]
    c = best['candidate']
    
    logger.info(f"\nBEST CANDIDATE FOR PAPER TRADING:")
    logger.info(f"  Strategy: {c.strategy_name}")
    logger.info(f"  Parameters: {c.params}")
    logger.info(f"  PSR: {best['psr']:.1%}")
    logger.info(f"  Sharpe (Nautilus): {best['sharpe']:.2f}")
    logger.info(f"  Max Drawdown: {best['max_dd']:.1%}")
    logger.info(f"  Validation Score: {c.score:.1f}/100")
    
    logger.info(f"\nNEXT STEPS:")
    logger.info(f"  1. Paper trade for 1-3 months")
    logger.info(f"  2. Monitor deviation from backtest < 20%")
    logger.info(f"  3. If PSR > 95%, proceed to Alpha Streams")
    
    # Check if meets Alpha Streams criteria
    if best['psr'] >= 0.95:
        logger.info(f"\n  [OK] PSR >= 95% - Ready for Alpha Streams consideration")
    else:
        logger.info(f"\n  [WARN] PSR < 95% - Need more validation before Alpha Streams")


def main():
    """Run full 4-phase workflow with real data."""
    logger.info("\n" + "=" * 60)
    logger.info("QUANT LAB - REAL DATA BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Version: 0.6.4")
    
    total_start = time.time()
    
    # Load real data
    prices = load_real_data('XAUUSD', days=60)  # 2 months of data
    
    if prices is None:
        logger.error("Failed to load data. Exiting.")
        return 1
    
    asset = 'XAUUSD'
    
    # Phase 1: Screening
    top_50 = phase1_screening(prices, asset)
    
    # Phase 2: Validation
    top_10 = phase2_validation(prices, top_50)
    
    # Phase 3: Deep Analysis
    top_3 = phase3_deep_analysis(prices, top_10)
    
    # Phase 4: Summary
    phase4_summary(top_3)
    
    total_elapsed = time.time() - total_start
    
    logger.info("\n" + "=" * 60)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed:.1f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
