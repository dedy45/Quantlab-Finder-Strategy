"""
Strategy Scanner for FASE 5: Production.

Systematically scans multiple strategies across assets to find
candidates with PSR > 95%.

Version: 0.6.1
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Setup path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.validation_engine import calculate_sharpe, calculate_psr, calculate_dsr
from strategies import MomentumStrategy, MeanReversionStrategy
from strategies.base import BaseStrategy, BacktestResult
from strategies.momentum_strategy import MomentumConfig
from strategies.mean_reversion import MeanReversionConfig

logger = logging.getLogger(__name__)

# Constants
MIN_DATA_POINTS = 100
MIN_TEST_OBSERVATIONS = 60
TRAIN_RATIO = 0.7
MAX_DAILY_RETURN = 0.5  # 50% max daily return (clip extreme)


@dataclass
class ScanResult:
    """Result of strategy scan."""
    
    strategy_name: str
    asset: str
    params: Dict[str, Any]
    sharpe_ratio: float
    psr: float
    dsr: float
    max_drawdown: float
    total_return: float
    n_trades: int
    win_rate: float
    returns: pd.Series = field(default=None, repr=False)
    
    @property
    def is_candidate(self) -> bool:
        """Check if meets PSR > 95% threshold."""
        return self.psr >= 0.95
    
    @property
    def meets_all_criteria(self) -> bool:
        """Check if meets all Alpha Streams criteria."""
        return (
            self.psr >= 0.95 and
            self.max_drawdown <= 0.20 and
            self.sharpe_ratio >= 0.5
        )


class StrategyScanner:
    """
    Systematic strategy scanner.
    
    Scans multiple strategies across assets to find candidates
    with PSR > 95%.
    
    Parameters
    ----------
    assets : List[str]
        List of asset symbols to scan
    min_date : str
        Minimum date for data
    max_date : str, optional
        Maximum date for data
    """
    
    # Strategy configurations to scan
    STRATEGY_CONFIGS = {
        'momentum_fast': {
            'class': MomentumStrategy,
            'config_class': MomentumConfig,
            'params': {'fast_period': 10, 'slow_period': 30, 'signal_type': 'ma_crossover'}
        },
        'momentum_medium': {
            'class': MomentumStrategy,
            'config_class': MomentumConfig,
            'params': {'fast_period': 20, 'slow_period': 50, 'signal_type': 'ma_crossover'}
        },
        'momentum_slow': {
            'class': MomentumStrategy,
            'config_class': MomentumConfig,
            'params': {'fast_period': 50, 'slow_period': 200, 'signal_type': 'ma_crossover'}
        },
        'momentum_dual': {
            'class': MomentumStrategy,
            'config_class': MomentumConfig,
            'params': {'fast_period': 20, 'slow_period': 60, 'signal_type': 'dual'}
        },
        'mean_rev_zscore': {
            'class': MeanReversionStrategy,
            'config_class': MeanReversionConfig,
            'params': {'lookback': 20, 'entry_z': 2.0, 'exit_z': 0.5, 'signal_type': 'zscore'}
        },
        'mean_rev_rsi': {
            'class': MeanReversionStrategy,
            'config_class': MeanReversionConfig,
            'params': {'lookback': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'signal_type': 'rsi'}
        },
        'mean_rev_bollinger': {
            'class': MeanReversionStrategy,
            'config_class': MeanReversionConfig,
            'params': {'lookback': 20, 'bollinger_std': 2.0, 'signal_type': 'bollinger'}
        },
    }
    
    def __init__(
        self,
        assets: List[str],
        min_date: str = '2015-01-01',
        max_date: Optional[str] = None
    ):
        assert assets is not None, "Assets cannot be None"
        assert len(assets) > 0, "Assets cannot be empty"
        
        self.assets = assets
        self.min_date = min_date
        self.max_date = max_date
        self.results: List[ScanResult] = []
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, use_synthetic: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load data for all assets.
        
        Parameters
        ----------
        use_synthetic : bool
            If True, use synthetic data instead of real data
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of asset -> price DataFrame
        """
        if self._data_cache:
            return self._data_cache
        
        try:
            if use_synthetic:
                raise ValueError("Force synthetic data")
            
            # Check if Quantiacs API key is available
            api_key = os.environ.get('API_KEY', '')
            if not api_key:
                raise ValueError("Quantiacs API_KEY not set, using synthetic data")
            
            # Try Quantiacs
            import qnt.data as qndata
            
            logger.info(f"Loading data for {self.assets} from Quantiacs...")
            
            data = qndata.futures.load_data(
                assets=self.assets,
                min_date=self.min_date
            )
            
            for asset in self.assets:
                if asset in data.asset.values:
                    close = data.sel(field='close', asset=asset).to_pandas()
                    df = pd.DataFrame({'close': close})
                    df.index = pd.to_datetime(df.index)
                    
                    # Validate loaded data
                    validation = self._validate_data(df, asset)
                    if validation:
                        self._data_cache[asset] = df
                        logger.info(f"Loaded {len(df)} days for {asset}")
                    else:
                        logger.warning(f"Data validation failed for {asset}")
            
        except Exception as e:
            logger.warning(f"Quantiacs unavailable: {e}")
            logger.info("Generating synthetic data...")
            
            self._data_cache = self._generate_synthetic_data()
        
        return self._data_cache
    
    def _validate_data(self, df: pd.DataFrame, asset: str) -> bool:
        """
        Validate loaded data for integrity.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
        asset : str
            Asset name for logging
            
        Returns
        -------
        bool
            True if valid
        """
        try:
            # Check 1: Not empty
            assert len(df) > 0, "DataFrame is empty"
            
            # Check 2: Has DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Check 3: Index is sorted
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
                logger.warning(f"{asset}: Index was not sorted, fixed")
            
            # Check 4: No duplicate timestamps
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                df = df[~df.index.duplicated(keep='first')]
                logger.warning(f"{asset}: Removed {duplicates} duplicate timestamps")
            
            # Check 5: Has required column
            assert 'close' in df.columns, "Missing 'close' column"
            
            # Check 6: No all-NaN
            assert not df['close'].isna().all(), "All close values are NaN"
            
            # Check 7: No infinity
            inf_count = np.isinf(df['close']).sum()
            if inf_count > 0:
                df['close'] = df['close'].replace([np.inf, -np.inf], np.nan)
                logger.warning(f"{asset}: Replaced {inf_count} inf values")
            
            # Check 8: Minimum data points
            assert len(df) >= MIN_DATA_POINTS, f"Insufficient data: {len(df)}"
            
            logger.info(f"{asset}: Data validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"{asset}: Data validation failed - {e}")
            return False
        except Exception as e:
            logger.error(f"{asset}: Unexpected validation error - {e}")
            return False
    
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic price data for testing."""
        np.random.seed(42)
        n_days = 252 * 8  # 8 years
        dates = pd.date_range(self.min_date, periods=n_days, freq='B')
        
        data = {}
        for i, asset in enumerate(self.assets):
            # Different characteristics per asset
            drift = 0.0002 + i * 0.0001
            vol = 0.015 + i * 0.005
            
            returns = np.random.randn(n_days) * vol + drift
            prices = 100 * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({'close': prices}, index=dates)
            data[asset] = df
            logger.info(f"Generated {len(df)} days for {asset} (synthetic)")
        
        return data
    
    def scan_strategy(
        self,
        strategy_name: str,
        asset: str,
        data: pd.DataFrame
    ) -> Optional[ScanResult]:
        """
        Scan a single strategy on a single asset.
        
        Parameters
        ----------
        strategy_name : str
            Name of strategy configuration
        asset : str
            Asset symbol
        data : pd.DataFrame
            Price data with 'close' column
            
        Returns
        -------
        ScanResult or None
            Scan result if successful
        """
        assert strategy_name in self.STRATEGY_CONFIGS, f"Unknown strategy: {strategy_name}"
        assert data is not None, "Data cannot be None"
        assert len(data) >= MIN_DATA_POINTS, f"Insufficient data: {len(data)}"
        assert 'close' in data.columns, "Missing 'close' column"
        
        config = self.STRATEGY_CONFIGS[strategy_name]
        strategy_class = config['class']
        config_class = config['config_class']
        params = config['params']
        
        try:
            # Split data: 70% train, 30% test (chronological)
            split_idx = int(len(data) * TRAIN_RATIO)
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
            
            # Validate train/test split (no look-ahead bias)
            assert train_data.index.max() < test_data.index.min(), \
                "Look-ahead bias: train/test overlap"
            
            original_test_len = len(test_data)
            
            # Create config and strategy
            strategy_config = config_class(**params)
            strategy = strategy_class(strategy_config)
            strategy.fit(train_data)
            
            # Generate signals on test data
            signals = strategy.predict(test_data)
            
            # Calculate returns (vectorized)
            price_returns = test_data['close'].pct_change()
            
            # Check for inf/NaN in returns
            if np.isinf(price_returns).any():
                price_returns = price_returns.replace([np.inf, -np.inf], np.nan)
                logger.warning(f"Replaced inf values in {strategy_name} returns")
            
            # Clip extreme returns
            price_returns = price_returns.clip(-MAX_DAILY_RETURN, MAX_DAILY_RETURN)
            
            # Strategy returns (signal * next period return)
            strategy_returns = signals.shift(1) * price_returns
            strategy_returns = strategy_returns.dropna()
            
            # Validate dimensionality
            if len(strategy_returns) < MIN_TEST_OBSERVATIONS:
                logger.warning(
                    f"Insufficient returns for {strategy_name} on {asset}: "
                    f"{len(strategy_returns)} < {MIN_TEST_OBSERVATIONS}"
                )
                return None
            
            # Check for inf/NaN in strategy returns
            assert not np.isinf(strategy_returns).any(), "Inf in strategy returns"
            assert not strategy_returns.isna().all(), "All strategy returns are NaN"
            
            # Calculate metrics (with safe division)
            sharpe = calculate_sharpe(strategy_returns)
            psr = calculate_psr(strategy_returns)
            dsr = calculate_dsr(sharpe, n_trials=10)
            
            # Calculate drawdown (vectorized)
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = np.divide(
                cumulative - rolling_max,
                rolling_max,
                out=np.zeros_like(cumulative),
                where=rolling_max != 0
            )
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
            
            # Calculate other metrics
            total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0
            
            # Trade statistics (vectorized)
            signal_changes = signals.diff().abs()
            n_trades = int(signal_changes.sum() / 2)
            
            # Win rate (vectorized)
            winning_days = (strategy_returns > 0).sum()
            total_days = len(strategy_returns)
            win_rate = winning_days / total_days if total_days > 0 else 0.0
            
            result = ScanResult(
                strategy_name=strategy_name,
                asset=asset,
                params=params,
                sharpe_ratio=sharpe,
                psr=psr,
                dsr=dsr,
                max_drawdown=max_dd,
                total_return=total_return,
                n_trades=n_trades,
                win_rate=win_rate,
                returns=strategy_returns
            )
            
            logger.info(
                f"[{strategy_name}] {asset}: "
                f"Sharpe={sharpe:.2f}, PSR={psr:.2%}, MaxDD={max_dd:.2%}"
            )
            
            return result
            
        except AssertionError as e:
            logger.error(f"Validation error in {strategy_name} on {asset}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scanning {strategy_name} on {asset}: {e}")
            return None
    
    def scan_all(self, use_synthetic: bool = False) -> List[ScanResult]:
        """
        Scan all strategies across all assets.
        
        Parameters
        ----------
        use_synthetic : bool
            If True, use synthetic data
            
        Returns
        -------
        List[ScanResult]
            List of all scan results
        """
        logger.info("=" * 60)
        logger.info("STRATEGY SCANNER - Starting scan")
        logger.info("=" * 60)
        
        # Load data
        data = self.load_data(use_synthetic=use_synthetic)
        
        if not data:
            logger.error("No data loaded")
            return []
        
        self.results = []
        
        # Scan each strategy on each asset
        for asset in self.assets:
            if asset not in data:
                logger.warning(f"No data for {asset}")
                continue
            
            asset_data = data[asset]
            logger.info(f"\nScanning {asset} ({len(asset_data)} days)...")
            
            for strategy_name in self.STRATEGY_CONFIGS:
                result = self.scan_strategy(strategy_name, asset, asset_data)
                if result:
                    self.results.append(result)
        
        logger.info(f"\nTotal results: {len(self.results)}")
        return self.results
    
    def filter_by_psr(self, threshold: float = 0.95) -> List[ScanResult]:
        """
        Filter results by PSR threshold.
        
        Parameters
        ----------
        threshold : float
            Minimum PSR threshold
            
        Returns
        -------
        List[ScanResult]
            Filtered results
        """
        candidates = [r for r in self.results if r.psr >= threshold]
        logger.info(f"Found {len(candidates)} candidates with PSR >= {threshold:.0%}")
        return candidates
    
    def filter_by_criteria(
        self,
        min_psr: float = 0.95,
        max_drawdown: float = 0.20,
        min_sharpe: float = 0.5
    ) -> List[ScanResult]:
        """
        Filter results by multiple criteria.
        
        Parameters
        ----------
        min_psr : float
            Minimum PSR
        max_drawdown : float
            Maximum drawdown
        min_sharpe : float
            Minimum Sharpe ratio
            
        Returns
        -------
        List[ScanResult]
            Filtered results
        """
        candidates = [
            r for r in self.results
            if r.psr >= min_psr
            and r.max_drawdown <= max_drawdown
            and r.sharpe_ratio >= min_sharpe
        ]
        
        logger.info(
            f"Found {len(candidates)} candidates meeting all criteria: "
            f"PSR>={min_psr:.0%}, MaxDD<={max_drawdown:.0%}, Sharpe>={min_sharpe}"
        )
        
        return candidates
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all results.
        
        Returns
        -------
        pd.DataFrame
            Summary of all scan results
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            data.append({
                'strategy': r.strategy_name,
                'asset': r.asset,
                'sharpe': r.sharpe_ratio,
                'psr': r.psr,
                'dsr': r.dsr,
                'max_dd': r.max_drawdown,
                'total_return': r.total_return,
                'n_trades': r.n_trades,
                'win_rate': r.win_rate,
                'is_candidate': r.is_candidate
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('psr', ascending=False)
        
        return df
    
    def print_summary(self) -> None:
        """Print formatted summary of results."""
        df = self.get_summary()
        
        if df.empty:
            logger.info("No results to display")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("SCAN SUMMARY")
        logger.info("=" * 80)
        
        # Top candidates
        candidates = df[df['is_candidate']]
        logger.info(f"\nCandidates (PSR >= 95%): {len(candidates)}")
        
        if not candidates.empty:
            for _, row in candidates.iterrows():
                logger.info(
                    f"  [{row['strategy']}] {row['asset']}: "
                    f"PSR={row['psr']:.2%}, Sharpe={row['sharpe']:.2f}, "
                    f"MaxDD={row['max_dd']:.2%}"
                )
        
        # Summary statistics
        logger.info(f"\nAll Results: {len(df)}")
        logger.info(f"  Avg PSR: {df['psr'].mean():.2%}")
        logger.info(f"  Avg Sharpe: {df['sharpe'].mean():.2f}")
        logger.info(f"  Avg MaxDD: {df['max_dd'].mean():.2%}")


def main():
    """Run strategy scanner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Initialize scanner
    scanner = StrategyScanner(
        assets=['F_GC', 'F_ES', 'F_CL'],
        min_date='2015-01-01'
    )
    
    # Run scan
    results = scanner.scan_all()
    
    # Print summary
    scanner.print_summary()
    
    # Get candidates
    candidates = scanner.filter_by_psr(threshold=0.95)
    
    return len(candidates) > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
