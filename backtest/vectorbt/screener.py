"""
Strategy Screener - Screen 1000+ strategy combinations using VectorBT.

Phase 1 of 4-Phase Research Workflow:
- Input: 1000+ strategy combinations
- Filter: Sharpe > 0.5, MaxDD < 30%
- Output: Top 50 candidates
- Time: ~30 minutes
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from ..base import BacktestMetrics, BacktestResult
from .adapter import VectorBTAdapter, VectorBTConfig

logger = logging.getLogger(__name__)


@dataclass
class ScreeningConfig:
    """Configuration for strategy screening."""
    
    # Filter criteria
    min_sharpe: float = 0.5
    max_drawdown: float = 0.30  # 30%
    min_trades: int = 10
    min_win_rate: float = 0.0  # No minimum by default
    
    # Output
    top_n: int = 50
    
    # Execution
    parallel: bool = False  # Future: parallel execution
    verbose: bool = True
    
    # Backtest config
    backtest_config: Optional[VectorBTConfig] = None


@dataclass
class ScreeningResult:
    """Result from strategy screening."""
    
    # Summary
    total_tested: int = 0
    total_passed: int = 0
    execution_time: float = 0.0
    
    # Results
    all_results: List[BacktestResult] = field(default_factory=list)
    passed_results: List[BacktestResult] = field(default_factory=list)
    top_results: List[BacktestResult] = field(default_factory=list)
    
    # Config used
    config: Optional[ScreeningConfig] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert top results to DataFrame."""
        if not self.top_results:
            return pd.DataFrame()
        
        rows = []
        for r in self.top_results:
            rows.append({
                'strategy': r.strategy_name,
                'asset': r.asset,
                'sharpe': r.metrics.sharpe_ratio,
                'sortino': r.metrics.sortino_ratio,
                'calmar': r.metrics.calmar_ratio,
                'total_return': r.metrics.total_return,
                'annual_return': r.metrics.annual_return,
                'max_drawdown': r.metrics.max_drawdown,
                'volatility': r.metrics.volatility,
                'total_trades': r.metrics.total_trades,
                'win_rate': r.metrics.win_rate,
                'profit_factor': r.metrics.profit_factor,
                'params': str(r.params)
            })
        
        return pd.DataFrame(rows)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "SCREENING RESULTS",
            "=" * 60,
            f"Total tested: {self.total_tested}",
            f"Total passed: {self.total_passed}",
            f"Top candidates: {len(self.top_results)}",
            f"Execution time: {self.execution_time:.1f}s",
            "-" * 60
        ]
        
        if self.top_results:
            lines.append("TOP 10 CANDIDATES:")
            for i, r in enumerate(self.top_results[:10], 1):
                lines.append(
                    f"  {i}. {r.strategy_name} ({r.asset}): "
                    f"Sharpe={r.metrics.sharpe_ratio:.2f}, "
                    f"MaxDD={r.metrics.max_drawdown:.1%}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)


class StrategyScreener:
    """
    Screen multiple strategy combinations using VectorBT.
    
    Cara Penggunaan:
    - Instantiate dengan ScreeningConfig
    - Call screen_strategies() dengan prices dan strategy list
    - Atau call screen_parameter_grid() untuk parameter optimization
    
    Nilai:
    - Screening cepat untuk 1000+ kombinasi
    - Filter otomatis berdasarkan Sharpe dan MaxDD
    - Output Top N candidates untuk Phase 2
    
    Manfaat:
    - Mempercepat research 10-100x
    - Menghindari manual testing
    - Systematic approach untuk menemukan alpha
    """
    
    def __init__(self, config: Optional[ScreeningConfig] = None):
        """Initialize screener."""
        self.config = config or ScreeningConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize adapter
        bt_config = self.config.backtest_config or VectorBTConfig()
        self.adapter = VectorBTAdapter(bt_config)
    
    def screen_strategies(
        self,
        prices: pd.DataFrame,
        strategies: List[Any],
        asset: str = "UNKNOWN"
    ) -> ScreeningResult:
        """
        Screen list of strategy objects.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        strategies : List[Any]
            List of strategy objects with fit/predict methods
        asset : str
            Asset identifier
            
        Returns
        -------
        ScreeningResult
            Screening results with top candidates
        """
        assert prices is not None, "Prices cannot be None"
        assert strategies is not None, "Strategies cannot be None"
        assert len(strategies) > 0, "Strategies list cannot be empty"
        
        start_time = time.time()
        all_results = []
        passed_results = []
        
        total = len(strategies)
        
        for i, strategy in enumerate(strategies):
            if self.config.verbose and (i + 1) % 10 == 0:
                self.logger.info(f"Screening {i+1}/{total}...")
            
            try:
                result = self.adapter.run_strategy(prices, strategy, asset)
                all_results.append(result)
                
                # Check if passes filter
                if self._passes_filter(result.metrics):
                    passed_results.append(result)
                    
            except Exception as e:
                self.logger.warning(f"Strategy {i} failed: {e}")
        
        # Sort by Sharpe ratio
        passed_results.sort(key=lambda x: x.metrics.sharpe_ratio, reverse=True)
        
        # Get top N
        top_results = passed_results[:self.config.top_n]
        
        execution_time = time.time() - start_time
        
        result = ScreeningResult(
            total_tested=len(all_results),
            total_passed=len(passed_results),
            execution_time=execution_time,
            all_results=all_results,
            passed_results=passed_results,
            top_results=top_results,
            config=self.config
        )
        
        if self.config.verbose:
            self.logger.info(result.summary())
        
        return result
    
    def screen_parameter_grid(
        self,
        prices: pd.DataFrame,
        strategy_class: Type,
        param_grid: Dict[str, List[Any]],
        asset: str = "UNKNOWN"
    ) -> ScreeningResult:
        """
        Screen strategy with parameter grid.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        strategy_class : Type
            Strategy class to instantiate
        param_grid : Dict[str, List[Any]]
            Parameter grid, e.g. {'fast_period': [10, 20], 'slow_period': [50, 100]}
        asset : str
            Asset identifier
            
        Returns
        -------
        ScreeningResult
            Screening results with top candidates
        """
        assert prices is not None, "Prices cannot be None"
        assert strategy_class is not None, "Strategy class cannot be None"
        assert param_grid is not None, "Parameter grid cannot be None"
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        self.logger.info(f"Screening {len(combinations)} parameter combinations...")
        
        start_time = time.time()
        all_results = []
        passed_results = []
        
        for i, combo in enumerate(combinations):
            if self.config.verbose and (i + 1) % 50 == 0:
                self.logger.info(f"Screening {i+1}/{len(combinations)}...")
            
            try:
                # Create parameter dict
                params = dict(zip(param_names, combo))
                
                # Instantiate strategy
                strategy = self._create_strategy(strategy_class, params)
                
                if strategy is None:
                    continue
                
                # Run backtest
                result = self.adapter.run_strategy(prices, strategy, asset)
                result.params = params
                all_results.append(result)
                
                # Check if passes filter
                if self._passes_filter(result.metrics):
                    passed_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Combination {i} failed: {e}")
        
        # Sort by Sharpe ratio
        passed_results.sort(key=lambda x: x.metrics.sharpe_ratio, reverse=True)
        
        # Get top N
        top_results = passed_results[:self.config.top_n]
        
        execution_time = time.time() - start_time
        
        result = ScreeningResult(
            total_tested=len(all_results),
            total_passed=len(passed_results),
            execution_time=execution_time,
            all_results=all_results,
            passed_results=passed_results,
            top_results=top_results,
            config=self.config
        )
        
        if self.config.verbose:
            self.logger.info(result.summary())
        
        return result
    
    def screen_signals(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame, Dict], pd.Series],
        param_grid: Dict[str, List[Any]],
        asset: str = "UNKNOWN",
        strategy_name: str = "Custom"
    ) -> ScreeningResult:
        """
        Screen with custom signal generator function.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        signal_generator : Callable
            Function that takes (prices, params) and returns signals
        param_grid : Dict[str, List[Any]]
            Parameter grid
        asset : str
            Asset identifier
        strategy_name : str
            Strategy name for identification
            
        Returns
        -------
        ScreeningResult
            Screening results
        """
        assert prices is not None, "Prices cannot be None"
        assert signal_generator is not None, "Signal generator cannot be None"
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        self.logger.info(f"Screening {len(combinations)} signal combinations...")
        
        start_time = time.time()
        all_results = []
        passed_results = []
        
        for i, combo in enumerate(combinations):
            if self.config.verbose and (i + 1) % 50 == 0:
                self.logger.info(f"Screening {i+1}/{len(combinations)}...")
            
            try:
                params = dict(zip(param_names, combo))
                
                # Generate signals
                signals = signal_generator(prices, params)
                
                # Run backtest
                name = f"{strategy_name}_{i}"
                result = self.adapter.run(prices, signals, asset, name)
                result.params = params
                all_results.append(result)
                
                # Check if passes filter
                if self._passes_filter(result.metrics):
                    passed_results.append(result)
                    
            except Exception as e:
                self.logger.debug(f"Combination {i} failed: {e}")
        
        # Sort and get top N
        passed_results.sort(key=lambda x: x.metrics.sharpe_ratio, reverse=True)
        top_results = passed_results[:self.config.top_n]
        
        execution_time = time.time() - start_time
        
        return ScreeningResult(
            total_tested=len(all_results),
            total_passed=len(passed_results),
            execution_time=execution_time,
            all_results=all_results,
            passed_results=passed_results,
            top_results=top_results,
            config=self.config
        )
    
    def _passes_filter(self, metrics: BacktestMetrics) -> bool:
        """Check if metrics pass screening filter."""
        return (
            metrics.sharpe_ratio >= self.config.min_sharpe and
            abs(metrics.max_drawdown) <= self.config.max_drawdown and
            metrics.total_trades >= self.config.min_trades and
            metrics.win_rate >= self.config.min_win_rate
        )
    
    def _create_strategy(
        self,
        strategy_class: Type,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """Create strategy instance with parameters."""
        try:
            # Try direct instantiation
            return strategy_class(**params)
        except TypeError:
            pass
        
        try:
            # Try with config class
            if hasattr(strategy_class, '__init__'):
                # Check if strategy expects a config object
                import inspect
                sig = inspect.signature(strategy_class.__init__)
                param_names = list(sig.parameters.keys())
                
                if 'config' in param_names:
                    # Find config class
                    config_class = None
                    
                    # Common config class names
                    strategy_name = strategy_class.__name__
                    if 'Momentum' in strategy_name:
                        from strategies.momentum_strategy import MomentumConfig
                        config_class = MomentumConfig
                    elif 'MeanReversion' in strategy_name:
                        from strategies.mean_reversion import MeanReversionConfig
                        config_class = MeanReversionConfig
                    elif 'ML' in strategy_name:
                        from strategies.ml_strategy import MLConfig
                        config_class = MLConfig
                    
                    if config_class:
                        config = config_class(**params)
                        return strategy_class(config=config)
            
            # Fallback: create with no args and set attributes
            strategy = strategy_class()
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
            return strategy
            
        except Exception as e:
            self.logger.debug(f"Failed to create strategy: {e}")
            return None
