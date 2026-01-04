"""
Parameter Optimizer for FASE 5: Production.

Walk-Forward Parameter Optimization untuk menemukan parameter
strategi yang robust dan konsisten di berbagai kondisi pasar.

Workflow:
1. Define parameter grid
2. Walk-forward optimization (expanding window)
3. Select parameters yang konsisten di SEMUA windows
4. Validate dengan PSR/DSR

Version: 0.6.2
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

# Setup path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.validation_engine import calculate_sharpe, calculate_psr
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Constants
MIN_TRAIN_SAMPLES = 252  # 1 year minimum
MIN_TEST_SAMPLES = 63    # 3 months minimum
DEFAULT_N_SPLITS = 4
DEFAULT_TRAIN_RATIO = 0.7
PURGE_GAP = 5            # Days between train/test
MAX_SHARPE_DEGRADATION = 0.30  # 30% max degradation
MIN_CONSISTENCY_SCORE = 0.60   # 60% windows must be profitable


@dataclass
class ParameterSet:
    """A single parameter configuration."""
    params: Dict[str, Any]
    
    def __hash__(self):
        return hash(tuple(sorted(self.params.items())))
    
    def __eq__(self, other):
        return self.params == other.params


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    sharpe_degradation: float
    is_profitable: bool


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    params: Dict[str, Any]
    
    # Aggregate metrics
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_degradation: float = 0.0
    std_test_sharpe: float = 0.0
    
    # Consistency metrics
    n_windows: int = 0
    n_profitable_windows: int = 0
    consistency_score: float = 0.0
    
    # Combined OOS metrics
    combined_oos_sharpe: float = 0.0
    combined_oos_psr: float = 0.0
    combined_oos_return: float = 0.0
    
    # Window details
    windows: List[WindowResult] = field(default_factory=list)
    
    # Validation
    is_robust: bool = False
    
    @property
    def score(self) -> float:
        """Calculate overall optimization score (0-100)."""
        score = 0.0
        
        # OOS Sharpe contribution (40 points)
        if self.avg_test_sharpe >= 1.5:
            score += 40
        elif self.avg_test_sharpe >= 1.0:
            score += 30
        elif self.avg_test_sharpe >= 0.5:
            score += 20
        elif self.avg_test_sharpe > 0:
            score += 10
        
        # Consistency contribution (30 points)
        score += self.consistency_score * 30
        
        # Low degradation contribution (20 points)
        if self.avg_degradation <= 0.10:
            score += 20
        elif self.avg_degradation <= 0.20:
            score += 15
        elif self.avg_degradation <= 0.30:
            score += 10
        
        # PSR contribution (10 points)
        if self.combined_oos_psr >= 0.95:
            score += 10
        elif self.combined_oos_psr >= 0.90:
            score += 7
        elif self.combined_oos_psr >= 0.80:
            score += 5
        
        return score


@dataclass
class GridSearchResult:
    """Result from grid search optimization."""
    best_params: Dict[str, Any]
    best_result: OptimizationResult
    all_results: List[OptimizationResult]
    
    # Summary
    n_combinations: int = 0
    n_valid: int = 0
    search_time: float = 0.0
    
    def get_top_n(self, n: int = 5) -> List[OptimizationResult]:
        """Get top N results by score."""
        sorted_results = sorted(
            self.all_results,
            key=lambda x: x.score,
            reverse=True
        )
        return sorted_results[:n]


class WalkForwardOptimizer:
    """
    Walk-Forward Parameter Optimizer.
    
    Optimizes strategy parameters using walk-forward validation
    to ensure robustness and avoid overfitting.
    
    Parameters
    ----------
    strategy_class : Type[BaseStrategy]
        Strategy class to optimize
    param_grid : Dict[str, List]
        Parameter grid to search
    n_splits : int
        Number of walk-forward splits
    train_ratio : float
        Ratio of data for training in each split
    min_train_samples : int
        Minimum training samples required
    min_test_samples : int
        Minimum test samples required
        
    Examples
    --------
    >>> from strategies import MomentumStrategy
    >>> optimizer = WalkForwardOptimizer(
    ...     strategy_class=MomentumStrategy,
    ...     param_grid={
    ...         'fast_period': [10, 20, 30],
    ...         'slow_period': [50, 100, 200]
    ...     }
    ... )
    >>> result = optimizer.optimize(price_data)
    >>> print(f"Best params: {result.best_params}")
    """
    
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_grid: Dict[str, List[Any]],
        n_splits: int = DEFAULT_N_SPLITS,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        min_train_samples: int = MIN_TRAIN_SAMPLES,
        min_test_samples: int = MIN_TEST_SAMPLES,
    ):
        assert strategy_class is not None, "strategy_class cannot be None"
        assert param_grid is not None, "param_grid cannot be None"
        assert len(param_grid) > 0, "param_grid cannot be empty"
        assert 2 <= n_splits <= 10, "n_splits must be between 2 and 10"
        assert 0.5 <= train_ratio <= 0.9, "train_ratio must be between 0.5 and 0.9"
        
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        
        # Generate all parameter combinations
        self.param_combinations = self._generate_combinations()
        
        logger.info(
            f"WalkForwardOptimizer initialized: "
            f"{len(self.param_combinations)} combinations, "
            f"{n_splits} splits"
        )
    
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combo in product(*values):
            params = dict(zip(keys, combo))
            combinations.append(params)
        
        return combinations
    
    def _generate_splits(
        self,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward split indices.
        
        Uses expanding window approach:
        - Window 1: Train [0:T1], Test [T1:T2]
        - Window 2: Train [0:T2], Test [T2:T3]
        - Window 3: Train [0:T3], Test [T3:T4]
        """
        splits = []
        split_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Expanding window: train on all data up to split point
            train_end = (i + 1) * split_size
            train_start = 0
            
            # Test on next split (with purge gap)
            test_start = train_end + PURGE_GAP
            test_end = min(test_start + split_size, n_samples)
            
            # Validate split sizes
            if test_start >= n_samples:
                break
            
            if train_end - train_start < self.min_train_samples:
                continue
            
            if test_end - test_start < self.min_test_samples:
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        data: pd.DataFrame,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        price_col: str = 'close'
    ) -> OptimizationResult:
        """
        Evaluate a single parameter set across all walk-forward windows.
        
        Parameters
        ----------
        params : Dict
            Parameter configuration
        data : pd.DataFrame
            Price data
        splits : List[Tuple]
            Walk-forward split indices
        price_col : str
            Price column name
            
        Returns
        -------
        OptimizationResult
            Evaluation result
        """
        windows = []
        all_oos_returns = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            try:
                # Get train/test data
                train_data = data.iloc[train_idx].copy()
                test_data = data.iloc[test_idx].copy()
                
                # Create strategy with params
                strategy = self._create_strategy(params)
                
                # Fit on training data
                strategy.fit(train_data)
                
                # Generate signals
                train_signals = strategy.predict(train_data)
                test_signals = strategy.predict(test_data)
                
                # Calculate returns
                train_returns = self._calculate_returns(
                    train_data, train_signals, price_col
                )
                test_returns = self._calculate_returns(
                    test_data, test_signals, price_col
                )
                
                # Calculate metrics
                train_sharpe = self._calculate_sharpe(train_returns)
                test_sharpe = self._calculate_sharpe(test_returns)
                
                # Degradation (safe division)
                if abs(train_sharpe) > 0.01:
                    degradation = 1 - (test_sharpe / train_sharpe)
                else:
                    degradation = 0.0 if test_sharpe >= 0 else 1.0
                
                # Clip extreme degradation
                degradation = np.clip(degradation, -1.0, 2.0)
                
                window = WindowResult(
                    window_id=i,
                    train_start=train_data.index[0],
                    train_end=train_data.index[-1],
                    test_start=test_data.index[0],
                    test_end=test_data.index[-1],
                    train_samples=len(train_data),
                    test_samples=len(test_data),
                    train_sharpe=train_sharpe,
                    test_sharpe=test_sharpe,
                    train_return=float(train_returns.sum()),
                    test_return=float(test_returns.sum()),
                    sharpe_degradation=degradation,
                    is_profitable=test_returns.sum() > 0
                )
                
                windows.append(window)
                all_oos_returns.append(test_returns)
                
            except Exception as e:
                logger.warning(f"Window {i} failed for params {params}: {e}")
                continue
        
        # Calculate aggregate metrics
        if len(windows) == 0:
            return OptimizationResult(params=params)
        
        # Combine OOS returns
        combined_oos = pd.concat(all_oos_returns) if all_oos_returns else pd.Series()
        
        # Calculate aggregates
        train_sharpes = [w.train_sharpe for w in windows]
        test_sharpes = [w.test_sharpe for w in windows]
        degradations = [w.sharpe_degradation for w in windows]
        
        avg_train_sharpe = float(np.mean(train_sharpes))
        avg_test_sharpe = float(np.mean(test_sharpes))
        avg_degradation = float(np.mean(degradations))
        std_test_sharpe = float(np.std(test_sharpes))
        
        # Consistency
        n_profitable = sum(1 for w in windows if w.is_profitable)
        consistency_score = n_profitable / len(windows)
        
        # Combined OOS metrics
        combined_oos_sharpe = self._calculate_sharpe(combined_oos)
        combined_oos_psr = calculate_psr(combined_oos) if len(combined_oos) > 30 else 0.0
        combined_oos_return = float(combined_oos.sum())
        
        # Is robust?
        is_robust = (
            avg_degradation <= MAX_SHARPE_DEGRADATION and
            consistency_score >= MIN_CONSISTENCY_SCORE and
            avg_test_sharpe > 0
        )
        
        return OptimizationResult(
            params=params,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            avg_degradation=avg_degradation,
            std_test_sharpe=std_test_sharpe,
            n_windows=len(windows),
            n_profitable_windows=n_profitable,
            consistency_score=consistency_score,
            combined_oos_sharpe=combined_oos_sharpe,
            combined_oos_psr=combined_oos_psr,
            combined_oos_return=combined_oos_return,
            windows=windows,
            is_robust=is_robust
        )
    
    def _create_strategy(self, params: Dict[str, Any]) -> BaseStrategy:
        """
        Create strategy instance with given parameters.
        
        Handles different strategy initialization patterns:
        1. MomentumStrategy(MomentumConfig(...))
        2. MeanReversionStrategy(MeanReversionConfig(...))
        3. Direct kwargs: SomeStrategy(**params)
        """
        try:
            strategy_name = self.strategy_class.__name__
            
            # Import config classes based on strategy type
            if strategy_name == 'MomentumStrategy':
                from strategies.momentum_strategy import MomentumConfig
                config = MomentumConfig(**params)
                return self.strategy_class(config)
            
            elif strategy_name == 'MeanReversionStrategy':
                from strategies.mean_reversion import MeanReversionConfig
                config = MeanReversionConfig(**params)
                return self.strategy_class(config)
            
            elif strategy_name == 'MLStrategy':
                from strategies.ml_strategy import MLConfig
                config = MLConfig(**params)
                return self.strategy_class(config)
            
            else:
                # Fallback: try direct kwargs
                return self.strategy_class(**params)
                
        except Exception as e:
            logger.error(f"Failed to create strategy {strategy_name}: {e}")
            raise
    
    def _calculate_returns(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        price_col: str
    ) -> pd.Series:
        """Calculate strategy returns from signals."""
        prices = data[price_col] if price_col in data.columns else data.iloc[:, 0]
        price_returns = prices.pct_change()
        
        # Replace inf with NaN
        price_returns = price_returns.replace([np.inf, -np.inf], np.nan)
        
        # Clip extreme returns
        price_returns = price_returns.clip(-0.5, 0.5)
        
        # Position at next bar
        positions = signals.shift(1)
        
        # Strategy returns
        strategy_returns = positions * price_returns
        
        return strategy_returns.dropna()
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        if std_ret == 0 or np.isnan(std_ret):
            return 0.0
        
        return float(mean_ret / std_ret * np.sqrt(252))
    
    def optimize(
        self,
        data: pd.DataFrame,
        price_col: str = 'close'
    ) -> GridSearchResult:
        """
        Run walk-forward optimization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data with DatetimeIndex
        price_col : str
            Price column name
            
        Returns
        -------
        GridSearchResult
            Optimization results
        """
        assert data is not None, "Data cannot be None"
        assert len(data) >= self.min_train_samples + self.min_test_samples, \
            f"Insufficient data: {len(data)}"
        
        logger.info("=" * 60)
        logger.info("WALK-FORWARD OPTIMIZATION")
        logger.info("=" * 60)
        logger.info(f"Data: {len(data)} samples")
        logger.info(f"Combinations: {len(self.param_combinations)}")
        logger.info(f"Splits: {self.n_splits}")
        
        start_time = datetime.now()
        
        # Validate data
        data = self._validate_data(data)
        
        # Generate splits
        splits = self._generate_splits(len(data))
        
        if len(splits) < 2:
            logger.error("Insufficient data for walk-forward optimization")
            return GridSearchResult(
                best_params={},
                best_result=OptimizationResult(params={}),
                all_results=[],
                n_combinations=len(self.param_combinations),
                n_valid=0
            )
        
        logger.info(f"Generated {len(splits)} walk-forward windows")
        
        # Evaluate all parameter combinations
        all_results = []
        
        for i, params in enumerate(self.param_combinations):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Evaluating combination {i + 1}/{len(self.param_combinations)}")
            
            result = self._evaluate_params(params, data, splits, price_col)
            all_results.append(result)
        
        # Find best result
        valid_results = [r for r in all_results if r.is_robust]
        
        if valid_results:
            best_result = max(valid_results, key=lambda x: x.score)
        else:
            # Fallback to best by score even if not robust
            best_result = max(all_results, key=lambda x: x.score)
            logger.warning("No robust parameters found, using best available")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        grid_result = GridSearchResult(
            best_params=best_result.params,
            best_result=best_result,
            all_results=all_results,
            n_combinations=len(self.param_combinations),
            n_valid=len(valid_results),
            search_time=elapsed
        )
        
        # Print summary
        self._print_summary(grid_result)
        
        return grid_result
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data."""
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data = data.set_index('date')
            else:
                data.index = pd.to_datetime(data.index)
        
        return data.sort_index()
    
    def _print_summary(self, result: GridSearchResult) -> None:
        """Print optimization summary."""
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"\nSearch completed in {result.search_time:.1f} seconds")
        logger.info(f"Combinations tested: {result.n_combinations}")
        logger.info(f"Robust combinations: {result.n_valid}")
        
        logger.info(f"\nBest Parameters:")
        for key, value in result.best_params.items():
            logger.info(f"  {key}: {value}")
        
        best = result.best_result
        logger.info(f"\nBest Result Metrics:")
        logger.info(f"  Score: {best.score:.1f}/100")
        logger.info(f"  Avg OOS Sharpe: {best.avg_test_sharpe:.2f}")
        logger.info(f"  Avg Degradation: {best.avg_degradation:.2%}")
        logger.info(f"  Consistency: {best.consistency_score:.2%}")
        logger.info(f"  Combined OOS PSR: {best.combined_oos_psr:.2%}")
        logger.info(f"  Is Robust: {best.is_robust}")
        
        # Top 5
        logger.info(f"\nTop 5 Configurations:")
        for i, r in enumerate(result.get_top_n(5)):
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            logger.info(
                f"  {i+1}. Score={r.score:.1f}, "
                f"OOS_SR={r.avg_test_sharpe:.2f}, "
                f"Params: {params_str}"
            )


class ParameterSpaces:
    """
    Predefined parameter spaces for different strategy types.
    
    Usage:
    >>> grid = ParameterSpaces.momentum_grid()
    >>> optimizer = WalkForwardOptimizer(MomentumStrategy, grid)
    """
    
    @staticmethod
    def momentum_grid() -> Dict[str, List[Any]]:
        """Parameter grid for Momentum Strategy."""
        return {
            'fast_period': [10, 20, 30, 50],
            'slow_period': [50, 100, 150, 200],
            'signal_type': ['ma_crossover', 'dual'],
        }
    
    @staticmethod
    def momentum_grid_fine() -> Dict[str, List[Any]]:
        """Fine-grained parameter grid for Momentum Strategy."""
        return {
            'fast_period': [5, 10, 15, 20, 25, 30, 40, 50],
            'slow_period': [40, 50, 60, 80, 100, 120, 150, 200],
            'signal_type': ['ma_crossover', 'dual', 'momentum'],
        }
    
    @staticmethod
    def mean_reversion_grid() -> Dict[str, List[Any]]:
        """Parameter grid for Mean Reversion Strategy."""
        return {
            'lookback': [10, 14, 20, 30],
            'entry_z': [1.5, 2.0, 2.5, 3.0],
            'exit_z': [0.0, 0.5, 1.0],
            'signal_type': ['zscore', 'rsi', 'bollinger'],
        }
    
    @staticmethod
    def mean_reversion_grid_fine() -> Dict[str, List[Any]]:
        """Fine-grained parameter grid for Mean Reversion Strategy."""
        return {
            'lookback': [5, 10, 14, 20, 25, 30, 40],
            'entry_z': [1.0, 1.5, 2.0, 2.5, 3.0],
            'exit_z': [0.0, 0.25, 0.5, 0.75, 1.0],
            'signal_type': ['zscore', 'rsi', 'bollinger'],
        }
    
    @staticmethod
    def ml_grid() -> Dict[str, List[Any]]:
        """Parameter grid for ML Strategy."""
        return {
            'model_type': ['random_forest', 'lightgbm'],
            'lookback': [20, 50, 100],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
        }


def optimize_strategy(
    strategy_class: Type[BaseStrategy],
    data: pd.DataFrame,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 4,
    price_col: str = 'close'
) -> GridSearchResult:
    """
    Quick function to optimize a strategy.
    
    Parameters
    ----------
    strategy_class : Type[BaseStrategy]
        Strategy class to optimize
    data : pd.DataFrame
        Price data
    param_grid : Dict, optional
        Parameter grid (uses default if None)
    n_splits : int
        Number of walk-forward splits
    price_col : str
        Price column name
        
    Returns
    -------
    GridSearchResult
        Optimization results
        
    Examples
    --------
    >>> from strategies import MomentumStrategy
    >>> result = optimize_strategy(MomentumStrategy, price_data)
    >>> print(f"Best: {result.best_params}")
    """
    # Use default grid if not provided
    if param_grid is None:
        strategy_name = strategy_class.__name__.lower()
        if 'momentum' in strategy_name:
            param_grid = ParameterSpaces.momentum_grid()
        elif 'mean' in strategy_name or 'reversion' in strategy_name:
            param_grid = ParameterSpaces.mean_reversion_grid()
        else:
            raise ValueError(
                f"No default grid for {strategy_class.__name__}. "
                "Please provide param_grid."
            )
    
    optimizer = WalkForwardOptimizer(
        strategy_class=strategy_class,
        param_grid=param_grid,
        n_splits=n_splits
    )
    
    return optimizer.optimize(data, price_col=price_col)


def main():
    """Test parameter optimizer."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_days = 252 * 5  # 5 years
    dates = pd.date_range('2019-01-01', periods=n_days, freq='B')
    
    # Trending price with noise
    drift = 0.0003
    vol = 0.015
    returns = np.random.randn(n_days) * vol + drift
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({'close': prices}, index=dates)
    
    logger.info(f"Generated {len(data)} days of synthetic data")
    
    # Import strategy
    from strategies import MomentumStrategy
    
    # Define small grid for testing
    param_grid = {
        'fast_period': [10, 20],
        'slow_period': [50, 100],
        'signal_type': ['ma_crossover'],
    }
    
    # Run optimization
    optimizer = WalkForwardOptimizer(
        strategy_class=MomentumStrategy,
        param_grid=param_grid,
        n_splits=3
    )
    
    result = optimizer.optimize(data)
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)
    
    return result.n_valid > 0 or result.best_result.score > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
