"""
Walk-Forward Validation.

Rolling out-of-sample validation for time-series strategies.
Prevents look-ahead bias by training only on past data.

Reference: Protokol Kausalitas - Validasi Strategi
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime


# Constants
DEFAULT_N_SPLITS = 5
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_PURGE_GAP = 5
MIN_TRAIN_SAMPLES = 100
MIN_TEST_SAMPLES = 20


@dataclass
class WalkForwardSplit:
    """Single walk-forward split result."""
    split_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    
    # Performance metrics
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    
    # Degradation
    sharpe_degradation: float = 0.0
    return_degradation: float = 0.0


@dataclass
class WalkForwardResult:
    """Container for walk-forward validation results."""
    splits: List[WalkForwardSplit]
    
    # Aggregate metrics
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_sharpe_degradation: float = 0.0
    std_test_sharpe: float = 0.0
    
    # Combined OOS performance
    combined_oos_returns: pd.Series = field(default_factory=pd.Series)
    combined_oos_sharpe: float = 0.0
    
    # Metadata
    n_splits: int = 0
    total_train_samples: int = 0
    total_test_samples: int = 0
    
    def __post_init__(self):
        """Compute aggregate metrics."""
        if len(self.splits) > 0:
            self._compute_aggregates()
    
    def _compute_aggregates(self):
        """Compute aggregate statistics."""
        self.n_splits = len(self.splits)
        
        train_sharpes = [s.train_sharpe for s in self.splits]
        test_sharpes = [s.test_sharpe for s in self.splits]
        degradations = [s.sharpe_degradation for s in self.splits]
        
        self.avg_train_sharpe = np.mean(train_sharpes)
        self.avg_test_sharpe = np.mean(test_sharpes)
        self.avg_sharpe_degradation = np.mean(degradations)
        self.std_test_sharpe = np.std(test_sharpes)
        
        self.total_train_samples = sum(s.train_size for s in self.splits)
        self.total_test_samples = sum(s.test_size for s in self.splits)
        
        # Combined OOS Sharpe
        if len(self.combined_oos_returns) > 0:
            oos = self.combined_oos_returns.dropna()
            if len(oos) > 1 and oos.std() > 0:
                self.combined_oos_sharpe = oos.mean() / oos.std() * np.sqrt(252)
    
    def is_valid(self, max_degradation: float = 0.3) -> bool:
        """
        Check if strategy passes walk-forward validation.
        
        Parameters
        ----------
        max_degradation : float
            Maximum allowed Sharpe degradation (default 30%)
            
        Returns
        -------
        bool
            True if strategy is valid
        """
        return self.avg_sharpe_degradation < max_degradation
    
    def summary(self) -> str:
        """Return summary string."""
        status = "PASS" if self.is_valid() else "FAIL"
        return (
            f"Walk-Forward Validation ({status}):\n"
            f"  Splits: {self.n_splits}\n"
            f"  Avg Train Sharpe: {self.avg_train_sharpe:.4f}\n"
            f"  Avg Test Sharpe: {self.avg_test_sharpe:.4f}\n"
            f"  Sharpe Degradation: {self.avg_sharpe_degradation:.2%}\n"
            f"  Combined OOS Sharpe: {self.combined_oos_sharpe:.4f}"
        )


class WalkForwardValidator:
    """
    Walk-Forward Validation Engine.
    
    Performs rolling out-of-sample validation by:
    1. Splitting data into sequential train/test periods
    2. Training strategy on each training period
    3. Evaluating on subsequent test period
    4. Aggregating results across all splits
    
    Examples
    --------
    >>> validator = WalkForwardValidator(n_splits=5, train_ratio=0.8)
    >>> result = validator.validate(strategy, price_data)
    >>> print(result.summary())
    """
    
    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        purge_gap: int = DEFAULT_PURGE_GAP,
        min_train_samples: int = MIN_TRAIN_SAMPLES,
        min_test_samples: int = MIN_TEST_SAMPLES,
    ):
        """
        Initialize validator.
        
        Parameters
        ----------
        n_splits : int
            Number of walk-forward splits
        train_ratio : float
            Ratio of data for training in each split
        purge_gap : int
            Number of bars to skip between train and test
        min_train_samples : int
            Minimum training samples required
        min_test_samples : int
            Minimum test samples required
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.purge_gap = purge_gap
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
    
    def validate(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str = 'close',
    ) -> WalkForwardResult:
        """
        Perform walk-forward validation.
        
        Parameters
        ----------
        strategy : BaseStrategy
            Strategy instance with fit/predict methods
        data : pd.DataFrame
            Price data
        price_col : str
            Column name for prices
            
        Returns
        -------
        WalkForwardResult
            Validation results
        """
        data = self._validate_data(data)
        
        # Generate splits
        splits_indices = self._generate_splits(len(data))
        
        results = []
        all_oos_returns = []
        
        for i, (train_idx, test_idx) in enumerate(splits_indices):
            # Get train/test data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Fit strategy on training data
            strategy.fit(train_data)
            
            # Generate signals for both periods
            train_signals = strategy.predict(train_data)
            test_signals = strategy.predict(test_data)
            
            # Calculate returns
            train_prices = train_data[price_col] if price_col in train_data.columns else train_data.iloc[:, 0]
            test_prices = test_data[price_col] if price_col in test_data.columns else test_data.iloc[:, 0]
            
            train_returns = self._calculate_strategy_returns(train_prices, train_signals)
            test_returns = self._calculate_strategy_returns(test_prices, test_signals)
            
            # Calculate metrics
            train_sharpe = self._calculate_sharpe(train_returns)
            test_sharpe = self._calculate_sharpe(test_returns)
            
            # Degradation
            if train_sharpe != 0:
                degradation = 1 - (test_sharpe / train_sharpe)
            else:
                degradation = 0.0
            
            split_result = WalkForwardSplit(
                split_id=i,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_size=len(train_data),
                test_size=len(test_data),
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                train_return=train_returns.sum(),
                test_return=test_returns.sum(),
                sharpe_degradation=degradation,
            )
            
            results.append(split_result)
            all_oos_returns.append(test_returns)
        
        # Combine OOS returns
        combined_oos = pd.concat(all_oos_returns) if all_oos_returns else pd.Series()
        
        return WalkForwardResult(
            splits=results,
            combined_oos_returns=combined_oos,
        )
    
    def _generate_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test split indices."""
        splits = []
        
        # Calculate split size
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Expanding window: train on all data up to split point
            train_end = (i + 1) * split_size
            train_start = 0
            
            # Test on next split
            test_start = train_end + self.purge_gap
            test_end = min(test_start + split_size, n_samples)
            
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
    
    def _calculate_strategy_returns(
        self,
        prices: pd.Series,
        signals: pd.Series
    ) -> pd.Series:
        """Calculate strategy returns from signals."""
        price_returns = prices.pct_change()
        positions = signals.shift(1)  # Enter at next bar
        strategy_returns = positions * price_returns
        return strategy_returns.dropna()
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        if std_ret == 0:
            return 0.0
        
        return mean_ret / std_ret * np.sqrt(252)
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data = data.set_index('date')
        
        return data.sort_index()


def walk_forward_validate(
    strategy,
    data: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.8,
    purge_gap: int = 5,
) -> WalkForwardResult:
    """
    Quick walk-forward validation.
    
    Parameters
    ----------
    strategy : BaseStrategy
        Strategy with fit/predict methods
    data : pd.DataFrame
        Price data
    n_splits : int
        Number of splits
    train_ratio : float
        Training ratio
    purge_gap : int
        Gap between train and test
        
    Returns
    -------
    WalkForwardResult
        Validation results
    """
    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_ratio=train_ratio,
        purge_gap=purge_gap,
    )
    return validator.validate(strategy, data)
