"""
Purged K-Fold Cross-Validation.

Time-series cross-validation with purging to prevent data leakage.
Essential for financial ML where observations are not independent.

Reference: Advances in Financial Machine Learning (de Prado)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Generator, Dict, Any
from dataclasses import dataclass, field


# Constants
DEFAULT_N_SPLITS = 5
DEFAULT_PURGE_PCT = 0.01
DEFAULT_EMBARGO_PCT = 0.01


@dataclass
class CVFold:
    """Single cross-validation fold."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_size: int
    test_size: int
    purged_samples: int = 0
    embargoed_samples: int = 0


@dataclass
class PurgedCVResult:
    """Container for purged CV results."""
    folds: List[CVFold]
    scores: List[float]
    
    # Aggregate metrics
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    
    # Metadata
    n_folds: int = 0
    total_purged: int = 0
    total_embargoed: int = 0
    
    def __post_init__(self):
        """Compute aggregate metrics."""
        if len(self.scores) > 0:
            self.mean_score = np.mean(self.scores)
            self.std_score = np.std(self.scores)
            self.min_score = np.min(self.scores)
            self.max_score = np.max(self.scores)
        
        if len(self.folds) > 0:
            self.n_folds = len(self.folds)
            self.total_purged = sum(f.purged_samples for f in self.folds)
            self.total_embargoed = sum(f.embargoed_samples for f in self.folds)
    
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Purged K-Fold CV Results:\n"
            f"  Folds: {self.n_folds}\n"
            f"  Mean Score: {self.mean_score:.4f}\n"
            f"  Std Score: {self.std_score:.4f}\n"
            f"  Range: [{self.min_score:.4f}, {self.max_score:.4f}]\n"
            f"  Purged Samples: {self.total_purged}\n"
            f"  Embargoed Samples: {self.total_embargoed}"
        )


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for Time Series.
    
    Standard K-Fold CV causes data leakage in time series because:
    - Training data may contain information from test period
    - Observations near test period are correlated
    
    Purged K-Fold solves this by:
    1. Purging: Remove training samples that overlap with test period
    2. Embargo: Remove training samples immediately after test period
    
    Examples
    --------
    >>> cv = PurgedKFold(n_splits=5, purge_pct=0.01)
    >>> for train_idx, test_idx in cv.split(data):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])
    """
    
    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        purge_pct: float = DEFAULT_PURGE_PCT,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
    ):
        """
        Initialize purged K-Fold.
        
        Parameters
        ----------
        n_splits : int
            Number of folds
        purge_pct : float
            Percentage of data to purge before test period
        embargo_pct : float
            Percentage of data to embargo after test period
        """
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray, optional
            Target vector (not used)
        groups : np.ndarray, optional
            Group labels (not used)
            
        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            Train and test indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate purge and embargo sizes
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Calculate fold size
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test indices for this fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            # Train indices: all except test, purge, and embargo
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Remove test indices
            train_mask[test_start:test_end] = False
            
            # Purge: remove samples before test that might leak
            purge_start = max(0, test_start - purge_size)
            train_mask[purge_start:test_start] = False
            
            # Embargo: remove samples after test
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits


class TimeSeriesSplit:
    """
    Time Series Split with expanding window.
    
    Unlike standard K-Fold, this ensures training data
    always comes before test data (no future leakage).
    
    Examples
    --------
    >>> cv = TimeSeriesSplit(n_splits=5, gap=5)
    >>> for train_idx, test_idx in cv.split(data):
    ...     # train_idx always before test_idx
    ...     pass
    """
    
    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        gap: int = 0,
        test_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
    ):
        """
        Initialize time series split.
        
        Parameters
        ----------
        n_splits : int
            Number of splits
        gap : int
            Gap between train and test
        test_size : int, optional
            Fixed test size (default: auto)
        max_train_size : int, optional
            Maximum training size (for rolling window)
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        self.max_train_size = max_train_size
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            Train and test indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # Generate splits
        for i in range(self.n_splits):
            # Test window
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            
            # Train window (everything before test minus gap)
            train_end = test_start - self.gap
            train_start = 0
            
            # Apply max train size if specified
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            
            if train_end <= train_start:
                continue
            
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits


def cross_val_score(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: Optional[PurgedKFold] = None,
    scoring: str = 'accuracy',
) -> PurgedCVResult:
    """
    Perform cross-validation with purged K-Fold.
    
    Parameters
    ----------
    model : estimator
        Model with fit/predict methods
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : PurgedKFold, optional
        Cross-validator
    scoring : str
        Scoring metric
        
    Returns
    -------
    PurgedCVResult
        Cross-validation results
    """
    if cv is None:
        cv = PurgedKFold(n_splits=5)
    
    folds = []
    scores = []
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Fit model
        model.fit(X[train_idx], y[train_idx])
        
        # Score
        if scoring == 'accuracy':
            predictions = model.predict(X[test_idx])
            score = np.mean(predictions == y[test_idx])
        elif scoring == 'sharpe':
            predictions = model.predict(X[test_idx])
            returns = predictions * y[test_idx]
            score = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            score = model.score(X[test_idx], y[test_idx])
        
        fold = CVFold(
            fold_id=i,
            train_indices=train_idx,
            test_indices=test_idx,
            train_size=len(train_idx),
            test_size=len(test_idx),
        )
        
        folds.append(fold)
        scores.append(score)
    
    return PurgedCVResult(folds=folds, scores=scores)
