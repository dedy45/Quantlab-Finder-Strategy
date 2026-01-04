"""
Base classes for Portfolio Engine.

Provides abstract base class and data structures for portfolio allocation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    """Configuration for portfolio allocation."""
    
    rebalance_frequency: str = 'monthly'  # daily, weekly, monthly
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_volatility: Optional[float] = 0.15  # 15% annual
    risk_free_rate: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        assert 0.0 <= self.min_weight <= 1.0, "min_weight must be in [0, 1]"
        assert 0.0 <= self.max_weight <= 1.0, "max_weight must be in [0, 1]"
        assert self.min_weight <= self.max_weight, "min_weight must be <= max_weight"
        if self.target_volatility is not None:
            assert self.target_volatility > 0, "target_volatility must be positive"


@dataclass
class AllocationResult:
    """Result of portfolio allocation."""
    
    weights: Dict[str, float]
    method: str
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weights_series(self) -> pd.Series:
        """Return weights as pandas Series."""
        return pd.Series(self.weights)
    
    @property
    def n_assets(self) -> int:
        """Number of assets in allocation."""
        return len(self.weights)
    
    @property
    def is_valid(self) -> bool:
        """Check if weights sum to approximately 1."""
        total = sum(self.weights.values())
        return abs(total - 1.0) < 1e-6
    
    def normalize(self) -> 'AllocationResult':
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total == 0:
            logger.warning("Total weight is 0, cannot normalize")
            return self
        
        normalized = {k: v / total for k, v in self.weights.items()}
        return AllocationResult(
            weights=normalized,
            method=self.method,
            timestamp=self.timestamp,
            metadata=self.metadata
        )


class BaseAllocator(ABC):
    """Abstract base class for portfolio allocators."""
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        """
        Initialize allocator.
        
        Parameters
        ----------
        config : AllocationConfig, optional
            Configuration for allocation
        """
        self.config = config or AllocationConfig()
        self._is_fitted = False
        self._weights: Optional[Dict[str, float]] = None
        
    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> 'BaseAllocator':
        """
        Fit allocator to historical returns.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns with assets as columns
            
        Returns
        -------
        BaseAllocator
            Fitted allocator instance
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """
        Get current allocation weights.
        
        Returns
        -------
        Dict[str, float]
            Asset weights
        """
        pass
    
    def fit_predict(self, returns: pd.DataFrame) -> AllocationResult:
        """
        Fit and return allocation result.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
            
        Returns
        -------
        AllocationResult
            Allocation result with weights
        """
        self.fit(returns)
        weights = self.get_weights()
        
        return AllocationResult(
            weights=weights,
            method=self.__class__.__name__,
            metadata={'n_observations': len(returns)}
        )
    
    def _validate_returns(self, returns: pd.DataFrame) -> None:
        """Validate returns DataFrame."""
        assert returns is not None, "Returns cannot be None"
        assert isinstance(returns, pd.DataFrame), "Returns must be DataFrame"
        assert len(returns) > 0, "Returns cannot be empty"
        assert returns.shape[1] > 0, "Returns must have at least one asset"
        
        # Check for NaN
        nan_count = returns.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Returns contain {nan_count} NaN values")
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        constrained = {}
        
        for asset, weight in weights.items():
            constrained[asset] = np.clip(
                weight, 
                self.config.min_weight, 
                self.config.max_weight
            )
        
        # Renormalize after constraints
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}
        
        return constrained
