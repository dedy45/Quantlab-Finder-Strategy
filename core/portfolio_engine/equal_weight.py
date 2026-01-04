"""
Equal Weight Allocator.

Simple 1/N allocation strategy that often outperforms
more complex optimization methods out-of-sample.

Reference: DeMiguel, Garlappi, Uppal (2009)
"Optimal Versus Naive Diversification"
"""

import logging
from typing import Dict, Optional

import pandas as pd

from .base import BaseAllocator, AllocationConfig

logger = logging.getLogger(__name__)


class EqualWeightAllocator(BaseAllocator):
    """
    Equal weight (1/N) allocator.
    
    Allocates equal weight to all assets. Despite its simplicity,
    this approach often outperforms MVO out-of-sample due to
    estimation error in expected returns and covariances.
    
    Parameters
    ----------
    config : AllocationConfig, optional
        Allocation configuration
    """
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        super().__init__(config)
        self._assets: list = []
        
    def fit(self, returns: pd.DataFrame) -> 'EqualWeightAllocator':
        """
        Fit equal weight allocator.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns with assets as columns
            
        Returns
        -------
        EqualWeightAllocator
            Fitted allocator
        """
        self._validate_returns(returns)
        
        self._assets = list(returns.columns)
        n_assets = len(self._assets)
        
        logger.info(f"Fitting Equal Weight with {n_assets} assets")
        
        # Equal weight for all assets
        weight = 1.0 / n_assets
        self._weights = {asset: weight for asset in self._assets}
        
        # Apply constraints (may adjust weights)
        self._weights = self._apply_constraints(self._weights)
        self._is_fitted = True
        
        logger.info(f"Equal Weight fitted: {weight:.4f} per asset")
        
        return self
    
    def get_weights(self) -> Dict[str, float]:
        """Get allocation weights."""
        assert self._is_fitted, "Allocator must be fitted first"
        return self._weights.copy()
    
    def rebalance(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate rebalancing trades to return to equal weight.
        
        Parameters
        ----------
        current_weights : Dict[str, float]
            Current portfolio weights
            
        Returns
        -------
        Dict[str, float]
            Required weight changes (positive = buy, negative = sell)
        """
        assert self._is_fitted, "Allocator must be fitted first"
        
        target = self._weights
        trades = {}
        
        for asset in self._assets:
            current = current_weights.get(asset, 0.0)
            target_weight = target.get(asset, 0.0)
            trades[asset] = target_weight - current
        
        return trades
