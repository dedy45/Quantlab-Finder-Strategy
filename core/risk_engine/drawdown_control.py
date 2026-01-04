"""
Drawdown Controller.

Monitors and controls portfolio drawdown.
Triggers risk reduction when drawdown exceeds threshold.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseRiskManager, RiskConfig, RiskMetrics, calculate_drawdown

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    """Current drawdown state."""
    
    current_dd: float
    max_dd: float
    peak_value: float
    current_value: float
    days_in_drawdown: int
    is_breach: bool


class DrawdownController(BaseRiskManager):
    """
    Drawdown controller for risk management.
    
    Monitors drawdown and provides signals for risk reduction
    when drawdown exceeds threshold.
    
    Parameters
    ----------
    config : RiskConfig, optional
        Risk configuration
    warning_threshold : float, default=0.5
        Fraction of max_drawdown to trigger warning
    reduction_factor : float, default=0.5
        Factor to reduce exposure when breached
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        warning_threshold: float = 0.5,
        reduction_factor: float = 0.5
    ):
        super().__init__(config)
        
        assert 0 < warning_threshold < 1.0, "warning_threshold must be in (0, 1)"
        assert 0 < reduction_factor <= 1.0, "reduction_factor must be in (0, 1]"
        
        self.warning_threshold = warning_threshold
        self.reduction_factor = reduction_factor
        
        self._state: Optional[DrawdownState] = None
        self._metrics: Optional[RiskMetrics] = None
        
    def check(self, portfolio_value: pd.Series) -> bool:
        """
        Check if drawdown is within limits.
        
        Parameters
        ----------
        portfolio_value : pd.Series
            Historical portfolio values
            
        Returns
        -------
        bool
            True if within limits
        """
        self._validate_data(portfolio_value)
        
        try:
            # Calculate drawdown
            dd_series = calculate_drawdown(portfolio_value)
            current_dd = abs(dd_series.iloc[-1])
            max_dd = abs(dd_series.min())
            
            # Calculate days in drawdown
            peak_idx = portfolio_value.idxmax()
            days_in_dd = len(portfolio_value) - portfolio_value.index.get_loc(peak_idx)
            
            # Check breach
            is_breach = current_dd >= self.config.max_drawdown
            
            # Update state
            self._state = DrawdownState(
                current_dd=current_dd,
                max_dd=max_dd,
                peak_value=portfolio_value.max(),
                current_value=portfolio_value.iloc[-1],
                days_in_drawdown=days_in_dd,
                is_breach=is_breach
            )
            
            # Update metrics
            self._metrics = RiskMetrics(
                current_drawdown=current_dd,
                max_drawdown=max_dd,
                is_drawdown_breach=is_breach
            )
            
            # Log status
            if is_breach:
                logger.warning(
                    f"Drawdown BREACH: {current_dd:.2%} >= {self.config.max_drawdown:.2%}"
                )
            elif current_dd >= self.config.max_drawdown * self.warning_threshold:
                logger.warning(
                    f"Drawdown WARNING: {current_dd:.2%} approaching limit"
                )
            else:
                logger.info(f"Drawdown OK: {current_dd:.2%}")
            
            return not is_breach
            
        except Exception as e:
            logger.error(f"Drawdown check failed: {e}")
            raise
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        if self._metrics is None:
            return RiskMetrics()
        return self._metrics
    
    def get_state(self) -> DrawdownState:
        """Get current drawdown state."""
        if self._state is None:
            raise ValueError("Must call check() first")
        return self._state
    
    def get_exposure_adjustment(self) -> float:
        """
        Get recommended exposure adjustment.
        
        Returns
        -------
        float
            Multiplier for exposure (1.0 = no change, 0.5 = reduce by half)
        """
        if self._state is None:
            return 1.0
        
        if self._state.is_breach:
            return self.reduction_factor
        
        # Gradual reduction as approaching limit
        dd_ratio = self._state.current_dd / self.config.max_drawdown
        
        if dd_ratio >= self.warning_threshold:
            # Linear reduction from warning to breach
            reduction = 1.0 - (dd_ratio - self.warning_threshold) / (1.0 - self.warning_threshold)
            return max(self.reduction_factor, reduction)
        
        return 1.0
    
    def reduce_exposure(
        self, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Reduce portfolio exposure based on drawdown.
        
        Parameters
        ----------
        weights : Dict[str, float]
            Current portfolio weights
            
        Returns
        -------
        Dict[str, float]
            Adjusted weights
        """
        adjustment = self.get_exposure_adjustment()
        
        if adjustment < 1.0:
            logger.info(f"Reducing exposure by {(1-adjustment):.1%}")
        
        return {asset: weight * adjustment for asset, weight in weights.items()}
