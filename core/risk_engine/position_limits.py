"""
Position Limits.

Enforces position size constraints for risk management.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseRiskManager, RiskConfig, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class PositionLimitResult:
    """Position limit check result."""
    
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    breaches: List[str]
    is_adjusted: bool


class PositionLimiter(BaseRiskManager):
    """
    Position limiter for risk management.
    
    Enforces maximum position sizes and concentration limits.
    
    Parameters
    ----------
    config : RiskConfig, optional
        Risk configuration
    min_position : float, default=0.0
        Minimum position size
    max_concentration : float, default=0.5
        Maximum concentration in top N positions
    top_n : int, default=3
        Number of top positions for concentration check
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        min_position: float = 0.0,
        max_concentration: float = 0.5,
        top_n: int = 3
    ):
        super().__init__(config)
        
        assert 0 <= min_position < 1.0, "min_position must be in [0, 1)"
        assert 0 < max_concentration <= 1.0, "max_concentration must be in (0, 1]"
        assert top_n > 0, "top_n must be positive"
        
        self.min_position = min_position
        self.max_concentration = max_concentration
        self.top_n = top_n
        
        self._result: Optional[PositionLimitResult] = None
        
    def check(self, portfolio_value: pd.Series) -> bool:
        """
        Check position limits (requires weights).
        
        Use check_weights() for weight-based checking.
        """
        logger.warning("check() requires weights. Use check_weights() instead.")
        return True
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return RiskMetrics()
    
    def check_weights(
        self, 
        weights: Dict[str, float]
    ) -> PositionLimitResult:
        """
        Check and adjust weights for position limits.
        
        Parameters
        ----------
        weights : Dict[str, float]
            Portfolio weights
            
        Returns
        -------
        PositionLimitResult
            Result with adjusted weights
        """
        assert weights is not None, "Weights cannot be None"
        assert len(weights) > 0, "Weights cannot be empty"
        
        try:
            breaches = []
            adjusted = weights.copy()
            
            # Apply limits using vectorized approach
            weights_series = pd.Series(adjusted)
            
            # Check individual position limits (vectorized)
            breach_mask = weights_series.abs() > self.config.max_position
            if breach_mask.any():
                breaches = weights_series[breach_mask].index.tolist()
                for asset in breaches:
                    logger.warning(
                        f"Position limit breach: {asset} "
                        f"{weights_series[asset]:.2%} -> {self.config.max_position:.2%}"
                    )
                # Clip to max position
                weights_series = weights_series.clip(
                    lower=-self.config.max_position,
                    upper=self.config.max_position
                )
            
            # Remove positions below minimum (vectorized)
            below_min_mask = (weights_series.abs() < self.min_position) & (weights_series != 0)
            weights_series[below_min_mask] = 0.0
            
            # Check concentration (vectorized)
            sorted_weights = weights_series.abs().sort_values(ascending=False)
            top_concentration = sorted_weights.iloc[:self.top_n].sum()
            
            if top_concentration > self.max_concentration:
                logger.warning(
                    f"Concentration breach: top {self.top_n} = "
                    f"{top_concentration:.2%} > {self.max_concentration:.2%}"
                )
                scale = self.max_concentration / top_concentration
                weights_series = weights_series * scale
            
            adjusted = weights_series.to_dict()
            
            # Final normalization (without exceeding limits)
            total = sum(adjusted.values())
            if total > 0 and total < 1.0:
                # Only scale up if it won't breach limits
                max_scale = min(
                    self.config.max_position / max(abs(w) for w in adjusted.values())
                    if max(abs(w) for w in adjusted.values()) > 0 else 1.0,
                    1.0 / total
                )
                if max_scale > 1.0:
                    scale = min(max_scale, 1.0 / total)
                    adjusted = {k: v * scale for k, v in adjusted.items()}
            
            is_adjusted = adjusted != weights
            
            result = PositionLimitResult(
                original_weights=weights,
                adjusted_weights=adjusted,
                breaches=breaches,
                is_adjusted=is_adjusted
            )
            
            self._result = result
            
            if is_adjusted:
                logger.info(f"Weights adjusted, {len(breaches)} breaches")
            else:
                logger.info("All positions within limits")
            
            return result
            
        except Exception as e:
            logger.error(f"Position limit check failed: {e}")
            raise
    
    def apply_limits(
        self, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply position limits and return adjusted weights.
        
        Parameters
        ----------
        weights : Dict[str, float]
            Original weights
            
        Returns
        -------
        Dict[str, float]
            Adjusted weights
        """
        result = self.check_weights(weights)
        return result.adjusted_weights
    
    def get_result(self) -> PositionLimitResult:
        """Get latest position limit result."""
        if self._result is None:
            raise ValueError("Must call check_weights() first")
        return self._result
