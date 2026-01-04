"""
Base classes for Risk Engine.

Provides abstract base class and data structures for risk management.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    max_drawdown: float = 0.20  # 20% max drawdown
    max_position: float = 0.25  # 25% max single position
    max_correlation: float = 0.30  # 30% max correlation with benchmark
    var_confidence: float = 0.95  # 95% VaR confidence
    lookback_days: int = 252  # 1 year lookback
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        assert 0 < self.max_drawdown <= 1.0, "max_drawdown must be in (0, 1]"
        assert 0 < self.max_position <= 1.0, "max_position must be in (0, 1]"
        assert 0 < self.max_correlation <= 1.0, "max_correlation must be in (0, 1]"
        assert 0 < self.var_confidence < 1.0, "var_confidence must be in (0, 1)"


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    volatility: float = 0.0
    correlation: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Status flags
    is_drawdown_breach: bool = False
    is_var_breach: bool = False
    is_correlation_breach: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_safe(self) -> bool:
        """Check if all risk metrics are within limits."""
        return not (
            self.is_drawdown_breach or 
            self.is_var_breach or 
            self.is_correlation_breach
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'volatility': self.volatility,
            'correlation': self.correlation,
            'sharpe_ratio': self.sharpe_ratio,
            'is_safe': self.is_safe,
        }


class BaseRiskManager(ABC):
    """Abstract base class for risk managers."""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk manager.
        
        Parameters
        ----------
        config : RiskConfig, optional
            Risk configuration
        """
        self.config = config or RiskConfig()
        
    @abstractmethod
    def check(self, portfolio_value: pd.Series) -> bool:
        """
        Check if portfolio is within risk limits.
        
        Parameters
        ----------
        portfolio_value : pd.Series
            Historical portfolio values
            
        Returns
        -------
        bool
            True if within limits, False otherwise
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> RiskMetrics:
        """
        Get current risk metrics.
        
        Returns
        -------
        RiskMetrics
            Current risk metrics
        """
        pass
    
    def _validate_data(self, data: pd.Series) -> None:
        """Validate input data."""
        assert data is not None, "Data cannot be None"
        assert len(data) > 0, "Data cannot be empty"
        
        nan_count = data.isna().sum()
        if nan_count > 0:
            logger.warning(f"Data contains {nan_count} NaN values")


def calculate_drawdown(portfolio_value: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    
    Parameters
    ----------
    portfolio_value : pd.Series
        Portfolio value series
        
    Returns
    -------
    pd.Series
        Drawdown series (negative values)
    """
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    return drawdown


def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    portfolio_value : pd.Series
        Portfolio value series
        
    Returns
    -------
    float
        Maximum drawdown (positive value)
    """
    drawdown = calculate_drawdown(portfolio_value)
    return abs(drawdown.min())
