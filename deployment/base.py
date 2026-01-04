"""
Base classes for Deployment Module.

Provides abstract base class and data structures for platform adapters.
Loads defaults from config module (no hardcoded values).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _get_deployment_defaults() -> Dict[str, Any]:
    """Get deployment defaults from config module."""
    try:
        from config import get_config
        cfg = get_config()
        return {
            'initial_capital': cfg.trading.initial_capital,
            'max_leverage': cfg.trading.max_leverage,
            'max_drawdown': cfg.trading.max_drawdown_pct,
            'slippage_bps': cfg.trading.slippage_pct * 10000,  # Convert to bps
            'commission_bps': cfg.trading.commission_pct * 10000,  # Convert to bps
        }
    except Exception as e:
        logger.warning(f"[WARN] Could not load config, using fallback defaults: {e}")
        return {
            'initial_capital': 10000.0,
            'max_leverage': 1.0,
            'max_drawdown': 0.15,
            'slippage_bps': 10.0,
            'commission_bps': 20.0,
        }


class PlatformType(Enum):
    """Supported deployment platforms."""
    QUANTIACS = "quantiacs"
    QUANTCONNECT = "quantconnect"
    PAPER = "paper"


@dataclass
class DeploymentConfig:
    """
    Configuration for deployment.
    
    Defaults are loaded from config module (config/default.yaml).
    """
    
    platform: PlatformType = PlatformType.QUANTIACS
    strategy_name: str = "QuantLabStrategy"
    version: str = "1.0.0"
    
    # Capital and risk - defaults loaded from config
    initial_capital: Optional[float] = None
    max_leverage: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Execution - defaults loaded from config
    slippage_bps: Optional[float] = None
    commission_bps: Optional[float] = None
    
    # Monitoring
    alert_email: Optional[str] = None
    decay_threshold: float = 0.50  # 50% Sharpe degradation
    
    def __post_init__(self) -> None:
        """Load defaults from config and validate."""
        defaults = _get_deployment_defaults()
        
        # Apply defaults if not set
        if self.initial_capital is None:
            self.initial_capital = defaults['initial_capital']
        if self.max_leverage is None:
            self.max_leverage = defaults['max_leverage']
        if self.max_drawdown is None:
            self.max_drawdown = defaults['max_drawdown']
        if self.slippage_bps is None:
            self.slippage_bps = defaults['slippage_bps']
        if self.commission_bps is None:
            self.commission_bps = defaults['commission_bps']
        
        # Validate
        assert self.initial_capital > 0, "initial_capital must be positive"
        assert 0 < self.max_leverage <= 10, "max_leverage must be in (0, 10]"
        assert 0 < self.max_drawdown <= 1.0, "max_drawdown must be in (0, 1]"
        assert self.slippage_bps >= 0, "slippage_bps must be non-negative"
        assert self.commission_bps >= 0, "commission_bps must be non-negative"


@dataclass
class DeploymentResult:
    """Result of deployment operation."""
    
    success: bool
    platform: PlatformType
    strategy_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Generated artifacts
    code: Optional[str] = None
    backtest_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'platform': self.platform.value,
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'metadata': self.metadata,
        }


class BaseAdapter(ABC):
    """Abstract base class for platform adapters."""
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize adapter.
        
        Parameters
        ----------
        config : DeploymentConfig, optional
            Deployment configuration
        """
        self.config = config or DeploymentConfig()
        
    @abstractmethod
    def convert(
        self, 
        strategy_func: Callable,
        **kwargs
    ) -> str:
        """
        Convert strategy to platform-specific format.
        
        Parameters
        ----------
        strategy_func : Callable
            Strategy function that takes data and returns signals
        **kwargs
            Additional platform-specific parameters
            
        Returns
        -------
        str
            Platform-specific code or configuration
        """
        pass
    
    @abstractmethod
    def validate(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Validate strategy meets platform requirements.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
            
        Returns
        -------
        Dict[str, Any]
            Validation results with metrics
        """
        pass
    
    @abstractmethod
    def deploy(self, code: str) -> DeploymentResult:
        """
        Deploy strategy to platform.
        
        Parameters
        ----------
        code : str
            Platform-specific code
            
        Returns
        -------
        DeploymentResult
            Deployment result
        """
        pass
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate standard performance metrics."""
        assert returns is not None, "Returns cannot be None"
        assert len(returns) > 0, "Returns cannot be empty"
        
        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * (252 ** 0.5)
            
            # Sharpe ratio
            sharpe = annual_return / volatility if volatility > 0 else 0.0
            
            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'n_observations': len(returns),
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            raise
