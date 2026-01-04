"""
Value at Risk (VaR) Calculator.

Calculates VaR and CVaR (Expected Shortfall) for risk budgeting.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseRiskManager, RiskConfig, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """VaR calculation result."""
    
    var: float  # Value at Risk
    cvar: float  # Conditional VaR (Expected Shortfall)
    confidence: float
    method: str
    n_observations: int


class VaRCalculator(BaseRiskManager):
    """
    Value at Risk calculator.
    
    Calculates VaR using historical, parametric, or Monte Carlo methods.
    
    Parameters
    ----------
    config : RiskConfig, optional
        Risk configuration
    method : str, default='historical'
        VaR calculation method: 'historical', 'parametric', 'cornish_fisher'
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        method: str = 'historical'
    ):
        super().__init__(config)
        
        valid_methods = ['historical', 'parametric', 'cornish_fisher']
        assert method in valid_methods, f"method must be one of {valid_methods}"
        
        self.method = method
        self._result: Optional[VaRResult] = None
        self._metrics: Optional[RiskMetrics] = None
        
    def check(self, portfolio_value: pd.Series) -> bool:
        """
        Check VaR against limits.
        
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
        
        # Calculate returns
        returns = portfolio_value.pct_change().dropna()
        
        # Calculate VaR
        result = self.calculate(returns)
        self._result = result
        
        # Update metrics
        self._metrics = RiskMetrics(
            var_95=result.var,
            cvar_95=result.cvar,
        )
        
        logger.info(
            f"VaR ({result.confidence:.0%}): {result.var:.2%}, "
            f"CVaR: {result.cvar:.2%}"
        )
        
        return True  # VaR doesn't have a hard limit by default
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        if self._metrics is None:
            return RiskMetrics()
        return self._metrics
    
    def calculate(
        self, 
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> VaRResult:
        """
        Calculate VaR and CVaR.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns
        confidence : float, optional
            Confidence level (default from config)
            
        Returns
        -------
        VaRResult
            VaR calculation result
        """
        assert returns is not None, "Returns cannot be None"
        assert len(returns) > 0, "Returns cannot be empty"
        
        confidence = confidence or self.config.var_confidence
        
        try:
            if self.method == 'historical':
                var, cvar = self._historical_var(returns, confidence)
            elif self.method == 'parametric':
                var, cvar = self._parametric_var(returns, confidence)
            elif self.method == 'cornish_fisher':
                var, cvar = self._cornish_fisher_var(returns, confidence)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            return VaRResult(
                var=var,
                cvar=cvar,
                confidence=confidence,
                method=self.method,
                n_observations=len(returns)
            )
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise
    
    def _historical_var(
        self, 
        returns: pd.Series, 
        confidence: float
    ) -> Tuple[float, float]:
        """Calculate historical VaR."""
        # VaR is the percentile of losses
        var = -np.percentile(returns, (1 - confidence) * 100)
        
        # CVaR is the mean of losses beyond VaR
        losses = -returns[returns < -var]
        cvar = losses.mean() if len(losses) > 0 else var
        
        return var, cvar
    
    def _parametric_var(
        self, 
        returns: pd.Series, 
        confidence: float
    ) -> Tuple[float, float]:
        """Calculate parametric (Gaussian) VaR."""
        mu = returns.mean()
        sigma = returns.std()
        
        # Z-score for confidence level
        z = stats.norm.ppf(1 - confidence)
        
        var = -(mu + z * sigma)
        
        # CVaR for normal distribution
        pdf_z = stats.norm.pdf(z)
        cvar = -mu + sigma * pdf_z / (1 - confidence)
        
        return var, cvar
    
    def _cornish_fisher_var(
        self, 
        returns: pd.Series, 
        confidence: float
    ) -> Tuple[float, float]:
        """
        Calculate Cornish-Fisher VaR.
        
        Adjusts for skewness and kurtosis.
        """
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Standard normal quantile
        z = stats.norm.ppf(1 - confidence)
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = -(mu + z_cf * sigma)
        
        # Approximate CVaR
        cvar = var * 1.1  # Simple approximation
        
        return var, cvar


def calculate_var_simple(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Simple historical VaR calculation.
    
    Parameters
    ----------
    returns : pd.Series
        Historical returns
    confidence : float, default=0.95
        Confidence level
        
    Returns
    -------
    float
        VaR value (positive)
    """
    return -np.percentile(returns, (1 - confidence) * 100)
