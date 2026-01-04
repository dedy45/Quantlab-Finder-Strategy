"""
Correlation Monitor.

Monitors correlation with benchmark to ensure alpha generation.
Target: correlation < 0.3 for Alpha Streams.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseRiskManager, RiskConfig, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Correlation analysis result."""
    
    correlation: float
    rolling_corr: pd.Series
    is_breach: bool
    threshold: float
    beta: float
    r_squared: float


class CorrelationMonitor(BaseRiskManager):
    """
    Correlation monitor for alpha verification.
    
    Monitors correlation with benchmark to ensure strategy
    generates alpha, not just beta exposure.
    
    Parameters
    ----------
    config : RiskConfig, optional
        Risk configuration
    lookback : int, default=60
        Lookback period for rolling correlation
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        lookback: int = 60
    ):
        super().__init__(config)
        
        assert lookback > 0, "lookback must be positive"
        self.lookback = lookback
        
        self._result: Optional[CorrelationResult] = None
        self._metrics: Optional[RiskMetrics] = None
        
    def check(self, portfolio_value: pd.Series) -> bool:
        """
        Check correlation (requires benchmark in metadata).
        
        Parameters
        ----------
        portfolio_value : pd.Series
            Historical portfolio values
            
        Returns
        -------
        bool
            True if within limits
        """
        logger.warning("check() requires benchmark. Use analyze() instead.")
        return True
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        if self._metrics is None:
            return RiskMetrics()
        return self._metrics
    
    def analyze(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> CorrelationResult:
        """
        Analyze correlation with benchmark.
        
        Parameters
        ----------
        strategy_returns : pd.Series
            Strategy returns
        benchmark_returns : pd.Series
            Benchmark returns (e.g., SPY)
            
        Returns
        -------
        CorrelationResult
            Correlation analysis result
        """
        assert strategy_returns is not None, "strategy_returns cannot be None"
        assert benchmark_returns is not None, "benchmark_returns cannot be None"
        
        try:
            # Align data
            aligned = pd.DataFrame({
                'strategy': strategy_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            assert len(aligned) > self.lookback, \
                f"Need at least {self.lookback} observations"
            
            # Calculate correlation
            correlation = aligned['strategy'].corr(aligned['benchmark'])
            
            # Rolling correlation
            rolling_corr = aligned['strategy'].rolling(
                window=self.lookback
            ).corr(aligned['benchmark'])
            
            # Calculate beta (regression coefficient)
            cov = aligned.cov().loc['strategy', 'benchmark']
            var_benchmark = aligned['benchmark'].var()
            beta = cov / var_benchmark if var_benchmark > 0 else 0.0
            
            # R-squared
            r_squared = correlation ** 2
            
            # Check breach
            is_breach = abs(correlation) >= self.config.max_correlation
            
            result = CorrelationResult(
                correlation=correlation,
                rolling_corr=rolling_corr,
                is_breach=is_breach,
                threshold=self.config.max_correlation,
                beta=beta,
                r_squared=r_squared
            )
            
            self._result = result
            self._metrics = RiskMetrics(
                correlation=correlation,
                is_correlation_breach=is_breach
            )
            
            # Log status
            status = "BREACH" if is_breach else "OK"
            logger.info(
                f"Correlation {status}: {correlation:.2f} "
                f"(threshold: {self.config.max_correlation:.2f}), "
                f"beta: {beta:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def get_result(self) -> CorrelationResult:
        """Get latest correlation result."""
        if self._result is None:
            raise ValueError("Must call analyze() first")
        return self._result


def calculate_correlation(
    returns1: pd.Series,
    returns2: pd.Series
) -> float:
    """
    Calculate correlation between two return series.
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
        
    Returns
    -------
    float
        Correlation coefficient
    """
    aligned = pd.DataFrame({
        'r1': returns1,
        'r2': returns2
    }).dropna()
    
    return aligned['r1'].corr(aligned['r2'])


def calculate_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate beta (market sensitivity).
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
        
    Returns
    -------
    float
        Beta coefficient
    """
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    cov = aligned.cov().loc['strategy', 'benchmark']
    var_benchmark = aligned['benchmark'].var()
    
    return cov / var_benchmark if var_benchmark > 0 else 0.0
