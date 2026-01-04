"""
Strategy Decay Detector.

Detects when strategy performance degrades significantly from backtest.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_SHARPE_DECAY_THRESHOLD = 0.50  # 50% degradation
DEFAULT_CORRELATION_THRESHOLD = 0.30  # Correlation with backtest
DEFAULT_LOOKBACK = 60  # Trading days


@dataclass
class DecayResult:
    """Result of decay detection."""
    
    is_decaying: bool
    sharpe_live: float
    sharpe_backtest: float
    sharpe_ratio: float  # live / backtest
    correlation: float
    p_value: float
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_decaying': self.is_decaying,
            'sharpe_live': self.sharpe_live,
            'sharpe_backtest': self.sharpe_backtest,
            'sharpe_ratio': self.sharpe_ratio,
            'correlation': self.correlation,
            'p_value': self.p_value,
            'recommendation': self.recommendation,
        }


class DecayDetector:
    """
    Detect strategy decay by comparing live vs backtest performance.
    
    Parameters
    ----------
    lookback : int, default=60
        Lookback period for comparison (trading days)
    sharpe_threshold : float, default=0.50
        Threshold for Sharpe degradation (0.5 = 50% of backtest)
    correlation_threshold : float, default=0.30
        Minimum correlation with backtest returns
    """
    
    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        sharpe_threshold: float = DEFAULT_SHARPE_DECAY_THRESHOLD,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    ):
        assert lookback > 0, "lookback must be positive"
        assert 0 < sharpe_threshold <= 1.0, "sharpe_threshold must be in (0, 1]"
        assert 0 <= correlation_threshold <= 1.0, "correlation_threshold must be in [0, 1]"
        
        self.lookback = lookback
        self.sharpe_threshold = sharpe_threshold
        self.correlation_threshold = correlation_threshold
        
    def check(
        self,
        live_returns: pd.Series,
        backtest_returns: pd.Series
    ) -> DecayResult:
        """
        Check for strategy decay.
        
        Parameters
        ----------
        live_returns : pd.Series
            Live trading returns
        backtest_returns : pd.Series
            Backtest returns (same period or representative)
            
        Returns
        -------
        DecayResult
            Decay detection result
        """
        assert live_returns is not None, "live_returns cannot be None"
        assert backtest_returns is not None, "backtest_returns cannot be None"
        assert len(live_returns) > 0, "live_returns cannot be empty"
        assert len(backtest_returns) > 0, "backtest_returns cannot be empty"
        
        logger.info(f"Checking decay with {len(live_returns)} live observations")
        
        try:
            # Use recent data
            live = live_returns.iloc[-self.lookback:] if len(live_returns) > self.lookback else live_returns
            bt = backtest_returns.iloc[-self.lookback:] if len(backtest_returns) > self.lookback else backtest_returns
            
            # Calculate Sharpe ratios
            sharpe_live = self._calculate_sharpe(live)
            sharpe_bt = self._calculate_sharpe(bt)
            
            # Sharpe ratio (live / backtest)
            sharpe_ratio = sharpe_live / sharpe_bt if sharpe_bt != 0 else 0.0
            
            # Correlation (if same length)
            if len(live) == len(bt):
                correlation = live.corr(bt)
            else:
                correlation = np.nan
            
            # Statistical test for difference
            p_value = self._test_difference(live, bt)
            
            # Determine if decaying
            is_decaying = (
                sharpe_ratio < self.sharpe_threshold or
                (not np.isnan(correlation) and correlation < self.correlation_threshold)
            )
            
            # Generate recommendation
            recommendation = self._get_recommendation(
                is_decaying, sharpe_ratio, correlation, p_value
            )
            
            result = DecayResult(
                is_decaying=is_decaying,
                sharpe_live=sharpe_live,
                sharpe_backtest=sharpe_bt,
                sharpe_ratio=sharpe_ratio,
                correlation=correlation if not np.isnan(correlation) else 0.0,
                p_value=p_value,
                recommendation=recommendation,
            )
            
            if is_decaying:
                logger.warning(f"Strategy decay detected: {recommendation}")
            else:
                logger.info(f"Strategy healthy: Sharpe ratio {sharpe_ratio:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Decay detection failed: {e}")
            raise
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            std = returns.std()
            if std == 0 or np.isnan(std):
                return 0.0
            
            return returns.mean() / std * np.sqrt(252)
        except Exception as e:
            logger.warning(f"Sharpe calculation failed: {e}")
            return 0.0
    
    def _test_difference(
        self,
        live: pd.Series,
        backtest: pd.Series
    ) -> float:
        """
        Test if live returns are significantly different from backtest.
        
        Uses Welch's t-test for unequal variances.
        """
        try:
            _, p_value = stats.ttest_ind(live, backtest, equal_var=False)
            return p_value
        except Exception:
            return 1.0  # Cannot reject null hypothesis
    
    def _get_recommendation(
        self,
        is_decaying: bool,
        sharpe_ratio: float,
        correlation: float,
        p_value: float
    ) -> str:
        """Generate recommendation based on decay analysis."""
        if not is_decaying:
            return "Strategy performing as expected. Continue monitoring."
        
        reasons = []
        
        if sharpe_ratio < self.sharpe_threshold:
            reasons.append(f"Sharpe degraded to {sharpe_ratio:.0%} of backtest")
        
        if not np.isnan(correlation) and correlation < self.correlation_threshold:
            reasons.append(f"Low correlation ({correlation:.2f}) with backtest")
        
        if p_value < 0.05:
            reasons.append("Statistically significant performance difference")
        
        recommendation = "DECAY DETECTED: " + "; ".join(reasons) + ". "
        
        if sharpe_ratio < 0.3:
            recommendation += "Consider stopping strategy."
        elif sharpe_ratio < 0.5:
            recommendation += "Reduce position size by 50%."
        else:
            recommendation += "Monitor closely for next 2 weeks."
        
        return recommendation


def quick_decay_check(
    live_returns: pd.Series,
    backtest_sharpe: float,
    threshold: float = 0.50
) -> Tuple[bool, float]:
    """
    Quick decay check without full backtest returns.
    
    Parameters
    ----------
    live_returns : pd.Series
        Live trading returns
    backtest_sharpe : float
        Expected Sharpe from backtest
    threshold : float
        Decay threshold
        
    Returns
    -------
    Tuple[bool, float]
        (is_decaying, live_sharpe)
    """
    assert live_returns is not None, "live_returns cannot be None"
    assert len(live_returns) > 0, "live_returns cannot be empty"
    
    live_sharpe = live_returns.mean() / live_returns.std() * np.sqrt(252)
    
    ratio = live_sharpe / backtest_sharpe if backtest_sharpe != 0 else 0.0
    is_decaying = ratio < threshold
    
    return is_decaying, live_sharpe
