"""
Performance Tracker.

Tracks live strategy performance and compares with backtest.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    
    timestamp: datetime
    portfolio_value: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    sharpe_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'drawdown': self.drawdown,
            'sharpe_ratio': self.sharpe_ratio,
        }


class PerformanceTracker:
    """
    Track live strategy performance.
    
    Compares live performance with backtest expectations
    and tracks key metrics over time.
    
    Parameters
    ----------
    strategy_name : str
        Name of the strategy
    backtest_sharpe : float, optional
        Expected Sharpe from backtest
    backtest_returns : pd.Series, optional
        Historical backtest returns for comparison
    """
    
    def __init__(
        self,
        strategy_name: str,
        backtest_sharpe: Optional[float] = None,
        backtest_returns: Optional[pd.Series] = None
    ):
        assert strategy_name, "strategy_name cannot be empty"
        
        self.strategy_name = strategy_name
        self.backtest_sharpe = backtest_sharpe
        self.backtest_returns = backtest_returns
        
        # Live tracking
        self._returns: List[float] = []
        self._dates: List[datetime] = []
        self._snapshots: List[PerformanceSnapshot] = []
        
        # Peak tracking for drawdown
        self._peak_value = 1.0
        self._current_value = 1.0
        
    def update(
        self,
        date: datetime,
        daily_return: float
    ) -> PerformanceSnapshot:
        """
        Update tracker with new daily return.
        
        Parameters
        ----------
        date : datetime
            Date of return
        daily_return : float
            Daily return (e.g., 0.01 for 1%)
            
        Returns
        -------
        PerformanceSnapshot
            Current performance snapshot
        """
        assert date is not None, "Date cannot be None"
        
        self._returns.append(daily_return)
        self._dates.append(date)
        
        # Update portfolio value
        self._current_value *= (1 + daily_return)
        self._peak_value = max(self._peak_value, self._current_value)
        
        # Calculate metrics
        returns_array = np.array(self._returns)
        cumulative = self._current_value - 1
        drawdown = (self._current_value - self._peak_value) / self._peak_value
        
        # Rolling Sharpe (annualized)
        if len(returns_array) >= 20:
            sharpe = returns_array.mean() / returns_array.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=date,
            portfolio_value=self._current_value,
            daily_return=daily_return,
            cumulative_return=cumulative,
            drawdown=drawdown,
            sharpe_ratio=sharpe,
        )
        
        self._snapshots.append(snapshot)
        
        logger.debug(
            f"{self.strategy_name} | {date.date()} | "
            f"Return: {daily_return:.2%} | Cum: {cumulative:.2%} | "
            f"DD: {drawdown:.2%}"
        )
        
        return snapshot
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns
        -------
        Dict[str, float]
            Performance metrics
        """
        if len(self._returns) == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'n_days': 0,
            }
        
        returns = np.array(self._returns)
        n_days = len(returns)
        
        assert n_days > 0, "Must have at least one return"
        
        total_return = self._current_value - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown from snapshots
        max_dd = min(s.drawdown for s in self._snapshots) if self._snapshots else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_dd),
            'n_days': n_days,
        }
    
    def compare_with_backtest(self) -> Dict[str, Any]:
        """
        Compare live performance with backtest.
        
        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        live_metrics = self.get_metrics()
        
        result = {
            'live': live_metrics,
            'backtest_sharpe': self.backtest_sharpe,
            'sharpe_deviation': None,
            'is_underperforming': False,
        }
        
        if self.backtest_sharpe is not None and live_metrics['sharpe_ratio'] != 0:
            deviation = (
                (live_metrics['sharpe_ratio'] - self.backtest_sharpe) 
                / self.backtest_sharpe
            )
            result['sharpe_deviation'] = deviation
            result['is_underperforming'] = deviation < -0.30  # 30% degradation
            
            if result['is_underperforming']:
                logger.warning(
                    f"{self.strategy_name}: Live Sharpe {live_metrics['sharpe_ratio']:.2f} "
                    f"vs Backtest {self.backtest_sharpe:.2f} "
                    f"(deviation: {deviation:.1%})"
                )
        
        return result
    
    def get_returns_series(self) -> pd.Series:
        """Get returns as pandas Series."""
        if not self._returns:
            return pd.Series(dtype=float)
        
        return pd.Series(self._returns, index=pd.DatetimeIndex(self._dates))
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        if not self._snapshots:
            return pd.Series(dtype=float)
        
        values = [s.portfolio_value for s in self._snapshots]
        dates = [s.timestamp for s in self._snapshots]
        
        return pd.Series(values, index=pd.DatetimeIndex(dates))
    
    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._returns = []
        self._dates = []
        self._snapshots = []
        self._peak_value = 1.0
        self._current_value = 1.0
        
        logger.info(f"Performance tracker reset for {self.strategy_name}")
