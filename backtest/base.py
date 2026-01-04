"""
Backtest Base - Base classes for backtest adapters.

Provides common interface for all backtest engines (VectorBT, Nautilus, LEAN).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine(Enum):
    """Supported backtest engines."""
    VECTORBT = "vectorbt"
    NAUTILUS = "nautilus"
    LEAN = "lean"
    QUANTIACS = "quantiacs"


def _load_config_defaults() -> Dict[str, Any]:
    """Load defaults from config module if available."""
    try:
        from config import get_config
        cfg = get_config()
        return {
            'initial_capital': cfg.backtest.initial_capital,
            'commission': cfg.backtest.commission_pct,
            'slippage': cfg.backtest.slippage_pct,
            'max_position_size': cfg.backtest.max_position_pct,
            'max_leverage': cfg.backtest.max_leverage,
            'max_drawdown': cfg.backtest.max_drawdown_pct,
            'target_volatility': cfg.backtest.target_volatility_pct,
            'vol_lookback': cfg.backtest.vol_lookback,
            'position_sizing': cfg.backtest.position_sizing,
        }
    except Exception:
        return {}


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    
    initial_capital: float = None
    commission: float = None
    slippage: float = None
    
    # Risk parameters
    max_position_size: float = None
    max_leverage: float = None
    max_drawdown: float = None
    
    # Volatility targeting
    target_volatility: float = None
    vol_lookback: int = None
    position_sizing: str = None
    
    # Time parameters
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Engine-specific
    engine: BacktestEngine = BacktestEngine.VECTORBT
    
    def __post_init__(self) -> None:
        """Load defaults from config and validate."""
        defaults = _load_config_defaults()
        
        # Apply defaults for None values
        if self.initial_capital is None:
            self.initial_capital = defaults.get('initial_capital', 100000.0)
        if self.commission is None:
            self.commission = defaults.get('commission', 0.001)
        if self.slippage is None:
            self.slippage = defaults.get('slippage', 0.0005)
        if self.max_position_size is None:
            self.max_position_size = defaults.get('max_position_size', 0.20)
        if self.max_leverage is None:
            self.max_leverage = defaults.get('max_leverage', 1.0)
        if self.max_drawdown is None:
            self.max_drawdown = defaults.get('max_drawdown', 0.20)
        if self.target_volatility is None:
            self.target_volatility = defaults.get('target_volatility', 0.25)
        if self.vol_lookback is None:
            self.vol_lookback = defaults.get('vol_lookback', 36)
        if self.position_sizing is None:
            self.position_sizing = defaults.get('position_sizing', 'volatility_target')
        
        # Validate
        assert self.initial_capital > 0, "Initial capital must be positive"
        assert 0 <= self.commission < 1, "Commission must be between 0 and 1"
        assert 0 <= self.slippage < 1, "Slippage must be between 0 and 1"
        assert 0 < self.max_position_size <= 1, "Max position size must be (0, 1]"


@dataclass
class BacktestMetrics:
    """Standard metrics from backtest results."""
    
    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    psr: float = 0.0  # Probabilistic Sharpe Ratio
    
    # Risk
    max_drawdown: float = 0.0
    volatility: float = 0.0
    
    # Trading
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    
    # Time
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'psr': self.psr,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'trading_days': self.trading_days
        }
    
    def passes_filter(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.30,
        min_trades: int = 10,
        min_psr: float = 0.0
    ) -> bool:
        """Check if metrics pass screening filter."""
        return (
            self.sharpe_ratio >= min_sharpe and
            abs(self.max_drawdown) <= max_drawdown and
            self.total_trades >= min_trades and
            self.psr >= min_psr
        )


@dataclass
class BacktestResult:
    """Complete backtest result."""
    
    # Identification
    strategy_name: str
    asset: str
    engine: BacktestEngine
    
    # Metrics
    metrics: BacktestMetrics
    
    # Time series
    equity_curve: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None
    positions: Optional[pd.Series] = None
    
    # Trades
    trades: Optional[pd.DataFrame] = None
    
    # Parameters used
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    config: Optional[BacktestConfig] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.strategy_name}, {self.asset}, "
            f"Sharpe={self.metrics.sharpe_ratio:.2f}, "
            f"MaxDD={self.metrics.max_drawdown:.1%})"
        )


class BaseBacktestAdapter(ABC):
    """Abstract base class for backtest adapters."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize adapter with configuration."""
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        asset: str = "UNKNOWN"
    ) -> BacktestResult:
        """
        Run backtest with given prices and signals.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
        signals : pd.Series
            Trading signals: 1 (long), -1 (short), 0 (flat)
        asset : str
            Asset identifier
            
        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        pass
    
    @abstractmethod
    def run_strategy(
        self,
        prices: pd.DataFrame,
        strategy: Any,
        asset: str = "UNKNOWN"
    ) -> BacktestResult:
        """
        Run backtest with strategy object.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        strategy : Any
            Strategy object with fit/predict methods
        asset : str
            Asset identifier
            
        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        pass
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None
    ) -> BacktestMetrics:
        """
        Calculate standard metrics from returns.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns series
        trades : pd.DataFrame, optional
            Trade log with pnl column
            
        Returns
        -------
        BacktestMetrics
            Calculated metrics
        """
        assert returns is not None, "Returns cannot be None"
        assert len(returns) > 0, "Returns cannot be empty"
        
        try:
            # Clean returns
            returns = returns.dropna()
            
            if len(returns) < 2:
                self.logger.warning("Insufficient returns for metrics calculation")
                return BacktestMetrics()
            
            # Basic returns
            total_return = (1 + returns).prod() - 1
            trading_days = len(returns)
            annual_factor = 252 / trading_days if trading_days > 0 else 1
            annual_return = (1 + total_return) ** annual_factor - 1
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            mean_return = returns.mean() * 252
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0
            
            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calmar Ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            # PSR (Probabilistic Sharpe Ratio)
            # PSR = Φ((SR - SR_benchmark) / SE_SR)
            # SE_SR = sqrt((1 - γ₃×SR + (γ₄-1)/4 × SR²) / (n-1))
            n = len(returns)
            if n > 1 and volatility > 0:
                # Calculate skewness and kurtosis
                skewness = returns.skew()
                kurtosis = returns.kurtosis()  # Excess kurtosis
                
                # Standard error of Sharpe Ratio
                se_sr = np.sqrt((1 - skewness * sharpe_ratio + 
                                ((kurtosis + 2) / 4) * sharpe_ratio**2) / (n - 1))
                
                # PSR with benchmark SR = 0
                if se_sr > 0:
                    from scipy import stats
                    psr = stats.norm.cdf(sharpe_ratio / se_sr)
                else:
                    psr = 0.5
            else:
                psr = 0.5
            
            # Trade metrics
            total_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0
            
            if trades is not None and len(trades) > 0 and 'pnl' in trades.columns:
                total_trades = len(trades)
                winning_trades = trades[trades['pnl'] > 0]
                losing_trades = trades[trades['pnl'] < 0]
                
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                
                total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
                total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
                profit_factor = total_profit / total_loss if total_loss > 0 else 0
                
                avg_trade_return = trades['pnl'].mean()
            
            return BacktestMetrics(
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                psr=psr,
                max_drawdown=max_drawdown,
                volatility=volatility,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_return=avg_trade_return,
                start_date=returns.index[0] if hasattr(returns.index[0], 'date') else None,
                end_date=returns.index[-1] if hasattr(returns.index[-1], 'date') else None,
                trading_days=trading_days
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return BacktestMetrics()
