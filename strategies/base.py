"""
Base Strategy Classes.

Abstract base classes for all trading strategies.
Follows the fit/predict pattern for consistency.

Reference: Protokol Kausalitas - Fase 2 (Strategy Development)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum


class StrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    PAIRS_TRADING = "pairs_trading"
    MARKET_NEUTRAL = "market_neutral"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"


@dataclass
class StrategyConfig:
    """Configuration for strategy."""
    name: str = "base_strategy"
    strategy_type: StrategyType = StrategyType.MOMENTUM
    lookback: int = 20
    rebalance_frequency: int = 1
    max_position: float = 1.0
    min_position: float = -1.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySignal:
    """Container for strategy signals."""
    timestamp: pd.Timestamp
    signal: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    returns: pd.Series
    positions: pd.Series
    signals: pd.Series
    equity_curve: pd.Series
    trades: pd.DataFrame
    
    sharpe_ratio: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute metrics after initialization."""
        if len(self.returns) > 0:
            self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute performance metrics."""
        returns = self.returns.dropna()
        if len(returns) < 2:
            return
        
        # Sharpe Ratio (annualized)
        mean_ret = returns.mean()
        std_ret = returns.std()
        if std_ret > 0:
            self.sharpe_ratio = mean_ret / std_ret * np.sqrt(252)
        
        # Annualized Return
        total_days = (returns.index[-1] - returns.index[0]).days
        if total_days > 0:
            total_return = (1 + returns).prod() - 1
            years = total_days / 365.25
            self.annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        self.max_drawdown = drawdown.min()
        
        # Trade statistics
        if len(self.trades) > 0:
            self.n_trades = len(self.trades)
            if 'pnl' in self.trades.columns:
                wins = self.trades[self.trades['pnl'] > 0]
                losses = self.trades[self.trades['pnl'] < 0]
                self.win_rate = len(wins) / self.n_trades if self.n_trades > 0 else 0
                
                total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
                total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
                self.profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
    
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Backtest Results:\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.4f}\n"
            f"  Annual Return: {self.annualized_return:.2%}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Win Rate: {self.win_rate:.2%}\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  N Trades: {self.n_trades}"
        )


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - fit(): Learn from historical data
    - predict(): Generate trading signals
    
    Optional:
    - validate(): Check if strategy is ready
    - get_params(): Return strategy parameters
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize strategy.
        
        Parameters
        ----------
        config : StrategyConfig, optional
            Strategy configuration
        """
        self.config = config or StrategyConfig()
        self._is_fitted = False
        self._fit_data = None
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseStrategy':
        """
        Fit strategy to historical data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical price data with columns: open, high, low, close, volume
            
        Returns
        -------
        BaseStrategy
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data
            
        Returns
        -------
        pd.Series
            Trading signals: 1 (long), -1 (short), 0 (flat)
        """
        pass
    
    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)
    
    @property
    def is_fitted(self) -> bool:
        """Check if strategy is fitted."""
        return self._is_fitted
    
    def validate(self) -> bool:
        """Validate strategy is ready for trading."""
        return self._is_fitted
    
    def get_params(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            'name': self.config.name,
            'type': self.config.strategy_type.value,
            'lookback': self.config.lookback,
            'params': self.config.params,
        }
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data = data.set_index('date')
        
        return data.sort_index()
    
    def _get_close(self, data: pd.DataFrame) -> pd.Series:
        """Extract close prices from data."""
        if 'close' in data.columns:
            return data['close']
        elif len(data.columns) == 1:
            return data.iloc[:, 0]
        else:
            raise ValueError("Cannot find 'close' column in data")
    
    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """
        Run simple vectorized backtest.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data
        initial_capital : float
            Starting capital
            
        Returns
        -------
        BacktestResult
            Backtest results
        """
        # Load default capital from config if not provided
        if initial_capital is None:
            try:
                from config import get_config
                initial_capital = get_config().backtest.initial_capital
            except Exception:
                initial_capital = 100000.0  # Fallback
        
        data = self._validate_data(data)
        close = self._get_close(data)
        
        # Generate signals
        signals = self.predict(data)
        
        # Calculate returns
        price_returns = close.pct_change()
        
        # Strategy returns (signal * next period return)
        positions = signals.shift(1)  # Enter at next bar
        strategy_returns = positions * price_returns
        
        # Apply transaction costs
        position_changes = positions.diff().abs()
        costs = position_changes * self.config.transaction_cost
        strategy_returns = strategy_returns - costs
        
        # Equity curve
        equity = initial_capital * (1 + strategy_returns).cumprod()
        
        # Generate trades DataFrame
        trades = self._generate_trades(signals, close, strategy_returns)
        
        return BacktestResult(
            returns=strategy_returns.dropna(),
            positions=positions.dropna(),
            signals=signals,
            equity_curve=equity.dropna(),
            trades=trades,
            metadata={
                'initial_capital': initial_capital,
                'config': self.get_params(),
            }
        )
    
    def _generate_trades(
        self,
        signals: pd.Series,
        prices: pd.Series,
        returns: pd.Series
    ) -> pd.DataFrame:
        """Generate trades DataFrame from signals."""
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for date, signal in signals.items():
            if signal != position:
                # Close existing position
                if position != 0 and entry_date is not None:
                    exit_price = prices.loc[date]
                    pnl = (exit_price / entry_price - 1) * position
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': 'long' if position > 0 else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                    })
                
                # Open new position
                if signal != 0:
                    entry_price = prices.loc[date]
                    entry_date = date
                
                position = signal
        
        return pd.DataFrame(trades)
