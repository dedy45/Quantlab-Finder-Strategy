"""
Paper Trader - Simulates live trading without real money.

Provides realistic order execution simulation with:
- Slippage modeling
- Commission costs
- Position tracking
- P&L calculation

Defaults loaded from config module (no hardcoded values).

Version: 0.7.1
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_paper_trading_defaults() -> Dict[str, float]:
    """Get paper trading defaults from config module."""
    try:
        from config import get_config
        cfg = get_config()
        return {
            'slippage_pct': cfg.trading.slippage_pct,
            'commission_pct': cfg.trading.commission_pct,
            'initial_capital': cfg.trading.initial_capital,
            'max_position_pct': cfg.trading.max_position_pct,
        }
    except Exception as e:
        logger.warning(f"[WARN] Could not load config, using fallback defaults: {e}")
        return {
            'slippage_pct': 0.001,
            'commission_pct': 0.002,
            'initial_capital': 10000.0,
            'max_position_pct': 0.10,
        }


# Load defaults from config
_DEFAULTS = _get_paper_trading_defaults()
DEFAULT_SLIPPAGE_PCT = _DEFAULTS['slippage_pct']
DEFAULT_COMMISSION_PCT = _DEFAULTS['commission_pct']
DEFAULT_INITIAL_CAPITAL = _DEFAULTS['initial_capital']
DEFAULT_MAX_POSITION_PCT = _DEFAULTS['max_position_pct']
MIN_TRADE_SIZE = 0.01


class OrderSide(Enum):
    """Order side enum."""
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class PaperTradingConfig:
    """
    Configuration for paper trading.
    
    Defaults loaded from config module (config/default.yaml).
    """
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT
    commission_pct: float = DEFAULT_COMMISSION_PCT
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    allow_short: bool = True
    log_trades: bool = True


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    
    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None
    
    @property
    def holding_period(self) -> Optional[float]:
        """Calculate holding period in days."""
        if self.exit_timestamp is None:
            return None
        delta = self.exit_timestamp - self.timestamp
        return delta.total_seconds() / 86400


class PaperTrader:
    """
    Paper Trading Simulator.
    
    Simulates live trading with realistic execution:
    - Slippage on entry and exit
    - Commission costs
    - Position tracking
    - P&L calculation
    
    Parameters
    ----------
    config : PaperTradingConfig
        Trading configuration
    strategy_name : str
        Name of strategy being tested
        
    Examples
    --------
    >>> config = PaperTradingConfig(initial_capital=100000)
    >>> trader = PaperTrader(config, "MomentumGold")
    >>> trader.execute_signal(datetime.now(), "XAUUSD", 1, 2650.0)
    >>> trader.execute_signal(datetime.now(), "XAUUSD", 0, 2660.0)
    >>> print(trader.get_summary())
    """
    
    def __init__(
        self,
        config: Optional[PaperTradingConfig] = None,
        strategy_name: str = "unnamed"
    ):
        """Initialize paper trader."""
        assert strategy_name, "Strategy name cannot be empty"
        
        self.config = config or PaperTradingConfig()
        self.strategy_name = strategy_name
        
        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.open_trades: Dict[str, TradeRecord] = {}  # symbol -> open trade
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        logger.info(
            f"PaperTrader initialized: {strategy_name}, "
            f"capital=${self.config.initial_capital:,.0f}"
        )
    
    def execute_signal(
        self,
        timestamp: datetime,
        symbol: str,
        signal: int,
        price: float,
        quantity: Optional[float] = None
    ) -> Optional[TradeRecord]:
        """
        Execute a trading signal.
        
        Parameters
        ----------
        timestamp : datetime
            Signal timestamp
        symbol : str
            Trading symbol
        signal : int
            Signal: 1 (long), -1 (short), 0 (flat/close)
        price : float
            Current market price
        quantity : float, optional
            Trade quantity (default: calculate from capital)
            
        Returns
        -------
        TradeRecord or None
            Trade record if executed
        """
        assert price > 0, "Price must be positive"
        assert signal in [-1, 0, 1], "Signal must be -1, 0, or 1"
        
        try:
            current_position = self.positions.get(symbol, 0)
            
            # Determine action
            if signal == 0:
                # Close position
                if current_position != 0:
                    return self._close_position(timestamp, symbol, price)
                return None
            
            elif signal == 1:
                # Long
                if current_position < 0:
                    # Close short first
                    self._close_position(timestamp, symbol, price)
                if current_position <= 0:
                    return self._open_position(
                        timestamp, symbol, OrderSide.BUY, price, quantity
                    )
            
            elif signal == -1:
                # Short
                if not self.config.allow_short:
                    logger.warning("Short selling not allowed")
                    return None
                if current_position > 0:
                    # Close long first
                    self._close_position(timestamp, symbol, price)
                if current_position >= 0:
                    return self._open_position(
                        timestamp, symbol, OrderSide.SELL, price, quantity
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return None
    
    def _open_position(
        self,
        timestamp: datetime,
        symbol: str,
        side: OrderSide,
        price: float,
        quantity: Optional[float] = None
    ) -> TradeRecord:
        """Open a new position."""
        # Calculate quantity if not provided
        if quantity is None:
            max_value = self.capital * self.config.max_position_pct
            quantity = max_value / price
        
        # Apply slippage (worse price)
        slippage = price * self.config.slippage_pct
        if side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        # Calculate commission
        commission = fill_price * quantity * self.config.commission_pct
        
        # Create trade record
        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            commission=commission,
            slippage=slippage * quantity,
            status=OrderStatus.FILLED
        )
        
        # Update state
        position_sign = 1 if side == OrderSide.BUY else -1
        self.positions[symbol] = position_sign * quantity
        self.open_trades[symbol] = trade
        self.capital -= commission
        self.total_commission += commission
        self.total_slippage += trade.slippage
        
        # Record equity
        self.equity_curve.append((timestamp, self._calculate_equity(price)))
        
        if self.config.log_trades:
            logger.info(
                f"[OPEN] {side.value} {symbol}: "
                f"qty={quantity:.4f}, price={fill_price:.2f}, "
                f"commission=${commission:.2f}"
            )
        
        return trade
    
    def _close_position(
        self,
        timestamp: datetime,
        symbol: str,
        price: float
    ) -> Optional[TradeRecord]:
        """Close an existing position."""
        if symbol not in self.open_trades:
            return None
        
        trade = self.open_trades[symbol]
        quantity = abs(self.positions.get(symbol, 0))
        
        if quantity < MIN_TRADE_SIZE:
            return None
        
        # Apply slippage (worse price)
        slippage = price * self.config.slippage_pct
        if trade.side == OrderSide.BUY:
            # Closing long = selling
            fill_price = price - slippage
        else:
            # Closing short = buying
            fill_price = price + slippage
        
        # Calculate commission
        commission = fill_price * quantity * self.config.commission_pct
        
        # Calculate P&L
        if trade.side == OrderSide.BUY:
            pnl = (fill_price - trade.entry_price) * quantity
        else:
            pnl = (trade.entry_price - fill_price) * quantity
        
        pnl -= commission + trade.commission  # Subtract total commissions
        pnl_pct = pnl / (trade.entry_price * quantity)
        
        # Update trade record
        trade.exit_price = fill_price
        trade.exit_timestamp = timestamp
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.commission += commission
        trade.slippage += slippage * quantity
        
        # Update state
        self.positions[symbol] = 0
        del self.open_trades[symbol]
        self.trades.append(trade)
        self.capital += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += slippage * quantity
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Record equity
        self.equity_curve.append((timestamp, self._calculate_equity(price)))
        
        if self.config.log_trades:
            status = "[WIN]" if pnl > 0 else "[LOSS]"
            logger.info(
                f"[CLOSE] {status} {symbol}: "
                f"pnl=${pnl:.2f} ({pnl_pct:.2%}), "
                f"holding={trade.holding_period:.1f}d"
            )
        
        return trade
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open positions."""
        equity = self.capital
        
        for symbol, qty in self.positions.items():
            if qty != 0 and symbol in self.open_trades:
                trade = self.open_trades[symbol]
                if trade.side == OrderSide.BUY:
                    unrealized = (current_price - trade.entry_price) * abs(qty)
                else:
                    unrealized = (trade.entry_price - current_price) * abs(qty)
                equity += unrealized
        
        return equity
    
    def get_metrics(self) -> Dict:
        """Get trading metrics."""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
            }
        
        # Calculate returns from equity curve
        if len(self.equity_curve) > 1:
            equities = [e[1] for e in self.equity_curve]
            returns = pd.Series(equities).pct_change().dropna()
            
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0 else 0.0
            )
            
            # Max drawdown
            peak = pd.Series(equities).expanding().max()
            drawdown = (pd.Series(equities) - peak) / peak
            max_dd = abs(drawdown.min())
        else:
            sharpe = 0.0
            max_dd = 0.0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl / self.config.initial_capital,
            'avg_pnl': self.total_pnl / self.total_trades,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_capital': self.capital,
        }
    
    def get_summary(self) -> str:
        """Get summary string."""
        metrics = self.get_metrics()
        
        return (
            f"\n{'='*50}\n"
            f"PAPER TRADING SUMMARY: {self.strategy_name}\n"
            f"{'='*50}\n"
            f"Total Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Total P&L: ${metrics['total_pnl']:,.2f} "
            f"({metrics.get('total_pnl_pct', 0):.2%})\n"
            f"Avg P&L per Trade: ${metrics['avg_pnl']:,.2f}\n"
            f"Total Commission: ${metrics['total_commission']:,.2f}\n"
            f"Total Slippage: ${metrics['total_slippage']:,.2f}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Final Capital: ${metrics['final_capital']:,.2f}\n"
            f"{'='*50}"
        )
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(columns=['timestamp', 'equity'])
        
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df = df.set_index('timestamp')
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for t in self.trades:
            records.append({
                'timestamp': t.timestamp,
                'exit_timestamp': t.exit_timestamp,
                'symbol': t.symbol,
                'side': t.side.value,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'commission': t.commission,
                'slippage': t.slippage,
                'holding_days': t.holding_period,
            })
        
        return pd.DataFrame(records)
    
    def reset(self) -> None:
        """Reset trader state."""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.open_trades = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        logger.info(f"PaperTrader reset: {self.strategy_name}")
