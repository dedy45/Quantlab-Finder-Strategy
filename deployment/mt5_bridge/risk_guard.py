"""
Risk Guard - Real-time risk limits and circuit breakers for MT5 trading.

Protects against:
- Excessive drawdown
- Over-leveraging
- Too many positions
- Daily loss limits
- Correlation risk
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .connector import MT5Connector
from .position_manager import PositionManager, PositionSummary

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level indicators."""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class RiskAction(Enum):
    """Actions to take when risk limit is breached."""
    NONE = "none"
    WARN = "warn"
    REDUCE = "reduce"  # Reduce position size
    CLOSE_PARTIAL = "close_partial"  # Close some positions
    CLOSE_ALL = "close_all"  # Close all positions
    HALT = "halt"  # Stop all trading


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    # Drawdown limits
    max_daily_drawdown_pct: float = 5.0  # 5% max daily loss
    max_total_drawdown_pct: float = 20.0  # 20% max total drawdown
    
    # Position limits
    max_positions: int = 5
    max_positions_per_symbol: int = 2
    max_total_volume: float = 1.0  # Total lots
    max_volume_per_symbol: float = 0.5
    
    # Margin limits
    min_margin_level: float = 150.0  # Minimum margin level %
    max_margin_used_pct: float = 50.0  # Max margin usage %
    
    # Correlation limits
    max_correlated_positions: int = 3  # Max positions in correlated pairs
    
    # Daily limits
    max_daily_trades: int = 20
    max_daily_loss: float = 500.0  # Absolute $ amount
    
    # Circuit breaker
    consecutive_losses_halt: int = 5  # Halt after N consecutive losses
    
    # Actions
    drawdown_action: RiskAction = RiskAction.CLOSE_ALL
    margin_action: RiskAction = RiskAction.WARN
    position_action: RiskAction = RiskAction.WARN
    
    # Callbacks
    on_risk_breach: Optional[Callable[[str, RiskLevel, str], None]] = None


@dataclass
class RiskStatus:
    """Current risk status."""
    
    level: RiskLevel = RiskLevel.SAFE
    
    # Metrics
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_drawdown_pct: float = 0.0
    margin_level: float = 0.0
    margin_used_pct: float = 0.0
    
    # Counts
    open_positions: int = 0
    daily_trades: int = 0
    consecutive_losses: int = 0
    
    # Flags
    can_trade: bool = True
    warnings: List[str] = field(default_factory=list)
    breaches: List[str] = field(default_factory=list)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level.value,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'total_drawdown_pct': self.total_drawdown_pct,
            'margin_level': self.margin_level,
            'margin_used_pct': self.margin_used_pct,
            'open_positions': self.open_positions,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'can_trade': self.can_trade,
            'warnings': self.warnings,
            'breaches': self.breaches,
            'timestamp': self.timestamp.isoformat(),
        }


class RiskGuard:
    """
    Real-time risk management and circuit breakers.
    
    Cara Penggunaan:
        connector = MT5Connector()
        connector.connect()
        
        guard = RiskGuard(connector)
        
        # Check before trading
        if guard.can_open_position('XAUUSD', 0.1):
            # Safe to trade
            executor.buy('XAUUSD', 0.1)
        
        # Get risk status
        status = guard.get_status()
        print(f"Risk Level: {status.level.value}")
        print(f"Daily P&L: ${status.daily_pnl:.2f}")
        
        # Monitor continuously
        guard.monitor()  # Call periodically
    
    Nilai:
    - Proteksi otomatis dari kerugian besar
    - Circuit breakers untuk kondisi ekstrem
    - Real-time risk monitoring
    
    Manfaat:
    - Mencegah blow-up account
    - Disiplin trading otomatis
    - Peace of mind saat trading
    """
    
    def __init__(
        self,
        connector: MT5Connector,
        config: Optional[RiskConfig] = None
    ):
        """Initialize risk guard."""
        self.connector = connector
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.position_manager = PositionManager(connector)
        
        # State tracking
        self._starting_balance: float = 0.0
        self._daily_starting_balance: float = 0.0
        self._current_date: Optional[date] = None
        self._daily_trades: int = 0
        self._consecutive_losses: int = 0
        self._is_halted: bool = False
        self._trade_history: List[Dict] = []
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize tracking variables."""
        if self.connector.is_connected:
            account = self.connector.refresh_account()
            if account:
                self._starting_balance = account.balance
                self._daily_starting_balance = account.balance
                self._current_date = date.today()
    
    def get_status(self) -> RiskStatus:
        """
        Get current risk status.
        
        Returns
        -------
        RiskStatus
            Current risk metrics and status
        """
        status = RiskStatus()
        
        if not self.connector.is_connected:
            status.can_trade = False
            status.breaches.append("MT5 not connected")
            return status
        
        # Refresh account
        account = self.connector.refresh_account()
        if account is None:
            status.can_trade = False
            status.breaches.append("Failed to get account info")
            return status
        
        # Check for new day
        today = date.today()
        if self._current_date != today:
            self._current_date = today
            self._daily_starting_balance = account.balance
            self._daily_trades = 0
        
        # Calculate metrics
        status.daily_pnl = account.equity - self._daily_starting_balance
        status.daily_pnl_pct = (status.daily_pnl / self._daily_starting_balance * 100
                                if self._daily_starting_balance > 0 else 0)
        
        status.total_drawdown_pct = ((self._starting_balance - account.equity) / 
                                      self._starting_balance * 100
                                      if self._starting_balance > 0 else 0)
        
        status.margin_level = account.margin_level
        status.margin_used_pct = (account.margin / account.equity * 100
                                   if account.equity > 0 else 0)
        
        # Get position info
        summary = self.position_manager.get_summary()
        status.open_positions = summary.total_positions
        status.daily_trades = self._daily_trades
        status.consecutive_losses = self._consecutive_losses
        
        # Check limits
        self._check_limits(status, account, summary)
        
        # Determine overall level
        if status.breaches:
            status.level = RiskLevel.CRITICAL
            status.can_trade = False
        elif len(status.warnings) >= 3:
            status.level = RiskLevel.DANGER
        elif status.warnings:
            status.level = RiskLevel.WARNING
        else:
            status.level = RiskLevel.SAFE
        
        # Check halt status
        if self._is_halted:
            status.can_trade = False
            status.breaches.append("Trading halted")
        
        return status
    
    def _check_limits(
        self,
        status: RiskStatus,
        account: Any,
        summary: PositionSummary
    ) -> None:
        """Check all risk limits."""
        # Daily drawdown
        if status.daily_pnl_pct <= -self.config.max_daily_drawdown_pct:
            status.breaches.append(
                f"Daily drawdown limit breached: {status.daily_pnl_pct:.1f}%"
            )
            self._trigger_action("daily_drawdown", self.config.drawdown_action)
        elif status.daily_pnl_pct <= -self.config.max_daily_drawdown_pct * 0.8:
            status.warnings.append(
                f"Approaching daily drawdown limit: {status.daily_pnl_pct:.1f}%"
            )
        
        # Total drawdown
        if status.total_drawdown_pct >= self.config.max_total_drawdown_pct:
            status.breaches.append(
                f"Total drawdown limit breached: {status.total_drawdown_pct:.1f}%"
            )
            self._trigger_action("total_drawdown", self.config.drawdown_action)
        elif status.total_drawdown_pct >= self.config.max_total_drawdown_pct * 0.8:
            status.warnings.append(
                f"Approaching total drawdown limit: {status.total_drawdown_pct:.1f}%"
            )
        
        # Margin level
        if status.margin_level > 0 and status.margin_level < self.config.min_margin_level:
            status.breaches.append(
                f"Margin level too low: {status.margin_level:.1f}%"
            )
            self._trigger_action("margin", self.config.margin_action)
        
        # Margin used
        if status.margin_used_pct > self.config.max_margin_used_pct:
            status.warnings.append(
                f"High margin usage: {status.margin_used_pct:.1f}%"
            )
        
        # Position count
        if summary.total_positions >= self.config.max_positions:
            status.warnings.append(
                f"Max positions reached: {summary.total_positions}"
            )
        
        # Total volume
        if summary.total_volume >= self.config.max_total_volume:
            status.warnings.append(
                f"Max volume reached: {summary.total_volume:.2f} lots"
            )
        
        # Daily trades
        if self._daily_trades >= self.config.max_daily_trades:
            status.warnings.append(
                f"Max daily trades reached: {self._daily_trades}"
            )
        
        # Daily loss (absolute)
        if status.daily_pnl <= -self.config.max_daily_loss:
            status.breaches.append(
                f"Daily loss limit breached: ${abs(status.daily_pnl):.2f}"
            )
        
        # Consecutive losses
        if self._consecutive_losses >= self.config.consecutive_losses_halt:
            status.breaches.append(
                f"Consecutive losses halt: {self._consecutive_losses} losses"
            )
            self._is_halted = True
    
    def _trigger_action(self, reason: str, action: RiskAction) -> None:
        """Trigger risk action."""
        self.logger.warning(f"Risk action triggered: {reason} -> {action.value}")
        
        if self.config.on_risk_breach:
            self.config.on_risk_breach(reason, RiskLevel.CRITICAL, action.value)
        
        if action == RiskAction.CLOSE_ALL:
            self._close_all_positions()
        elif action == RiskAction.HALT:
            self._is_halted = True
    
    def _close_all_positions(self) -> None:
        """Close all positions (emergency)."""
        self.logger.warning("EMERGENCY: Closing all positions")
        
        from .order_executor import OrderExecutor
        executor = OrderExecutor(self.connector)
        executor.close_all()
    
    def can_open_position(
        self,
        symbol: str,
        volume: float,
        check_correlation: bool = True
    ) -> bool:
        """
        Check if it's safe to open a new position.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        volume : float
            Position volume in lots
        check_correlation : bool
            Check correlation with existing positions
            
        Returns
        -------
        bool
            True if safe to open position
        """
        status = self.get_status()
        
        if not status.can_trade:
            self.logger.warning(f"Cannot trade: {status.breaches}")
            return False
        
        # Check position limits
        summary = self.position_manager.get_summary()
        
        if summary.total_positions >= self.config.max_positions:
            self.logger.warning("Max positions reached")
            return False
        
        if summary.total_volume + volume > self.config.max_total_volume:
            self.logger.warning("Would exceed max volume")
            return False
        
        # Check per-symbol limits
        symbol_info = summary.by_symbol.get(symbol, {})
        symbol_positions = symbol_info.get('positions', 0)
        symbol_volume = symbol_info.get('volume', 0.0)
        
        if symbol_positions >= self.config.max_positions_per_symbol:
            self.logger.warning(f"Max positions for {symbol} reached")
            return False
        
        if symbol_volume + volume > self.config.max_volume_per_symbol:
            self.logger.warning(f"Would exceed max volume for {symbol}")
            return False
        
        # Check daily trades
        if self._daily_trades >= self.config.max_daily_trades:
            self.logger.warning("Max daily trades reached")
            return False
        
        # Check correlation (simplified)
        if check_correlation:
            correlated = self._count_correlated_positions(symbol)
            if correlated >= self.config.max_correlated_positions:
                self.logger.warning(f"Too many correlated positions: {correlated}")
                return False
        
        return True
    
    def _count_correlated_positions(self, symbol: str) -> int:
        """Count positions in correlated pairs."""
        # Simplified correlation groups
        correlation_groups = {
            'USD': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
            'GOLD': ['XAUUSD', 'XAGUSD'],
            'JPY': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY'],
            'EUR': ['EURUSD', 'EURGBP', 'EURJPY', 'EURAUD'],
        }
        
        # Find which groups this symbol belongs to
        symbol_groups = []
        for group, symbols in correlation_groups.items():
            if symbol in symbols:
                symbol_groups.append(group)
        
        if not symbol_groups:
            return 0
        
        # Count positions in same groups
        positions = self.position_manager.get_positions()
        correlated_count = 0
        
        for pos in positions:
            for group in symbol_groups:
                if pos.symbol in correlation_groups.get(group, []):
                    correlated_count += 1
                    break
        
        return correlated_count
    
    def record_trade(self, profit: float) -> None:
        """
        Record a completed trade for tracking.
        
        Parameters
        ----------
        profit : float
            Trade profit/loss
        """
        self._daily_trades += 1
        
        if profit < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        self._trade_history.append({
            'timestamp': datetime.now(),
            'profit': profit,
        })
    
    def reset_halt(self) -> None:
        """Reset halt status (manual override)."""
        self._is_halted = False
        self._consecutive_losses = 0
        self.logger.info("Trading halt reset")
    
    def monitor(self) -> RiskStatus:
        """
        Perform monitoring check (call periodically).
        
        Returns
        -------
        RiskStatus
            Current risk status
        """
        status = self.get_status()
        
        # Log status
        if status.level == RiskLevel.CRITICAL:
            self.logger.error(f"CRITICAL: {status.breaches}")
        elif status.level == RiskLevel.DANGER:
            self.logger.warning(f"DANGER: {status.warnings}")
        elif status.level == RiskLevel.WARNING:
            self.logger.info(f"WARNING: {status.warnings}")
        
        return status
