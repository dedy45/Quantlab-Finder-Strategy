"""
Trade Logger - Records all trades for analysis and audit.

Provides:
- Trade logging with timestamps
- Export to CSV/JSON
- Trade statistics
- Audit trail

Version: 0.6.3
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Constants
DEFAULT_LOG_DIR = "output/paper_trading"


@dataclass
class TradeLogEntry:
    """Single trade log entry."""
    trade_id: int
    timestamp: datetime
    symbol: str
    action: str  # OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT
    quantity: float
    price: float
    value: float
    commission: float
    slippage: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    cumulative_pnl: float = 0.0
    capital_after: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class TradeLogger:
    """
    Trade Logger for paper trading.
    
    Records all trades with full details for:
    - Performance analysis
    - Audit trail
    - Regulatory compliance
    - Strategy debugging
    
    Parameters
    ----------
    strategy_name : str
        Name of strategy being logged
    log_dir : str, optional
        Directory for log files
    auto_save : bool, default=True
        Auto-save after each trade
        
    Examples
    --------
    >>> logger = TradeLogger("MomentumGold")
    >>> logger.log_trade(
    ...     timestamp=datetime.now(),
    ...     symbol="XAUUSD",
    ...     action="OPEN_LONG",
    ...     quantity=1.0,
    ...     price=2650.0,
    ...     commission=2.65,
    ...     slippage=2.65
    ... )
    >>> logger.export_csv()
    """
    
    def __init__(
        self,
        strategy_name: str,
        log_dir: Optional[str] = None,
        auto_save: bool = True
    ):
        """Initialize trade logger."""
        assert strategy_name, "Strategy name cannot be empty"
        
        self.strategy_name = strategy_name
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.auto_save = auto_save
        
        # State
        self._entries: List[TradeLogEntry] = []
        self._trade_counter = 0
        self._cumulative_pnl = 0.0
        self._session_start = datetime.now()
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"TradeLogger initialized: {strategy_name}, "
            f"log_dir={self.log_dir}"
        )
    
    def log_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        pnl: Optional[float] = None,
        capital_after: float = 0.0,
        notes: str = ""
    ) -> TradeLogEntry:
        """
        Log a trade.
        
        Parameters
        ----------
        timestamp : datetime
            Trade timestamp
        symbol : str
            Trading symbol
        action : str
            Trade action (OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT)
        quantity : float
            Trade quantity
        price : float
            Execution price
        commission : float
            Commission paid
        slippage : float
            Slippage cost
        pnl : float, optional
            Realized P&L (for closing trades)
        capital_after : float
            Capital after trade
        notes : str
            Additional notes
            
        Returns
        -------
        TradeLogEntry
            Created log entry
        """
        assert timestamp is not None, "Timestamp cannot be None"
        assert symbol, "Symbol cannot be empty"
        assert action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"], \
            f"Invalid action: {action}"
        assert quantity > 0, "Quantity must be positive"
        assert price > 0, "Price must be positive"
        
        self._trade_counter += 1
        
        # Update cumulative P&L
        if pnl is not None:
            self._cumulative_pnl += pnl
        
        # Calculate P&L percentage
        pnl_pct = None
        if pnl is not None and quantity > 0 and price > 0:
            trade_value = quantity * price
            pnl_pct = pnl / trade_value if trade_value > 0 else 0.0
        
        entry = TradeLogEntry(
            trade_id=self._trade_counter,
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            value=quantity * price,
            commission=commission,
            slippage=slippage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            cumulative_pnl=self._cumulative_pnl,
            capital_after=capital_after,
            notes=notes
        )
        
        self._entries.append(entry)
        
        pnl_str = f"${pnl:.2f}" if pnl is not None else "N/A"
        logger.info(
            f"[TRADE #{self._trade_counter}] {action} {symbol}: "
            f"qty={quantity:.4f}, price={price:.2f}, "
            f"pnl={pnl_str}"
        )
        
        if self.auto_save:
            self._auto_save()
        
        return entry
    
    def _auto_save(self) -> None:
        """Auto-save to JSON."""
        try:
            filepath = self.log_dir / f"{self.strategy_name}_trades.json"
            self.export_json(str(filepath))
        except Exception as e:
            logger.warning(f"Auto-save failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trade statistics.
        
        Returns
        -------
        Dict[str, Any]
            Trade statistics
        """
        if not self._entries:
            return {
                'total_trades': 0,
                'open_trades': 0,
                'close_trades': 0,
                'total_pnl': 0.0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
            }
        
        # Count trades by type
        open_trades = sum(1 for e in self._entries if e.action.startswith("OPEN"))
        close_trades = sum(1 for e in self._entries if e.action.startswith("CLOSE"))
        
        # P&L statistics
        pnls = [e.pnl for e in self._entries if e.pnl is not None]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]
        
        total_commission = sum(e.commission for e in self._entries)
        total_slippage = sum(e.slippage for e in self._entries)
        
        return {
            'total_trades': len(self._entries),
            'open_trades': open_trades,
            'close_trades': close_trades,
            'total_pnl': self._cumulative_pnl,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(pnls) if pnls else 0.0,
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0.0,
            'avg_win': sum(winning) / len(winning) if winning else 0.0,
            'avg_loss': sum(losing) / len(losing) if losing else 0.0,
        }
    
    def get_entries(self) -> List[TradeLogEntry]:
        """Get all log entries."""
        return self._entries.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Trade log as DataFrame
        """
        if not self._entries:
            return pd.DataFrame()
        
        records = [e.to_dict() for e in self._entries]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def export_csv(self, filepath: Optional[str] = None) -> str:
        """
        Export to CSV file.
        
        Parameters
        ----------
        filepath : str, optional
            Output file path
            
        Returns
        -------
        str
            Path to exported file
        """
        if filepath is None:
            timestamp = self._session_start.strftime("%Y%m%d_%H%M%S")
            filepath = str(self.log_dir / f"{self.strategy_name}_{timestamp}.csv")
        
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} trades to {filepath}")
        
        return filepath
    
    def export_json(self, filepath: Optional[str] = None) -> str:
        """
        Export to JSON file.
        
        Parameters
        ----------
        filepath : str, optional
            Output file path
            
        Returns
        -------
        str
            Path to exported file
        """
        if filepath is None:
            timestamp = self._session_start.strftime("%Y%m%d_%H%M%S")
            filepath = str(self.log_dir / f"{self.strategy_name}_{timestamp}.json")
        
        data = {
            'strategy_name': self.strategy_name,
            'session_start': self._session_start.isoformat(),
            'export_time': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'trades': [e.to_dict() for e in self._entries]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self._entries)} trades to {filepath}")
        return filepath
    
    def get_summary(self) -> str:
        """Get summary string."""
        stats = self.get_statistics()
        
        return (
            f"\n{'='*50}\n"
            f"TRADE LOG SUMMARY: {self.strategy_name}\n"
            f"{'='*50}\n"
            f"Total Trades: {stats['total_trades']}\n"
            f"  - Opens: {stats['open_trades']}\n"
            f"  - Closes: {stats['close_trades']}\n"
            f"Total P&L: ${stats['total_pnl']:,.2f}\n"
            f"Win Rate: {stats['win_rate']:.2%}\n"
            f"  - Winners: {stats['winning_trades']} (avg ${stats['avg_win']:,.2f})\n"
            f"  - Losers: {stats['losing_trades']} (avg ${stats['avg_loss']:,.2f})\n"
            f"Total Commission: ${stats['total_commission']:,.2f}\n"
            f"Total Slippage: ${stats['total_slippage']:,.2f}\n"
            f"{'='*50}"
        )
    
    def clear(self) -> None:
        """Clear all entries."""
        self._entries = []
        self._trade_counter = 0
        self._cumulative_pnl = 0.0
        self._session_start = datetime.now()
        
        logger.info(f"TradeLogger cleared: {self.strategy_name}")
