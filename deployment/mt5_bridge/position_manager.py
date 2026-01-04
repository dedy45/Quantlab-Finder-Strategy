"""
Position Manager - Track and manage open positions on MT5.

Handles:
- Position tracking
- P&L monitoring
- Position modification (SL/TP adjustment)
- Trailing stops
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .connector import MT5Connector

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about an open position."""
    
    ticket: int = 0
    symbol: str = ""
    type: str = ""  # "buy" or "sell"
    volume: float = 0.0
    open_price: float = 0.0
    current_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    
    # P&L
    profit: float = 0.0
    profit_pips: float = 0.0
    
    # Timing
    open_time: Optional[datetime] = None
    duration_hours: float = 0.0
    
    # Metadata
    magic: int = 0
    comment: str = ""
    
    @property
    def is_long(self) -> bool:
        return self.type == "buy"
    
    @property
    def is_short(self) -> bool:
        return self.type == "sell"
    
    @property
    def is_profitable(self) -> bool:
        return self.profit > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ticket': self.ticket,
            'symbol': self.symbol,
            'type': self.type,
            'volume': self.volume,
            'open_price': self.open_price,
            'current_price': self.current_price,
            'sl': self.sl,
            'tp': self.tp,
            'profit': self.profit,
            'profit_pips': self.profit_pips,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'duration_hours': self.duration_hours,
            'magic': self.magic,
            'comment': self.comment,
        }


@dataclass
class PositionSummary:
    """Summary of all positions."""
    
    total_positions: int = 0
    total_volume: float = 0.0
    total_profit: float = 0.0
    
    long_positions: int = 0
    long_volume: float = 0.0
    long_profit: float = 0.0
    
    short_positions: int = 0
    short_volume: float = 0.0
    short_profit: float = 0.0
    
    by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_positions': self.total_positions,
            'total_volume': self.total_volume,
            'total_profit': self.total_profit,
            'long_positions': self.long_positions,
            'long_volume': self.long_volume,
            'long_profit': self.long_profit,
            'short_positions': self.short_positions,
            'short_volume': self.short_volume,
            'short_profit': self.short_profit,
            'by_symbol': self.by_symbol,
        }


class PositionManager:
    """
    Manage and monitor open positions on MT5.
    
    Cara Penggunaan:
        connector = MT5Connector()
        connector.connect()
        
        manager = PositionManager(connector)
        
        # Get all positions
        positions = manager.get_positions()
        
        # Get positions for symbol
        gold_positions = manager.get_positions('XAUUSD')
        
        # Get summary
        summary = manager.get_summary()
        print(f"Total P&L: ${summary.total_profit:.2f}")
        
        # Modify SL/TP
        manager.modify_sl_tp(ticket=12345, sl=1900.0, tp=2000.0)
        
        # Set trailing stop
        manager.set_trailing_stop(ticket=12345, trail_pips=30)
    
    Nilai:
    - Real-time position monitoring
    - Easy SL/TP modification
    - Trailing stop support
    
    Manfaat:
    - Centralized position management
    - Risk monitoring
    - Automated position adjustments
    """
    
    def __init__(self, connector: MT5Connector, magic_number: int = 123456):
        """Initialize position manager."""
        self.connector = connector
        self.magic_number = magic_number
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._positions_cache: Dict[int, PositionInfo] = {}
        self._last_update: Optional[datetime] = None
    
    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic_only: bool = True
    ) -> List[PositionInfo]:
        """
        Get open positions.
        
        Parameters
        ----------
        symbol : str, optional
            Filter by symbol
        magic_only : bool
            Only return positions with our magic number
            
        Returns
        -------
        List[PositionInfo]
            List of open positions
        """
        if not self.connector.is_connected:
            return []
        
        mt5 = self.connector._mt5
        
        # Get positions
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        result = []
        now = datetime.now()
        
        for pos in positions:
            # Filter by magic number
            if magic_only and pos.magic != self.magic_number:
                continue
            
            # Get symbol info for pip calculation
            symbol_info = self.connector.get_symbol_info(pos.symbol)
            point = symbol_info.point if symbol_info else 0.00001
            
            # Calculate profit in pips
            if pos.type == 0:  # Buy
                profit_pips = (pos.price_current - pos.price_open) / point / 10
                pos_type = "buy"
            else:  # Sell
                profit_pips = (pos.price_open - pos.price_current) / point / 10
                pos_type = "sell"
            
            # Calculate duration
            open_time = datetime.fromtimestamp(pos.time)
            duration = (now - open_time).total_seconds() / 3600
            
            position_info = PositionInfo(
                ticket=pos.ticket,
                symbol=pos.symbol,
                type=pos_type,
                volume=pos.volume,
                open_price=pos.price_open,
                current_price=pos.price_current,
                sl=pos.sl,
                tp=pos.tp,
                profit=pos.profit,
                profit_pips=profit_pips,
                open_time=open_time,
                duration_hours=duration,
                magic=pos.magic,
                comment=pos.comment,
            )
            
            result.append(position_info)
            self._positions_cache[pos.ticket] = position_info
        
        self._last_update = now
        return result
    
    def get_position(self, ticket: int) -> Optional[PositionInfo]:
        """
        Get specific position by ticket.
        
        Parameters
        ----------
        ticket : int
            Position ticket
            
        Returns
        -------
        PositionInfo or None
            Position information
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket == ticket:
                return pos
        return None
    
    def get_summary(self, symbol: Optional[str] = None) -> PositionSummary:
        """
        Get summary of all positions.
        
        Parameters
        ----------
        symbol : str, optional
            Filter by symbol
            
        Returns
        -------
        PositionSummary
            Summary statistics
        """
        positions = self.get_positions(symbol)
        
        summary = PositionSummary()
        by_symbol: Dict[str, Dict[str, Any]] = {}
        
        for pos in positions:
            summary.total_positions += 1
            summary.total_volume += pos.volume
            summary.total_profit += pos.profit
            
            if pos.is_long:
                summary.long_positions += 1
                summary.long_volume += pos.volume
                summary.long_profit += pos.profit
            else:
                summary.short_positions += 1
                summary.short_volume += pos.volume
                summary.short_profit += pos.profit
            
            # By symbol
            if pos.symbol not in by_symbol:
                by_symbol[pos.symbol] = {
                    'positions': 0,
                    'volume': 0.0,
                    'profit': 0.0,
                }
            by_symbol[pos.symbol]['positions'] += 1
            by_symbol[pos.symbol]['volume'] += pos.volume
            by_symbol[pos.symbol]['profit'] += pos.profit
        
        summary.by_symbol = by_symbol
        return summary
    
    def modify_sl_tp(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> bool:
        """
        Modify SL/TP for a position.
        
        Parameters
        ----------
        ticket : int
            Position ticket
        sl : float, optional
            New stop loss price
        tp : float, optional
            New take profit price
            
        Returns
        -------
        bool
            True if successful
        """
        if not self.connector.is_connected:
            return False
        
        mt5 = self.connector._mt5
        
        # Get current position
        position = self.get_position(ticket)
        if position is None:
            self.logger.error(f"Position {ticket} not found")
            return False
        
        # Use current values if not specified
        new_sl = sl if sl is not None else position.sl
        new_tp = tp if tp is not None else position.tp
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Modified position {ticket}: SL={new_sl}, TP={new_tp}")
            return True
        else:
            error = result.comment if result else "Unknown error"
            self.logger.error(f"Failed to modify position {ticket}: {error}")
            return False
    
    def set_trailing_stop(
        self,
        ticket: int,
        trail_pips: float,
        step_pips: float = 5.0
    ) -> bool:
        """
        Set trailing stop for a position.
        
        Parameters
        ----------
        ticket : int
            Position ticket
        trail_pips : float
            Trailing distance in pips
        step_pips : float
            Minimum step to move SL
            
        Returns
        -------
        bool
            True if SL was updated
        """
        position = self.get_position(ticket)
        if position is None:
            return False
        
        symbol_info = self.connector.get_symbol_info(position.symbol)
        if symbol_info is None:
            return False
        
        point = symbol_info.point
        trail_distance = trail_pips * point * 10
        step_distance = step_pips * point * 10
        
        # Calculate new SL
        if position.is_long:
            new_sl = position.current_price - trail_distance
            # Only move SL up
            if position.sl > 0 and new_sl <= position.sl + step_distance:
                return False
        else:
            new_sl = position.current_price + trail_distance
            # Only move SL down
            if position.sl > 0 and new_sl >= position.sl - step_distance:
                return False
        
        new_sl = round(new_sl, symbol_info.digits)
        return self.modify_sl_tp(ticket, sl=new_sl)
    
    def move_to_breakeven(
        self,
        ticket: int,
        min_profit_pips: float = 20.0,
        offset_pips: float = 2.0
    ) -> bool:
        """
        Move SL to breakeven when in profit.
        
        Parameters
        ----------
        ticket : int
            Position ticket
        min_profit_pips : float
            Minimum profit in pips before moving to BE
        offset_pips : float
            Offset from entry price (to cover spread)
            
        Returns
        -------
        bool
            True if SL was moved to breakeven
        """
        position = self.get_position(ticket)
        if position is None:
            return False
        
        # Check if already at breakeven or better
        if position.is_long and position.sl >= position.open_price:
            return False
        if position.is_short and position.sl > 0 and position.sl <= position.open_price:
            return False
        
        # Check minimum profit
        if position.profit_pips < min_profit_pips:
            return False
        
        symbol_info = self.connector.get_symbol_info(position.symbol)
        if symbol_info is None:
            return False
        
        offset = offset_pips * symbol_info.point * 10
        
        if position.is_long:
            new_sl = position.open_price + offset
        else:
            new_sl = position.open_price - offset
        
        new_sl = round(new_sl, symbol_info.digits)
        
        self.logger.info(f"Moving position {ticket} to breakeven: SL={new_sl}")
        return self.modify_sl_tp(ticket, sl=new_sl)
    
    def to_dataframe(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get positions as DataFrame.
        
        Parameters
        ----------
        symbol : str, optional
            Filter by symbol
            
        Returns
        -------
        pd.DataFrame
            Positions data
        """
        positions = self.get_positions(symbol)
        
        if not positions:
            return pd.DataFrame()
        
        return pd.DataFrame([p.to_dict() for p in positions])
