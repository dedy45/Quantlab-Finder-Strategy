"""
Order Executor - Execute orders on MT5 with proper risk management.

Handles:
- Market orders (buy/sell)
- Pending orders (limit/stop)
- Order modification
- Position closing
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .connector import MT5Connector, SymbolInfo

logger = logging.getLogger(__name__)


class OrderAction(Enum):
    """Order actions."""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"


@dataclass
class OrderConfig:
    """Configuration for order execution."""
    
    # Default values
    default_lot: float = 0.01
    max_lot: float = 1.0
    
    # Slippage
    max_deviation: int = 20  # Points
    
    # Risk management
    default_sl_pips: float = 50.0
    default_tp_pips: float = 100.0
    use_default_sl: bool = True
    use_default_tp: bool = True
    
    # Execution
    magic_number: int = 123456
    retry_count: int = 3
    retry_delay_ms: int = 500
    
    # Validation
    check_margin: bool = True
    min_margin_level: float = 150.0  # Minimum margin level %


@dataclass
class OrderResult:
    """Result of order execution."""
    
    success: bool = False
    order_id: int = 0
    ticket: int = 0
    
    # Execution details
    symbol: str = ""
    action: str = ""
    lot: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    
    # Error info
    error_code: int = 0
    error_message: str = ""
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'order_id': self.order_id,
            'ticket': self.ticket,
            'symbol': self.symbol,
            'action': self.action,
            'lot': self.lot,
            'price': self.price,
            'sl': self.sl,
            'tp': self.tp,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
            'execution_time_ms': self.execution_time_ms,
        }


class OrderExecutor:
    """
    Execute orders on MT5 with risk management.
    
    Cara Penggunaan:
        connector = MT5Connector()
        connector.connect()
        
        executor = OrderExecutor(connector)
        
        # Simple buy
        result = executor.buy('XAUUSD', lot=0.1)
        
        # Buy with SL/TP
        result = executor.buy('XAUUSD', lot=0.1, sl_pips=50, tp_pips=100)
        
        # Sell
        result = executor.sell('EURUSD', lot=0.05, sl_pips=30, tp_pips=60)
        
        # Close position
        executor.close_position(ticket=12345)
        
        # Close all positions for symbol
        executor.close_all('XAUUSD')
    
    Nilai:
    - Safe order execution dengan validation
    - Automatic SL/TP calculation
    - Retry logic untuk reliability
    
    Manfaat:
    - Mengurangi human error
    - Consistent risk management
    - Audit trail untuk semua orders
    """
    
    def __init__(
        self,
        connector: MT5Connector,
        config: Optional[OrderConfig] = None
    ):
        """Initialize order executor."""
        self.connector = connector
        self.config = config or OrderConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._order_history: List[OrderResult] = []
    
    def buy(
        self,
        symbol: str,
        lot: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        comment: str = ""
    ) -> Optional[OrderResult]:
        """
        Execute market buy order.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        lot : float, optional
            Position size (uses default if None)
        sl_pips : float, optional
            Stop loss in pips
        tp_pips : float, optional
            Take profit in pips
        sl_price : float, optional
            Stop loss price (overrides sl_pips)
        tp_price : float, optional
            Take profit price (overrides tp_pips)
        comment : str
            Order comment
            
        Returns
        -------
        OrderResult or None
            Execution result
        """
        return self._execute_market_order(
            symbol=symbol,
            action=OrderAction.BUY,
            lot=lot,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            sl_price=sl_price,
            tp_price=tp_price,
            comment=comment
        )
    
    def sell(
        self,
        symbol: str,
        lot: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        comment: str = ""
    ) -> Optional[OrderResult]:
        """
        Execute market sell order.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        lot : float, optional
            Position size (uses default if None)
        sl_pips : float, optional
            Stop loss in pips
        tp_pips : float, optional
            Take profit in pips
        sl_price : float, optional
            Stop loss price (overrides sl_pips)
        tp_price : float, optional
            Take profit price (overrides tp_pips)
        comment : str
            Order comment
            
        Returns
        -------
        OrderResult or None
            Execution result
        """
        return self._execute_market_order(
            symbol=symbol,
            action=OrderAction.SELL,
            lot=lot,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            sl_price=sl_price,
            tp_price=tp_price,
            comment=comment
        )
    
    def _execute_market_order(
        self,
        symbol: str,
        action: OrderAction,
        lot: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        comment: str = ""
    ) -> Optional[OrderResult]:
        """Execute market order with retry logic."""
        if not self.connector.is_connected:
            self.logger.error("MT5 not connected")
            return None
        
        start_time = time.time()
        
        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        if not symbol_info.is_tradeable:
            self.logger.error(f"Symbol {symbol} is not tradeable")
            return None
        
        # Validate and normalize lot size
        lot = lot or self.config.default_lot
        lot = self._normalize_lot(lot, symbol_info)
        
        if lot is None:
            self.logger.error("Invalid lot size")
            return None
        
        # Check margin
        if self.config.check_margin:
            if not self._check_margin(symbol, lot, action):
                self.logger.error("Insufficient margin")
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    action=action.value,
                    error_message="Insufficient margin"
                )
        
        # Get current price
        tick = self.connector.get_tick(symbol)
        if tick is None:
            self.logger.error(f"Failed to get tick for {symbol}")
            return None
        
        if action == OrderAction.BUY:
            price = tick['ask']
            order_type = 0  # ORDER_TYPE_BUY
        else:
            price = tick['bid']
            order_type = 1  # ORDER_TYPE_SELL
        
        # Calculate SL/TP
        sl, tp = self._calculate_sl_tp(
            symbol_info=symbol_info,
            action=action,
            price=price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            sl_price=sl_price,
            tp_price=tp_price
        )
        
        # Prepare request
        mt5 = self.connector._mt5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config.max_deviation,
            "magic": self.config.magic_number,
            "comment": comment or f"QuantLab {action.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Execute with retry
        result = None
        for attempt in range(self.config.retry_count):
            try:
                result = mt5.order_send(request)
                
                if result is None:
                    error = mt5.last_error()
                    self.logger.warning(f"Order failed (attempt {attempt+1}): {error}")
                    time.sleep(self.config.retry_delay_ms / 1000)
                    continue
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Success
                    execution_time = (time.time() - start_time) * 1000
                    
                    order_result = OrderResult(
                        success=True,
                        order_id=result.order,
                        ticket=result.deal,
                        symbol=symbol,
                        action=action.value,
                        lot=lot,
                        price=result.price,
                        sl=sl,
                        tp=tp,
                        execution_time_ms=execution_time
                    )
                    
                    self._order_history.append(order_result)
                    self.logger.info(
                        f"Order executed: {action.value} {lot} {symbol} @ {result.price}"
                    )
                    
                    return order_result
                else:
                    # Error
                    self.logger.warning(
                        f"Order rejected (attempt {attempt+1}): "
                        f"{result.retcode} - {result.comment}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Order execution error: {e}")
            
            time.sleep(self.config.retry_delay_ms / 1000)
        
        # All retries failed
        error_msg = result.comment if result else "Unknown error"
        error_code = result.retcode if result else 0
        
        return OrderResult(
            success=False,
            symbol=symbol,
            action=action.value,
            lot=lot,
            error_code=error_code,
            error_message=error_msg,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def close_position(self, ticket: int) -> Optional[OrderResult]:
        """
        Close specific position by ticket.
        
        Parameters
        ----------
        ticket : int
            Position ticket number
            
        Returns
        -------
        OrderResult or None
            Execution result
        """
        if not self.connector.is_connected:
            return None
        
        mt5 = self.connector._mt5
        
        # Get position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            self.logger.warning(f"Position {ticket} not found")
            return None
        
        position = position[0]
        
        # Determine close action
        if position.type == 0:  # Buy position
            order_type = 1  # Sell to close
            price = mt5.symbol_info_tick(position.symbol).bid
        else:  # Sell position
            order_type = 0  # Buy to close
            price = mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.config.max_deviation,
            "magic": self.config.magic_number,
            "comment": "QuantLab close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Position {ticket} closed")
            return OrderResult(
                success=True,
                ticket=ticket,
                symbol=position.symbol,
                action="close",
                lot=position.volume,
                price=result.price
            )
        else:
            error_msg = result.comment if result else "Unknown error"
            self.logger.error(f"Failed to close position {ticket}: {error_msg}")
            return OrderResult(
                success=False,
                ticket=ticket,
                error_message=error_msg
            )
    
    def close_all(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Close all positions (optionally filtered by symbol).
        
        Parameters
        ----------
        symbol : str, optional
            Filter by symbol
            
        Returns
        -------
        List[OrderResult]
            Results for each closed position
        """
        if not self.connector.is_connected:
            return []
        
        mt5 = self.connector._mt5
        
        # Get positions
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if not positions:
            self.logger.info("No positions to close")
            return []
        
        results = []
        for position in positions:
            result = self.close_position(position.ticket)
            if result:
                results.append(result)
        
        return results
    
    def _normalize_lot(
        self,
        lot: float,
        symbol_info: SymbolInfo
    ) -> Optional[float]:
        """Normalize lot size to valid value."""
        # Clamp to min/max
        lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
        lot = min(lot, self.config.max_lot)
        
        # Round to step
        if symbol_info.volume_step > 0:
            lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
        
        # Final validation
        if lot < symbol_info.volume_min:
            return None
        
        return round(lot, 2)
    
    def _calculate_sl_tp(
        self,
        symbol_info: SymbolInfo,
        action: OrderAction,
        price: float,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None
    ) -> Tuple[float, float]:
        """Calculate SL and TP prices."""
        # Use price if provided
        if sl_price is not None:
            sl = sl_price
        elif sl_pips is not None:
            sl_distance = sl_pips * symbol_info.point * 10  # Convert pips to price
            if action == OrderAction.BUY:
                sl = price - sl_distance
            else:
                sl = price + sl_distance
        elif self.config.use_default_sl:
            sl_distance = self.config.default_sl_pips * symbol_info.point * 10
            if action == OrderAction.BUY:
                sl = price - sl_distance
            else:
                sl = price + sl_distance
        else:
            sl = 0.0
        
        if tp_price is not None:
            tp = tp_price
        elif tp_pips is not None:
            tp_distance = tp_pips * symbol_info.point * 10
            if action == OrderAction.BUY:
                tp = price + tp_distance
            else:
                tp = price - tp_distance
        elif self.config.use_default_tp:
            tp_distance = self.config.default_tp_pips * symbol_info.point * 10
            if action == OrderAction.BUY:
                tp = price + tp_distance
            else:
                tp = price - tp_distance
        else:
            tp = 0.0
        
        # Round to symbol digits
        sl = round(sl, symbol_info.digits)
        tp = round(tp, symbol_info.digits)
        
        return sl, tp
    
    def _check_margin(
        self,
        symbol: str,
        lot: float,
        action: OrderAction
    ) -> bool:
        """Check if there's sufficient margin."""
        account = self.connector.refresh_account()
        if account is None:
            return False
        
        # Check margin level
        if account.margin_level > 0 and account.margin_level < self.config.min_margin_level:
            self.logger.warning(
                f"Margin level too low: {account.margin_level:.1f}% "
                f"(min: {self.config.min_margin_level}%)"
            )
            return False
        
        # Check free margin (simplified check)
        # In production, use mt5.order_calc_margin() for accurate calculation
        if account.free_margin < account.balance * 0.1:  # Less than 10% free
            self.logger.warning("Free margin too low")
            return False
        
        return True
    
    def get_order_history(self, limit: int = 100) -> List[OrderResult]:
        """Get order execution history."""
        return self._order_history[-limit:]
