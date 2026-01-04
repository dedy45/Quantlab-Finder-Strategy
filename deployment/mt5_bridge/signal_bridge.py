"""
Signal Bridge - Transfer signals from QuantLab to MT5.

Converts QuantLab strategy signals to MT5 trading actions.
Supports multiple signal delivery methods:
- Direct MT5 API calls
- File-based communication
- Socket-based communication (for EA integration)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import pandas as pd
import numpy as np

from .connector import MT5Connector

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types."""
    LONG = 1
    SHORT = -1
    FLAT = 0
    CLOSE_LONG = 2
    CLOSE_SHORT = -2


class DeliveryMethod(Enum):
    """Signal delivery methods."""
    DIRECT = "direct"  # Direct MT5 API
    FILE = "file"  # File-based (for EA)
    SOCKET = "socket"  # Socket-based


@dataclass
class Signal:
    """Trading signal from QuantLab."""
    
    # Identification
    id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Signal details
    symbol: str = ""
    signal_type: SignalType = SignalType.FLAT
    
    # Position sizing
    lot_size: Optional[float] = None  # If None, use default
    
    # Risk management
    stop_loss: Optional[float] = None  # Price or pips
    take_profit: Optional[float] = None  # Price or pips
    sl_pips: Optional[float] = None  # SL in pips
    tp_pips: Optional[float] = None  # TP in pips
    
    # Metadata
    strategy_name: str = ""
    confidence: float = 1.0  # 0-1, for position sizing
    comment: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'lot_size': self.lot_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'strategy_name': self.strategy_name,
            'confidence': self.confidence,
            'comment': self.comment,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create from dictionary."""
        return cls(
            id=data.get('id', ''),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            symbol=data.get('symbol', ''),
            signal_type=SignalType(data.get('signal_type', 0)),
            lot_size=data.get('lot_size'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            sl_pips=data.get('sl_pips'),
            tp_pips=data.get('tp_pips'),
            strategy_name=data.get('strategy_name', ''),
            confidence=data.get('confidence', 1.0),
            comment=data.get('comment', ''),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class SignalBridgeConfig:
    """Configuration for signal bridge."""
    
    # Delivery
    method: DeliveryMethod = DeliveryMethod.DIRECT
    
    # File-based settings
    signal_file_path: str = "signals/quantlab_signal.json"
    signal_history_path: str = "signals/history/"
    
    # Socket settings (for future use)
    socket_host: str = "localhost"
    socket_port: int = 5555
    
    # Processing
    signal_timeout_sec: int = 60  # Signal expires after this
    confirm_execution: bool = True
    
    # Callbacks
    on_signal_sent: Optional[Callable[[Signal], None]] = None
    on_signal_executed: Optional[Callable[[Signal, Dict], None]] = None
    on_signal_failed: Optional[Callable[[Signal, str], None]] = None


class SignalBridge:
    """
    Bridge for transferring signals from QuantLab to MT5.
    
    Cara Penggunaan:
        connector = MT5Connector()
        connector.connect()
        
        bridge = SignalBridge(connector)
        
        # Send simple signal
        bridge.send_signal('XAUUSD', SignalType.LONG, lot_size=0.1)
        
        # Send signal with SL/TP
        signal = Signal(
            symbol='XAUUSD',
            signal_type=SignalType.LONG,
            lot_size=0.1,
            sl_pips=50,
            tp_pips=100,
            strategy_name='MomentumGold'
        )
        bridge.send(signal)
        
        # Convert QuantLab signals to MT5
        bridge.process_strategy_signals(prices, signals, 'XAUUSD')
    
    Nilai:
    - Abstraksi signal delivery
    - Multiple delivery methods
    - Signal history tracking
    
    Manfaat:
    - Seamless integration dengan QuantLab strategies
    - Flexible deployment options
    - Audit trail untuk signals
    """
    
    def __init__(
        self,
        connector: MT5Connector,
        config: Optional[SignalBridgeConfig] = None
    ):
        """Initialize signal bridge."""
        self.connector = connector
        self.config = config or SignalBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._signal_counter = 0
        self._signal_history: List[Signal] = []
        
        # Create signal directories if using file method
        if self.config.method == DeliveryMethod.FILE:
            Path(self.config.signal_file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.signal_history_path).mkdir(parents=True, exist_ok=True)
    
    def send_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        lot_size: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        strategy_name: str = "",
        confidence: float = 1.0
    ) -> Optional[Signal]:
        """
        Send a trading signal.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        signal_type : SignalType
            Signal type (LONG, SHORT, FLAT)
        lot_size : float, optional
            Position size in lots
        sl_pips : float, optional
            Stop loss in pips
        tp_pips : float, optional
            Take profit in pips
        strategy_name : str
            Strategy identifier
        confidence : float
            Signal confidence (0-1)
            
        Returns
        -------
        Signal or None
            Sent signal if successful
        """
        # Create signal
        self._signal_counter += 1
        signal = Signal(
            id=f"SIG_{self._signal_counter:06d}",
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            lot_size=lot_size,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            strategy_name=strategy_name,
            confidence=confidence,
            comment=f"QuantLab {strategy_name}"
        )
        
        return self.send(signal)
    
    def send(self, signal: Signal) -> Optional[Signal]:
        """
        Send a signal object.
        
        Parameters
        ----------
        signal : Signal
            Signal to send
            
        Returns
        -------
        Signal or None
            Sent signal if successful
        """
        try:
            if self.config.method == DeliveryMethod.DIRECT:
                success = self._send_direct(signal)
            elif self.config.method == DeliveryMethod.FILE:
                success = self._send_file(signal)
            elif self.config.method == DeliveryMethod.SOCKET:
                success = self._send_socket(signal)
            else:
                self.logger.error(f"Unknown delivery method: {self.config.method}")
                return None
            
            if success:
                self._signal_history.append(signal)
                self.logger.info(f"Signal sent: {signal.id} {signal.symbol} {signal.signal_type.name}")
                
                if self.config.on_signal_sent:
                    self.config.on_signal_sent(signal)
                
                return signal
            else:
                if self.config.on_signal_failed:
                    self.config.on_signal_failed(signal, "Delivery failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to send signal: {e}")
            if self.config.on_signal_failed:
                self.config.on_signal_failed(signal, str(e))
            return None
    
    def _send_direct(self, signal: Signal) -> bool:
        """Send signal directly via MT5 API."""
        if not self.connector.is_connected:
            self.logger.error("MT5 not connected")
            return False
        
        # Import order executor
        from .order_executor import OrderExecutor
        
        executor = OrderExecutor(self.connector)
        
        if signal.signal_type == SignalType.LONG:
            result = executor.buy(
                symbol=signal.symbol,
                lot=signal.lot_size,
                sl_pips=signal.sl_pips,
                tp_pips=signal.tp_pips,
                comment=signal.comment
            )
        elif signal.signal_type == SignalType.SHORT:
            result = executor.sell(
                symbol=signal.symbol,
                lot=signal.lot_size,
                sl_pips=signal.sl_pips,
                tp_pips=signal.tp_pips,
                comment=signal.comment
            )
        elif signal.signal_type in [SignalType.FLAT, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            result = executor.close_all(symbol=signal.symbol)
        else:
            self.logger.warning(f"Unknown signal type: {signal.signal_type}")
            return False
        
        if result and self.config.on_signal_executed:
            self.config.on_signal_executed(signal, result)
        
        return result is not None
    
    def _send_file(self, signal: Signal) -> bool:
        """Send signal via file (for EA integration)."""
        try:
            # Write current signal
            with open(self.config.signal_file_path, 'w') as f:
                json.dump(signal.to_dict(), f, indent=2)
            
            # Archive to history
            history_file = os.path.join(
                self.config.signal_history_path,
                f"{signal.id}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(history_file, 'w') as f:
                json.dump(signal.to_dict(), f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write signal file: {e}")
            return False
    
    def _send_socket(self, signal: Signal) -> bool:
        """Send signal via socket (for future use)."""
        # TODO: Implement socket-based delivery
        self.logger.warning("Socket delivery not yet implemented")
        return False
    
    def process_strategy_signals(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        strategy_name: str = "QuantLab",
        lot_size: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None
    ) -> List[Signal]:
        """
        Process QuantLab strategy signals and send to MT5.
        
        Only sends signal when there's a change from previous state.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        signals : pd.Series
            Strategy signals (1=long, -1=short, 0=flat)
        symbol : str
            Trading symbol
        strategy_name : str
            Strategy identifier
        lot_size : float, optional
            Position size
        sl_pips : float, optional
            Stop loss in pips
        tp_pips : float, optional
            Take profit in pips
            
        Returns
        -------
        List[Signal]
            List of sent signals
        """
        sent_signals = []
        
        # Get latest signal
        if len(signals) == 0:
            return sent_signals
        
        current_signal = signals.iloc[-1]
        prev_signal = signals.iloc[-2] if len(signals) > 1 else 0
        
        # Only send if signal changed
        if current_signal != prev_signal:
            if current_signal == 1:
                signal_type = SignalType.LONG
            elif current_signal == -1:
                signal_type = SignalType.SHORT
            else:
                signal_type = SignalType.FLAT
            
            signal = self.send_signal(
                symbol=symbol,
                signal_type=signal_type,
                lot_size=lot_size,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                strategy_name=strategy_name
            )
            
            if signal:
                sent_signals.append(signal)
        
        return sent_signals
    
    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100
    ) -> List[Signal]:
        """
        Get signal history.
        
        Parameters
        ----------
        symbol : str, optional
            Filter by symbol
        strategy : str, optional
            Filter by strategy
        limit : int
            Maximum number of signals to return
            
        Returns
        -------
        List[Signal]
            Signal history
        """
        history = self._signal_history.copy()
        
        if symbol:
            history = [s for s in history if s.symbol == symbol]
        
        if strategy:
            history = [s for s in history if s.strategy_name == strategy]
        
        return history[-limit:]
    
    def clear_history(self) -> None:
        """Clear signal history."""
        self._signal_history.clear()
        self.logger.info("Signal history cleared")
