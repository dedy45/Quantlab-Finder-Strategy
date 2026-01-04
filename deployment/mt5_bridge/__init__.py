"""
MT5 Bridge - Live trading bridge for MetaTrader 5.

Connects QuantLab strategies to MT5 for live derivatives trading.
Supports Forex, Gold, CFD, and other instruments available on MT5.

Components:
- MT5Connector: Connection management to MT5 terminal
- SignalBridge: Transfer signals from QuantLab to MT5
- OrderExecutor: Execute orders with proper risk management
- PositionManager: Track and manage open positions
- RiskGuard: Real-time risk limits and circuit breakers

Cara Penggunaan:
    from deployment.mt5_bridge import MT5Connector, SignalBridge
    
    # Connect to MT5
    connector = MT5Connector()
    connector.connect()
    
    # Send signal
    bridge = SignalBridge(connector)
    bridge.send_signal('XAUUSD', 1, lot_size=0.1)

Requirements:
    - MetaTrader5 Python package: pip install MetaTrader5
    - MT5 terminal installed and logged in
    - Broker account with API trading enabled
"""

from .connector import MT5Connector, MT5Config
from .signal_bridge import SignalBridge, Signal, SignalType
from .order_executor import OrderExecutor, OrderConfig
from .position_manager import PositionManager
from .risk_guard import RiskGuard, RiskConfig

__all__ = [
    'MT5Connector',
    'MT5Config',
    'SignalBridge',
    'Signal',
    'SignalType',
    'OrderExecutor',
    'OrderConfig',
    'PositionManager',
    'RiskGuard',
    'RiskConfig',
]
