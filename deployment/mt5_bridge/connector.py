"""
MT5 Connector - Connection management to MetaTrader 5 terminal.

Handles:
- Terminal connection/disconnection
- Account information retrieval
- Symbol information and market data
- Connection health monitoring
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MT5Config:
    """Configuration for MT5 connection."""
    
    # Connection
    path: Optional[str] = None  # Path to MT5 terminal (auto-detect if None)
    login: Optional[int] = None  # Account login (use terminal default if None)
    password: Optional[str] = None
    server: Optional[str] = None
    
    # Timeouts
    timeout_ms: int = 60000  # Connection timeout
    retry_count: int = 3
    retry_delay_ms: int = 1000
    
    # Trading
    magic_number: int = 123456  # EA magic number for order identification
    deviation: int = 20  # Max price deviation in points
    
    # Risk defaults (can be overridden per trade)
    default_lot: float = 0.01
    max_lot: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.timeout_ms > 0, "Timeout must be positive"
        assert self.retry_count >= 0, "Retry count must be non-negative"
        assert self.default_lot > 0, "Default lot must be positive"
        assert self.max_lot >= self.default_lot, "Max lot must be >= default lot"


@dataclass
class AccountInfo:
    """MT5 account information."""
    login: int = 0
    name: str = ""
    server: str = ""
    currency: str = "USD"
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    leverage: int = 1
    trade_allowed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'login': self.login,
            'name': self.name,
            'server': self.server,
            'currency': self.currency,
            'balance': self.balance,
            'equity': self.equity,
            'margin': self.margin,
            'free_margin': self.free_margin,
            'margin_level': self.margin_level,
            'leverage': self.leverage,
            'trade_allowed': self.trade_allowed,
        }


@dataclass
class SymbolInfo:
    """MT5 symbol information."""
    name: str = ""
    description: str = ""
    currency_base: str = ""
    currency_profit: str = ""
    digits: int = 5
    point: float = 0.00001
    tick_size: float = 0.00001
    tick_value: float = 1.0
    contract_size: float = 100000.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    spread: int = 0
    trade_mode: int = 0  # 0=disabled, 1=long only, 2=short only, 3=full
    
    @property
    def is_tradeable(self) -> bool:
        """Check if symbol is tradeable."""
        return self.trade_mode > 0


class MT5Connector:
    """
    Connection manager for MetaTrader 5.
    
    Cara Penggunaan:
        connector = MT5Connector()
        
        # Connect
        if connector.connect():
            print(f"Connected: {connector.account.login}")
            
            # Get symbol info
            info = connector.get_symbol_info('XAUUSD')
            print(f"Spread: {info.spread}")
            
            # Get prices
            prices = connector.get_prices('XAUUSD', 'H1', 1000)
            
        # Disconnect
        connector.disconnect()
    
    Nilai:
    - Abstraksi MT5 API yang clean
    - Error handling dan retry logic
    - Health monitoring
    
    Manfaat:
    - Mudah digunakan dari QuantLab
    - Robust connection management
    - Logging untuk debugging
    """
    
    def __init__(self, config: Optional[MT5Config] = None):
        """Initialize connector."""
        self.config = config or MT5Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._status = ConnectionStatus.DISCONNECTED
        self._mt5 = None
        self._account: Optional[AccountInfo] = None
        self._symbols_cache: Dict[str, SymbolInfo] = {}
        
        # Check if MT5 library is available
        self._mt5_available = False
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            self._mt5_available = True
            self.logger.info("MetaTrader5 library available")
        except ImportError:
            self.logger.warning(
                "MetaTrader5 library not found. "
                "Install with: pip install MetaTrader5"
            )
    
    @property
    def status(self) -> ConnectionStatus:
        """Get connection status."""
        return self._status
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status == ConnectionStatus.CONNECTED
    
    @property
    def account(self) -> Optional[AccountInfo]:
        """Get account information."""
        return self._account
    
    def connect(self) -> bool:
        """
        Connect to MT5 terminal.
        
        Returns
        -------
        bool
            True if connected successfully
        """
        if not self._mt5_available:
            self.logger.error("MT5 library not available")
            return False
        
        if self.is_connected:
            self.logger.info("Already connected")
            return True
        
        self._status = ConnectionStatus.CONNECTING
        
        for attempt in range(self.config.retry_count + 1):
            try:
                # Initialize MT5
                init_kwargs = {}
                if self.config.path:
                    init_kwargs['path'] = self.config.path
                if self.config.login:
                    init_kwargs['login'] = self.config.login
                if self.config.password:
                    init_kwargs['password'] = self.config.password
                if self.config.server:
                    init_kwargs['server'] = self.config.server
                init_kwargs['timeout'] = self.config.timeout_ms
                
                if not self._mt5.initialize(**init_kwargs):
                    error = self._mt5.last_error()
                    self.logger.warning(f"MT5 init failed (attempt {attempt+1}): {error}")
                    continue
                
                # Get account info
                account = self._mt5.account_info()
                if account is None:
                    self.logger.warning("Failed to get account info")
                    self._mt5.shutdown()
                    continue
                
                self._account = AccountInfo(
                    login=account.login,
                    name=account.name,
                    server=account.server,
                    currency=account.currency,
                    balance=account.balance,
                    equity=account.equity,
                    margin=account.margin,
                    free_margin=account.margin_free,
                    margin_level=account.margin_level if account.margin_level else 0,
                    leverage=account.leverage,
                    trade_allowed=account.trade_allowed,
                )
                
                self._status = ConnectionStatus.CONNECTED
                self.logger.info(
                    f"Connected to MT5: {self._account.login} @ {self._account.server}"
                )
                return True
                
            except Exception as e:
                self.logger.error(f"Connection error (attempt {attempt+1}): {e}")
                
            # Wait before retry
            if attempt < self.config.retry_count:
                import time
                time.sleep(self.config.retry_delay_ms / 1000)
        
        self._status = ConnectionStatus.ERROR
        self.logger.error("Failed to connect to MT5 after all retries")
        return False
    
    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self._mt5_available and self.is_connected:
            self._mt5.shutdown()
            self.logger.info("Disconnected from MT5")
        
        self._status = ConnectionStatus.DISCONNECTED
        self._account = None
        self._symbols_cache.clear()
    
    def refresh_account(self) -> Optional[AccountInfo]:
        """Refresh account information."""
        if not self.is_connected:
            return None
        
        try:
            account = self._mt5.account_info()
            if account:
                self._account = AccountInfo(
                    login=account.login,
                    name=account.name,
                    server=account.server,
                    currency=account.currency,
                    balance=account.balance,
                    equity=account.equity,
                    margin=account.margin,
                    free_margin=account.margin_free,
                    margin_level=account.margin_level if account.margin_level else 0,
                    leverage=account.leverage,
                    trade_allowed=account.trade_allowed,
                )
            return self._account
        except Exception as e:
            self.logger.error(f"Failed to refresh account: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get symbol information.
        
        Parameters
        ----------
        symbol : str
            Symbol name (e.g., 'XAUUSD', 'EURUSD')
            
        Returns
        -------
        SymbolInfo or None
            Symbol information if available
        """
        if not self.is_connected:
            return None
        
        # Check cache
        if symbol in self._symbols_cache:
            return self._symbols_cache[symbol]
        
        try:
            # Select symbol (required before getting info)
            if not self._mt5.symbol_select(symbol, True):
                self.logger.warning(f"Failed to select symbol: {symbol}")
                return None
            
            info = self._mt5.symbol_info(symbol)
            if info is None:
                return None
            
            symbol_info = SymbolInfo(
                name=info.name,
                description=info.description,
                currency_base=info.currency_base,
                currency_profit=info.currency_profit,
                digits=info.digits,
                point=info.point,
                tick_size=info.trade_tick_size,
                tick_value=info.trade_tick_value,
                contract_size=info.trade_contract_size,
                volume_min=info.volume_min,
                volume_max=info.volume_max,
                volume_step=info.volume_step,
                spread=info.spread,
                trade_mode=info.trade_mode,
            )
            
            # Cache it
            self._symbols_cache[symbol] = symbol_info
            return symbol_info
            
        except Exception as e:
            self.logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def get_prices(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000,
        start_pos: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV prices.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        timeframe : str
            Timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN1
        count : int
            Number of bars to retrieve
        start_pos : int
            Starting position (0 = current bar)
            
        Returns
        -------
        pd.DataFrame or None
            OHLCV data with datetime index
        """
        if not self.is_connected:
            return None
        
        try:
            # Map timeframe string to MT5 constant
            tf_map = {
                'M1': self._mt5.TIMEFRAME_M1,
                'M5': self._mt5.TIMEFRAME_M5,
                'M15': self._mt5.TIMEFRAME_M15,
                'M30': self._mt5.TIMEFRAME_M30,
                'H1': self._mt5.TIMEFRAME_H1,
                'H4': self._mt5.TIMEFRAME_H4,
                'D1': self._mt5.TIMEFRAME_D1,
                'W1': self._mt5.TIMEFRAME_W1,
                'MN1': self._mt5.TIMEFRAME_MN1,
            }
            
            tf = tf_map.get(timeframe.upper())
            if tf is None:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Get rates
            rates = self._mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
            })
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"Failed to get prices for {symbol}: {e}")
            return None
    
    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current tick (bid/ask).
        
        Parameters
        ----------
        symbol : str
            Symbol name
            
        Returns
        -------
        dict or None
            Tick data with bid, ask, last, volume, time
        """
        if not self.is_connected:
            return None
        
        try:
            tick = self._mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get tick for {symbol}: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns
        -------
        dict
            Health status with details
        """
        result = {
            'status': self._status.value,
            'mt5_available': self._mt5_available,
            'connected': self.is_connected,
            'account': None,
            'terminal_info': None,
            'timestamp': datetime.now().isoformat(),
        }
        
        if self.is_connected:
            # Refresh account
            self.refresh_account()
            if self._account:
                result['account'] = self._account.to_dict()
            
            # Get terminal info
            try:
                term = self._mt5.terminal_info()
                if term:
                    result['terminal_info'] = {
                        'connected': term.connected,
                        'trade_allowed': term.trade_allowed,
                        'dlls_allowed': term.dlls_allowed,
                        'company': term.company,
                        'name': term.name,
                        'path': term.path,
                    }
            except Exception:
                pass
        
        return result
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
