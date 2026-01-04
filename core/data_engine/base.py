"""
Base classes and configurations for Data Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd


class DataSource(Enum):
    """Supported data sources."""
    QUANTIACS = "quantiacs"
    BINANCE = "binance"
    TIINGO = "tiingo"
    FRED = "fred"
    YAHOO = "yahoo"
    LOCAL = "local"


class AssetClass(Enum):
    """Asset class types."""
    FUTURES = "futures"
    CRYPTO = "crypto"
    EQUITY = "equity"
    MACRO = "macro"


class Resolution(Enum):
    """Data resolution/timeframe."""
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"


@dataclass
class DataConfig:
    """Configuration for data operations."""
    source: DataSource
    asset_class: AssetClass
    symbols: List[str]
    start_date: str
    end_date: Optional[str] = None
    resolution: Resolution = Resolution.DAILY
    
    # Storage options
    cache_enabled: bool = True
    cache_dir: str = "data/cache"
    output_dir: str = "data/processed"
    
    # Cleaning options
    fill_method: str = "ffill"  # ffill, bfill, interpolate
    remove_weekends: bool = True  # For non-crypto
    
    # Validation options
    check_survivorship: bool = True
    check_lookahead: bool = True
    
    # API keys (loaded from .env)
    api_keys: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataResult:
    """Container for data operation results."""
    data: pd.DataFrame
    source: DataSource
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    resolution: Resolution
    
    # Quality metrics
    missing_pct: float = 0.0
    gaps_detected: int = 0
    outliers_removed: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if data passes quality thresholds."""
        return (
            len(self.data) > 0 and
            self.missing_pct < 0.05 and  # Max 5% missing
            self.gaps_detected < 10  # Max 10 gaps
        )


class BaseDataHandler(ABC):
    """Abstract base class for data handlers."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    @abstractmethod
    def execute(self) -> DataResult:
        """Execute the data operation."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the operation can be performed."""
        pass


# Standard column names for internal data format
STANDARD_COLUMNS = {
    'timestamp': 'timestamp',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'symbol': 'symbol',
}

# Column mappings for different sources
COLUMN_MAPPINGS = {
    DataSource.QUANTIACS: {
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'vol': 'volume',
        'asset': 'symbol',
    },
    DataSource.BINANCE: {
        'open_time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    },
    DataSource.TIINGO: {
        'date': 'timestamp',
        'adjOpen': 'open',
        'adjHigh': 'high',
        'adjLow': 'low',
        'adjClose': 'close',
        'adjVolume': 'volume',
    },
    DataSource.YAHOO: {
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    },
}
