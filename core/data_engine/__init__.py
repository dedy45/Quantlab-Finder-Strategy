"""
Data Engine - Data ingestion, cleaning, and storage.

This module handles all data-related operations including:
- Loading REAL data from Quantiacs API (primary source)
- Tick data from Dukascopy (Forex/CFD - FREE)
- Downloading from other sources (Tiingo, FRED)
- Data cleaning and validation
- Point-in-time data management (no look-ahead bias)
- ArcticDB storage (primary) with Parquet fallback

IMPORTANT: Always use REAL data, never dummy/sample data!

Storage:
- ArcticDB: Primary storage (fast queries, versioning, append)
- Parquet: Legacy fallback

Data Sources:
- Quantiacs: Futures, Crypto, Stocks (daily)
- Dukascopy: Forex, CFD, Gold tick data (FREE, no API)
- Tiingo: US Equities (backup)
- FRED: Macro/Economic indicators
"""

from .base import DataSource, DataConfig
from .downloader import DataDownloader
from .cleaner import DataCleaner
from .validator import DataValidator
from .storage import DataStorage
from .quantiacs_loader import (
    QuantiacsLoader,
    load_futures,
    QUANTIACS_API_KEY,
)
from .dukascopy_loader import (
    DukascopyLoader,
    load_forex_ticks,
    load_forex_ohlcv,
    INSTRUMENTS as DUKASCOPY_INSTRUMENTS,
)
from .data_manager import DataManager, load_data
from .data_validator import DataValidator as OHLCValidator, validate_ohlcv, validate_no_lookahead
from .csv_loader import CSVTickLoader, load_csv_ticks

# ArcticDB (optional - graceful fallback if not installed)
try:
    from .arctic_store import ArcticStore, get_arctic_store, ARCTICDB_AVAILABLE
except ImportError:
    ArcticStore = None
    get_arctic_store = None
    ARCTICDB_AVAILABLE = False

__all__ = [
    # Base
    'DataSource',
    'DataConfig',
    # Loaders
    'DataDownloader',
    'QuantiacsLoader',
    'DukascopyLoader',
    'CSVTickLoader',
    'load_futures',
    'load_forex_ticks',
    'load_forex_ohlcv',
    'load_csv_ticks',
    # Processing
    'DataCleaner',
    'DataValidator',
    'DataStorage',
    # Validation
    'OHLCValidator',
    'validate_ohlcv',
    'validate_no_lookahead',
    # Constants
    'QUANTIACS_API_KEY',
    'DUKASCOPY_INSTRUMENTS',
    # Data Manager
    'DataManager',
    'load_data',
    # ArcticDB
    'ArcticStore',
    'get_arctic_store',
    'ARCTICDB_AVAILABLE',
]
