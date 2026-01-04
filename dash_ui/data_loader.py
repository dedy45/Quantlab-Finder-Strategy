"""
Data Loader dengan Server-Side Caching untuk Dash UI.

PENTING: Selalu gunakan data REAL dari ArcticDB!
TIDAK BOLEH menggunakan dummy/synthetic/demo data.

Usage:
    from dash_ui.data_loader import load_ohlcv_cached, get_available_symbols
    
    # Get available symbols
    symbols = get_available_symbols()
    
    # Load data with caching
    df = load_ohlcv_cached('XAUUSD', '2024-01-01', '2024-12-31', '1H')
"""
import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import pandas as pd

from .cache import (
    generate_data_key,
    save_to_cache,
    load_from_cache,
    cache_exists,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_available_symbols() -> List[str]:
    """
    Get list of available symbols from ArcticDB.
    
    Cached in memory (LRU) since symbols don't change often.
    
    Returns
    -------
    list of str
        Available symbols (e.g., ['XAUUSD', 'EURUSD'])
        
    Raises
    ------
    RuntimeError
        If ArcticDB is not available or no data found
    """
    try:
        from core.data_engine import ArcticStore
        
        store = ArcticStore()
        symbols = store.list_symbols()
        
        if not symbols:
            logger.warning("No symbols found in ArcticDB")
            return []
        
        # Extract unique base symbols (remove timeframe suffix)
        unique_symbols = set()
        for sym in symbols:
            # Symbol format: XAUUSD_1H, XAUUSD_4H, etc.
            parts = sym.rsplit('_', 1)
            if len(parts) >= 1:
                unique_symbols.add(parts[0])
        
        result = sorted(list(unique_symbols))
        logger.info(f"Found {len(result)} symbols in ArcticDB: {result}")
        return result
        
    except ImportError:
        logger.error("core.data_engine not available")
        raise RuntimeError("Data engine not available. Please check installation.")
    except Exception as e:
        logger.error(f"Failed to get symbols from ArcticDB: {e}")
        raise RuntimeError(f"Failed to connect to ArcticDB: {e}")


@lru_cache(maxsize=1)
def get_available_timeframes() -> List[str]:
    """
    Get list of available timeframes from config.
    
    Returns
    -------
    list of str
        Available timeframes (e.g., ['15T', '1H', '4H', '1D'])
    """
    try:
        from config import get_config
        cfg = get_config()
        return cfg.data.available_timeframes
    except Exception:
        # Fallback to defaults
        return ['15T', '1H', '4H', '1D']


def load_ohlcv_cached(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = '1H'
) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data with server-side caching.
    
    First checks cache, then loads from ArcticDB if not cached.
    
    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., 'XAUUSD')
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD)
    timeframe : str, default='1H'
        Timeframe (e.g., '1H', '4H', '1D')
        
    Returns
    -------
    pd.DataFrame or None
        OHLCV DataFrame with columns: open, high, low, close, volume (optional)
        Returns None if data not available
        
    Raises
    ------
    RuntimeError
        If ArcticDB connection fails
        
    Examples
    --------
    >>> df = load_ohlcv_cached('XAUUSD', '2024-01-01', '2024-12-31', '1H')
    >>> print(f"Loaded {len(df)} bars")
    """
    # Generate cache key
    cache_key = generate_data_key(symbol, timeframe, start, end)
    
    # Check cache first
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        logger.info(f"Cache hit for {symbol} {timeframe}: {len(cached_data)} bars")
        return cached_data
    
    # Load from ArcticDB
    logger.info(f"Loading {symbol} {timeframe} from ArcticDB ({start} to {end})")
    
    try:
        from core.data_engine import load_data
        
        df = load_data(symbol, start, end, timeframe)
        
        if df is None or len(df) == 0:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return None
        
        # Save to cache
        save_to_cache(cache_key, df)
        logger.info(f"Loaded and cached {len(df)} bars for {symbol} {timeframe}")
        
        return df
        
    except ImportError:
        logger.error("core.data_engine not available")
        raise RuntimeError("Data engine not available")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise RuntimeError(f"Failed to load data from ArcticDB: {e}")


def load_backtest_result_cached(job_id: str) -> Optional[dict]:
    """
    Load backtest result from cache.
    
    Parameters
    ----------
    job_id : str
        Backtest job ID
        
    Returns
    -------
    dict or None
        Backtest result dictionary, or None if not found
    """
    cache_key = f"backtest_{job_id}"
    return load_from_cache(cache_key)


def save_backtest_result(job_id: str, result: dict) -> bool:
    """
    Save backtest result to cache.
    
    Parameters
    ----------
    job_id : str
        Backtest job ID
    result : dict
        Backtest result dictionary
        
    Returns
    -------
    bool
        True if saved successfully
    """
    cache_key = f"backtest_{job_id}"
    return save_to_cache(cache_key, result)


def get_data_info(
    symbol: str,
    timeframe: str
) -> Optional[dict]:
    """
    Get information about available data for a symbol/timeframe.
    
    Parameters
    ----------
    symbol : str
        Trading symbol
    timeframe : str
        Timeframe
        
    Returns
    -------
    dict or None
        Data info including date range, row count, etc.
    """
    try:
        from core.data_engine import ArcticStore
        
        store = ArcticStore()
        
        # Try to read metadata or small sample
        df = store.read(symbol, timeframe)
        
        if df is None or len(df) == 0:
            return None
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'rows': len(df),
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'columns': list(df.columns),
        }
        
    except Exception as e:
        logger.error(f"Failed to get data info: {e}")
        return None


def clear_symbol_cache():
    """Clear the LRU cache for symbols (call when data is updated)."""
    get_available_symbols.cache_clear()
    logger.info("Symbol cache cleared")


def get_default_date_range() -> Tuple[str, str]:
    """
    Get default date range for data loading.
    
    Uses data available in ArcticDB to determine sensible defaults.
    
    Returns
    -------
    tuple of (str, str)
        (start_date, end_date) in YYYY-MM-DD format
    """
    from datetime import datetime, timedelta
    
    # Try to get actual data range from ArcticDB
    try:
        from core.data_engine import ArcticStore
        store = ArcticStore()
        
        # Try XAUUSD first (primary symbol)
        for symbol in ['XAUUSD', 'EURUSD']:
            for tf in ['1H', '4H', '1D']:
                try:
                    df = store.read(symbol, tf)
                    if df is not None and len(df) > 0:
                        # Use actual data range
                        start = df.index.min().strftime('%Y-%m-%d')
                        end = df.index.max().strftime('%Y-%m-%d')
                        logger.info(f"Using data range from {symbol}_{tf}: {start} to {end}")
                        return start, end
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Could not get data range from ArcticDB: {e}")
    
    # Fallback: use last 2 years
    end = datetime.now()
    start = end - timedelta(days=730)
    
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def get_default_symbol() -> str:
    """
    Get default symbol from config.
    
    Returns
    -------
    str
        Default symbol (e.g., 'XAUUSD')
    """
    try:
        # Try to get from config
        from config import get_config
        cfg = get_config()
        
        # Check if default_symbol exists in config
        if hasattr(cfg.data, 'default_symbol'):
            return cfg.data.default_symbol
    except Exception:
        pass
    
    # Fallback: get first available symbol
    symbols = get_available_symbols()
    if symbols:
        return symbols[0]
    
    return 'XAUUSD'  # Ultimate fallback


def get_default_timeframe() -> str:
    """
    Get default timeframe from config.
    
    Returns
    -------
    str
        Default timeframe (e.g., '1H')
    """
    try:
        from config import get_config
        cfg = get_config()
        return cfg.data.default_timeframe
    except Exception:
        return '1H'


def load_all_data_for_symbol(symbol: str, timeframe: str = None) -> tuple:
    """
    Load ALL available data for a symbol from ArcticDB.
    
    If timeframe is specified, loads data for that specific timeframe.
    If timeframe is None, tries timeframes in order: 1H, 4H, 1D.
    
    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., 'XAUUSD')
    timeframe : str, optional
        Specific timeframe to load (e.g., '15T', '1H', '4H', '1D')
        If None, tries multiple timeframes in order of preference
        
    Returns
    -------
    tuple of (pd.DataFrame, str)
        (DataFrame with DatetimeIndex, timeframe used)
        Returns (None, None) if no data found
        
    Examples
    --------
    >>> df, tf = load_all_data_for_symbol('XAUUSD', '15T')
    >>> print(f"Loaded {len(df)} bars at {tf}")
    """
    try:
        from core.data_engine import ArcticStore
        
        store = ArcticStore()
        
        # If specific timeframe requested, try only that one
        if timeframe:
            timeframes_to_try = [timeframe, timeframe.upper(), timeframe.lower()]
        else:
            # Fallback: try timeframes in order of preference
            timeframes_to_try = ['1H', '4H', '1D', '15T']
        
        for tf in timeframes_to_try:
            try:
                df = store.read(symbol, tf)
                if df is not None and len(df) > 0:
                    logger.debug(f"Raw data columns: {df.columns.tolist()}")
                    logger.debug(f"Raw data index type: {type(df.index)}")
                    
                    # ArcticStore.read() returns timestamp as COLUMN (not index)
                    # We need to set it as index for proper date range extraction
                    if 'timestamp' in df.columns:
                        # Convert to datetime if needed
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        logger.debug(f"Set timestamp column as index")
                    elif not isinstance(df.index, pd.DatetimeIndex):
                        # Try to convert existing index to datetime
                        try:
                            df.index = pd.to_datetime(df.index)
                            logger.debug(f"Converted index to DatetimeIndex")
                        except Exception as conv_err:
                            logger.warning(f"Could not convert index to datetime: {conv_err}")
                    
                    # Verify we have DatetimeIndex
                    if isinstance(df.index, pd.DatetimeIndex):
                        logger.info(f"Loaded ALL data for {symbol}_{tf}: {len(df):,} bars")
                        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                        return df, tf
                    else:
                        logger.warning(f"Index is not DatetimeIndex: {type(df.index)}")
                        # Still return the data, but log warning
                        return df, tf
                        
            except Exception as e:
                logger.debug(f"No data for {symbol}_{tf}: {e}")
                continue
        
        # If specific timeframe was requested but not found
        if timeframe:
            logger.warning(f"No data found for {symbol}_{timeframe}")
        else:
            logger.warning(f"No data found for {symbol} in any timeframe")
        return None, None
        
    except ImportError:
        logger.error("core.data_engine not available")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load all data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
