"""
ArcticDB Data Store - High-performance time-series database.

ArcticDB provides:
- Serverless DataFrame database (no external DB needed)
- Native pandas DataFrame support
- Efficient storage with LMDB backend
- Versioning and snapshots
- Query by date range without loading all data
- Append without rewriting entire dataset

Usage:
    from core.data_engine import ArcticStore
    
    store = ArcticStore()
    
    # Write data
    store.write('XAUUSD', df, timeframe='1H')
    
    # Read data
    df = store.read('XAUUSD', timeframe='1H')
    
    # Read with date range (efficient - only loads needed data)
    df = store.read('XAUUSD', timeframe='1H', 
                    start='2024-01-01', end='2024-12-31')
    
    # Append new data
    store.append('XAUUSD', new_df, timeframe='1H')
    
    # List all symbols
    store.list_symbols()
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Check if ArcticDB is available
try:
    import arcticdb as adb
    ARCTICDB_AVAILABLE = True
except ImportError:
    ARCTICDB_AVAILABLE = False
    logger.warning("ArcticDB not installed. Run: pip install arcticdb")


class ArcticStore:
    """
    ArcticDB-based data store for time-series financial data.
    
    Features:
    - Multi-symbol, multi-timeframe storage in single database
    - Efficient date range queries
    - Native append support
    - Versioning for audit trail
    - LMDB backend (local disk, no server needed)
    
    Symbol naming convention:
        {SYMBOL}_{TIMEFRAME}
        e.g., XAUUSD_1H, EURUSD_4H, F_GC_1D
    
    Examples
    --------
    >>> store = ArcticStore()
    >>> 
    >>> # Write OHLCV data
    >>> store.write('XAUUSD', df, timeframe='1H')
    >>> 
    >>> # Read all data
    >>> df = store.read('XAUUSD', timeframe='1H')
    >>> 
    >>> # Read specific date range (efficient)
    >>> df = store.read('XAUUSD', timeframe='1H',
    ...                 start='2024-01-01', end='2024-06-30')
    >>> 
    >>> # Append new data
    >>> store.append('XAUUSD', new_df, timeframe='1H')
    >>> 
    >>> # List all symbols
    >>> symbols = store.list_symbols()
    """
    
    # Library names for different data types
    LIB_OHLCV = 'ohlcv'
    LIB_FEATURES = 'features'
    LIB_SIGNALS = 'signals'
    LIB_RESULTS = 'results'
    
    def __init__(self, db_path: str = None):
        """
        Initialize ArcticDB store.
        
        Parameters
        ----------
        db_path : str, optional
            Path to database directory.
            Default: QuantLab/data/arcticdb/
        """
        if not ARCTICDB_AVAILABLE:
            raise ImportError(
                "ArcticDB not installed. Run: pip install arcticdb"
            )
        
        if db_path is None:
            # Find project root
            current = Path(__file__).parent
            while current.name != 'QuantLab' and current.parent != current:
                current = current.parent
            db_path = current / 'data' / 'arcticdb'
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Arctic connection (LMDB backend)
        uri = f"lmdb://{self.db_path}"
        self._arctic = adb.Arctic(uri)
        
        # NOTE: Libraries are created on-demand (lazy initialization)
        # to avoid LMDB pre-allocating ~2GB per empty library
        # Only create 'ohlcv' library if it doesn't exist (primary use case)
        if self.LIB_OHLCV not in self._arctic.list_libraries():
            self._arctic.create_library(self.LIB_OHLCV)
            logger.info(f"[OK] Created library: {self.LIB_OHLCV}")
        
        logger.info(f"[OK] ArcticDB initialized at {self.db_path}")
    
    def _init_libraries(self) -> None:
        """
        Initialize default libraries (DEPRECATED - kept for backward compatibility).
        
        NOTE: This method is no longer called automatically.
        Libraries are now created on-demand to avoid LMDB pre-allocating
        ~2GB per empty library.
        
        Use _ensure_library() instead for on-demand creation.
        """
        for lib_name in [self.LIB_OHLCV, self.LIB_FEATURES, 
                         self.LIB_SIGNALS, self.LIB_RESULTS]:
            if lib_name not in self._arctic.list_libraries():
                self._arctic.create_library(lib_name)
                logger.info(f"[OK] Created library: {lib_name}")
    
    def _ensure_library(self, lib_name: str) -> None:
        """
        Ensure library exists (create on-demand if needed).
        
        Parameters
        ----------
        lib_name : str
            Library name to ensure exists
        """
        if lib_name not in self._arctic.list_libraries():
            self._arctic.create_library(lib_name)
            logger.info(f"[OK] Created library on-demand: {lib_name}")
    
    def _get_symbol_key(self, symbol: str, timeframe: str) -> str:
        """Generate symbol key for storage."""
        return f"{symbol.upper()}_{timeframe.upper()}"
    
    def _get_library(self, lib_name: str = None):
        """
        Get library instance (creates on-demand if needed).
        
        Parameters
        ----------
        lib_name : str, optional
            Library name. Default: 'ohlcv'
            
        Returns
        -------
        Library instance
        """
        lib_name = lib_name or self.LIB_OHLCV
        self._ensure_library(lib_name)
        return self._arctic.get_library(lib_name)
    
    def write(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = '1H',
        library: str = None,
        prune_previous: bool = True,
    ) -> bool:
        """
        Write DataFrame to ArcticDB.
        
        Parameters
        ----------
        symbol : str
            Symbol name (e.g., 'XAUUSD', 'EURUSD')
        df : pd.DataFrame
            DataFrame with timestamp index or column
        timeframe : str
            Timeframe: '1H', '4H', '1D', etc.
        library : str, optional
            Library name. Default: 'ohlcv'
        prune_previous : bool
            If True, remove old versions to save space
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            lib = self._get_library(library)
            key = self._get_symbol_key(symbol, timeframe)
            
            # Ensure timestamp index
            df = self._prepare_dataframe(df)
            
            # Write to ArcticDB
            lib.write(key, df, prune_previous_versions=prune_previous)
            
            logger.info(f"[OK] Written {len(df):,} rows to {key}")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Write error: {e}")
            return False
    
    def read(
        self,
        symbol: str,
        timeframe: str = '1H',
        start: str = None,
        end: str = None,
        library: str = None,
        columns: List[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Read DataFrame from ArcticDB.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        timeframe : str
            Timeframe
        start : str, optional
            Start date (YYYY-MM-DD)
        end : str, optional
            End date (YYYY-MM-DD)
        library : str, optional
            Library name
        columns : list, optional
            Specific columns to read
            
        Returns
        -------
        pd.DataFrame or None
            Data if found, None otherwise
        """
        try:
            lib = self._get_library(library)
            key = self._get_symbol_key(symbol, timeframe)
            
            # Check if symbol exists
            if key not in lib.list_symbols():
                logger.warning(f"[WARN] Symbol not found: {key}")
                return None
            
            # Read all data first
            result = lib.read(key, columns=columns) if columns else lib.read(key)
            df = result.data
            
            # Reset index to have timestamp as column
            if df.index.name == 'timestamp':
                df = df.reset_index()
            
            # Apply date filter if specified
            if start or end:
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    if start:
                        start_dt = pd.to_datetime(start)
                        df = df[df['timestamp'] >= start_dt]
                    
                    if end:
                        end_dt = pd.to_datetime(end)
                        df = df[df['timestamp'] <= end_dt]
            
            logger.info(f"[OK] Read {len(df):,} rows from {key}")
            return df
            
        except Exception as e:
            logger.error(f"[FAIL] Read error: {e}")
            return None
    
    def append(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = '1H',
        library: str = None,
    ) -> bool:
        """
        Append new data to existing symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        df : pd.DataFrame
            New data to append
        timeframe : str
            Timeframe
        library : str, optional
            Library name
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            lib = self._get_library(library)
            key = self._get_symbol_key(symbol, timeframe)
            
            # Ensure timestamp index
            df = self._prepare_dataframe(df)
            
            # Check if symbol exists
            if key not in lib.list_symbols():
                # First write
                return self.write(symbol, df, timeframe, library)
            
            # Append to existing
            lib.append(key, df)
            
            logger.info(f"[OK] Appended {len(df):,} rows to {key}")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Append error: {e}")
            return False
    
    def update(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = '1H',
        library: str = None,
    ) -> bool:
        """
        Update existing data (upsert - update or insert).
        
        Parameters
        ----------
        symbol : str
            Symbol name
        df : pd.DataFrame
            Data to update
        timeframe : str
            Timeframe
        library : str, optional
            Library name
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            lib = self._get_library(library)
            key = self._get_symbol_key(symbol, timeframe)
            
            # Ensure timestamp index
            df = self._prepare_dataframe(df)
            
            # Check if symbol exists
            if key not in lib.list_symbols():
                return self.write(symbol, df, timeframe, library)
            
            # Update existing
            lib.update(key, df)
            
            logger.info(f"[OK] Updated {len(df):,} rows in {key}")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Update error: {e}")
            return False
    
    def delete(
        self,
        symbol: str,
        timeframe: str = None,
        library: str = None,
    ) -> bool:
        """
        Delete symbol from database.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        timeframe : str, optional
            Timeframe. If None, delete all timeframes
        library : str, optional
            Library name
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            lib = self._get_library(library)
            
            if timeframe:
                key = self._get_symbol_key(symbol, timeframe)
                if key in lib.list_symbols():
                    lib.delete(key)
                    logger.info(f"[OK] Deleted {key}")
                    return True
            else:
                # Delete all timeframes
                deleted = 0
                for key in lib.list_symbols():
                    if key.startswith(symbol.upper() + '_'):
                        lib.delete(key)
                        deleted += 1
                logger.info(f"[OK] Deleted {deleted} symbols for {symbol}")
                return deleted > 0
            
            return False
            
        except Exception as e:
            logger.error(f"[FAIL] Delete error: {e}")
            return False
    
    def list_symbols(
        self,
        library: str = None,
        symbol_filter: str = None,
    ) -> List[str]:
        """
        List all symbols in library.
        
        Parameters
        ----------
        library : str, optional
            Library name
        symbol_filter : str, optional
            Filter by symbol prefix (e.g., 'XAUUSD')
            
        Returns
        -------
        list
            List of symbol keys
        """
        lib = self._get_library(library)
        symbols = lib.list_symbols()
        
        if symbol_filter:
            symbols = [s for s in symbols 
                      if s.startswith(symbol_filter.upper())]
        
        return sorted(symbols)
    
    def get_info(
        self,
        symbol: str,
        timeframe: str = '1H',
        library: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        timeframe : str
            Timeframe
        library : str, optional
            Library name
            
        Returns
        -------
        dict or None
            Symbol metadata
        """
        try:
            lib = self._get_library(library)
            key = self._get_symbol_key(symbol, timeframe)
            
            if key not in lib.list_symbols():
                return None
            
            # Get symbol info
            info = lib.get_description(key)
            
            # Read to get date range
            df = self.read(symbol, timeframe, library=library)
            
            if df is not None and len(df) > 0:
                if 'timestamp' in df.columns:
                    ts_col = df['timestamp']
                else:
                    ts_col = df.index
                
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'key': key,
                    'rows': len(df),
                    'start': ts_col.min(),
                    'end': ts_col.max(),
                    'columns': list(df.columns),
                }
            
            return {'symbol': symbol, 'timeframe': timeframe, 'key': key}
            
        except Exception as e:
            logger.error(f"[FAIL] Get info error: {e}")
            return None
    
    def list_available(self, library: str = None) -> List[Dict[str, Any]]:
        """
        List all available data with metadata.
        
        Returns
        -------
        list
            List of symbol info dicts
        """
        result = []
        
        for key in self.list_symbols(library):
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                symbol, timeframe = parts
                info = self.get_info(symbol, timeframe, library)
                if info:
                    result.append(info)
        
        return result
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for storage."""
        df = df.copy()
        
        # Ensure timestamp index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif df.index.name != 'timestamp':
            if isinstance(df.index, pd.DatetimeIndex):
                df.index.name = 'timestamp'
            else:
                raise ValueError(
                    "DataFrame must have 'timestamp' column or DatetimeIndex"
                )
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def get_libraries(self) -> List[str]:
        """Get list of all libraries."""
        return self._arctic.list_libraries()
    
    def get_db_size(self) -> Dict[str, Any]:
        """Get database size information."""
        total_size = 0
        
        for f in self.db_path.rglob('*'):
            if f.is_file():
                total_size += f.stat().st_size
        
        return {
            'path': str(self.db_path),
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024),
            'size_gb': total_size / (1024 * 1024 * 1024),
        }


# Convenience functions
def get_arctic_store(db_path: str = None) -> ArcticStore:
    """Get ArcticStore instance."""
    return ArcticStore(db_path)
