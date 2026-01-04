"""
Data Manager - Unified data loading with ArcticDB-first strategy.

Strategy:
1. Check ArcticDB first (primary storage)
2. Fallback to Parquet files (legacy)
3. If not found, download from source
4. Save to ArcticDB for future use
5. Return clean, validated data

This ensures:
- Fast loading with efficient date range queries
- Multi-symbol, multi-timeframe in single database
- Native append support
- Versioning for audit trail
- Offline capability
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Check storage backends
ARCTICDB_AVAILABLE = False
try:
    from .arctic_store import ArcticStore, ARCTICDB_AVAILABLE as _ADB
    ARCTICDB_AVAILABLE = _ADB
except ImportError:
    pass


class DataManager:
    """
    Unified data manager with ArcticDB-first loading strategy.
    
    Storage Priority:
    1. ArcticDB (primary - fast, efficient queries)
    2. Parquet files (legacy fallback)
    3. Download from source
    
    Examples
    --------
    >>> dm = DataManager()
    >>> 
    >>> # Load data (checks ArcticDB first)
    >>> df = dm.load('XAUUSD', '2024-01-01', '2024-12-31', timeframe='1H')
    >>> 
    >>> # Force download (refresh data)
    >>> df = dm.load('XAUUSD', '2024-01-01', '2024-12-31', force_download=True)
    >>> 
    >>> # List available data
    >>> dm.list_available()
    >>> 
    >>> # Append new data
    >>> dm.append('XAUUSD', new_df, timeframe='1H')
    """
    
    def __init__(
        self,
        data_dir: str = None,
        use_arctic: bool = True,
    ):
        """
        Initialize DataManager.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory for storing data files.
            Default: data/processed/
        use_arctic : bool
            Use ArcticDB as primary storage. Default: True
        """
        # Find project root
        current = Path(__file__).parent
        while current.name != 'QuantLab' and current.parent != current:
            current = current.parent
        self.project_root = current
        
        if data_dir is None:
            data_dir = current / 'data' / 'processed'
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ArcticDB if available and requested
        self._arctic = None
        self.use_arctic = use_arctic and ARCTICDB_AVAILABLE
        
        if self.use_arctic:
            try:
                arctic_path = current / 'data' / 'arcticdb'
                self._arctic = ArcticStore(str(arctic_path))
                logger.info("[OK] Using ArcticDB as primary storage")
            except Exception as e:
                logger.warning(f"[WARN] ArcticDB init failed: {e}")
                logger.warning("[WARN] Falling back to Parquet storage")
                self.use_arctic = False
        else:
            if not ARCTICDB_AVAILABLE:
                logger.info("[INFO] ArcticDB not available, using Parquet storage")
            else:
                logger.info("[INFO] Using Parquet storage (ArcticDB disabled)")
        
        # Initialize loaders (lazy)
        self._dukascopy = None
    
    @property
    def arctic(self) -> Optional['ArcticStore']:
        """Get ArcticDB store."""
        return self._arctic
    
    @property
    def dukascopy(self):
        """Lazy load Dukascopy loader."""
        if self._dukascopy is None:
            from .dukascopy_loader import DukascopyLoader
            self._dukascopy = DukascopyLoader()
        return self._dukascopy
    
    def get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get path for Parquet file (legacy)."""
        return self.data_dir / f"{symbol.upper()}_{timeframe.upper()}.parquet"
    
    def load(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        timeframe: str = '1H',
        force_download: bool = False,
        source: str = 'dukascopy',
    ) -> Optional[pd.DataFrame]:
        """
        Load data with ArcticDB-first strategy.
        
        Parameters
        ----------
        symbol : str
            Symbol to load (e.g., 'XAUUSD', 'EURUSD')
        start_date : str, optional
            Start date (YYYY-MM-DD). Default: 6 months ago
        end_date : str, optional
            End date (YYYY-MM-DD). Default: today
        timeframe : str
            Timeframe: '1H', '4H', '1D'
        force_download : bool
            Force download even if data exists
        source : str
            Data source: 'dukascopy' (default)
            
        Returns
        -------
        pd.DataFrame or None
            OHLCV data with columns: timestamp, open, high, low, close, volume
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        
        # Default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start = datetime.now() - timedelta(days=180)
            start_date = start.strftime('%Y-%m-%d')
        
        # Check storage (unless force_download)
        if not force_download:
            # Try ArcticDB first
            if self.use_arctic and self._arctic:
                df = self._load_from_arctic(symbol, timeframe, start_date, end_date)
                if df is not None and len(df) > 0:
                    logger.info(f"[ARCTIC] Loaded {len(df):,} bars of {symbol} {timeframe}")
                    return df
            
            # Fallback to Parquet
            df = self._load_from_parquet(symbol, timeframe, start_date, end_date)
            if df is not None and len(df) > 0:
                logger.info(f"[PARQUET] Loaded {len(df):,} bars of {symbol} {timeframe}")
                
                # Migrate to ArcticDB if available
                if self.use_arctic and self._arctic:
                    self._arctic.write(symbol, df, timeframe)
                    logger.info(f"[MIGRATE] Migrated {symbol} {timeframe} to ArcticDB")
                
                return df
        
        # Download from source
        logger.info(f"[DOWNLOAD] Fetching {symbol} {timeframe} from {source}...")
        df = self._download(symbol, start_date, end_date, timeframe, source)
        
        if df is None or len(df) == 0:
            logger.warning(f"[FAIL] No data available for {symbol}")
            return None
        
        # Clean and validate
        df = self._clean_data(df)
        
        # Save to storage
        self._save(symbol, df, timeframe)
        
        logger.info(f"[OK] Loaded {len(df):,} bars of {symbol} {timeframe}")
        return df
    
    def _load_from_arctic(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Load data from ArcticDB."""
        try:
            df = self._arctic.read(
                symbol, timeframe,
                start=start_date, end=end_date
            )
            return df
        except Exception as e:
            logger.debug(f"ArcticDB load failed: {e}")
            return None
    
    def _load_from_parquet(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Load data from Parquet file (legacy)."""
        path = self.get_parquet_path(symbol, timeframe)
        
        if not path.exists():
            return None
        
        try:
            df = pd.read_parquet(path)
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()
            
            if 'timestamp' not in df.columns:
                logger.warning(f"No timestamp column in {path}")
                return None
            
            # Convert to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading from Parquet: {e}")
            return None
    
    def _download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
        source: str,
    ) -> Optional[pd.DataFrame]:
        """Download data from source."""
        if source == 'dukascopy':
            return self.dukascopy.load_ohlcv(symbol, start_date, end_date, timeframe)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        # Ensure required columns
        required = ['timestamp', 'open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove NaN in OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate OHLC relationships
        valid_high = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
        valid_low = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
        
        invalid = ~(valid_high & valid_low)
        if invalid.sum() > 0:
            logger.warning(f"Removed {invalid.sum()} invalid OHLC rows")
            df = df[~invalid]
        
        # Remove zero/negative prices
        df = df[df['close'] > 0]
        
        return df
    
    def _save(self, symbol: str, df: pd.DataFrame, timeframe: str) -> None:
        """Save data to storage."""
        # Save to ArcticDB (primary)
        if self.use_arctic and self._arctic:
            try:
                self._arctic.write(symbol, df, timeframe)
                logger.info(f"[SAVE] Saved {len(df):,} bars to ArcticDB")
                return
            except Exception as e:
                logger.warning(f"[WARN] ArcticDB save failed: {e}")
        
        # Fallback to Parquet
        self._save_to_parquet(symbol, df, timeframe)
    
    def _save_to_parquet(self, symbol: str, df: pd.DataFrame, timeframe: str) -> None:
        """Save data to Parquet file."""
        try:
            path = self.get_parquet_path(symbol, timeframe)
            df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
            logger.info(f"[SAVE] Saved {len(df):,} bars to {path.name}")
        except Exception as e:
            logger.error(f"Error saving to Parquet: {e}")
    
    def append(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = '1H',
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
            
        Returns
        -------
        bool
            True if successful
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        
        # Clean data
        df = self._clean_data(df)
        
        if self.use_arctic and self._arctic:
            return self._arctic.append(symbol, df, timeframe)
        else:
            # For Parquet, load existing, concat, and save
            existing = self._load_from_parquet(
                symbol, timeframe,
                '1900-01-01', '2100-01-01'
            )
            
            if existing is not None:
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            self._save_to_parquet(symbol, df, timeframe)
            return True
    
    def list_available(self) -> List[Dict[str, Any]]:
        """List all available data."""
        result = []
        
        # List from ArcticDB
        if self.use_arctic and self._arctic:
            arctic_data = self._arctic.list_available()
            for item in arctic_data:
                item['source'] = 'arcticdb'
                result.append(item)
        
        # List from Parquet (if not already in ArcticDB)
        arctic_keys = {f"{r['symbol']}_{r['timeframe']}" for r in result}
        
        for f in self.data_dir.glob("*.parquet"):
            parts = f.stem.rsplit('_', 1)
            if len(parts) == 2:
                symbol, timeframe = parts
                key = f"{symbol}_{timeframe}"
                
                if key not in arctic_keys:
                    try:
                        df = pd.read_parquet(f)
                        
                        if 'timestamp' in df.columns:
                            ts_col = df['timestamp']
                        elif df.index.name == 'timestamp':
                            ts_col = df.index
                        else:
                            ts_col = None
                        
                        info = {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'key': key,
                            'rows': len(df),
                            'source': 'parquet',
                            'file': f.name,
                        }
                        
                        if ts_col is not None:
                            info['start'] = pd.to_datetime(ts_col.iloc[0])
                            info['end'] = pd.to_datetime(ts_col.iloc[-1])
                        
                        result.append(info)
                        
                    except Exception as e:
                        logger.warning(f"Error reading {f}: {e}")
        
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        symbols = set()
        
        for item in self.list_available():
            symbols.add(item['symbol'])
        
        return sorted(list(symbols))
    
    def delete(self, symbol: str, timeframe: str = None) -> bool:
        """Delete data for a symbol."""
        symbol = symbol.upper()
        deleted = False
        
        # Delete from ArcticDB
        if self.use_arctic and self._arctic:
            deleted = self._arctic.delete(symbol, timeframe) or deleted
        
        # Delete Parquet files
        if timeframe:
            path = self.get_parquet_path(symbol, timeframe)
            if path.exists():
                path.unlink()
                deleted = True
        else:
            for f in self.data_dir.glob(f"{symbol}_*.parquet"):
                f.unlink()
                deleted = True
        
        return deleted
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information."""
        info = {
            'backend': 'arcticdb' if self.use_arctic else 'parquet',
            'data_dir': str(self.data_dir),
        }
        
        if self.use_arctic and self._arctic:
            db_size = self._arctic.get_db_size()
            info['arcticdb'] = db_size
        
        # Parquet size
        parquet_size = sum(f.stat().st_size for f in self.data_dir.glob("*.parquet"))
        info['parquet_size_mb'] = parquet_size / (1024 * 1024)
        
        return info


# Convenience function
def load_data(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    timeframe: str = '1H',
) -> Optional[pd.DataFrame]:
    """
    Quick function to load data.
    
    Examples
    --------
    >>> df = load_data('XAUUSD', '2024-01-01', '2024-12-31')
    >>> print(f"Loaded {len(df)} bars")
    """
    dm = DataManager()
    return dm.load(symbol, start_date, end_date, timeframe)
