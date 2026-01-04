"""
CSV Tick Data Loader - Load and resample tick data from CSV files.

Supports various CSV formats:
- Standard: timestamp, bid, ask, volume
- MT5 export: time, bid, ask, last, volume
- Custom formats with configurable columns

Usage:
    loader = CSVTickLoader()
    df = loader.load_ticks('data/raw/XAUUSD_ticks.csv')
    ohlcv = loader.resample_to_ohlcv(df, timeframe='1H')
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CSVConfig:
    """Configuration for CSV loading."""
    
    # Column mapping
    timestamp_col: str = 'timestamp'
    bid_col: str = 'bid'
    ask_col: str = 'ask'
    volume_col: str = 'volume'
    
    # Timestamp format
    timestamp_format: Optional[str] = None  # Auto-detect if None
    
    # CSV options
    delimiter: str = ','
    encoding: str = 'utf-8'
    skip_rows: int = 0
    
    # Data options
    use_mid_price: bool = True  # Use (bid+ask)/2 for OHLCV
    fill_missing: bool = True


class CSVTickLoader:
    """
    Load tick data from CSV files and resample to OHLCV.
    
    Examples
    --------
    >>> loader = CSVTickLoader()
    >>> 
    >>> # Load ticks
    >>> ticks = loader.load_ticks('data/raw/XAUUSD_ticks.csv')
    >>> 
    >>> # Resample to 1H OHLCV
    >>> ohlcv = loader.resample_to_ohlcv(ticks, '1H')
    >>> 
    >>> # Or load and resample in one step
    >>> ohlcv = loader.load_ohlcv('data/raw/XAUUSD_ticks.csv', '1H')
    """
    
    # Common column name mappings
    COLUMN_ALIASES = {
        'timestamp': ['timestamp', 'time', 'datetime', 'date', 'Time', 'DateTime'],
        'bid': ['bid', 'Bid', 'bid_price', 'BidPrice'],
        'ask': ['ask', 'Ask', 'ask_price', 'AskPrice'],
        'volume': ['volume', 'Volume', 'vol', 'Vol', 'tick_volume', 'TickVolume'],
        'last': ['last', 'Last', 'close', 'Close', 'price', 'Price'],
    }
    
    def __init__(self, config: Optional[CSVConfig] = None):
        """Initialize loader with configuration."""
        self.config = config or CSVConfig()
    
    def load_ticks(
        self,
        filepath: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load tick data from CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to CSV file
        start_date : str, optional
            Filter start date (YYYY-MM-DD)
        end_date : str, optional
            Filter end date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            Tick data with columns: timestamp, bid, ask, volume
        """
        filepath = Path(filepath)
        
        assert filepath.exists(), f"File not found: {filepath}"
        
        logger.info(f"Loading ticks from {filepath.name}...")
        
        try:
            # Read CSV
            df = pd.read_csv(
                filepath,
                delimiter=self.config.delimiter,
                encoding=self.config.encoding,
                skiprows=self.config.skip_rows
            )
            
            logger.info(f"  Raw rows: {len(df):,}")
            logger.info(f"  Columns: {list(df.columns)}")
            
            # Map columns
            df = self._map_columns(df)
            
            # Parse timestamp
            df = self._parse_timestamp(df)
            
            # Filter by date range
            if start_date or end_date:
                df = self._filter_date_range(df, start_date, end_date)
            
            # Clean data
            df = self._clean_ticks(df)
            
            logger.info(f"  Loaded {len(df):,} ticks")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def resample_to_ohlcv(
        self,
        ticks: pd.DataFrame,
        timeframe: str = '1H'
    ) -> pd.DataFrame:
        """
        Resample tick data to OHLCV.
        
        Parameters
        ----------
        ticks : pd.DataFrame
            Tick data with timestamp, bid, ask columns
        timeframe : str
            Target timeframe: '1T' (1min), '5T', '15T', '1H', '4H', '1D'
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        assert ticks is not None, "Ticks cannot be None"
        assert len(ticks) > 0, "Ticks cannot be empty"
        assert 'timestamp' in ticks.columns, "Missing timestamp column"
        
        logger.info(f"Resampling {len(ticks):,} ticks to {timeframe}...")
        
        # Set timestamp as index
        df = ticks.copy()
        df = df.set_index('timestamp')
        
        # Calculate price for OHLCV
        if self.config.use_mid_price and 'bid' in df.columns and 'ask' in df.columns:
            df['price'] = (df['bid'] + df['ask']) / 2
        elif 'last' in df.columns:
            df['price'] = df['last']
        elif 'bid' in df.columns:
            df['price'] = df['bid']
        else:
            raise ValueError("No price column found (bid, ask, or last)")
        
        # Resample
        tf = timeframe.lower()
        ohlcv = df['price'].resample(tf).ohlc()
        ohlcv.columns = ['open', 'high', 'low', 'close']
        
        # Add volume
        if 'volume' in df.columns:
            ohlcv['volume'] = df['volume'].resample(tf).sum()
        else:
            ohlcv['volume'] = df['price'].resample(tf).count()
        
        # Add spread if available
        if 'bid' in df.columns and 'ask' in df.columns:
            ohlcv['spread'] = (df['ask'] - df['bid']).resample(tf).mean()
        
        # Drop NaN rows
        ohlcv = ohlcv.dropna()
        
        # Reset index
        ohlcv = ohlcv.reset_index()
        ohlcv = ohlcv.rename(columns={'index': 'timestamp'})
        
        logger.info(f"  Generated {len(ohlcv):,} bars")
        
        return ohlcv
    
    def load_ohlcv(
        self,
        filepath: Union[str, Path],
        timeframe: str = '1H',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load ticks and resample to OHLCV in one step.
        
        Parameters
        ----------
        filepath : str or Path
            Path to CSV file
        timeframe : str
            Target timeframe
        start_date : str, optional
            Filter start date
        end_date : str, optional
            Filter end date
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        ticks = self.load_ticks(filepath, start_date, end_date)
        return self.resample_to_ohlcv(ticks, timeframe)
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map column names to standard names."""
        column_map = {}
        
        for standard_name, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in df.columns:
                    column_map[alias] = standard_name
                    break
        
        if column_map:
            df = df.rename(columns=column_map)
        
        return df
    
    def _parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp column to datetime."""
        if 'timestamp' not in df.columns:
            # Try to find timestamp column
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    df = df.rename(columns={col: 'timestamp'})
                    break
        
        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp column found")
        
        # Parse timestamp
        if self.config.timestamp_format:
            df['timestamp'] = pd.to_datetime(
                df['timestamp'],
                format=self.config.timestamp_format
            )
        else:
            # Auto-detect format
            df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        
        return df
    
    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter by date range."""
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start]
        
        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end]
        
        return df
    
    def _clean_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean tick data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove invalid prices
        price_cols = ['bid', 'ask', 'last']
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # Remove NaN in critical columns
        critical = ['timestamp']
        if 'bid' in df.columns:
            critical.append('bid')
        if 'ask' in df.columns:
            critical.append('ask')
        
        df = df.dropna(subset=critical)
        
        return df
    
    def get_file_info(self, filepath: Union[str, Path]) -> Dict:
        """Get information about CSV file without loading all data."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {'error': 'File not found'}
        
        # Read first few rows
        df_head = pd.read_csv(
            filepath,
            delimiter=self.config.delimiter,
            nrows=100
        )
        
        # Count total rows
        with open(filepath, 'r', encoding=self.config.encoding) as f:
            total_rows = sum(1 for _ in f) - 1  # Subtract header
        
        return {
            'file': filepath.name,
            'size_mb': filepath.stat().st_size / 1024 / 1024,
            'total_rows': total_rows,
            'columns': list(df_head.columns),
            'sample': df_head.head(5).to_dict()
        }


def load_csv_ticks(filepath: str, timeframe: str = '1H') -> pd.DataFrame:
    """Convenience function to load CSV ticks and resample."""
    loader = CSVTickLoader()
    return loader.load_ohlcv(filepath, timeframe)
