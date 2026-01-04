"""
Dukascopy Tick Data Loader - FREE tick data for Forex/CFD.

NO API KEY REQUIRED - Public data from Dukascopy bank.

Data available:
- Forex: EUR/USD, GBP/USD, USD/JPY, etc (50+ pairs)
- Commodities: XAU/USD (Gold), XAG/USD (Silver), Oil
- Indices: US30, US500, DE30, etc
- History: 2003 - present (20+ years tick data!)

Reference: https://www.dukascopy.com/swiss/english/marketwatch/historical/
"""

import os
import struct
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from io import BytesIO
import lzma

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Dukascopy instrument specifications
INSTRUMENTS = {
    # Forex Major
    'EURUSD': {'pip': 0.00001, 'name': 'Euro/US Dollar'},
    'GBPUSD': {'pip': 0.00001, 'name': 'British Pound/US Dollar'},
    'USDJPY': {'pip': 0.001, 'name': 'US Dollar/Japanese Yen'},
    'USDCHF': {'pip': 0.00001, 'name': 'US Dollar/Swiss Franc'},
    'AUDUSD': {'pip': 0.00001, 'name': 'Australian Dollar/US Dollar'},
    'USDCAD': {'pip': 0.00001, 'name': 'US Dollar/Canadian Dollar'},
    'NZDUSD': {'pip': 0.00001, 'name': 'New Zealand Dollar/US Dollar'},
    
    # Forex Cross
    'EURGBP': {'pip': 0.00001, 'name': 'Euro/British Pound'},
    'EURJPY': {'pip': 0.001, 'name': 'Euro/Japanese Yen'},
    'GBPJPY': {'pip': 0.001, 'name': 'British Pound/Japanese Yen'},
    
    # Commodities
    'XAUUSD': {'pip': 0.01, 'name': 'Gold/US Dollar'},
    'XAGUSD': {'pip': 0.001, 'name': 'Silver/US Dollar'},
    
    # Indices (CFD)
    'USA500IDXUSD': {'pip': 0.1, 'name': 'S&P 500 Index'},
    'USA30IDXUSD': {'pip': 1.0, 'name': 'Dow Jones 30 Index'},
    'USATECHIDXUSD': {'pip': 0.1, 'name': 'NASDAQ 100 Index'},
    'DEUIDXEUR': {'pip': 0.1, 'name': 'DAX 30 Index'},
    
    # Crypto
    'BTCUSD': {'pip': 0.01, 'name': 'Bitcoin/US Dollar'},
    'ETHUSD': {'pip': 0.01, 'name': 'Ethereum/US Dollar'},
}

# Base URL for Dukascopy data
BASE_URL = "https://datafeed.dukascopy.com/datafeed"


class DukascopyLoader:
    """
    Load FREE tick data from Dukascopy.
    
    No API key required - public data.
    
    Examples
    --------
    >>> loader = DukascopyLoader()
    >>> df = loader.load_ticks('EURUSD', '2024-01-01', '2024-01-02')
    >>> print(df.head())
    
    >>> # Load and resample to 1-minute OHLCV
    >>> ohlcv = loader.load_ohlcv('XAUUSD', '2024-01-01', '2024-01-31', timeframe='1H')
    """
    
    def __init__(self, cache_dir: str = None):
        """Initialize loader with optional cache directory."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache/dukascopy')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        
    def list_instruments(self) -> dict:
        """List available instruments."""
        return INSTRUMENTS
    
    def load_ticks(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load tick data for a symbol.
        
        Parameters
        ----------
        symbol : str
            Instrument symbol (e.g., 'EURUSD', 'XAUUSD')
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        use_cache : bool
            Use cached data if available
            
        Returns
        -------
        pd.DataFrame
            Tick data with columns: timestamp, bid, ask, bid_volume, ask_volume
        """
        symbol = symbol.upper()
        if symbol not in INSTRUMENTS:
            raise ValueError(f"Unknown symbol: {symbol}. Available: {list(INSTRUMENTS.keys())}")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_ticks = []
        current = start
        
        while current < end:
            # Download hour by hour
            for hour in range(24):
                try:
                    ticks = self._download_hour(symbol, current.year, current.month, current.day, hour, use_cache)
                    if ticks is not None and len(ticks) > 0:
                        all_ticks.append(ticks)
                except Exception as e:
                    logger.debug(f"No data for {symbol} {current.date()} {hour:02d}:00 - {e}")
            
            current += timedelta(days=1)
            
            # Progress indicator
            if (current - start).days % 7 == 0:
                logger.info(f"Loading {symbol}: {current.date()}")
        
        if not all_ticks:
            logger.warning(f"No tick data found for {symbol} from {start_date} to {end_date}")
            return pd.DataFrame()
        
        df = pd.concat(all_ticks, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Filter to exact date range
        df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        
        logger.info(f"Loaded {len(df):,} ticks for {symbol}")
        return df
    
    def load_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = '1H',
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load OHLCV data by resampling tick data.
        
        Parameters
        ----------
        symbol : str
            Instrument symbol
        start_date : str
            Start date
        end_date : str
            End date
        timeframe : str
            Resample timeframe: '1T' (1min), '5T', '15T', '1H', '4H', '1D'
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        # Load ticks
        ticks = self.load_ticks(symbol, start_date, end_date, use_cache)
        
        if ticks.empty:
            return pd.DataFrame()
        
        # Use mid price for OHLCV
        ticks['mid'] = (ticks['bid'] + ticks['ask']) / 2
        ticks = ticks.set_index('timestamp')
        
        # Resample (use lowercase for pandas 2.0+)
        tf = timeframe.lower()
        ohlcv = ticks['mid'].resample(tf).ohlc()
        ohlcv.columns = ['open', 'high', 'low', 'close']
        
        # Add volume (tick count as proxy)
        ohlcv['volume'] = ticks['mid'].resample(tf).count()
        
        # Add spread
        ohlcv['spread'] = (ticks['ask'] - ticks['bid']).resample(tf).mean()
        
        ohlcv = ohlcv.dropna()
        ohlcv = ohlcv.reset_index()
        ohlcv['symbol'] = symbol
        
        return ohlcv
    
    def _download_hour(
        self,
        symbol: str,
        year: int,
        month: int,
        day: int,
        hour: int,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Download tick data for a specific hour."""
        
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}.parquet"
        
        if use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)
        
        # Build URL
        # Format: /EURUSD/2024/00/01/00h_ticks.bi5
        # Note: month is 0-indexed in Dukascopy
        url = f"{BASE_URL}/{symbol}/{year}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            
            # Decompress LZMA
            try:
                data = lzma.decompress(response.content)
            except lzma.LZMAError:
                # Sometimes data is not compressed
                data = response.content
            
            if len(data) == 0:
                return None
            
            # Parse binary data
            ticks = self._parse_ticks(data, symbol, year, month, day, hour)
            
            # Cache the result
            if use_cache and ticks is not None and len(ticks) > 0:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                ticks.to_parquet(cache_file, index=False)
            
            return ticks
            
        except requests.RequestException as e:
            logger.debug(f"Download failed: {url} - {e}")
            return None
    
    def _parse_ticks(
        self,
        data: bytes,
        symbol: str,
        year: int,
        month: int,
        day: int,
        hour: int,
    ) -> pd.DataFrame:
        """Parse binary tick data from Dukascopy."""
        
        # Each tick is 20 bytes:
        # - 4 bytes: milliseconds from hour start (uint32)
        # - 4 bytes: ask price (uint32, needs scaling)
        # - 4 bytes: bid price (uint32, needs scaling)
        # - 4 bytes: ask volume (float32)
        # - 4 bytes: bid volume (float32)
        
        tick_size = 20
        n_ticks = len(data) // tick_size
        
        if n_ticks == 0:
            return None
        
        pip = INSTRUMENTS[symbol]['pip']
        base_time = datetime(year, month, day, hour)
        
        ticks = []
        
        for i in range(n_ticks):
            offset = i * tick_size
            chunk = data[offset:offset + tick_size]
            
            if len(chunk) < tick_size:
                break
            
            # Unpack binary data (big-endian)
            ms, ask_int, bid_int, ask_vol, bid_vol = struct.unpack('>IIIff', chunk)
            
            # Convert to actual values
            timestamp = base_time + timedelta(milliseconds=ms)
            ask = ask_int * pip
            bid = bid_int * pip
            
            ticks.append({
                'timestamp': timestamp,
                'bid': bid,
                'ask': ask,
                'bid_volume': bid_vol,
                'ask_volume': ask_vol,
            })
        
        return pd.DataFrame(ticks)
    
    def save_parquet(self, df: pd.DataFrame, path: str) -> str:
        """Save DataFrame to Parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return str(path)


# Convenience function
def load_forex_ticks(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Quick function to load forex tick data."""
    loader = DukascopyLoader()
    return loader.load_ticks(symbol, start_date, end_date)


def load_forex_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1H',
) -> pd.DataFrame:
    """Quick function to load forex OHLCV data."""
    loader = DukascopyLoader()
    return loader.load_ohlcv(symbol, start_date, end_date, timeframe)
