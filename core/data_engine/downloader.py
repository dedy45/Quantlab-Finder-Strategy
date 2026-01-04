"""
Data Downloader - Download data from various sources.

Supported sources:
- Quantiacs: Futures data (F_GC, F_ES, F_CL, F_BTC, etc.)
- Binance Vision: Crypto OHLCV data
- Tiingo: US Equities (adjusted prices)
- Yahoo Finance: Backup source for equities
- FRED: Macro economic indicators
"""

import os
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from .base import (
    DataSource, DataConfig, DataResult, BaseDataHandler,
    Resolution, AssetClass, COLUMN_MAPPINGS
)

logger = logging.getLogger(__name__)


class DataDownloader(BaseDataHandler):
    """
    Universal data downloader for multiple sources.
    
    Examples
    --------
    >>> config = DataConfig(
    ...     source=DataSource.QUANTIACS,
    ...     asset_class=AssetClass.FUTURES,
    ...     symbols=['F_GC', 'F_ES'],
    ...     start_date='2015-01-01'
    ... )
    >>> downloader = DataDownloader(config)
    >>> result = downloader.execute()
    >>> print(result.data.head())
    """
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self._handlers = {
            DataSource.QUANTIACS: self._download_quantiacs,
            DataSource.BINANCE: self._download_binance,
            DataSource.TIINGO: self._download_tiingo,
            DataSource.YAHOO: self._download_yahoo,
            DataSource.FRED: self._download_fred,
            DataSource.LOCAL: self._load_local,
        }
    
    def validate(self) -> bool:
        """Validate download can be performed."""
        if self.config.source not in self._handlers:
            logger.error(f"Unsupported source: {self.config.source}")
            return False
        
        if not self.config.symbols:
            logger.error("No symbols specified")
            return False
        
        # Check API keys for sources that require them
        if self.config.source == DataSource.TIINGO:
            if 'TIINGO_API_KEY' not in self.config.api_keys:
                api_key = os.getenv('TIINGO_API_KEY')
                if not api_key:
                    logger.error("TIINGO_API_KEY not found")
                    return False
                self.config.api_keys['TIINGO_API_KEY'] = api_key
        
        return True
    
    def execute(self) -> DataResult:
        """Execute download from configured source."""
        if not self.validate():
            raise ValueError("Validation failed. Check logs for details.")
        
        handler = self._handlers[self.config.source]
        df = handler()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Create result
        result = DataResult(
            data=df,
            source=self.config.source,
            symbols=self.config.symbols,
            start_date=df['timestamp'].min() if 'timestamp' in df.columns else datetime.now(),
            end_date=df['timestamp'].max() if 'timestamp' in df.columns else datetime.now(),
            resolution=self.config.resolution,
            missing_pct=df.isnull().sum().sum() / df.size if df.size > 0 else 0,
        )
        
        logger.info(f"Downloaded {len(df)} rows from {self.config.source.value}")
        return result
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to internal format."""
        mapping = COLUMN_MAPPINGS.get(self.config.source, {})
        df = df.rename(columns=mapping)
        return df
    
    # =========================================================================
    # QUANTIACS DOWNLOADER
    # =========================================================================
    def _download_quantiacs(self) -> pd.DataFrame:
        """
        Download futures data from Quantiacs.
        
        Requires: pip install qnt
        """
        try:
            import qnt.data as qndata
        except ImportError:
            logger.warning("qnt not installed. Using fallback method.")
            return self._download_quantiacs_fallback()
        
        # Load futures data
        data = qndata.futures.load_data(
            assets=self.config.symbols,
            min_date=self.config.start_date,
            max_date=self.config.end_date,
        )
        
        # Convert xarray to DataFrame
        records = []
        for asset in data.asset.values:
            asset_data = data.sel(asset=asset)
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data.time.values),
                'open': asset_data.sel(field='open').values,
                'high': asset_data.sel(field='high').values,
                'low': asset_data.sel(field='low').values,
                'close': asset_data.sel(field='close').values,
                'volume': asset_data.sel(field='vol').values,
                'symbol': asset,
            })
            records.append(df)
        
        return pd.concat(records, ignore_index=True)
    
    def _download_quantiacs_fallback(self) -> pd.DataFrame:
        """Fallback method if qnt is not installed."""
        logger.info("Using Yahoo Finance as fallback for futures data")
        
        # Map Quantiacs symbols to Yahoo tickers
        symbol_map = {
            'F_GC': 'GC=F',  # Gold
            'F_ES': 'ES=F',  # S&P 500 E-mini
            'F_CL': 'CL=F',  # Crude Oil
            'F_NG': 'NG=F',  # Natural Gas
            'F_ZN': 'ZN=F',  # 10-Year Treasury
        }
        
        yahoo_symbols = [symbol_map.get(s, s) for s in self.config.symbols]
        self.config.symbols = yahoo_symbols
        return self._download_yahoo()
    
    # =========================================================================
    # BINANCE DOWNLOADER
    # =========================================================================
    def _download_binance(self) -> pd.DataFrame:
        """
        Download crypto data from Binance Vision.
        
        Data source: https://data.binance.vision/
        """
        all_data = []
        
        for symbol in tqdm(self.config.symbols, desc="Downloading Binance"):
            df = self._download_binance_symbol(symbol)
            if df is not None and len(df) > 0:
                df['symbol'] = symbol
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def _download_binance_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download single symbol from Binance Vision."""
        interval = self._resolution_to_binance(self.config.resolution)
        base_url = "https://data.binance.vision/data/spot/monthly/klines"
        
        # Parse date range
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date or datetime.now())
        
        all_data = []
        current = start
        
        while current <= end:
            year = current.year
            month = current.month
            
            url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Save and extract ZIP
                    zip_path = Path(self.config.cache_dir) / f"{symbol}_{year}_{month:02d}.zip"
                    zip_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Read CSV from ZIP
                    with zipfile.ZipFile(zip_path) as z:
                        csv_name = z.namelist()[0]
                        df = pd.read_csv(
                            z.open(csv_name),
                            names=['open_time', 'open', 'high', 'low', 'close',
                                   'volume', 'close_time', 'quote_volume',
                                   'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
                        )
                        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                        all_data.append(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
                    
                    # Clean up
                    zip_path.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to download {symbol} {year}-{month}: {e}")
            
            # Move to next month
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)
        
        if not all_data:
            return None
        
        return pd.concat(all_data, ignore_index=True)
    
    def _resolution_to_binance(self, resolution: Resolution) -> str:
        """Convert Resolution enum to Binance interval string."""
        mapping = {
            Resolution.MINUTE_1: '1m',
            Resolution.MINUTE_5: '5m',
            Resolution.MINUTE_15: '15m',
            Resolution.HOUR_1: '1h',
            Resolution.HOUR_4: '4h',
            Resolution.DAILY: '1d',
            Resolution.WEEKLY: '1w',
        }
        return mapping.get(resolution, '1d')
    
    # =========================================================================
    # TIINGO DOWNLOADER
    # =========================================================================
    def _download_tiingo(self) -> pd.DataFrame:
        """Download US equity data from Tiingo API."""
        api_key = self.config.api_keys.get('TIINGO_API_KEY', os.getenv('TIINGO_API_KEY'))
        
        all_data = []
        
        for symbol in tqdm(self.config.symbols, desc="Downloading Tiingo"):
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
            params = {
                'startDate': self.config.start_date,
                'endDate': self.config.end_date or datetime.now().strftime('%Y-%m-%d'),
                'token': api_key,
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data)
                    df['symbol'] = symbol
                    all_data.append(df)
                else:
                    logger.warning(f"Tiingo API error for {symbol}: {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to download {symbol} from Tiingo: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    # =========================================================================
    # YAHOO FINANCE DOWNLOADER
    # =========================================================================
    def _download_yahoo(self) -> pd.DataFrame:
        """Download data from Yahoo Finance (backup source)."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        # Download multiple tickers
        tickers = ' '.join(self.config.symbols)
        df = yf.download(
            tickers,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=True,
        )
        
        # Handle single vs multiple tickers
        if len(self.config.symbols) == 1:
            df['symbol'] = self.config.symbols[0]
            df = df.reset_index()
        else:
            # Reshape multi-ticker data
            records = []
            for symbol in self.config.symbols:
                try:
                    symbol_df = df.xs(symbol, axis=1, level=1) if isinstance(df.columns, pd.MultiIndex) else df
                    symbol_df = symbol_df.reset_index()
                    symbol_df['symbol'] = symbol
                    records.append(symbol_df)
                except KeyError:
                    logger.warning(f"Symbol {symbol} not found in Yahoo data")
            df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        
        return df
    
    # =========================================================================
    # FRED DOWNLOADER
    # =========================================================================
    def _download_fred(self) -> pd.DataFrame:
        """Download macro data from FRED."""
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("fredapi not installed. Run: pip install fredapi")
        
        api_key = self.config.api_keys.get('FRED_API_KEY', os.getenv('FRED_API_KEY'))
        fred = Fred(api_key=api_key)
        
        all_data = []
        
        for series_id in tqdm(self.config.symbols, desc="Downloading FRED"):
            try:
                series = fred.get_series(
                    series_id,
                    observation_start=self.config.start_date,
                    observation_end=self.config.end_date,
                )
                df = pd.DataFrame({
                    'timestamp': series.index,
                    'close': series.values,  # FRED only has single value
                    'symbol': series_id,
                })
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to download {series_id} from FRED: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    # =========================================================================
    # LOCAL FILE LOADER
    # =========================================================================
    def _load_local(self) -> pd.DataFrame:
        """Load data from local files (CSV or Parquet)."""
        all_data = []
        
        for file_path in self.config.symbols:  # symbols = file paths for local
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
            
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix == '.csv':
                df = pd.read_csv(path)
            else:
                logger.warning(f"Unsupported file format: {path.suffix}")
                continue
            
            df['symbol'] = path.stem
            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
