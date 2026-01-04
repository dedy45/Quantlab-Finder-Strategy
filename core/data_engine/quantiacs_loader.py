"""
Quantiacs Data Loader - Load REAL data from Quantiacs ONLY.

Installation (via conda):
    conda install -c quantiacs-source qnt

Reference: https://quantiacs.com/documentation/en/user_guide/local_development.html

NO FALLBACK - Quantiacs data only for credibility.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Quantiacs API Key
QUANTIACS_API_KEY = os.getenv('API_KEY', '8db51257-9dc3-4297-8c48-034a1f5ff9b5')


def setup_quantiacs_api(api_key: str = None):
    """Setup Quantiacs API key in environment."""
    key = api_key or QUANTIACS_API_KEY
    os.environ['API_KEY'] = key
    logger.info("Quantiacs API key configured")


def check_qnt_available() -> bool:
    """Check if qnt library is available."""
    try:
        import qnt.data
        return True
    except ImportError:
        return False


class QuantiacsLoader:
    """
    Load REAL data from Quantiacs.
    
    NO FALLBACK - Only Quantiacs data for credibility.
    
    Installation:
        conda install -c quantiacs-source qnt
    
    Examples
    --------
    >>> loader = QuantiacsLoader()
    >>> df = loader.load_futures(['F_GC', 'F_ES'], min_date='2015-01-01')
    """
    
    # Available futures assets
    FUTURES_ASSETS = [
        'F_GC',   # Gold
        'F_ES',   # S&P 500 E-mini
        'F_CL',   # Crude Oil
        'F_NG',   # Natural Gas
        'F_NQ',   # NASDAQ-100 E-mini
        'F_SI',   # Silver
        'F_HG',   # Copper
        'F_ZN',   # 10-Year Treasury
        'F_ZB',   # 30-Year Treasury
        'F_YM',   # Mini Dow
        'F_BC',   # Brent Crude
        'F_AX',   # DAX
    ]
    
    def __init__(self, api_key: str = None):
        """Initialize loader with API key."""
        self.api_key = api_key or QUANTIACS_API_KEY
        os.environ['API_KEY'] = self.api_key
        
        if not check_qnt_available():
            raise ImportError(
                "qnt library not installed!\n"
                "Install with: conda install -c quantiacs-source qnt"
            )
        
        logger.info("QuantiacsLoader initialized with REAL data source")
    
    def load_futures(
        self,
        assets: List[str] = None,
        min_date: str = '2015-01-01',
        max_date: str = None,
        tail: int = None,
    ) -> pd.DataFrame:
        """
        Load futures data from Quantiacs.
        
        Parameters
        ----------
        assets : List[str]
            Futures symbols (e.g., ['F_GC', 'F_ES'])
        min_date : str
            Start date (YYYY-MM-DD)
        max_date : str, optional
            End date
        tail : int, optional
            Number of days from end
            
        Returns
        -------
        pd.DataFrame
            OHLCV data with columns: timestamp, open, high, low, close, volume, symbol
        """
        import qnt.data as qndata
        
        if assets is None:
            assets = ['F_GC', 'F_ES', 'F_CL']
        
        logger.info(f"Loading Quantiacs futures: {assets}")
        
        kwargs = {'assets': assets}
        if tail:
            kwargs['tail'] = tail
        else:
            kwargs['min_date'] = min_date
            if max_date:
                kwargs['max_date'] = max_date
        
        data = qndata.futures.load_data(**kwargs)
        return self._xarray_to_dataframe(data)
    
    def load_futures_xarray(
        self,
        assets: List[str] = None,
        min_date: str = '2015-01-01',
        tail: int = None,
    ):
        """
        Load futures data as xarray (native Quantiacs format).
        
        Use this for backtesting with qnt.backtester.
        """
        import qnt.data as qndata
        
        if assets is None:
            assets = ['F_GC', 'F_ES', 'F_CL']
        
        kwargs = {'assets': assets}
        if tail:
            kwargs['tail'] = tail
        else:
            kwargs['min_date'] = min_date
        
        return qndata.futures.load_data(**kwargs)
    
    def load_crypto(
        self,
        assets: List[str] = None,
        tail: int = None,
        min_date: str = None,
    ):
        """
        Load crypto data from Quantiacs.
        
        Available assets: BTC, ETH, and 70+ other cryptos.
        """
        import qnt.data as qndata
        
        kwargs = {}
        if assets:
            kwargs['assets'] = assets
        if tail:
            kwargs['tail'] = tail
        if min_date:
            kwargs['min_date'] = min_date
        
        logger.info(f"Loading Quantiacs crypto data")
        return qndata.cryptodaily.load_data(**kwargs)
    
    def load_stocks(
        self,
        market: str = 'nasdaq100',
        tail: int = None,
        min_date: str = None,
    ):
        """
        Load stock data from Quantiacs.
        
        Parameters
        ----------
        market : str
            'nasdaq100' or 'sp500'
        """
        import qnt.data as qndata
        
        kwargs = {}
        if tail:
            kwargs['tail'] = tail
        if min_date:
            kwargs['min_date'] = min_date
        
        logger.info(f"Loading Quantiacs {market} stocks")
        
        if market == 'nasdaq100':
            return qndata.stocks.load_ndx_data(**kwargs)
        elif market == 'sp500':
            return qndata.stocks.load_spx_data(**kwargs)
        else:
            raise ValueError(f"Unknown market: {market}. Use 'nasdaq100' or 'sp500'")
    
    def list_futures_assets(self) -> List[dict]:
        """List available futures assets."""
        import urllib.request
        import json
        
        url = 'https://data-api.quantiacs.io/futures/list'
        headers = {'API-Key': self.api_key}
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    
    def _xarray_to_dataframe(self, data) -> pd.DataFrame:
        """Convert Quantiacs xarray to DataFrame."""
        records = []
        
        for asset in data.asset.values:
            asset_data = data.sel(asset=asset)
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data.time.values),
                'open': asset_data.sel(field='open').values,
                'high': asset_data.sel(field='high').values,
                'low': asset_data.sel(field='low').values,
                'close': asset_data.sel(field='close').values,
                'volume': asset_data.sel(field='vol').values if 'vol' in data.field.values else np.nan,
                'symbol': asset,
            })
            
            df = df.dropna(subset=['close'])
            records.append(df)
        
        if not records:
            return pd.DataFrame()
        
        return pd.concat(records, ignore_index=True).sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    def save_parquet(self, df: pd.DataFrame, path: str) -> str:
        """Save to Parquet format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return str(path)
    
    def load_parquet(self, path: str) -> pd.DataFrame:
        """Load from Parquet format."""
        return pd.read_parquet(path, engine='pyarrow')


# Convenience functions
def load_futures(
    assets: List[str] = None,
    min_date: str = '2015-01-01',
) -> pd.DataFrame:
    """Quick function to load futures data."""
    loader = QuantiacsLoader()
    return loader.load_futures(assets, min_date)
