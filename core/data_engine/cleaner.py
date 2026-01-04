"""
Data Cleaner - Clean and preprocess raw data.

Handles:
- Missing value imputation
- Outlier detection and removal
- Weekend/holiday filtering
- Data normalization
- Survivorship bias awareness
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats

from .base import DataConfig, DataResult, BaseDataHandler, AssetClass

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    rows_before: int
    rows_after: int
    missing_filled: int
    outliers_removed: int
    duplicates_removed: int
    gaps_filled: int
    
    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after
    
    @property
    def removal_pct(self) -> float:
        if self.rows_before == 0:
            return 0.0
        return self.rows_removed / self.rows_before * 100


class DataCleaner(BaseDataHandler):
    """
    Clean and preprocess raw market data.
    
    Examples
    --------
    >>> cleaner = DataCleaner(config)
    >>> result = cleaner.clean(raw_data)
    >>> print(result.data.head())
    """
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.report: Optional[CleaningReport] = None
    
    def validate(self) -> bool:
        """Validate cleaning can be performed."""
        return True
    
    def execute(self, data: pd.DataFrame) -> DataResult:
        """Execute cleaning pipeline."""
        return self.clean(data)
    
    def clean(self, df: pd.DataFrame) -> DataResult:
        """
        Main cleaning pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data with columns: timestamp, open, high, low, close, volume, symbol
            
        Returns
        -------
        DataResult
            Cleaned data with quality metrics
        """
        rows_before = len(df)
        missing_filled = 0
        outliers_removed = 0
        duplicates_removed = 0
        gaps_filled = 0
        
        # 1. Ensure timestamp is datetime
        df = self._ensure_datetime(df)
        
        # 2. Sort by timestamp and symbol
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # 3. Remove duplicates
        dup_count = df.duplicated(subset=['timestamp', 'symbol']).sum()
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        duplicates_removed = dup_count
        
        # 4. Remove weekends for non-crypto
        if self.config.remove_weekends and self.config.asset_class != AssetClass.CRYPTO:
            df = self._remove_weekends(df)
        
        # 5. Handle missing values
        df, filled = self._handle_missing(df)
        missing_filled = filled
        
        # 6. Remove outliers
        df, removed = self._remove_outliers(df)
        outliers_removed = removed
        
        # 7. Validate OHLC consistency
        df = self._validate_ohlc(df)
        
        # 8. Fill gaps in time series
        df, gaps = self._fill_time_gaps(df)
        gaps_filled = gaps
        
        # Create report
        self.report = CleaningReport(
            rows_before=rows_before,
            rows_after=len(df),
            missing_filled=missing_filled,
            outliers_removed=outliers_removed,
            duplicates_removed=duplicates_removed,
            gaps_filled=gaps_filled,
        )
        
        logger.info(f"Cleaning complete: {self.report.rows_removed} rows removed "
                   f"({self.report.removal_pct:.2f}%)")
        
        # Create result
        result = DataResult(
            data=df,
            source=self.config.source,
            symbols=df['symbol'].unique().tolist() if 'symbol' in df.columns else [],
            start_date=df['timestamp'].min(),
            end_date=df['timestamp'].max(),
            resolution=self.config.resolution,
            missing_pct=df.isnull().sum().sum() / df.size if df.size > 0 else 0,
            outliers_removed=outliers_removed,
            metadata={'cleaning_report': self.report.__dict__},
        )
        
        return result
    
    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamp column is datetime type."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _remove_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove weekend data (Saturday=5, Sunday=6)."""
        if 'timestamp' not in df.columns:
            return df
        
        weekday = df['timestamp'].dt.dayofweek
        return df[weekday < 5].copy()
    
    def _handle_missing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Handle missing values based on config."""
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        existing_cols = [c for c in numeric_cols if c in df.columns]
        
        missing_before = df[existing_cols].isnull().sum().sum()
        
        if self.config.fill_method == 'ffill':
            # Forward fill within each symbol
            df[existing_cols] = df.groupby('symbol')[existing_cols].ffill()
            # Backward fill remaining (start of series)
            df[existing_cols] = df.groupby('symbol')[existing_cols].bfill()
        
        elif self.config.fill_method == 'interpolate':
            df[existing_cols] = df.groupby('symbol')[existing_cols].transform(
                lambda x: x.interpolate(method='linear')
            )
        
        elif self.config.fill_method == 'drop':
            df = df.dropna(subset=existing_cols)
        
        missing_after = df[existing_cols].isnull().sum().sum()
        filled = missing_before - missing_after
        
        return df, filled
    
    def _remove_outliers(
        self, 
        df: pd.DataFrame, 
        z_threshold: float = 5.0,
        method: str = 'zscore'
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove outliers from price data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        z_threshold : float
            Z-score threshold for outlier detection
        method : str
            'zscore' or 'iqr'
            
        Returns
        -------
        Tuple[pd.DataFrame, int]
            Cleaned data and count of removed outliers
        """
        if 'close' not in df.columns:
            return df, 0
        
        rows_before = len(df)
        
        if method == 'zscore':
            # Calculate returns for outlier detection
            df['_returns'] = df.groupby('symbol')['close'].pct_change()
            
            # Z-score of returns
            df['_zscore'] = df.groupby('symbol')['_returns'].transform(
                lambda x: np.abs(stats.zscore(x, nan_policy='omit'))
            )
            
            # Remove extreme outliers
            df = df[(df['_zscore'] < z_threshold) | df['_zscore'].isna()]
            
            # Clean up temp columns
            df = df.drop(columns=['_returns', '_zscore'])
        
        elif method == 'iqr':
            # IQR method per symbol
            def remove_iqr_outliers(group):
                returns = group['close'].pct_change()
                q1 = returns.quantile(0.25)
                q3 = returns.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                mask = (returns >= lower) & (returns <= upper) | returns.isna()
                return group[mask]
            
            df = df.groupby('symbol', group_keys=False).apply(remove_iqr_outliers)
        
        removed = rows_before - len(df)
        return df.reset_index(drop=True), removed
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC consistency.
        
        Rules:
        - High >= max(Open, Close)
        - Low <= min(Open, Close)
        - High >= Low
        - All prices > 0
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required_cols):
            return df
        
        # Fix inconsistencies
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Remove rows with non-positive prices
        price_cols = ['open', 'high', 'low', 'close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        return df
    
    def _fill_time_gaps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Fill gaps in time series.
        
        For daily data, fills missing trading days.
        """
        if 'timestamp' not in df.columns or 'symbol' not in df.columns:
            return df, 0
        
        gaps_filled = 0
        filled_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if len(symbol_df) < 2:
                filled_dfs.append(symbol_df)
                continue
            
            # Create complete date range
            date_range = pd.date_range(
                start=symbol_df['timestamp'].min(),
                end=symbol_df['timestamp'].max(),
                freq='D' if self.config.resolution.value == '1d' else 'h'
            )
            
            # Remove weekends if needed
            if self.config.remove_weekends and self.config.asset_class != AssetClass.CRYPTO:
                date_range = date_range[date_range.dayofweek < 5]
            
            # Reindex to fill gaps
            symbol_df = symbol_df.set_index('timestamp')
            original_len = len(symbol_df)
            
            symbol_df = symbol_df.reindex(date_range)
            symbol_df['symbol'] = symbol
            
            # Forward fill prices
            price_cols = ['open', 'high', 'low', 'close']
            existing_price_cols = [c for c in price_cols if c in symbol_df.columns]
            symbol_df[existing_price_cols] = symbol_df[existing_price_cols].ffill()
            
            # Fill volume with 0 for gaps
            if 'volume' in symbol_df.columns:
                symbol_df['volume'] = symbol_df['volume'].fillna(0)
            
            gaps_filled += len(symbol_df) - original_len
            
            symbol_df = symbol_df.reset_index().rename(columns={'index': 'timestamp'})
            filled_dfs.append(symbol_df)
        
        return pd.concat(filled_dfs, ignore_index=True), gaps_filled


def clean_data(
    df: pd.DataFrame,
    fill_method: str = 'ffill',
    remove_weekends: bool = True,
    outlier_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Convenience function for quick data cleaning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    fill_method : str
        Method for filling missing values
    remove_weekends : bool
        Whether to remove weekend data
    outlier_threshold : float
        Z-score threshold for outlier removal
        
    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    from .base import DataSource, AssetClass
    
    config = DataConfig(
        source=DataSource.LOCAL,
        asset_class=AssetClass.EQUITY,
        symbols=[],
        start_date='2000-01-01',
        fill_method=fill_method,
        remove_weekends=remove_weekends,
    )
    
    cleaner = DataCleaner(config)
    result = cleaner.clean(df)
    return result.data
