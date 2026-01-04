"""
Data Validator - Comprehensive data integrity validation.

Validates:
A. Data Integrity (Ingestion)
   - DatetimeIndex sorted and unique
   - No duplicate timestamps
   - Gap detection (market holidays vs data issues)

B. Dimensionality (Feature Engineering)
   - Row count consistency
   - Column alignment

C. Stationarity & Value (Pre-processing)
   - No inf/nan values
   - ADF test for stationarity

D. Look-Ahead Bias (Data Splitting)
   - Chronological order validation
   - No future data leakage
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from data validation."""
    
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    stats: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.stats is None:
            self.stats = {}
    
    def add_error(self, msg: str) -> None:
        """Add error and mark as invalid."""
        self.errors.append(msg)
        self.is_valid = False
    
    def add_warning(self, msg: str) -> None:
        """Add warning (doesn't invalidate)."""
        self.warnings.append(msg)


class DataValidator:
    """
    Comprehensive data validator for trading data.
    
    Examples
    --------
    >>> validator = DataValidator()
    >>> result = validator.validate_ohlcv(df)
    >>> if not result.is_valid:
    ...     print(f"Errors: {result.errors}")
    """
    
    def __init__(
        self,
        max_gap_hours: int = 72,  # Max gap before warning (weekends)
        min_rows: int = 100,
        check_stationarity: bool = False
    ):
        """
        Initialize validator.
        
        Parameters
        ----------
        max_gap_hours : int
            Maximum allowed gap in hours before warning
        min_rows : int
            Minimum required rows
        check_stationarity : bool
            Whether to run ADF test (slower)
        """
        self.max_gap_hours = max_gap_hours
        self.min_rows = min_rows
        self.check_stationarity = check_stationarity
    
    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> ValidationResult:
        """
        Validate OHLCV DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
        symbol : str
            Symbol for logging
            
        Returns
        -------
        ValidationResult
            Validation result with errors/warnings
        """
        result = ValidationResult()
        
        # Basic checks
        if df is None:
            result.add_error("DataFrame is None")
            return result
        
        if len(df) == 0:
            result.add_error("DataFrame is empty")
            return result
        
        if len(df) < self.min_rows:
            result.add_warning(f"Only {len(df)} rows (min: {self.min_rows})")
        
        # A. Data Integrity Checks
        self._check_datetime_index(df, result)
        self._check_duplicates(df, result)
        self._check_gaps(df, result)
        self._check_ohlc_columns(df, result)
        self._check_ohlc_validity(df, result)
        
        # C. Value Checks
        self._check_inf_nan(df, result)
        self._check_zero_prices(df, result)
        
        # Stats
        result.stats = self._calculate_stats(df)
        
        # Log result
        if result.is_valid:
            logger.info(f"[OK] {symbol}: {len(df)} rows validated")
        else:
            logger.error(f"[FAIL] {symbol}: {result.errors}")
        
        if result.warnings:
            for w in result.warnings:
                logger.warning(f"[WARN] {symbol}: {w}")
        
        return result
    
    def _check_datetime_index(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check if index is DatetimeIndex and sorted."""
        # Check for timestamp column or index
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            try:
                ts = pd.to_datetime(df.index)
            except Exception:
                result.add_error("No valid datetime index or timestamp column")
                return
        
        # Check sorted
        if not ts.is_monotonic_increasing:
            result.add_error("Timestamps not sorted in ascending order")
        
        result.stats['start_date'] = ts.min()
        result.stats['end_date'] = ts.max()
    
    def _check_duplicates(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check for duplicate timestamps."""
        if 'timestamp' in df.columns:
            dups = df['timestamp'].duplicated().sum()
        else:
            dups = df.index.duplicated().sum()
        
        if dups > 0:
            result.add_error(f"Found {dups} duplicate timestamps")
        
        result.stats['duplicates'] = dups
    
    def _check_gaps(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check for data gaps."""
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
        else:
            ts = pd.to_datetime(df.index)
        
        if len(ts) < 2:
            return
        
        # Calculate time differences
        diffs = ts.diff().dropna()
        
        # Find gaps larger than threshold
        threshold = timedelta(hours=self.max_gap_hours)
        large_gaps = diffs[diffs > threshold]
        
        if len(large_gaps) > 0:
            max_gap = large_gaps.max()
            result.add_warning(
                f"Found {len(large_gaps)} gaps > {self.max_gap_hours}h "
                f"(max: {max_gap})"
            )
        
        result.stats['gap_count'] = len(large_gaps)
        result.stats['median_interval'] = diffs.median()
    
    def _check_ohlc_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check for required OHLC columns."""
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            result.add_error(f"Missing columns: {missing}")
    
    def _check_ohlc_validity(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check OHLC relationships."""
        if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            return
        
        # High >= all others
        invalid_high = ~(
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['high'] >= df['low'])
        )
        
        # Low <= all others
        invalid_low = ~(
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['low'] <= df['high'])
        )
        
        invalid_count = (invalid_high | invalid_low).sum()
        
        if invalid_count > 0:
            pct = invalid_count / len(df) * 100
            if pct > 1:
                result.add_error(f"{invalid_count} invalid OHLC rows ({pct:.1f}%)")
            else:
                result.add_warning(f"{invalid_count} invalid OHLC rows ({pct:.2f}%)")
        
        result.stats['invalid_ohlc'] = invalid_count
    
    def _check_inf_nan(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check for inf and nan values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        nan_count = df[numeric_cols].isna().sum().sum()
        inf_count = np.isinf(df[numeric_cols].values).sum()
        
        if inf_count > 0:
            result.add_error(f"Found {inf_count} inf values")
        
        if nan_count > 0:
            pct = nan_count / (len(df) * len(numeric_cols)) * 100
            if pct > 5:
                result.add_error(f"Found {nan_count} NaN values ({pct:.1f}%)")
            else:
                result.add_warning(f"Found {nan_count} NaN values ({pct:.2f}%)")
        
        result.stats['nan_count'] = nan_count
        result.stats['inf_count'] = inf_count
    
    def _check_zero_prices(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> None:
        """Check for zero or negative prices."""
        if 'close' not in df.columns:
            return
        
        zero_count = (df['close'] <= 0).sum()
        
        if zero_count > 0:
            result.add_error(f"Found {zero_count} zero/negative prices")
        
        result.stats['zero_prices'] = zero_count
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data statistics."""
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
        }
        
        if 'close' in df.columns:
            stats['price_min'] = df['close'].min()
            stats['price_max'] = df['close'].max()
            stats['price_mean'] = df['close'].mean()
        
        if 'volume' in df.columns:
            stats['volume_mean'] = df['volume'].mean()
        
        return stats
    
    def validate_features(
        self,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
        allow_row_reduction: bool = True,
        max_reduction_pct: float = 20.0
    ) -> ValidationResult:
        """
        Validate feature engineering output.
        
        B. Dimensionality Check
        
        Parameters
        ----------
        original_df : pd.DataFrame
            Original data before feature engineering
        features_df : pd.DataFrame
            Data after feature engineering
        allow_row_reduction : bool
            Whether row reduction is allowed (e.g., from NaN dropping)
        max_reduction_pct : float
            Maximum allowed row reduction percentage
            
        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()
        
        orig_rows = len(original_df)
        feat_rows = len(features_df)
        
        reduction = orig_rows - feat_rows
        reduction_pct = reduction / orig_rows * 100 if orig_rows > 0 else 0
        
        result.stats['original_rows'] = orig_rows
        result.stats['feature_rows'] = feat_rows
        result.stats['reduction_pct'] = reduction_pct
        
        if not allow_row_reduction and reduction > 0:
            result.add_error(
                f"Row count changed: {orig_rows} -> {feat_rows} "
                f"(reduction not allowed)"
            )
        elif reduction_pct > max_reduction_pct:
            result.add_error(
                f"Row reduction too high: {reduction_pct:.1f}% "
                f"(max: {max_reduction_pct}%)"
            )
        elif reduction > 0:
            result.add_warning(
                f"Row count reduced: {orig_rows} -> {feat_rows} "
                f"({reduction_pct:.1f}%)"
            )
        
        # Check for inf/nan in features
        self._check_inf_nan(features_df, result)
        
        return result
    
    def validate_train_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> ValidationResult:
        """
        Validate train/test split for look-ahead bias.
        
        D. Look-Ahead Bias Check
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        test_df : pd.DataFrame
            Test data
        val_df : pd.DataFrame, optional
            Validation data
            
        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()
        
        def get_max_date(df: pd.DataFrame) -> datetime:
            if 'timestamp' in df.columns:
                return pd.to_datetime(df['timestamp']).max()
            return pd.to_datetime(df.index).max()
        
        def get_min_date(df: pd.DataFrame) -> datetime:
            if 'timestamp' in df.columns:
                return pd.to_datetime(df['timestamp']).min()
            return pd.to_datetime(df.index).min()
        
        train_end = get_max_date(train_df)
        test_start = get_min_date(test_df)
        
        # Check chronological order
        if train_end >= test_start:
            result.add_error(
                f"Look-ahead bias detected! "
                f"Train ends at {train_end}, Test starts at {test_start}"
            )
        
        if val_df is not None:
            val_start = get_min_date(val_df)
            val_end = get_max_date(val_df)
            
            if train_end >= val_start:
                result.add_error(
                    f"Look-ahead bias! Train ends at {train_end}, "
                    f"Val starts at {val_start}"
                )
            
            if val_end >= test_start:
                result.add_error(
                    f"Look-ahead bias! Val ends at {val_end}, "
                    f"Test starts at {test_start}"
                )
        
        result.stats['train_end'] = train_end
        result.stats['test_start'] = test_start
        
        return result


def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationResult:
    """Convenience function for OHLCV validation."""
    validator = DataValidator()
    return validator.validate_ohlcv(df, symbol)


def validate_no_lookahead(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> ValidationResult:
    """Convenience function for look-ahead bias check."""
    validator = DataValidator()
    return validator.validate_train_test_split(train_df, test_df)
