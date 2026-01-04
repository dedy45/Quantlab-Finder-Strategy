"""
Data Validator for FASE 5: Production.

Comprehensive data validation utilities for research pipeline.
Implements all required checks:
- A. Data Integrity (DatetimeIndex, duplicates, gaps)
- B. Dimensionality Check (row count validation)
- C. Stationarity & Value Check (inf, NaN, ADF)
- D. Look-Ahead Bias Check (chronological split)

Version: 0.6.0
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
MIN_DATA_POINTS = 100
MAX_GAP_DAYS = 5  # Maximum allowed gap in business days
MIN_OBSERVATIONS_FOR_ADF = 50


@dataclass
class ValidationReport:
    """Report from data validation."""
    
    is_valid: bool = True
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_pass(self, check: str) -> None:
        """Add passed check."""
        self.checks_passed.append(check)
    
    def add_fail(self, check: str) -> None:
        """Add failed check and mark invalid."""
        self.checks_failed.append(check)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add warning."""
        self.warnings.append(warning)
    
    def summary(self) -> str:
        """Generate summary string."""
        status = "[OK]" if self.is_valid else "[FAIL]"
        return (
            f"{status} Validation: "
            f"{len(self.checks_passed)} passed, "
            f"{len(self.checks_failed)} failed, "
            f"{len(self.warnings)} warnings"
        )


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_gaps: bool = True,
    max_gap_days: int = MAX_GAP_DAYS
) -> ValidationReport:
    """
    Validate DataFrame for data integrity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str], optional
        Required column names
    check_gaps : bool
        Whether to check for data gaps
    max_gap_days : int
        Maximum allowed gap in business days
        
    Returns
    -------
    ValidationReport
        Validation report with results
    """
    assert df is not None, "DataFrame cannot be None"
    
    report = ValidationReport()
    
    # Check 1: Not empty
    if len(df) == 0:
        report.add_fail("DataFrame is empty")
        return report
    report.add_pass("DataFrame not empty")
    
    # Check 2: Minimum data points
    if len(df) < MIN_DATA_POINTS:
        report.add_fail(f"Insufficient data: {len(df)} < {MIN_DATA_POINTS}")
    else:
        report.add_pass(f"Sufficient data: {len(df)} rows")
    
    # Check 3: DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        report.add_fail("Index is not DatetimeIndex")
    else:
        report.add_pass("Index is DatetimeIndex")
    
    # Check 4: Index is sorted
    if not df.index.is_monotonic_increasing:
        report.add_fail("Index is not sorted chronologically")
    else:
        report.add_pass("Index is sorted")
    
    # Check 5: No duplicate timestamps
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        report.add_fail(f"Found {duplicates} duplicate timestamps")
    else:
        report.add_pass("No duplicate timestamps")
    
    # Check 6: Required columns
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            report.add_fail(f"Missing columns: {missing}")
        else:
            report.add_pass("All required columns present")
    
    # Check 7: Data gaps (only if DatetimeIndex)
    if check_gaps and isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        gaps = _check_data_gaps(df.index, max_gap_days)
        if gaps:
            report.add_warning(f"Found {len(gaps)} gaps > {max_gap_days} days")
            report.metrics['n_gaps'] = len(gaps)
        else:
            report.add_pass("No significant data gaps")
    
    # Check 8: No all-NaN columns
    nan_cols = df.columns[df.isna().all()].tolist()
    if nan_cols:
        report.add_fail(f"All-NaN columns: {nan_cols}")
    else:
        report.add_pass("No all-NaN columns")
    
    # Metrics
    report.metrics['n_rows'] = len(df)
    report.metrics['n_cols'] = len(df.columns)
    report.metrics['nan_pct'] = df.isna().sum().sum() / df.size * 100
    
    return report


def _check_data_gaps(
    index: pd.DatetimeIndex,
    max_gap_days: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Check for gaps in DatetimeIndex.
    
    Returns list of (start, end, gap_days) tuples.
    """
    gaps = []
    
    # Calculate differences
    diffs = pd.Series(index).diff()
    
    # Find gaps larger than threshold (accounting for weekends)
    threshold = timedelta(days=max_gap_days)
    
    for i, diff in enumerate(diffs):
        if pd.notna(diff) and diff > threshold:
            gap_days = diff.days
            gaps.append((index[i-1], index[i], gap_days))
    
    return gaps


def validate_returns(
    returns: pd.Series,
    check_stationarity: bool = False
) -> ValidationReport:
    """
    Validate returns series for numerical issues.
    
    Parameters
    ----------
    returns : pd.Series
        Returns series to validate
    check_stationarity : bool
        Whether to run ADF test
        
    Returns
    -------
    ValidationReport
        Validation report
    """
    assert returns is not None, "Returns cannot be None"
    
    report = ValidationReport()
    
    # Check 1: Not empty
    if len(returns) == 0:
        report.add_fail("Returns series is empty")
        return report
    report.add_pass("Returns not empty")
    
    # Check 2: No infinity values
    inf_count = np.isinf(returns).sum()
    if inf_count > 0:
        report.add_fail(f"Found {inf_count} infinity values")
    else:
        report.add_pass("No infinity values")
    
    # Check 3: NaN handling
    nan_count = returns.isna().sum()
    nan_pct = nan_count / len(returns) * 100
    if nan_pct > 10:
        report.add_fail(f"Too many NaN: {nan_pct:.1f}%")
    elif nan_pct > 0:
        report.add_warning(f"Contains {nan_count} NaN ({nan_pct:.1f}%)")
    else:
        report.add_pass("No NaN values")
    
    # Check 4: Reasonable return range
    clean_returns = returns.dropna()
    if len(clean_returns) > 0:
        max_abs_return = clean_returns.abs().max()
        if max_abs_return > 0.5:  # 50% daily return is suspicious
            report.add_warning(f"Extreme return detected: {max_abs_return:.2%}")
        else:
            report.add_pass("Returns in reasonable range")
    
    # Check 5: Not all zeros
    if (clean_returns == 0).all():
        report.add_fail("All returns are zero")
    else:
        report.add_pass("Returns have variance")
    
    # Check 6: Stationarity (optional)
    if check_stationarity and len(clean_returns) >= MIN_OBSERVATIONS_FOR_ADF:
        try:
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(clean_returns, autolag='AIC')
            p_value = adf_result[1]
            
            report.metrics['adf_pvalue'] = p_value
            
            if p_value < 0.05:
                report.add_pass(f"Stationary (ADF p={p_value:.4f})")
            else:
                report.add_warning(f"May not be stationary (ADF p={p_value:.4f})")
                
        except Exception as e:
            report.add_warning(f"ADF test failed: {e}")
    
    # Metrics
    report.metrics['n_observations'] = len(returns)
    report.metrics['nan_count'] = nan_count
    report.metrics['mean'] = clean_returns.mean() if len(clean_returns) > 0 else 0
    report.metrics['std'] = clean_returns.std() if len(clean_returns) > 0 else 0
    
    return report


def validate_dimensionality(
    original_len: int,
    current_len: int,
    operation: str,
    expected_reduction: Optional[int] = None,
    tolerance: int = 5
) -> ValidationReport:
    """
    Validate that dimensionality changes are expected.
    
    Parameters
    ----------
    original_len : int
        Original row count
    current_len : int
        Current row count after operation
    operation : str
        Name of operation performed
    expected_reduction : int, optional
        Expected number of rows to lose
    tolerance : int
        Allowed deviation from expected
        
    Returns
    -------
    ValidationReport
        Validation report
    """
    report = ValidationReport()
    
    actual_reduction = original_len - current_len
    
    if expected_reduction is not None:
        deviation = abs(actual_reduction - expected_reduction)
        if deviation <= tolerance:
            report.add_pass(
                f"{operation}: {original_len} -> {current_len} "
                f"(expected -{expected_reduction})"
            )
        else:
            report.add_fail(
                f"{operation}: unexpected reduction. "
                f"Expected -{expected_reduction}, got -{actual_reduction}"
            )
    else:
        # Just log the change
        if actual_reduction > 0:
            report.add_warning(
                f"{operation}: lost {actual_reduction} rows "
                f"({original_len} -> {current_len})"
            )
        else:
            report.add_pass(f"{operation}: no rows lost")
    
    report.metrics['original_len'] = original_len
    report.metrics['current_len'] = current_len
    report.metrics['reduction'] = actual_reduction
    
    return report


def validate_train_test_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> ValidationReport:
    """
    Validate train/test split for look-ahead bias.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
        
    Returns
    -------
    ValidationReport
        Validation report
    """
    report = ValidationReport()
    
    # Check 1: Both have DatetimeIndex
    if not isinstance(train_data.index, pd.DatetimeIndex):
        report.add_fail("Train data index is not DatetimeIndex")
        return report
    if not isinstance(test_data.index, pd.DatetimeIndex):
        report.add_fail("Test data index is not DatetimeIndex")
        return report
    report.add_pass("Both datasets have DatetimeIndex")
    
    # Check 2: No overlap
    train_max = train_data.index.max()
    test_min = test_data.index.min()
    
    if train_max >= test_min:
        report.add_fail(
            f"Look-ahead bias: train ends {train_max}, "
            f"test starts {test_min}"
        )
    else:
        report.add_pass("No temporal overlap (train < test)")
    
    # Check 3: Chronological order
    if train_data.index.max() < test_data.index.min():
        report.add_pass("Chronological order maintained")
    else:
        report.add_fail("Chronological order violated")
    
    # Metrics
    report.metrics['train_start'] = str(train_data.index.min())
    report.metrics['train_end'] = str(train_data.index.max())
    report.metrics['test_start'] = str(test_data.index.min())
    report.metrics['test_end'] = str(test_data.index.max())
    report.metrics['train_size'] = len(train_data)
    report.metrics['test_size'] = len(test_data)
    
    return report


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    fill_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe division that handles zero denominators.
    
    Parameters
    ----------
    numerator : float, array, or Series
        Numerator
    denominator : float, array, or Series
        Denominator
    fill_value : float
        Value to use when denominator is zero
        
    Returns
    -------
    float, array, or Series
        Result of division
    """
    if isinstance(denominator, (np.ndarray, pd.Series)):
        result = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, fill_value, dtype=float),
            where=denominator != 0
        )
        return result
    else:
        if denominator == 0:
            logger.warning("Division by zero, returning fill_value")
            return fill_value
        return numerator / denominator


def clean_returns(
    returns: pd.Series,
    remove_inf: bool = True,
    remove_nan: bool = True,
    clip_extreme: Optional[float] = 0.5
) -> pd.Series:
    """
    Clean returns series by removing problematic values.
    
    Parameters
    ----------
    returns : pd.Series
        Raw returns
    remove_inf : bool
        Remove infinity values
    remove_nan : bool
        Remove NaN values
    clip_extreme : float, optional
        Clip returns to +/- this value
        
    Returns
    -------
    pd.Series
        Cleaned returns
    """
    assert returns is not None, "Returns cannot be None"
    
    cleaned = returns.copy()
    original_len = len(cleaned)
    
    # Replace inf with NaN
    if remove_inf:
        inf_mask = np.isinf(cleaned)
        if inf_mask.any():
            logger.warning(f"Replacing {inf_mask.sum()} inf values with NaN")
            cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    
    # Remove NaN
    if remove_nan:
        nan_count = cleaned.isna().sum()
        if nan_count > 0:
            logger.warning(f"Dropping {nan_count} NaN values")
            cleaned = cleaned.dropna()
    
    # Clip extreme values
    if clip_extreme is not None:
        extreme_mask = cleaned.abs() > clip_extreme
        if extreme_mask.any():
            logger.warning(f"Clipping {extreme_mask.sum()} extreme values")
            cleaned = cleaned.clip(-clip_extreme, clip_extreme)
    
    logger.info(
        f"Cleaned returns: {original_len} -> {len(cleaned)} "
        f"({original_len - len(cleaned)} removed)"
    )
    
    return cleaned
