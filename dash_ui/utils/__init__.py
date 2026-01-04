"""
QuantLab Dash UI - Utility Functions.

Modules:
- downsampling: LTTB algorithm for chart performance
- validators: Input validation functions
- data_quality: Statistical tests for data validation
"""

from .downsampling import (
    lttb_downsample,
    lttb_downsample_ohlc,
    auto_downsample,
    NUMBA_AVAILABLE,
)
from .validators import (
    validate_symbol,
    validate_date_range,
    validate_numeric,
    validate_timeframe,
    validate_strategy_params,
)
from .data_quality import (
    validate_data_quality,
    QualityReport,
    CategoryResult,
    TestResult,
    calculate_grade,
    check_completeness,
    check_distribution,
    check_stationarity,
    check_autocorrelation,
    check_outliers,
    check_sample_size,
    check_ohlc_integrity,
)

__all__ = [
    # Downsampling
    'lttb_downsample',
    'lttb_downsample_ohlc',
    'auto_downsample',
    'NUMBA_AVAILABLE',
    # Validators
    'validate_symbol',
    'validate_date_range',
    'validate_numeric',
    'validate_timeframe',
    'validate_strategy_params',
    # Data Quality
    'validate_data_quality',
    'QualityReport',
    'CategoryResult',
    'TestResult',
    'calculate_grade',
    'check_completeness',
    'check_distribution',
    'check_stationarity',
    'check_autocorrelation',
    'check_outliers',
    'check_sample_size',
    'check_ohlc_integrity',
]
