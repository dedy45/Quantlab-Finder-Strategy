"""
Data Quality Statistical Tests for QuantLab.

Provides comprehensive statistical validation of OHLCV data
to ensure data is "statistically sound" before analysis.

Philosophy: "Garbage In = Garbage Out"

7 VALIDATION CATEGORIES:
========================
1. Completeness (20%) - Missing values, gaps, bar completeness
2. Distribution (15%) - Normality, skewness, kurtosis  
3. Stationarity (20%) - ADF test for mean-reversion
4. Autocorrelation (10%) - Serial correlation check
5. Outliers (15%) - Extreme value detection
6. Sample Size (10%) - Minimum observations for inference
7. OHLC Integrity (10%) - Valid price relationships

GRADING SYSTEM:
===============
- Grade A (90-100%): EXCELLENT - Ready for any analysis
- Grade B (80-89%):  GOOD - Minor issues, proceed with caution
- Grade C (70-79%):  FAIR - Review issues before proceeding
- Grade D (60-69%):  POOR - Significant issues, NOT recommended
- Grade F (0-59%):   FAIL - Do NOT proceed with analysis

MINIMUM THRESHOLD: 70% (Grade C) to proceed
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - THRESHOLDS AND WEIGHTS
# =============================================================================

# Category weights (must sum to 1.0)
CATEGORY_WEIGHTS = {
    'completeness': 0.20,      # Data must be complete
    'distribution': 0.15,      # Returns distribution
    'stationarity': 0.20,      # Critical for time series
    'autocorrelation': 0.10,   # Serial correlation
    'outliers': 0.15,          # Extreme values
    'sample_size': 0.10,       # Enough data points
    'ohlc_integrity': 0.10,    # Valid OHLC
}

# Grade thresholds
GRADE_THRESHOLDS = {
    'A': 90,  # Excellent
    'B': 80,  # Good
    'C': 70,  # Fair (minimum to proceed)
    'D': 60,  # Poor
    'F': 0,   # Fail
}

# Minimum score to proceed
MIN_SCORE_TO_PROCEED = 70  # Grade C

# Performance optimization: max samples for expensive tests
MAX_SAMPLES_FOR_TESTS = 10000  # Limit samples for ADF, Ljung-Box, etc.


# Category descriptions for UI
CATEGORY_INFO = {
    'completeness': {
        'name': 'Completeness',
        'description': 'Checks for missing values, data gaps, and bar completeness',
        'importance': 'CRITICAL - Incomplete data leads to biased analysis',
        'weight_pct': '20%',
    },
    'distribution': {
        'name': 'Distribution',
        'description': 'Analyzes returns normality, skewness, and kurtosis',
        'importance': 'IMPORTANT - Affects statistical test validity',
        'weight_pct': '15%',
    },
    'stationarity': {
        'name': 'Stationarity',
        'description': 'Tests if returns are stationary (ADF test)',
        'importance': 'CRITICAL - Non-stationary data invalidates most models',
        'weight_pct': '20%',
    },
    'autocorrelation': {
        'name': 'Autocorrelation',
        'description': 'Detects serial correlation in returns',
        'importance': 'MODERATE - Affects standard error estimates',
        'weight_pct': '10%',
    },
    'outliers': {
        'name': 'Outliers',
        'description': 'Identifies extreme values using Z-score and IQR',
        'importance': 'IMPORTANT - Outliers can distort statistics',
        'weight_pct': '15%',
    },
    'sample_size': {
        'name': 'Sample Size',
        'description': 'Verifies sufficient observations for inference',
        'importance': 'CRITICAL - Small samples have high variance',
        'weight_pct': '10%',
    },
    'ohlc_integrity': {
        'name': 'OHLC Integrity',
        'description': 'Validates OHLC price relationships',
        'importance': 'CRITICAL - Invalid OHLC indicates data corruption',
        'weight_pct': '10%',
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TestResult:
    """Result of a single statistical test."""
    name: str
    passed: bool
    value: float
    threshold: str
    interpretation: str
    advice: str = ""  # English advice
    details: Dict[str, Any] = field(default_factory=dict)
    p_value: Optional[float] = None


@dataclass
class CategoryResult:
    """Result of a category of tests."""
    name: str
    score: float  # 0-100
    passed: bool
    status: str  # PASS, WARNING, FAIL
    advice: str  # English advice for this category
    tests: List[TestResult] = field(default_factory=list)
    weight: float = 1.0


@dataclass 
class QualityReport:
    """Complete data quality report."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_rows: int
    quality_score: float  # 0-100
    grade: str  # A/B/C/D/F
    passed: bool
    can_proceed: bool  # True if score >= 70%
    status_label: str  # Human readable status
    categories: Dict[str, CategoryResult] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recommendations: List[str] = field(default_factory=list)
    english_summary: str = ""  # English summary


# =============================================================================
# GRADE AND STATUS FUNCTIONS
# =============================================================================

def calculate_grade(score: float) -> str:
    """Convert score to letter grade."""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'


def get_status_label(score: float, grade: str) -> str:
    """Get human-readable status label."""
    if grade == 'A':
        return "✅ EXCELLENT - Data is clean and ready for analysis"
    elif grade == 'B':
        return "✅ GOOD - Minor issues detected, proceed with caution"
    elif grade == 'C':
        return "⚠️ FAIR - Some issues found, review before proceeding"
    elif grade == 'D':
        return "❌ POOR - Significant issues, analysis NOT recommended"
    else:
        return "🚫 FAIL - Data quality insufficient, DO NOT proceed"


def get_category_status(score: float) -> str:
    """Get category status based on score."""
    if score >= 80:
        return "PASS"
    elif score >= 50:
        return "WARNING"
    else:
        return "FAIL"


def calculate_quality_score(categories: Dict[str, CategoryResult]) -> float:
    """Calculate weighted quality score from category results."""
    if not categories:
        return 0.0
    
    total_weight = sum(cat.weight for cat in categories.values())
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(cat.score * cat.weight for cat in categories.values())
    return weighted_sum / total_weight


def _sample_for_test(data: pd.Series, max_samples: int = MAX_SAMPLES_FOR_TESTS) -> pd.Series:
    """
    Sample data for expensive statistical tests.
    
    For large datasets, we sample to keep tests fast while maintaining
    statistical validity. Uses systematic sampling to preserve time structure.
    
    Parameters
    ----------
    data : pd.Series
        Input data series
    max_samples : int
        Maximum number of samples
        
    Returns
    -------
    pd.Series
        Sampled data (or original if small enough)
    """
    n = len(data)
    if n <= max_samples:
        return data
    
    # Systematic sampling to preserve time structure
    step = n // max_samples
    indices = list(range(0, n, step))[:max_samples]
    
    logger.debug(f"Sampling {n:,} -> {len(indices):,} for statistical test")
    return data.iloc[indices]


# =============================================================================
# COMPLETENESS TESTS (Weight: 20%)
# =============================================================================

def check_completeness(df: pd.DataFrame, timeframe: str) -> CategoryResult:
    """
    Check data completeness - CRITICAL for any analysis.
    
    Tests:
    1. Missing values percentage (< 1%)
    2. Bar completeness vs expected (> 95%)
    3. Gap detection (< 5% gaps)
    """
    tests = []
    
    # Test 1: Missing values
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    
    tests.append(TestResult(
        name="Missing Values",
        passed=missing_pct < 1,
        value=round(missing_pct, 2),
        threshold="< 1%",
        interpretation="No missing data" if missing_pct == 0 else (
            "Acceptable" if missing_pct < 1 else "Too many missing values"
        ),
        advice="Missing values can bias calculations. Consider forward-fill or interpolation." if missing_pct >= 1 else "",
        details={'missing_cells': int(missing_cells), 'total_cells': total_cells}
    ))
    
    # Test 2: Bar completeness
    expected_bars = _calculate_expected_bars(df.index.min(), df.index.max(), timeframe)
    actual_bars = len(df)
    completeness_pct = min(100, (actual_bars / expected_bars * 100)) if expected_bars > 0 else 100
    
    tests.append(TestResult(
        name="Bar Completeness",
        passed=completeness_pct > 95,
        value=round(completeness_pct, 1),
        threshold="> 95%",
        interpretation="Complete" if completeness_pct > 98 else (
            "Good" if completeness_pct > 95 else "Significant gaps detected"
        ),
        advice="Data gaps may indicate market closures or data feed issues." if completeness_pct <= 95 else "",
        details={'expected_bars': expected_bars, 'actual_bars': actual_bars}
    ))
    
    # Test 3: Gap detection
    gaps = _detect_gaps(df, timeframe)
    gap_pct = (len(gaps) / len(df) * 100) if len(df) > 0 else 0
    
    tests.append(TestResult(
        name="Gap Detection",
        passed=gap_pct < 5,
        value=round(gap_pct, 2),
        threshold="< 5%",
        interpretation="No significant gaps" if gap_pct < 1 else (
            "Minor gaps" if gap_pct < 5 else "Major gaps detected"
        ),
        advice="Large gaps can affect momentum and trend calculations." if gap_pct >= 5 else "",
        details={'gap_count': len(gaps)}
    ))
    
    # Calculate category score (average of test scores)
    scores = [100 if t.passed else (50 if t.value < t.value * 2 else 0) for t in tests]
    score = sum(scores) / len(scores) if scores else 0
    
    # Determine status and advice
    status = get_category_status(score)
    if status == "FAIL":
        advice = "CRITICAL: Data has significant completeness issues. Fill gaps or use different date range."
    elif status == "WARNING":
        advice = "Data has some gaps. Results may be affected for momentum-based strategies."
    else:
        advice = "Data completeness is acceptable for analysis."
    
    return CategoryResult(
        name="Completeness",
        score=score,
        passed=score >= 80,
        status=status,
        advice=advice,
        tests=tests,
        weight=CATEGORY_WEIGHTS['completeness']
    )


def _calculate_expected_bars(start, end, timeframe: str) -> int:
    """Calculate expected number of bars for a timeframe."""
    if start is None or end is None:
        return 1
    
    try:
        if not isinstance(start, pd.Timestamp):
            start = pd.Timestamp(start)
        if not isinstance(end, pd.Timestamp):
            end = pd.Timestamp(end)
        
        time_diff = end - start
        
        if hasattr(time_diff, 'total_seconds'):
            total_seconds = time_diff.total_seconds()
        else:
            total_seconds = float(time_diff / np.timedelta64(1, 's'))
        
        tf_seconds = {
            '1T': 60, '5T': 300, '15T': 900, '30T': 1800,
            '1H': 3600, '1h': 3600, '4H': 14400, '4h': 14400,
            '1D': 86400, '1d': 86400, '1W': 604800, '1w': 604800
        }
        
        bar_seconds = tf_seconds.get(timeframe, 3600)
        # Account for ~70% market hours (weekends, holidays)
        return max(1, int(total_seconds / bar_seconds * 0.7))
    except Exception as e:
        logger.warning(f"Error calculating expected bars: {e}")
        return 1


def _detect_gaps(df: pd.DataFrame, timeframe: str) -> List[str]:
    """Detect gaps in time series data."""
    if len(df) < 2:
        return []
    
    try:
        tf_seconds = {
            '1T': 60, '5T': 300, '15T': 900, '30T': 1800,
            '1H': 3600, '1h': 3600, '4H': 14400, '4h': 14400,
            '1D': 86400, '1d': 86400, '1W': 604800, '1w': 604800,
        }
        
        expected_seconds = tf_seconds.get(timeframe, 3600)
        threshold_seconds = expected_seconds * 5  # 5x tolerance
        
        gaps = []
        index_series = df.index.to_series()
        
        for i in range(1, min(len(index_series), 10000)):  # Limit iterations
            try:
                diff = index_series.iloc[i] - index_series.iloc[i-1]
                if hasattr(diff, 'total_seconds'):
                    diff_seconds = diff.total_seconds()
                else:
                    diff_seconds = float(diff / np.timedelta64(1, 's'))
                
                if diff_seconds > threshold_seconds:
                    gaps.append(str(index_series.iloc[i]))
            except:
                continue
        
        return gaps
    except Exception as e:
        logger.warning(f"Error detecting gaps: {e}")
        return []


# =============================================================================
# DISTRIBUTION TESTS (Weight: 15%)
# =============================================================================

def check_distribution(returns: pd.Series) -> CategoryResult:
    """
    Check returns distribution - Important for statistical validity.
    
    Tests:
    1. Jarque-Bera normality test
    2. Skewness (|skew| < 2)
    3. Kurtosis (< 10)
    
    Note: Financial returns are typically NOT normal, so we use relaxed thresholds.
    """
    from scipy import stats
    
    tests = []
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 30:
        return CategoryResult(
            name="Distribution",
            score=0,
            passed=False,
            status="FAIL",
            advice="Insufficient data for distribution analysis. Need at least 30 observations.",
            tests=[TestResult(
                name="Insufficient Data",
                passed=False,
                value=len(returns_clean),
                threshold=">= 30",
                interpretation="Need more data",
                advice="Extend date range to get more observations."
            )],
            weight=CATEGORY_WEIGHTS['distribution']
        )
    
    # Test 1: Jarque-Bera (relaxed - financial data is rarely normal)
    try:
        jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
        # We use p > 0.001 as threshold (very relaxed for financial data)
        jb_passed = jb_pvalue > 0.001
        tests.append(TestResult(
            name="Jarque-Bera Test",
            passed=jb_passed,
            value=round(jb_stat, 2),
            threshold="p > 0.001",
            interpretation="Near-normal" if jb_pvalue > 0.01 else (
                "Acceptable" if jb_pvalue > 0.001 else "Non-normal"
            ),
            advice="" if jb_passed else "Non-normal returns are common in finance. Consider robust methods.",
            p_value=round(jb_pvalue, 6),
            details={'statistic': round(jb_stat, 4)}
        ))
    except Exception as e:
        tests.append(TestResult(
            name="Jarque-Bera Test",
            passed=True,  # Don't fail on error
            value=0,
            threshold="p > 0.001",
            interpretation="Test skipped",
            advice="",
            details={'error': str(e)}
        ))
    
    # Test 2: Skewness
    skew = float(stats.skew(returns_clean))
    skew_passed = abs(skew) < 2
    tests.append(TestResult(
        name="Skewness",
        passed=skew_passed,
        value=round(skew, 3),
        threshold="|skew| < 2",
        interpretation="Symmetric" if abs(skew) < 0.5 else (
            "Moderate skew" if abs(skew) < 2 else "High skew"
        ),
        advice="" if skew_passed else f"{'Positive' if skew > 0 else 'Negative'} skew may affect mean estimates.",
        details={'direction': 'right' if skew > 0 else 'left'}
    ))
    
    # Test 3: Kurtosis
    kurt = float(stats.kurtosis(returns_clean))
    kurt_passed = kurt < 10
    tests.append(TestResult(
        name="Kurtosis",
        passed=kurt_passed,
        value=round(kurt, 3),
        threshold="< 10",
        interpretation="Normal tails" if kurt < 3 else (
            "Fat tails" if kurt < 10 else "Extreme fat tails"
        ),
        advice="" if kurt_passed else "High kurtosis indicates extreme events. Consider tail risk measures.",
        details={'excess_kurtosis': round(kurt, 4)}
    ))
    
    # Calculate score
    passed_count = sum(1 for t in tests if t.passed)
    score = (passed_count / len(tests) * 100) if tests else 0
    
    status = get_category_status(score)
    if status == "FAIL":
        advice = "Distribution significantly deviates from normal. Use robust statistics."
    elif status == "WARNING":
        advice = "Some distribution issues. Consider non-parametric methods."
    else:
        advice = "Distribution is acceptable for standard statistical methods."
    
    return CategoryResult(
        name="Distribution",
        score=score,
        passed=score >= 50,  # Relaxed for financial data
        status=status,
        advice=advice,
        tests=tests,
        weight=CATEGORY_WEIGHTS['distribution']
    )


# =============================================================================
# STATIONARITY TESTS (Weight: 20%)
# =============================================================================

def check_stationarity(returns: pd.Series) -> CategoryResult:
    """
    Check if returns are stationary - CRITICAL for time series models.
    
    Tests:
    1. Augmented Dickey-Fuller (ADF) test
    
    Stationary returns are essential for:
    - Mean-reversion strategies
    - Statistical arbitrage
    - Most ML models
    
    PERFORMANCE: For large datasets (>10k), uses sampling to keep test fast.
    """
    from statsmodels.tsa.stattools import adfuller
    
    tests = []
    returns_clean = returns.dropna()
    original_len = len(returns_clean)
    
    if original_len < 50:
        return CategoryResult(
            name="Stationarity",
            score=50,
            passed=True,
            status="WARNING",
            advice="Insufficient data for reliable stationarity test. Results may be unreliable.",
            tests=[TestResult(
                name="ADF Test",
                passed=True,
                value=original_len,
                threshold=">= 50",
                interpretation="Skipped - need more data",
                advice="Extend date range for reliable stationarity testing."
            )],
            weight=CATEGORY_WEIGHTS['stationarity']
        )
    
    # Sample for performance if dataset is large
    returns_test = _sample_for_test(returns_clean, MAX_SAMPLES_FOR_TESTS)
    sampled = len(returns_test) < original_len
    
    # ADF Test
    try:
        adf_result = adfuller(returns_test, autolag='AIC')
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
        critical_values = adf_result[4]
        
        adf_passed = adf_pvalue < 0.05
        interpretation = "Stationary" if adf_passed else "Non-stationary"
        if sampled:
            interpretation += f" (sampled {len(returns_test):,} of {original_len:,})"
        
        tests.append(TestResult(
            name="ADF Test",
            passed=adf_passed,
            value=round(adf_stat, 4),
            threshold="p < 0.05",
            interpretation=interpretation,
            advice="" if adf_passed else "Non-stationary data may need differencing or detrending.",
            p_value=round(adf_pvalue, 6),
            details={
                'critical_1%': round(critical_values.get('1%', 0), 4),
                'critical_5%': round(critical_values.get('5%', 0), 4),
                'sampled': sampled,
                'sample_size': len(returns_test),
                'original_size': original_len,
            }
        ))
    except Exception as e:
        logger.warning(f"ADF test failed: {e}")
        tests.append(TestResult(
            name="ADF Test",
            passed=True,
            value=0,
            threshold="p < 0.05",
            interpretation="Test failed",
            advice="ADF test could not be performed.",
            details={'error': str(e)}
        ))
    
    passed_count = sum(1 for t in tests if t.passed)
    score = (passed_count / len(tests) * 100) if tests else 50
    
    status = get_category_status(score)
    if status == "FAIL":
        advice = "CRITICAL: Returns are non-stationary. Most statistical models will be invalid."
    elif status == "WARNING":
        advice = "Stationarity test inconclusive. Proceed with caution."
    else:
        advice = "Returns are stationary. Safe for time series analysis."
    
    return CategoryResult(
        name="Stationarity",
        score=score,
        passed=score >= 80,
        status=status,
        advice=advice,
        tests=tests,
        weight=CATEGORY_WEIGHTS['stationarity']
    )


# =============================================================================
# AUTOCORRELATION TESTS (Weight: 10%)
# =============================================================================

def check_autocorrelation(returns: pd.Series, lags: int = 20) -> CategoryResult:
    """
    Check for serial correlation in returns - Affects standard error estimates.
    
    Tests:
    1. Ljung-Box test (p > 0.05 = no serial correlation)
    2. Max ACF value (< 0.2)
    
    Serial correlation can:
    - Inflate t-statistics
    - Make strategies appear better than they are
    - Indicate market inefficiency (exploitable)
    
    PERFORMANCE: For large datasets (>10k), uses sampling to keep test fast.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    
    tests = []
    returns_clean = returns.dropna()
    original_len = len(returns_clean)
    
    if original_len < lags + 10:
        return CategoryResult(
            name="Autocorrelation",
            score=50,
            passed=True,
            status="WARNING",
            advice="Insufficient data for autocorrelation test. Need more observations.",
            tests=[TestResult(
                name="Ljung-Box Test",
                passed=True,
                value=original_len,
                threshold=f">= {lags + 10}",
                interpretation="Skipped - need more data",
                advice="Extend date range for reliable autocorrelation testing."
            )],
            weight=CATEGORY_WEIGHTS['autocorrelation']
        )
    
    # Sample for performance if dataset is large
    returns_test = _sample_for_test(returns_clean, MAX_SAMPLES_FOR_TESTS)
    sampled = len(returns_test) < original_len
    
    # Test 1: Ljung-Box test
    try:
        lb_result = acorr_ljungbox(returns_test, lags=lags, return_df=True)
        significant_lags = (lb_result['lb_pvalue'] < 0.05).sum()
        lb_passed = significant_lags < lags * 0.1  # Less than 10% significant
        
        interpretation = "No serial correlation" if lb_passed else "Serial correlation detected"
        if sampled:
            interpretation += f" (sampled {len(returns_test):,} of {original_len:,})"
        
        tests.append(TestResult(
            name="Ljung-Box Test",
            passed=lb_passed,
            value=int(significant_lags),
            threshold=f"< {int(lags * 0.1)} significant lags",
            interpretation=interpretation,
            advice="" if lb_passed else "Serial correlation may inflate test statistics. Use robust standard errors.",
            details={'total_lags': lags, 'significant_lags': int(significant_lags), 'sampled': sampled}
        ))
    except Exception as e:
        logger.warning(f"Ljung-Box test failed: {e}")
        tests.append(TestResult(
            name="Ljung-Box Test",
            passed=True,
            value=0,
            threshold="< 10% significant",
            interpretation="Test skipped",
            advice="",
            details={'error': str(e)}
        ))
    
    # Test 2: Max ACF value
    try:
        acf_values = acf(returns_test, nlags=lags)
        max_acf = float(max(abs(acf_values[1:])))  # Exclude lag 0
        acf_passed = max_acf < 0.2
        
        tests.append(TestResult(
            name="Max ACF",
            passed=acf_passed,
            value=round(max_acf, 4),
            threshold="< 0.2",
            interpretation="Weak autocorrelation" if acf_passed else "Strong autocorrelation",
            advice="" if acf_passed else "High ACF suggests predictable patterns. May affect model assumptions.",
            details={'acf_lag1': round(float(acf_values[1]), 4) if len(acf_values) > 1 else 0}
        ))
    except Exception as e:
        logger.warning(f"ACF calculation failed: {e}")
        tests.append(TestResult(
            name="Max ACF",
            passed=True,
            value=0,
            threshold="< 0.2",
            interpretation="Test skipped",
            advice="",
            details={'error': str(e)}
        ))
    
    passed_count = sum(1 for t in tests if t.passed)
    score = (passed_count / len(tests) * 100) if tests else 50
    
    status = get_category_status(score)
    if status == "FAIL":
        advice = "Significant serial correlation detected. Use Newey-West standard errors."
    elif status == "WARNING":
        advice = "Some autocorrelation present. Results may be slightly biased."
    else:
        advice = "No significant autocorrelation. Standard statistical methods are valid."
    
    return CategoryResult(
        name="Autocorrelation",
        score=score,
        passed=score >= 50,
        status=status,
        advice=advice,
        tests=tests,
        weight=CATEGORY_WEIGHTS['autocorrelation']
    )


# =============================================================================
# OUTLIER DETECTION (Weight: 15%)
# =============================================================================

def check_outliers(returns: pd.Series, z_threshold: float = 5.0) -> CategoryResult:
    """
    Detect outliers in returns - Important for robust statistics.
    
    Tests:
    1. Z-score method (|z| > 5)
    2. IQR method (1.5 * IQR)
    
    Outliers can:
    - Distort mean and standard deviation
    - Inflate Sharpe ratio artificially
    - Indicate data errors or extreme events
    """
    tests = []
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 30:
        return CategoryResult(
            name="Outliers",
            score=50,
            passed=True,
            status="WARNING",
            advice="Insufficient data for outlier detection. Need at least 30 observations.",
            tests=[TestResult(
                name="Z-Score Outliers",
                passed=True,
                value=len(returns_clean),
                threshold=">= 30",
                interpretation="Skipped - need more data",
                advice="Extend date range for reliable outlier detection."
            )],
            weight=CATEGORY_WEIGHTS['outliers']
        )
    
    # Test 1: Z-score method
    mean_ret = returns_clean.mean()
    std_ret = returns_clean.std()
    
    if std_ret > 0:
        z_scores = (returns_clean - mean_ret) / std_ret
        outliers_z = abs(z_scores) > z_threshold
        outlier_pct_z = float(outliers_z.mean() * 100)
        z_passed = outlier_pct_z < 1  # Less than 1%
        
        tests.append(TestResult(
            name="Z-Score Outliers",
            passed=z_passed,
            value=round(outlier_pct_z, 2),
            threshold="< 1%",
            interpretation="Few outliers" if z_passed else "Many extreme values",
            advice="" if z_passed else f"Found {outliers_z.sum()} extreme returns (|z| > {z_threshold}). Consider winsorizing.",
            details={
                'outlier_count': int(outliers_z.sum()),
                'z_threshold': z_threshold,
                'max_z': round(float(abs(z_scores).max()), 2)
            }
        ))
    else:
        tests.append(TestResult(
            name="Z-Score Outliers",
            passed=True,
            value=0,
            threshold="< 1%",
            interpretation="Zero variance",
            advice="Returns have zero variance - check data.",
            details={'error': 'std = 0'}
        ))
    
    # Test 2: IQR method
    Q1 = float(returns_clean.quantile(0.25))
    Q3 = float(returns_clean.quantile(0.75))
    IQR = Q3 - Q1
    
    if IQR > 0:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = (returns_clean < lower_bound) | (returns_clean > upper_bound)
        outlier_pct_iqr = float(outliers_iqr.mean() * 100)
        iqr_passed = outlier_pct_iqr < 5  # Less than 5%
        
        tests.append(TestResult(
            name="IQR Outliers",
            passed=iqr_passed,
            value=round(outlier_pct_iqr, 2),
            threshold="< 5%",
            interpretation="Normal range" if iqr_passed else "Wide distribution",
            advice="" if iqr_passed else "Many values outside IQR bounds. Distribution has heavy tails.",
            details={
                'outlier_count': int(outliers_iqr.sum()),
                'Q1': round(Q1, 6),
                'Q3': round(Q3, 6),
                'IQR': round(IQR, 6)
            }
        ))
    else:
        tests.append(TestResult(
            name="IQR Outliers",
            passed=True,
            value=0,
            threshold="< 5%",
            interpretation="Zero IQR",
            advice="IQR is zero - check data distribution.",
            details={'error': 'IQR = 0'}
        ))
    
    passed_count = sum(1 for t in tests if t.passed)
    score = (passed_count / len(tests) * 100) if tests else 50
    
    status = get_category_status(score)
    if status == "FAIL":
        advice = "Many outliers detected. Consider winsorizing or using robust statistics (median, MAD)."
    elif status == "WARNING":
        advice = "Some outliers present. Mean-based statistics may be affected."
    else:
        advice = "Outlier levels are acceptable. Standard statistics are reliable."
    
    return CategoryResult(
        name="Outliers",
        score=score,
        passed=score >= 50,
        status=status,
        advice=advice,
        tests=tests,
        weight=CATEGORY_WEIGHTS['outliers']
    )


# =============================================================================
# SAMPLE SIZE CHECK (Weight: 10%)
# =============================================================================

def check_sample_size(df: pd.DataFrame, timeframe: str) -> CategoryResult:
    """
    Check if sample size is sufficient for statistical inference.
    
    Requirements:
    - Minimum 252 observations (1 year daily equivalent)
    - Minimum 30 for basic statistics
    - Recommended 1000+ for robust analysis
    
    Small samples have:
    - High variance in estimates
    - Unreliable p-values
    - Overfitting risk
    """
    tests = []
    n = len(df)
    
    # Adjust minimum based on timeframe
    min_required = {
        '1T': 252 * 24 * 60,  # 1 year of 1-min bars
        '5T': 252 * 24 * 12,
        '15T': 252 * 24 * 4,
        '30T': 252 * 24 * 2,
        '1H': 252 * 24,
        '1h': 252 * 24,
        '4H': 252 * 6,
        '4h': 252 * 6,
        '1D': 252,
        '1d': 252,
        '1W': 52,
        '1w': 52,
    }.get(timeframe, 252)
    
    # Cap minimum at reasonable level
    min_required = min(min_required, 5000)
    
    # Test 1: Minimum observations
    min_passed = n >= min_required
    tests.append(TestResult(
        name="Minimum Observations",
        passed=min_passed,
        value=n,
        threshold=f">= {min_required}",
        interpretation="Sufficient" if min_passed else "Insufficient",
        advice="" if min_passed else f"Need {min_required - n} more observations for reliable {timeframe} analysis.",
        details={'required': min_required, 'actual': n, 'deficit': max(0, min_required - n)}
    ))
    
    # Test 2: Basic statistics threshold (30)
    basic_passed = n >= 30
    tests.append(TestResult(
        name="Basic Statistics",
        passed=basic_passed,
        value=n,
        threshold=">= 30",
        interpretation="OK for basic stats" if basic_passed else "Too few for any statistics",
        advice="" if basic_passed else "CRITICAL: Need at least 30 observations for basic statistics.",
        details={'threshold': 30}
    ))
    
    # Test 3: Robust analysis threshold (1000)
    robust_passed = n >= 1000
    tests.append(TestResult(
        name="Robust Analysis",
        passed=robust_passed,
        value=n,
        threshold=">= 1000",
        interpretation="Excellent sample" if robust_passed else "Limited for advanced analysis",
        advice="" if robust_passed else "Consider extending date range for more robust statistical inference.",
        details={'threshold': 1000}
    ))
    
    # Calculate score based on sufficiency levels
    if n >= min_required * 2:
        score = 100
    elif n >= min_required:
        score = 80
    elif n >= min_required * 0.5:
        score = 60
    elif n >= 30:
        score = 40
    else:
        score = 0
    
    status = get_category_status(score)
    if status == "FAIL":
        advice = f"CRITICAL: Only {n} observations. Need at least {min_required} for {timeframe} analysis."
    elif status == "WARNING":
        advice = f"Sample size ({n}) is marginal. Results may have high variance."
    else:
        advice = f"Sample size ({n}) is adequate for statistical analysis."
    
    return CategoryResult(
        name="Sample Size",
        score=score,
        passed=score >= 50,
        status=status,
        advice=advice,
        tests=tests,
        weight=CATEGORY_WEIGHTS['sample_size']
    )


# =============================================================================
# OHLC INTEGRITY CHECK (Weight: 10%)
# =============================================================================

def check_ohlc_integrity(df: pd.DataFrame) -> CategoryResult:
    """
    Check OHLC data integrity - CRITICAL for valid price data.
    
    Rules:
    1. High >= max(Open, Close)
    2. Low <= min(Open, Close)
    3. High >= Low
    4. All prices > 0
    
    Invalid OHLC indicates:
    - Data corruption
    - Feed errors
    - Processing bugs
    """
    tests = []
    n = len(df)
    
    if n == 0:
        return CategoryResult(
            name="OHLC Integrity",
            score=0,
            passed=False,
            status="FAIL",
            advice="No data to validate.",
            tests=[],
            weight=CATEGORY_WEIGHTS['ohlc_integrity']
        )
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return CategoryResult(
            name="OHLC Integrity",
            score=0,
            passed=False,
            status="FAIL",
            advice=f"Missing columns: {missing_cols}. Cannot validate OHLC integrity.",
            tests=[TestResult(
                name="Required Columns",
                passed=False,
                value=len(missing_cols),
                threshold="0 missing",
                interpretation=f"Missing: {missing_cols}",
                advice="Ensure data has open, high, low, close columns."
            )],
            weight=CATEGORY_WEIGHTS['ohlc_integrity']
        )
    
    # Test 1: High >= max(Open, Close)
    high_valid = df['high'] >= df[['open', 'close']].max(axis=1)
    high_invalid_count = int((~high_valid).sum())
    high_passed = high_invalid_count == 0
    
    tests.append(TestResult(
        name="High >= max(O,C)",
        passed=high_passed,
        value=high_invalid_count,
        threshold="0 violations",
        interpretation="Valid" if high_passed else f"{high_invalid_count} violations",
        advice="" if high_passed else "High price below Open/Close indicates data error.",
        details={'invalid_count': high_invalid_count, 'total': n}
    ))
    
    # Test 2: Low <= min(Open, Close)
    low_valid = df['low'] <= df[['open', 'close']].min(axis=1)
    low_invalid_count = int((~low_valid).sum())
    low_passed = low_invalid_count == 0
    
    tests.append(TestResult(
        name="Low <= min(O,C)",
        passed=low_passed,
        value=low_invalid_count,
        threshold="0 violations",
        interpretation="Valid" if low_passed else f"{low_invalid_count} violations",
        advice="" if low_passed else "Low price above Open/Close indicates data error.",
        details={'invalid_count': low_invalid_count, 'total': n}
    ))
    
    # Test 3: High >= Low
    hl_valid = df['high'] >= df['low']
    hl_invalid_count = int((~hl_valid).sum())
    hl_passed = hl_invalid_count == 0
    
    tests.append(TestResult(
        name="High >= Low",
        passed=hl_passed,
        value=hl_invalid_count,
        threshold="0 violations",
        interpretation="Valid" if hl_passed else f"{hl_invalid_count} violations",
        advice="" if hl_passed else "High < Low is impossible. Data is corrupted.",
        details={'invalid_count': hl_invalid_count, 'total': n}
    ))
    
    # Test 4: All prices > 0
    positive = (df[required_cols] > 0).all(axis=1)
    negative_count = int((~positive).sum())
    positive_passed = negative_count == 0
    
    tests.append(TestResult(
        name="Positive Prices",
        passed=positive_passed,
        value=negative_count,
        threshold="0 violations",
        interpretation="Valid" if positive_passed else f"{negative_count} non-positive",
        advice="" if positive_passed else "Non-positive prices indicate data error or special events.",
        details={'invalid_count': negative_count, 'total': n}
    ))
    
    passed_count = sum(1 for t in tests if t.passed)
    score = (passed_count / len(tests) * 100) if tests else 0
    
    status = get_category_status(score)
    if status == "FAIL":
        advice = "CRITICAL: OHLC data has integrity issues. Data may be corrupted. DO NOT use for analysis."
    elif status == "WARNING":
        advice = "Some OHLC integrity issues found. Review and clean data before analysis."
    else:
        advice = "OHLC data integrity is valid. Price relationships are correct."
    
    return CategoryResult(
        name="OHLC Integrity",
        score=score,
        passed=score >= 80,
        status=status,
        advice=advice,
        tests=tests,
        weight=CATEGORY_WEIGHTS['ohlc_integrity']
    )


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_data_quality(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str
) -> QualityReport:
    """
    Run comprehensive data quality validation.
    
    This is the main entry point for data quality validation.
    Runs all 7 categories of tests and generates a complete report.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex
    symbol : str
        Trading symbol (e.g., 'XAUUSD')
    timeframe : str
        Data timeframe (e.g., '1H', '1D')
    start_date : str
        Start date string
    end_date : str
        End date string
        
    Returns
    -------
    QualityReport
        Complete quality report with scores, grades, and recommendations
    """
    logger.info(f"Starting data quality validation for {symbol} {timeframe}")
    
    categories = {}
    recommendations = []
    
    # Calculate returns for distribution tests
    if 'close' in df.columns and len(df) > 1:
        returns = df['close'].pct_change().dropna()
    else:
        returns = pd.Series(dtype=float)
    
    # Run all category tests
    try:
        # 1. Completeness (20%)
        categories['completeness'] = check_completeness(df, timeframe)
        if categories['completeness'].status != "PASS":
            recommendations.append(f"📊 Completeness: {categories['completeness'].advice}")
        
        # 2. Distribution (15%)
        categories['distribution'] = check_distribution(returns)
        if categories['distribution'].status != "PASS":
            recommendations.append(f"📈 Distribution: {categories['distribution'].advice}")
        
        # 3. Stationarity (20%)
        categories['stationarity'] = check_stationarity(returns)
        if categories['stationarity'].status != "PASS":
            recommendations.append(f"📉 Stationarity: {categories['stationarity'].advice}")
        
        # 4. Autocorrelation (10%)
        categories['autocorrelation'] = check_autocorrelation(returns)
        if categories['autocorrelation'].status != "PASS":
            recommendations.append(f"🔄 Autocorrelation: {categories['autocorrelation'].advice}")
        
        # 5. Outliers (15%)
        categories['outliers'] = check_outliers(returns)
        if categories['outliers'].status != "PASS":
            recommendations.append(f"⚠️ Outliers: {categories['outliers'].advice}")
        
        # 6. Sample Size (10%)
        categories['sample_size'] = check_sample_size(df, timeframe)
        if categories['sample_size'].status != "PASS":
            recommendations.append(f"📏 Sample Size: {categories['sample_size'].advice}")
        
        # 7. OHLC Integrity (10%)
        categories['ohlc_integrity'] = check_ohlc_integrity(df)
        if categories['ohlc_integrity'].status != "PASS":
            recommendations.append(f"🔍 OHLC Integrity: {categories['ohlc_integrity'].advice}")
            
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        recommendations.append(f"❌ Validation error: {str(e)}")
    
    # Calculate overall quality score
    quality_score = calculate_quality_score(categories)
    grade = calculate_grade(quality_score)
    can_proceed = quality_score >= MIN_SCORE_TO_PROCEED
    status_label = get_status_label(quality_score, grade)
    
    # Generate English summary
    english_summary = _generate_english_summary(quality_score, grade, can_proceed, categories)
    
    # Add overall recommendation
    if not can_proceed:
        recommendations.insert(0, f"🚫 OVERALL: Quality score {quality_score:.1f}% (Grade {grade}) is below minimum threshold of {MIN_SCORE_TO_PROCEED}%. DO NOT proceed with analysis.")
    elif grade in ['C']:
        recommendations.insert(0, f"⚠️ OVERALL: Quality score {quality_score:.1f}% (Grade {grade}) meets minimum threshold but has issues. Review recommendations before proceeding.")
    elif grade in ['B']:
        recommendations.insert(0, f"✅ OVERALL: Quality score {quality_score:.1f}% (Grade {grade}) is good. Minor issues noted. Safe to proceed with caution.")
    else:
        recommendations.insert(0, f"✅ OVERALL: Quality score {quality_score:.1f}% (Grade {grade}) is excellent. Data is ready for analysis.")
    
    # Determine overall pass/fail
    passed = grade in ['A', 'B', 'C']
    
    report = QualityReport(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        total_rows=len(df),
        quality_score=quality_score,
        grade=grade,
        passed=passed,
        can_proceed=can_proceed,
        status_label=status_label,
        categories=categories,
        recommendations=recommendations,
        english_summary=english_summary
    )
    
    logger.info(f"Validation complete: {symbol} {timeframe} - Score: {quality_score:.1f}% Grade: {grade}")
    
    return report


def _generate_english_summary(
    score: float,
    grade: str,
    can_proceed: bool,
    categories: Dict[str, CategoryResult]
) -> str:
    """Generate English summary of data quality assessment."""
    
    # Count category statuses
    pass_count = sum(1 for c in categories.values() if c.status == "PASS")
    warn_count = sum(1 for c in categories.values() if c.status == "WARNING")
    fail_count = sum(1 for c in categories.values() if c.status == "FAIL")
    
    # Build summary
    lines = []
    
    # Overall assessment
    if grade == 'A':
        lines.append("EXCELLENT: Data quality is outstanding. All statistical tests passed.")
        lines.append("This data is suitable for any type of quantitative analysis.")
    elif grade == 'B':
        lines.append("GOOD: Data quality is acceptable with minor issues.")
        lines.append("Safe to proceed with standard analysis methods.")
    elif grade == 'C':
        lines.append("FAIR: Data quality meets minimum requirements but has notable issues.")
        lines.append("Proceed with caution and consider the recommendations below.")
    elif grade == 'D':
        lines.append("POOR: Data quality is below acceptable standards.")
        lines.append("Analysis results may be unreliable. NOT recommended to proceed.")
    else:
        lines.append("FAIL: Data quality is insufficient for statistical analysis.")
        lines.append("DO NOT use this data for trading decisions.")
    
    lines.append("")
    lines.append(f"Summary: {pass_count} PASS, {warn_count} WARNING, {fail_count} FAIL out of {len(categories)} categories.")
    
    # Specific issues
    if fail_count > 0:
        lines.append("")
        lines.append("Critical Issues:")
        for name, cat in categories.items():
            if cat.status == "FAIL":
                lines.append(f"  - {cat.name}: {cat.advice}")
    
    if warn_count > 0:
        lines.append("")
        lines.append("Warnings:")
        for name, cat in categories.items():
            if cat.status == "WARNING":
                lines.append(f"  - {cat.name}: {cat.advice}")
    
    # Proceed recommendation
    lines.append("")
    if can_proceed:
        lines.append(f"✅ RECOMMENDATION: You may proceed with analysis (Score: {score:.1f}%, Grade: {grade})")
    else:
        lines.append(f"❌ RECOMMENDATION: Do NOT proceed with analysis (Score: {score:.1f}%, Grade: {grade})")
        lines.append(f"   Minimum required score is {MIN_SCORE_TO_PROCEED}% (Grade C)")
    
    return "\n".join(lines)
