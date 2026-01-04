"""
Data Validator - Validate data quality and integrity.

Critical validations:
- Point-in-time correctness (no look-ahead bias)
- Survivorship bias detection
- Data completeness checks
- Statistical anomaly detection
"""

import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats

from .base import DataConfig, BaseDataHandler, Resolution

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    score: float  # 0-100 quality score
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"[{status}] (Score: {self.score:.1f}/100)\n" + \
               f"Issues: {len(self.issues)}, Warnings: {len(self.warnings)}"


class DataValidator(BaseDataHandler):
    """
    Validate data quality for quantitative research.
    
    Performs critical checks to ensure data integrity:
    1. Point-in-time validation (no look-ahead bias)
    2. Survivorship bias detection
    3. Data completeness
    4. Statistical anomalies
    
    Examples
    --------
    >>> validator = DataValidator(config)
    >>> result = validator.validate_data(df)
    >>> if not result.is_valid:
    ...     print(result.issues)
    """
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.validation_result: Optional[ValidationResult] = None
    
    def validate(self) -> bool:
        """Validate that validation can be performed."""
        return True
    
    def execute(self, data: pd.DataFrame) -> ValidationResult:
        """Execute validation pipeline."""
        return self.validate_data(data)
    
    def validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run full validation pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
            
        Returns
        -------
        ValidationResult
            Validation results with issues and score
        """
        issues = []
        warnings = []
        metrics = {}
        
        # 1. Basic structure validation
        struct_issues = self._validate_structure(df)
        issues.extend(struct_issues)
        
        if struct_issues:
            # Can't continue without basic structure
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=issues,
                warnings=warnings,
                metrics=metrics,
            )
        
        # 2. Completeness check
        completeness, comp_warnings = self._check_completeness(df)
        metrics['completeness'] = completeness
        warnings.extend(comp_warnings)
        
        # 3. Point-in-time validation
        if self.config.check_lookahead:
            pit_issues = self._check_point_in_time(df)
            issues.extend(pit_issues)
        
        # 4. Survivorship bias check
        if self.config.check_survivorship:
            surv_warnings = self._check_survivorship_bias(df)
            warnings.extend(surv_warnings)
        
        # 5. Statistical anomalies
        anomaly_warnings, anomaly_metrics = self._check_statistical_anomalies(df)
        warnings.extend(anomaly_warnings)
        metrics.update(anomaly_metrics)
        
        # 6. Data freshness
        freshness_warnings = self._check_freshness(df)
        warnings.extend(freshness_warnings)
        
        # Calculate overall score
        score = self._calculate_score(issues, warnings, metrics)
        
        # Determine validity
        is_valid = len(issues) == 0 and score >= 70
        
        self.validation_result = ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            metrics=metrics,
        )
        
        logger.info(f"Validation complete: {self.validation_result}")
        return self.validation_result
    
    def _validate_structure(self, df: pd.DataFrame) -> List[str]:
        """Validate basic data structure."""
        issues = []
        
        # Check required columns
        required = ['timestamp', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")
        
        # Check data types
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                issues.append("timestamp column is not datetime type")
        
        # Check for empty data
        if len(df) == 0:
            issues.append("DataFrame is empty")
        
        # Check for all-NaN columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and df[col].isna().all():
                issues.append(f"Column {col} is all NaN")
        
        return issues
    
    def _check_completeness(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data completeness."""
        warnings = []
        
        # Calculate missing percentage
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        existing_cols = [c for c in numeric_cols if c in df.columns]
        
        if not existing_cols:
            return 0.0, ["No numeric columns found"]
        
        missing_pct = df[existing_cols].isnull().sum().sum() / (len(df) * len(existing_cols))
        completeness = (1 - missing_pct) * 100
        
        if missing_pct > 0.05:
            warnings.append(f"High missing data: {missing_pct*100:.2f}%")
        
        # Check for gaps in time series
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
                if len(symbol_df) > 1:
                    time_diff = symbol_df['timestamp'].diff()
                    expected_diff = self._get_expected_diff()
                    
                    # Count gaps (more than 2x expected)
                    gaps = (time_diff > expected_diff * 2).sum()
                    if gaps > 10:
                        warnings.append(f"Symbol {symbol} has {gaps} time gaps")
        
        return completeness, warnings
    
    def _get_expected_diff(self) -> timedelta:
        """Get expected time difference based on resolution."""
        mapping = {
            Resolution.MINUTE_1: timedelta(minutes=1),
            Resolution.MINUTE_5: timedelta(minutes=5),
            Resolution.MINUTE_15: timedelta(minutes=15),
            Resolution.HOUR_1: timedelta(hours=1),
            Resolution.HOUR_4: timedelta(hours=4),
            Resolution.DAILY: timedelta(days=1),
            Resolution.WEEKLY: timedelta(weeks=1),
        }
        return mapping.get(self.config.resolution, timedelta(days=1))
    
    def _check_point_in_time(self, df: pd.DataFrame) -> List[str]:
        """
        Check for potential look-ahead bias.
        
        Detects:
        - Future data in historical records
        - Adjusted prices without proper handling
        - Corporate actions not properly handled
        """
        issues = []
        
        if 'timestamp' not in df.columns:
            return issues
        
        # Check for future dates
        now = datetime.now()
        future_data = df[df['timestamp'] > now]
        if len(future_data) > 0:
            issues.append(f"Found {len(future_data)} rows with future timestamps")
        
        # Check for suspicious patterns (perfect hindsight)
        if 'close' in df.columns and 'symbol' in df.columns:
            for symbol in df['symbol'].unique()[:5]:  # Check first 5 symbols
                symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_df) < 100:
                    continue
                
                # Check for unrealistic returns (might indicate adjusted data issues)
                returns = symbol_df['close'].pct_change()
                
                # More than 50% daily return is suspicious
                extreme_returns = (returns.abs() > 0.5).sum()
                if extreme_returns > 5:
                    issues.append(
                        f"Symbol {symbol} has {extreme_returns} extreme returns (>50%). "
                        "Check for stock splits or data issues."
                    )
        
        return issues
    
    def _check_survivorship_bias(self, df: pd.DataFrame) -> List[str]:
        """
        Check for potential survivorship bias.
        
        Warnings if:
        - Only successful/existing companies in dataset
        - No delisted securities
        - Suspiciously high average returns
        """
        warnings = []
        
        if 'symbol' not in df.columns or 'close' not in df.columns:
            return warnings
        
        # Check average returns across all symbols
        returns_by_symbol = df.groupby('symbol')['close'].apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) > 1 and x.iloc[0] > 0 else 0
        )
        
        avg_return = returns_by_symbol.mean()
        positive_pct = (returns_by_symbol > 0).mean()
        
        # If >80% of symbols have positive returns, might be survivorship bias
        if positive_pct > 0.8:
            warnings.append(
                f"Potential survivorship bias: {positive_pct*100:.1f}% of symbols "
                "have positive total returns. Consider including delisted securities."
            )
        
        # Check if all symbols exist until the end
        if 'timestamp' in df.columns:
            max_date = df['timestamp'].max()
            symbols_at_end = df[df['timestamp'] == max_date]['symbol'].nunique()
            total_symbols = df['symbol'].nunique()
            
            if symbols_at_end == total_symbols:
                warnings.append(
                    "All symbols exist until the last date. "
                    "This might indicate survivorship bias."
                )
        
        return warnings
    
    def _check_statistical_anomalies(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
        """Check for statistical anomalies in the data."""
        warnings = []
        metrics = {}
        
        if 'close' not in df.columns:
            return warnings, metrics
        
        # Calculate returns
        if 'symbol' in df.columns:
            returns = df.groupby('symbol')['close'].pct_change()
        else:
            returns = df['close'].pct_change()
        
        returns = returns.dropna()
        
        if len(returns) < 30:
            return warnings, metrics
        
        # Basic statistics
        metrics['returns_mean'] = returns.mean()
        metrics['returns_std'] = returns.std()
        metrics['returns_skew'] = returns.skew()
        metrics['returns_kurtosis'] = returns.kurtosis()
        
        # Check for abnormal statistics
        if abs(metrics['returns_skew']) > 2:
            warnings.append(f"High skewness in returns: {metrics['returns_skew']:.2f}")
        
        if metrics['returns_kurtosis'] > 10:
            warnings.append(f"High kurtosis (fat tails): {metrics['returns_kurtosis']:.2f}")
        
        # Check for zero variance periods
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique()[:10]:
                symbol_returns = df[df['symbol'] == symbol]['close'].pct_change()
                
                # Rolling variance
                rolling_var = symbol_returns.rolling(20).var()
                zero_var_periods = (rolling_var == 0).sum()
                
                if zero_var_periods > 10:
                    warnings.append(
                        f"Symbol {symbol} has {zero_var_periods} periods with zero variance"
                    )
        
        return warnings, metrics
    
    def _check_freshness(self, df: pd.DataFrame) -> List[str]:
        """Check data freshness."""
        warnings = []
        
        if 'timestamp' not in df.columns:
            return warnings
        
        max_date = df['timestamp'].max()
        days_old = (datetime.now() - max_date).days
        
        if days_old > 30:
            warnings.append(f"Data is {days_old} days old. Consider updating.")
        
        return warnings
    
    def _calculate_score(
        self, 
        issues: List[str], 
        warnings: List[str], 
        metrics: Dict[str, float]
    ) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Deduct for issues (critical)
        score -= len(issues) * 20
        
        # Deduct for warnings (minor)
        score -= len(warnings) * 5
        
        # Adjust for completeness
        completeness = metrics.get('completeness', 100)
        if completeness < 95:
            score -= (95 - completeness)
        
        # Ensure score is in valid range
        return max(0.0, min(100.0, score))


def validate_data(df: pd.DataFrame) -> ValidationResult:
    """
    Convenience function for quick data validation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
        
    Returns
    -------
    ValidationResult
        Validation results
    """
    from .base import DataSource, AssetClass
    
    config = DataConfig(
        source=DataSource.LOCAL,
        asset_class=AssetClass.EQUITY,
        symbols=[],
        start_date='2000-01-01',
    )
    
    validator = DataValidator(config)
    return validator.validate_data(df)
