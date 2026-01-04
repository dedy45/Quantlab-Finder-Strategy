"""
Fractional Differencing for Stationarity with Memory Preservation.

Standard differencing (d=1) makes data stationary but removes ALL memory.
Fractional differencing (0 < d < 1) achieves stationarity while preserving
some memory/predictive power.

Reference:
- Protokol Kausalitas - Fase 2 (Sebab Statistik)
- Lopez de Prado - Advances in Financial Machine Learning, Chapter 5

Key Insight:
- d=0: Original series (non-stationary, full memory)
- d=1: First difference (stationary, no memory)
- 0<d<1: Fractional (stationary, partial memory)

The goal is to find minimum d that makes series stationary (ADF test).
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Dict
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller

from .base import FeatureGenerator, FeatureConfig, FeatureResult, FeatureType

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class FracDiffResult:
    """Result of fractional differencing."""
    series: pd.Series
    d: float
    adf_stat: float
    adf_pvalue: float
    is_stationary: bool
    weights: np.ndarray
    
    def __str__(self) -> str:
        status = "STATIONARY" if self.is_stationary else "NON-STATIONARY"
        return (
            f"FracDiff Result:\n"
            f"  d: {self.d:.4f}\n"
            f"  ADF Stat: {self.adf_stat:.4f}\n"
            f"  ADF p-value: {self.adf_pvalue:.4f}\n"
            f"  Status: {status}"
        )


def get_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate weights for fractional differencing.
    
    The weights follow the binomial series expansion:
    w_k = -w_{k-1} * (d - k + 1) / k
    
    Parameters
    ----------
    d : float
        Differencing order (0 < d < 1)
    size : int
        Number of weights to compute
    threshold : float
        Minimum weight magnitude (for truncation)
        
    Returns
    -------
    np.ndarray
        Array of weights
        
    Raises
    ------
    ValueError
        If d is out of valid range
    """
    # Input validation
    assert size > 0, "Size must be positive"
    assert threshold > 0, "Threshold must be positive"
    
    try:
        weights = [1.0]
        k = 1
        
        while k < size:
            # Vectorized weight calculation
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1
        
        return np.array(weights[::-1], dtype=np.float64)  # Reverse for convolution
        
    except Exception as e:
        logger.error(f"Error calculating weights: {e}")
        raise


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Get weights for Fixed-width window Fractional Differencing (FFD).
    
    FFD uses a fixed window size determined by weight threshold,
    making it more practical for real-time applications.
    
    Parameters
    ----------
    d : float
        Differencing order
    threshold : float
        Minimum weight magnitude
        
    Returns
    -------
    np.ndarray
        Array of weights
    """
    weights = [1.0]
    k = 1
    
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    
    return np.array(weights[::-1])


def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Apply fractional differencing to a series.
    
    Parameters
    ----------
    series : pd.Series
        Input price series
    d : float
        Differencing order (0 < d < 1)
    threshold : float
        Weight threshold for truncation
        
    Returns
    -------
    pd.Series
        Fractionally differenced series
        
    Raises
    ------
    ValueError
        If series is empty or invalid
    """
    # Input validation
    assert series is not None, "Series cannot be None"
    assert len(series) > 0, "Series cannot be empty"
    
    try:
        weights = get_weights_ffd(d, threshold)
        width = len(weights)
        
        # Vectorized convolution using numpy
        values = series.values.astype(np.float64)
        result_values = np.full(len(values), np.nan)
        
        # Apply weights using vectorized dot product
        for i in range(width - 1, len(values)):
            result_values[i] = np.dot(weights, values[i - width + 1:i + 1])
        
        result = pd.Series(result_values, index=series.index, dtype=np.float64)
        
        logger.debug(f"Fractional diff applied: d={d:.4f}, width={width}")
        return result
        
    except Exception as e:
        logger.error(f"Error in frac_diff: {e}")
        raise


def frac_diff_expanding(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Apply fractional differencing with expanding window.
    
    Uses all available history (expanding window) rather than
    fixed window. More accurate but slower.
    
    Parameters
    ----------
    series : pd.Series
        Input price series
    d : float
        Differencing order
    threshold : float
        Weight threshold
        
    Returns
    -------
    pd.Series
        Fractionally differenced series
    """
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(len(series)):
        weights = get_weights(d, i + 1, threshold)
        width = len(weights)
        
        if width > i + 1:
            continue
            
        result.iloc[i] = np.dot(weights, series.iloc[i - width + 1:i + 1].values)
    
    return result


def find_min_d(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    n_steps: int = 20,
    significance: float = 0.05,
    threshold: float = 1e-5
) -> Tuple[float, float, float]:
    """
    Find minimum d that makes series stationary.
    
    Uses binary search to find the smallest d value that
    passes the ADF test at the given significance level.
    
    Parameters
    ----------
    series : pd.Series
        Input price series
    d_range : Tuple[float, float]
        Range of d values to search
    n_steps : int
        Number of steps in grid search
    significance : float
        Significance level for ADF test
    threshold : float
        Weight threshold
        
    Returns
    -------
    Tuple[float, float, float]
        (optimal_d, adf_stat, adf_pvalue)
    """
    # Input validation
    assert series is not None, "Series cannot be None"
    assert len(series) > 20, "Series must have at least 20 observations"
    assert 0 <= d_range[0] < d_range[1] <= 1, "d_range must be in [0, 1]"
    
    try:
        d_values = np.linspace(d_range[0], d_range[1], n_steps)
        
        for d in d_values:
            if d == 0:
                continue
                
            diff_series = frac_diff(series, d, threshold)
            diff_series = diff_series.dropna()
            
            if len(diff_series) < 20:
                continue
            
            try:
                adf_result = adfuller(diff_series, maxlag=1, regression='c')
                adf_stat = adf_result[0]
                adf_pvalue = adf_result[1]
                
                if adf_pvalue < significance:
                    logger.info(f"Found optimal d={d:.4f} (ADF p-value={adf_pvalue:.4f})")
                    return d, adf_stat, adf_pvalue
            except Exception as e:
                logger.warning(f"ADF test failed for d={d:.4f}: {e}")
                continue
        
        # If no d found, return d=1
        logger.warning("No optimal d found, returning d=1.0")
        return 1.0, np.nan, np.nan
        
    except Exception as e:
        logger.error(f"Error finding min d: {e}")
        raise


class FractionalDifferencer(FeatureGenerator):
    """
    Fractional Differencing Feature Generator.
    
    Transforms price series to stationary series while preserving
    memory/predictive power.
    
    Examples
    --------
    >>> differ = FractionalDifferencer(d=0.4)
    >>> result = differ.fit_transform(price_df)
    >>> print(f"Stationary: {result.metadata['is_stationary']}")
    
    >>> # Auto-find optimal d
    >>> differ = FractionalDifferencer(d='auto')
    >>> result = differ.fit_transform(price_df)
    >>> print(f"Optimal d: {result.metadata['d']}")
    """
    
    def __init__(
        self,
        d: Union[float, str] = 'auto',
        threshold: float = 1e-5,
        significance: float = 0.05,
        method: str = 'ffd',  # 'ffd' or 'expanding'
    ):
        """
        Initialize fractional differencer.
        
        Parameters
        ----------
        d : float or 'auto'
            Differencing order. If 'auto', finds minimum d.
        threshold : float
            Weight threshold for truncation
        significance : float
            Significance level for ADF test (when d='auto')
        method : str
            'ffd' for fixed-width, 'expanding' for expanding window
        """
        config = FeatureConfig(
            name='fractional_diff',
            feature_type=FeatureType.STATISTICAL,
            params={'d': d, 'threshold': threshold, 'method': method}
        )
        super().__init__(config)
        
        self.d = d
        self.threshold = threshold
        self.significance = significance
        self.method = method
        
        self._optimal_d: Optional[float] = None
        self._adf_stats: Dict[str, float] = {}
    
    def fit(self, data: pd.DataFrame) -> 'FractionalDifferencer':
        """
        Fit the differencer (find optimal d if auto).
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with price columns
            
        Returns
        -------
        FractionalDifferencer
            Fitted differencer
        """
        data = self._validate_data(data)
        
        # Get price column
        if 'close' in data.columns:
            prices = data['close']
        else:
            prices = data.iloc[:, 0]
        
        if self.d == 'auto':
            self._optimal_d, adf_stat, adf_pvalue = find_min_d(
                prices,
                significance=self.significance,
                threshold=self.threshold
            )
            self._adf_stats = {
                'adf_stat': adf_stat,
                'adf_pvalue': adf_pvalue,
                'is_stationary': adf_pvalue < self.significance
            }
        else:
            self._optimal_d = self.d
        
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """
        Apply fractional differencing.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with price columns
            
        Returns
        -------
        FeatureResult
            Fractionally differenced features
        """
        if not self._is_fitted:
            raise ValueError("Differencer not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        
        # Apply to all numeric columns
        result_df = pd.DataFrame(index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if self.method == 'expanding':
                diff_series = frac_diff_expanding(
                    data[col], self._optimal_d, self.threshold
                )
            else:
                diff_series = frac_diff(
                    data[col], self._optimal_d, self.threshold
                )
            
            result_df[f'{col}_fracdiff'] = diff_series
        
        # Handle NaN
        result_df = self._handle_nan(result_df, 'ffill')
        
        return FeatureResult(
            features=result_df,
            feature_names=list(result_df.columns),
            config=self.config,
            metadata={
                'd': self._optimal_d,
                'threshold': self.threshold,
                'method': self.method,
                **self._adf_stats
            }
        )
    
    def transform_series(self, series: pd.Series) -> FracDiffResult:
        """
        Transform a single series with detailed result.
        
        Parameters
        ----------
        series : pd.Series
            Price series
            
        Returns
        -------
        FracDiffResult
            Detailed result with ADF statistics
        """
        if self.d == 'auto':
            d, adf_stat, adf_pvalue = find_min_d(
                series,
                significance=self.significance,
                threshold=self.threshold
            )
        else:
            d = self.d
            diff_series = frac_diff(series, d, self.threshold)
            diff_series = diff_series.dropna()
            
            if len(diff_series) > 20:
                adf_result = adfuller(diff_series, maxlag=1, regression='c')
                adf_stat = adf_result[0]
                adf_pvalue = adf_result[1]
            else:
                adf_stat = np.nan
                adf_pvalue = np.nan
        
        diff_series = frac_diff(series, d, self.threshold)
        weights = get_weights_ffd(d, self.threshold)
        
        return FracDiffResult(
            series=diff_series,
            d=d,
            adf_stat=adf_stat,
            adf_pvalue=adf_pvalue,
            is_stationary=adf_pvalue < self.significance if not np.isnan(adf_pvalue) else False,
            weights=weights
        )


def fractional_difference(
    prices: pd.Series,
    d: Union[float, str] = 'auto',
    threshold: float = 1e-5
) -> pd.Series:
    """
    Quick fractional differencing function.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    d : float or 'auto'
        Differencing order
    threshold : float
        Weight threshold
        
    Returns
    -------
    pd.Series
        Fractionally differenced series
    """
    differ = FractionalDifferencer(d=d, threshold=threshold)
    data = pd.DataFrame({'close': prices})
    result = differ.fit_transform(data)
    return result.features.iloc[:, 0]
