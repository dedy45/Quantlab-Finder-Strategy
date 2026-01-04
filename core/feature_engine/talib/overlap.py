"""
TA-Lib Overlap Studies - Moving Averages and Bands.

Overlap studies are indicators that overlay on price charts:
- Moving Averages (SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA)
- Bollinger Bands (BBANDS)
- Parabolic SAR (SAR)
- Midpoint/Midprice

Reference: https://ta-lib.org/functions/
"""

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib

from ..base import FeatureGenerator, FeatureConfig, FeatureResult, FeatureType

logger = logging.getLogger(__name__)


# =============================================================================
# Moving Averages
# =============================================================================

def SMA(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Simple Moving Average."""
    result = talib.SMA(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'sma_{timeperiod}')


def EMA(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Exponential Moving Average."""
    result = talib.EMA(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'ema_{timeperiod}')


def WMA(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Weighted Moving Average."""
    result = talib.WMA(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'wma_{timeperiod}')


def DEMA(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Double Exponential Moving Average."""
    result = talib.DEMA(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'dema_{timeperiod}')


def TEMA(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Triple Exponential Moving Average."""
    result = talib.TEMA(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'tema_{timeperiod}')


def TRIMA(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Triangular Moving Average."""
    result = talib.TRIMA(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'trima_{timeperiod}')


def KAMA(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    result = talib.KAMA(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'kama_{timeperiod}')


# =============================================================================
# Bollinger Bands
# =============================================================================

@dataclass
class BBANDSResult:
    """Bollinger Bands result."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series


def BBANDS(
    close: pd.Series,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0
) -> BBANDSResult:
    """
    Bollinger Bands.
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    timeperiod : int
        MA period (default 20)
    nbdevup : float
        Upper band std multiplier
    nbdevdn : float
        Lower band std multiplier
    matype : int
        MA type (0=SMA, 1=EMA, 2=WMA, etc.)
        
    Returns
    -------
    BBANDSResult
        Upper, middle, lower bands + bandwidth + %B
    """
    upper, middle, lower = talib.BBANDS(
        close.values,
        timeperiod=timeperiod,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
        matype=matype
    )
    
    upper = pd.Series(upper, index=close.index, name='bb_upper')
    middle = pd.Series(middle, index=close.index, name='bb_middle')
    lower = pd.Series(lower, index=close.index, name='bb_lower')
    
    # Derived metrics
    bandwidth = (upper - lower) / middle
    bandwidth.name = 'bb_bandwidth'
    
    percent_b = (close - lower) / (upper - lower)
    percent_b.name = 'bb_percent_b'
    
    return BBANDSResult(
        upper=upper,
        middle=middle,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b
    )


# =============================================================================
# Parabolic SAR
# =============================================================================

def SAR(
    high: pd.Series,
    low: pd.Series,
    acceleration: float = 0.02,
    maximum: float = 0.2
) -> pd.Series:
    """
    Parabolic SAR (Stop and Reverse).
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    acceleration : float
        Acceleration factor (default 0.02)
    maximum : float
        Maximum acceleration (default 0.2)
        
    Returns
    -------
    pd.Series
        SAR values
    """
    result = talib.SAR(
        high.values, low.values,
        acceleration=acceleration,
        maximum=maximum
    )
    return pd.Series(result, index=high.index, name='sar')


# =============================================================================
# Midpoint / Midprice
# =============================================================================

def MIDPOINT(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """MidPoint over period."""
    result = talib.MIDPOINT(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'midpoint_{timeperiod}')


def MIDPRICE(high: pd.Series, low: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Midpoint Price over period."""
    result = talib.MIDPRICE(high.values, low.values, timeperiod=timeperiod)
    return pd.Series(result, index=high.index, name=f'midprice_{timeperiod}')


# =============================================================================
# Overlap Feature Generator
# =============================================================================

class OverlapFeatureGenerator(FeatureGenerator):
    """
    Generate overlap study features using TA-Lib.
    
    Features:
    - Multiple MA types (SMA, EMA, WMA, DEMA, TEMA, KAMA)
    - Bollinger Bands (%B, bandwidth)
    - Parabolic SAR
    - MA crossover signals
    
    Examples
    --------
    >>> gen = OverlapFeatureGenerator(ma_periods=[10, 20, 50])
    >>> result = gen.fit_transform(ohlcv_df)
    """
    
    def __init__(
        self,
        ma_periods: List[int] = None,
        ma_types: List[str] = None,
        bb_period: int = 20,
        bb_std: float = 2.0,
        include_sar: bool = True,
    ):
        """
        Initialize overlap feature generator.
        
        Parameters
        ----------
        ma_periods : List[int]
            MA periods to calculate (default [10, 20, 50, 200])
        ma_types : List[str]
            MA types: 'sma', 'ema', 'wma', 'dema', 'tema', 'kama'
        bb_period : int
            Bollinger Bands period
        bb_std : float
            Bollinger Bands std multiplier
        include_sar : bool
            Include Parabolic SAR
        """
        config = FeatureConfig(
            name='talib_overlap',
            feature_type=FeatureType.TECHNICAL,
            params={
                'ma_periods': ma_periods or [10, 20, 50, 200],
                'ma_types': ma_types or ['sma', 'ema'],
            }
        )
        super().__init__(config)
        
        self.ma_periods = ma_periods or [10, 20, 50, 200]
        self.ma_types = ma_types or ['sma', 'ema']
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.include_sar = include_sar
        
        # MA function mapping
        self._ma_funcs = {
            'sma': SMA,
            'ema': EMA,
            'wma': WMA,
            'dema': DEMA,
            'tema': TEMA,
            'kama': KAMA,
        }
    
    def fit(self, data: pd.DataFrame) -> 'OverlapFeatureGenerator':
        """Fit (no-op for technical indicators)."""
        self._validate_data(data)
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """Generate overlap features."""
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)
        
        # Moving Averages
        for ma_type in self.ma_types:
            if ma_type in self._ma_funcs:
                ma_func = self._ma_funcs[ma_type]
                for period in self.ma_periods:
                    features[f'{ma_type}_{period}'] = ma_func(close, period)
        
        # MA ratios (price relative to MA)
        for period in self.ma_periods:
            sma = SMA(close, period)
            features[f'price_sma_{period}_ratio'] = close / sma
        
        # Bollinger Bands
        bb = BBANDS(close, self.bb_period, self.bb_std, self.bb_std)
        features['bb_percent_b'] = bb.percent_b
        features['bb_bandwidth'] = bb.bandwidth
        
        # Parabolic SAR
        if self.include_sar:
            sar = SAR(high, low)
            features['sar'] = sar
            features['sar_signal'] = np.where(close > sar, 1, -1)
        
        # Handle NaN
        features = self._handle_nan(features, 'ffill')
        
        return FeatureResult(
            features=features,
            feature_names=list(features.columns),
            config=self.config,
            metadata={
                'ma_periods': self.ma_periods,
                'ma_types': self.ma_types,
                'n_features': len(features.columns),
            }
        )
