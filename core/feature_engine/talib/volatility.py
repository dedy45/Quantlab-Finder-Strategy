"""
TA-Lib Volatility Indicators.

Volatility indicators measure the rate of price movement:
- ATR (Average True Range)
- NATR (Normalized ATR)
- TRANGE (True Range)

Reference: https://ta-lib.org/functions/
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import talib

from ..base import FeatureGenerator, FeatureConfig, FeatureResult, FeatureType

logger = logging.getLogger(__name__)


# =============================================================================
# Volatility Indicators
# =============================================================================

def ATR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """
    Average True Range.
    
    ATR = SMA(TR, timeperiod)
    TR = max(H-L, |H-Cp|, |L-Cp|)
    """
    result = talib.ATR(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'atr_{timeperiod}')


def NATR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """
    Normalized Average True Range.
    
    NATR = ATR / Close * 100
    """
    result = talib.NATR(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'natr_{timeperiod}')


def TRANGE(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    True Range.
    
    TR = max(H-L, |H-Cp|, |L-Cp|)
    """
    result = talib.TRANGE(high.values, low.values, close.values)
    return pd.Series(result, index=close.index, name='trange')


# =============================================================================
# Volatility Feature Generator
# =============================================================================

class VolatilityFeatureGenerator(FeatureGenerator):
    """
    Generate volatility features using TA-Lib.
    
    Features:
    - ATR (multiple periods)
    - NATR (normalized)
    - True Range
    - ATR ratio (ATR/Close)
    - Volatility regime signals
    
    Examples
    --------
    >>> gen = VolatilityFeatureGenerator(atr_periods=[7, 14, 21])
    >>> result = gen.fit_transform(ohlcv_df)
    """
    
    def __init__(
        self,
        atr_periods: List[int] = None,
        include_natr: bool = True,
        include_trange: bool = True,
        include_regime: bool = True,
        high_vol_threshold: float = 1.5,
        low_vol_threshold: float = 0.5,
    ):
        """
        Initialize volatility feature generator.
        
        Parameters
        ----------
        atr_periods : List[int]
            ATR periods (default [7, 14, 21])
        include_natr : bool
            Include Normalized ATR
        include_trange : bool
            Include True Range
        include_regime : bool
            Include volatility regime signals
        high_vol_threshold : float
            High volatility threshold (ATR ratio)
        low_vol_threshold : float
            Low volatility threshold (ATR ratio)
        """
        config = FeatureConfig(
            name='talib_volatility',
            feature_type=FeatureType.TECHNICAL,
            params={'atr_periods': atr_periods or [7, 14, 21]}
        )
        super().__init__(config)
        
        self.atr_periods = atr_periods or [7, 14, 21]
        self.include_natr = include_natr
        self.include_trange = include_trange
        self.include_regime = include_regime
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
    
    def fit(self, data: pd.DataFrame) -> 'VolatilityFeatureGenerator':
        """Fit (no-op for technical indicators)."""
        self._validate_data(data)
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """Generate volatility features."""
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)
        
        # ATR (multiple periods)
        for period in self.atr_periods:
            features[f'atr_{period}'] = ATR(high, low, close, period)
            features[f'atr_{period}_ratio'] = features[f'atr_{period}'] / close
        
        # NATR
        if self.include_natr:
            for period in self.atr_periods:
                features[f'natr_{period}'] = NATR(high, low, close, period)
        
        # True Range
        if self.include_trange:
            features['trange'] = TRANGE(high, low, close)
            features['trange_ratio'] = features['trange'] / close
        
        # Volatility Regime
        if self.include_regime:
            # Use 14-period ATR ratio for regime detection
            atr_ratio = features.get('atr_14_ratio', features[f'atr_{self.atr_periods[0]}_ratio'])
            atr_ratio_ma = atr_ratio.rolling(20).mean()
            
            # Relative volatility
            rel_vol = atr_ratio / atr_ratio_ma
            features['vol_regime'] = np.where(
                rel_vol > self.high_vol_threshold, 2,  # High vol
                np.where(rel_vol < self.low_vol_threshold, 0, 1)  # Low vol / Normal
            )
            features['rel_volatility'] = rel_vol
        
        # Handle NaN
        features = self._handle_nan(features, 'ffill')
        
        return FeatureResult(
            features=features,
            feature_names=list(features.columns),
            config=self.config,
            metadata={
                'atr_periods': self.atr_periods,
                'n_features': len(features.columns),
            }
        )
