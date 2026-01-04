"""
TA-Lib Volume Indicators.

Volume indicators analyze trading volume:
- OBV (On Balance Volume)
- AD (Accumulation/Distribution)
- ADOSC (A/D Oscillator)

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
# Volume Indicators
# =============================================================================

def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume.
    
    OBV = cumsum(volume * sign(close_change))
    """
    result = talib.OBV(close.values.astype(float), volume.values.astype(float))
    return pd.Series(result, index=close.index, name='obv')


def AD(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Chaikin A/D Line (Accumulation/Distribution).
    
    AD = cumsum(((C-L) - (H-C)) / (H-L) * V)
    """
    result = talib.AD(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        volume.values.astype(float)
    )
    return pd.Series(result, index=close.index, name='ad')


def ADOSC(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fastperiod: int = 3,
    slowperiod: int = 10
) -> pd.Series:
    """
    Chaikin A/D Oscillator.
    
    ADOSC = EMA(AD, fast) - EMA(AD, slow)
    """
    result = talib.ADOSC(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        volume.values.astype(float),
        fastperiod=fastperiod,
        slowperiod=slowperiod
    )
    return pd.Series(result, index=close.index, name='adosc')


# =============================================================================
# Volume Feature Generator
# =============================================================================

class VolumeFeatureGenerator(FeatureGenerator):
    """
    Generate volume indicator features using TA-Lib.
    
    Features:
    - OBV (On Balance Volume)
    - AD (Accumulation/Distribution)
    - ADOSC (A/D Oscillator)
    - Volume ratios and signals
    
    Examples
    --------
    >>> gen = VolumeFeatureGenerator()
    >>> result = gen.fit_transform(ohlcv_df)
    """
    
    def __init__(
        self,
        include_obv: bool = True,
        include_ad: bool = True,
        include_adosc: bool = True,
        volume_ma_periods: List[int] = None,
    ):
        """
        Initialize volume feature generator.
        
        Parameters
        ----------
        include_obv : bool
            Include On Balance Volume
        include_ad : bool
            Include Accumulation/Distribution
        include_adosc : bool
            Include A/D Oscillator
        volume_ma_periods : List[int]
            Volume MA periods for ratio calculation
        """
        config = FeatureConfig(
            name='talib_volume',
            feature_type=FeatureType.TECHNICAL,
            params={'volume_ma_periods': volume_ma_periods or [10, 20]}
        )
        super().__init__(config)
        
        self.include_obv = include_obv
        self.include_ad = include_ad
        self.include_adosc = include_adosc
        self.volume_ma_periods = volume_ma_periods or [10, 20]
    
    def fit(self, data: pd.DataFrame) -> 'VolumeFeatureGenerator':
        """Fit (no-op for technical indicators)."""
        self._validate_data(data)
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """Generate volume features."""
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)
        volume = data.get('volume', None)
        
        if volume is None:
            logger.warning("Volume data not available. Skipping volume features.")
            return FeatureResult(
                features=features,
                feature_names=[],
                config=self.config,
                metadata={'error': 'No volume data'}
            )
        
        # OBV
        if self.include_obv:
            features['obv'] = OBV(close, volume)
            # OBV rate of change
            features['obv_roc'] = features['obv'].pct_change(10)
        
        # AD
        if self.include_ad:
            features['ad'] = AD(high, low, close, volume)
            features['ad_roc'] = features['ad'].pct_change(10)
        
        # ADOSC
        if self.include_adosc:
            features['adosc'] = ADOSC(high, low, close, volume)
        
        # Volume ratios
        for period in self.volume_ma_periods:
            vol_ma = volume.rolling(period).mean()
            features[f'volume_ratio_{period}'] = volume / vol_ma
        
        # Handle NaN
        features = self._handle_nan(features, 'ffill')
        
        return FeatureResult(
            features=features,
            feature_names=list(features.columns),
            config=self.config,
            metadata={
                'volume_ma_periods': self.volume_ma_periods,
                'n_features': len(features.columns),
            }
        )
