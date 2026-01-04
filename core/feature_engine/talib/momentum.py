"""
TA-Lib Momentum Indicators.

Momentum indicators measure the rate of change in prices:
- RSI, Stochastic, MACD
- ADX, CCI, MFI, Williams %R
- AROON, ROC, MOM, TRIX

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
# RSI Family
# =============================================================================

def RSI(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    RSI = 100 - 100/(1 + RS)
    RS = Average Gain / Average Loss
    """
    result = talib.RSI(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'rsi_{timeperiod}')


@dataclass
class STOCHResult:
    """Stochastic result."""
    slowk: pd.Series
    slowd: pd.Series


def STOCH(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0
) -> STOCHResult:
    """Stochastic Oscillator."""
    slowk, slowd = talib.STOCH(
        high.values, low.values, close.values,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=slowk_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype
    )
    return STOCHResult(
        slowk=pd.Series(slowk, index=close.index, name='stoch_k'),
        slowd=pd.Series(slowd, index=close.index, name='stoch_d')
    )


@dataclass
class STOCHFResult:
    """Fast Stochastic result."""
    fastk: pd.Series
    fastd: pd.Series


def STOCHF(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0
) -> STOCHFResult:
    """Fast Stochastic Oscillator."""
    fastk, fastd = talib.STOCHF(
        high.values, low.values, close.values,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    return STOCHFResult(
        fastk=pd.Series(fastk, index=close.index, name='stochf_k'),
        fastd=pd.Series(fastd, index=close.index, name='stochf_d')
    )


@dataclass
class STOCHRSIResult:
    """Stochastic RSI result."""
    fastk: pd.Series
    fastd: pd.Series


def STOCHRSI(
    close: pd.Series,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0
) -> STOCHRSIResult:
    """Stochastic RSI."""
    fastk, fastd = talib.STOCHRSI(
        close.values,
        timeperiod=timeperiod,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    return STOCHRSIResult(
        fastk=pd.Series(fastk, index=close.index, name='stochrsi_k'),
        fastd=pd.Series(fastd, index=close.index, name='stochrsi_d')
    )


# =============================================================================
# MACD Family
# =============================================================================

@dataclass
class MACDResult:
    """MACD result."""
    macd: pd.Series
    signal: pd.Series
    hist: pd.Series


def MACD(
    close: pd.Series,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9
) -> MACDResult:
    """Moving Average Convergence/Divergence."""
    macd, signal, hist = talib.MACD(
        close.values,
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod
    )
    return MACDResult(
        macd=pd.Series(macd, index=close.index, name='macd'),
        signal=pd.Series(signal, index=close.index, name='macd_signal'),
        hist=pd.Series(hist, index=close.index, name='macd_hist')
    )


def MACDEXT(
    close: pd.Series,
    fastperiod: int = 12,
    fastmatype: int = 0,
    slowperiod: int = 26,
    slowmatype: int = 0,
    signalperiod: int = 9,
    signalmatype: int = 0
) -> MACDResult:
    """MACD with controllable MA type."""
    macd, signal, hist = talib.MACDEXT(
        close.values,
        fastperiod=fastperiod,
        fastmatype=fastmatype,
        slowperiod=slowperiod,
        slowmatype=slowmatype,
        signalperiod=signalperiod,
        signalmatype=signalmatype
    )
    return MACDResult(
        macd=pd.Series(macd, index=close.index, name='macdext'),
        signal=pd.Series(signal, index=close.index, name='macdext_signal'),
        hist=pd.Series(hist, index=close.index, name='macdext_hist')
    )


def MACDFIX(close: pd.Series, signalperiod: int = 9) -> MACDResult:
    """MACD Fix 12/26."""
    macd, signal, hist = talib.MACDFIX(close.values, signalperiod=signalperiod)
    return MACDResult(
        macd=pd.Series(macd, index=close.index, name='macdfix'),
        signal=pd.Series(signal, index=close.index, name='macdfix_signal'),
        hist=pd.Series(hist, index=close.index, name='macdfix_hist')
    )


# =============================================================================
# ADX Family (Trend Strength)
# =============================================================================

def ADX(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Average Directional Movement Index."""
    result = talib.ADX(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'adx_{timeperiod}')


def ADXR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Average Directional Movement Index Rating."""
    result = talib.ADXR(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'adxr_{timeperiod}')


def DX(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Directional Movement Index."""
    result = talib.DX(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'dx_{timeperiod}')


def MINUS_DI(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Minus Directional Indicator."""
    result = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'minus_di_{timeperiod}')


def PLUS_DI(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Plus Directional Indicator."""
    result = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'plus_di_{timeperiod}')


# =============================================================================
# Other Momentum Indicators
# =============================================================================

def APO(close: pd.Series, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> pd.Series:
    """Absolute Price Oscillator."""
    result = talib.APO(close.values, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return pd.Series(result, index=close.index, name='apo')


def PPO(close: pd.Series, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> pd.Series:
    """Percentage Price Oscillator."""
    result = talib.PPO(close.values, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return pd.Series(result, index=close.index, name='ppo')


def MOM(close: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Momentum."""
    result = talib.MOM(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'mom_{timeperiod}')


def ROC(close: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Rate of Change."""
    result = talib.ROC(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'roc_{timeperiod}')


def ROCP(close: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Rate of Change Percentage."""
    result = talib.ROCP(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'rocp_{timeperiod}')


def ROCR(close: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Rate of Change Ratio."""
    result = talib.ROCR(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'rocr_{timeperiod}')


def CCI(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Commodity Channel Index."""
    result = talib.CCI(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'cci_{timeperiod}')


def CMO(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Chande Momentum Oscillator."""
    result = talib.CMO(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'cmo_{timeperiod}')


def MFI(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Money Flow Index."""
    result = talib.MFI(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        volume.values.astype(float),
        timeperiod=timeperiod
    )
    return pd.Series(result, index=close.index, name=f'mfi_{timeperiod}')


def WILLR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14
) -> pd.Series:
    """Williams' %R."""
    result = talib.WILLR(high.values, low.values, close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'willr_{timeperiod}')


def ULTOSC(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28
) -> pd.Series:
    """Ultimate Oscillator."""
    result = talib.ULTOSC(
        high.values, low.values, close.values,
        timeperiod1=timeperiod1,
        timeperiod2=timeperiod2,
        timeperiod3=timeperiod3
    )
    return pd.Series(result, index=close.index, name='ultosc')


def TRIX(close: pd.Series, timeperiod: int = 30) -> pd.Series:
    """1-day Rate-Of-Change of Triple Smooth EMA."""
    result = talib.TRIX(close.values, timeperiod=timeperiod)
    return pd.Series(result, index=close.index, name=f'trix_{timeperiod}')


# =============================================================================
# AROON
# =============================================================================

@dataclass
class AROONResult:
    """AROON result."""
    aroondown: pd.Series
    aroonup: pd.Series


def AROON(
    high: pd.Series,
    low: pd.Series,
    timeperiod: int = 14
) -> AROONResult:
    """Aroon Indicator."""
    aroondown, aroonup = talib.AROON(high.values, low.values, timeperiod=timeperiod)
    return AROONResult(
        aroondown=pd.Series(aroondown, index=high.index, name='aroon_down'),
        aroonup=pd.Series(aroonup, index=high.index, name='aroon_up')
    )


def AROONOSC(high: pd.Series, low: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Aroon Oscillator."""
    result = talib.AROONOSC(high.values, low.values, timeperiod=timeperiod)
    return pd.Series(result, index=high.index, name=f'aroonosc_{timeperiod}')


def BOP(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """Balance Of Power."""
    result = talib.BOP(open_.values, high.values, low.values, close.values)
    return pd.Series(result, index=close.index, name='bop')


# =============================================================================
# Momentum Feature Generator
# =============================================================================

class MomentumFeatureGenerator(FeatureGenerator):
    """
    Generate momentum indicator features using TA-Lib.
    
    Features:
    - RSI (multiple periods)
    - Stochastic (K, D)
    - MACD (line, signal, histogram)
    - ADX (trend strength)
    - CCI, MFI, Williams %R
    - ROC, MOM
    
    Examples
    --------
    >>> gen = MomentumFeatureGenerator(rsi_periods=[7, 14, 21])
    >>> result = gen.fit_transform(ohlcv_df)
    """
    
    def __init__(
        self,
        rsi_periods: List[int] = None,
        include_stoch: bool = True,
        include_macd: bool = True,
        include_adx: bool = True,
        include_cci: bool = True,
        include_mfi: bool = True,
        include_roc: bool = True,
    ):
        """
        Initialize momentum feature generator.
        
        Parameters
        ----------
        rsi_periods : List[int]
            RSI periods (default [7, 14, 21])
        include_stoch : bool
            Include Stochastic
        include_macd : bool
            Include MACD
        include_adx : bool
            Include ADX
        include_cci : bool
            Include CCI
        include_mfi : bool
            Include MFI (requires volume)
        include_roc : bool
            Include ROC/MOM
        """
        config = FeatureConfig(
            name='talib_momentum',
            feature_type=FeatureType.TECHNICAL,
            params={'rsi_periods': rsi_periods or [7, 14, 21]}
        )
        super().__init__(config)
        
        self.rsi_periods = rsi_periods or [7, 14, 21]
        self.include_stoch = include_stoch
        self.include_macd = include_macd
        self.include_adx = include_adx
        self.include_cci = include_cci
        self.include_mfi = include_mfi
        self.include_roc = include_roc
    
    def fit(self, data: pd.DataFrame) -> 'MomentumFeatureGenerator':
        """Fit (no-op for technical indicators)."""
        self._validate_data(data)
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """Generate momentum features."""
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)
        volume = data.get('volume', None)
        
        # RSI
        for period in self.rsi_periods:
            features[f'rsi_{period}'] = RSI(close, period)
        
        # Stochastic
        if self.include_stoch:
            stoch = STOCH(high, low, close)
            features['stoch_k'] = stoch.slowk
            features['stoch_d'] = stoch.slowd
        
        # MACD
        if self.include_macd:
            macd = MACD(close)
            features['macd'] = macd.macd
            features['macd_signal'] = macd.signal
            features['macd_hist'] = macd.hist
        
        # ADX
        if self.include_adx:
            features['adx'] = ADX(high, low, close)
            features['plus_di'] = PLUS_DI(high, low, close)
            features['minus_di'] = MINUS_DI(high, low, close)
        
        # CCI
        if self.include_cci:
            features['cci'] = CCI(high, low, close)
        
        # MFI (requires volume)
        if self.include_mfi and volume is not None:
            features['mfi'] = MFI(high, low, close, volume)
        
        # ROC / MOM
        if self.include_roc:
            features['roc_10'] = ROC(close, 10)
            features['mom_10'] = MOM(close, 10)
        
        # Handle NaN
        features = self._handle_nan(features, 'ffill')
        
        return FeatureResult(
            features=features,
            feature_names=list(features.columns),
            config=self.config,
            metadata={
                'rsi_periods': self.rsi_periods,
                'n_features': len(features.columns),
            }
        )
