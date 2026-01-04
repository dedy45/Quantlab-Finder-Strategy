"""
Technical Indicators for Feature Engineering.

Implements classic technical indicators as features:
- RSI (Relative Strength Index) - Kompendium #20
- Bollinger Bands - Kompendium #21
- Z-Score - Kompendium #22
- Moving Averages (SMA, EMA)
- ATR (Average True Range)
- MACD

Reference: Kompendium Rumus #20-22
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from .base import FeatureGenerator, FeatureConfig, FeatureResult, FeatureType


# =============================================================================
# RSI - Relative Strength Index (Kompendium #20)
# =============================================================================

def calculate_rsi(
    prices: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Formula:
        RSI = 100 - 100/(1 + RS)
        RS = Average Gain / Average Loss
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int
        Lookback period (default 14)
        
    Returns
    -------
    pd.Series
        RSI values (0-100)
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral when undefined


def rsi_signal(
    prices: pd.Series,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70
) -> pd.Series:
    """
    Generate trading signal from RSI.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int
        RSI period
    oversold : float
        Oversold threshold (buy signal)
    overbought : float
        Overbought threshold (sell signal)
        
    Returns
    -------
    pd.Series
        Signal: 1 (buy), -1 (sell), 0 (neutral)
    """
    rsi = calculate_rsi(prices, period)
    
    signal = pd.Series(0, index=prices.index)
    signal[rsi < oversold] = 1    # Oversold → Buy
    signal[rsi > overbought] = -1  # Overbought → Sell
    
    return signal


# =============================================================================
# Bollinger Bands (Kompendium #21)
# =============================================================================

@dataclass
class BollingerBands:
    """Bollinger Bands result."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> BollingerBands:
    """
    Calculate Bollinger Bands.
    
    Formula:
        Upper = MA + k×σ
        Lower = MA - k×σ
        %B = (Price - Lower) / (Upper - Lower)
        Bandwidth = (Upper - Lower) / MA
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    window : int
        Moving average window
    num_std : float
        Number of standard deviations
        
    Returns
    -------
    BollingerBands
        Bollinger bands components
    """
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper = ma + num_std * std
    lower = ma - num_std * std
    
    bandwidth = (upper - lower) / ma
    percent_b = (prices - lower) / (upper - lower)
    
    return BollingerBands(
        upper=upper,
        middle=ma,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b
    )


def bollinger_signal(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> pd.Series:
    """
    Generate trading signal from Bollinger Bands.
    
    Mean reversion strategy:
    - Buy when price touches lower band
    - Sell when price touches upper band
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    window : int
        MA window
    num_std : float
        Number of std deviations
        
    Returns
    -------
    pd.Series
        Signal: 1 (buy), -1 (sell), 0 (neutral)
    """
    bb = calculate_bollinger_bands(prices, window, num_std)
    
    signal = pd.Series(0, index=prices.index)
    signal[prices <= bb.lower] = 1   # At lower band → Buy
    signal[prices >= bb.upper] = -1  # At upper band → Sell
    
    return signal


# =============================================================================
# Z-Score (Kompendium #22)
# =============================================================================

def calculate_zscore(
    series: pd.Series,
    window: Optional[int] = None
) -> pd.Series:
    """
    Calculate Z-Score.
    
    Formula:
        Z = (x - μ) / σ
    
    Parameters
    ----------
    series : pd.Series
        Input series
    window : int, optional
        Rolling window. If None, uses full history.
        
    Returns
    -------
    pd.Series
        Z-score values
    """
    if window:
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
    else:
        mean = series.expanding().mean()
        std = series.expanding().std()
    
    zscore = (series - mean) / std.replace(0, np.nan)
    return zscore.fillna(0)


def zscore_signal(
    series: pd.Series,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5
) -> pd.Series:
    """
    Generate mean reversion signal from Z-Score.
    
    Parameters
    ----------
    series : pd.Series
        Input series (prices or spread)
    window : int
        Rolling window
    entry_z : float
        Entry threshold (absolute)
    exit_z : float
        Exit threshold (absolute)
        
    Returns
    -------
    pd.Series
        Signal: 1 (buy), -1 (sell), 0 (neutral)
    """
    z = calculate_zscore(series, window)
    
    signal = pd.Series(0, index=series.index)
    signal[z < -entry_z] = 1   # Very oversold → Buy
    signal[z > entry_z] = -1   # Very overbought → Sell
    
    return signal


# =============================================================================
# Moving Averages
# =============================================================================

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window=window).mean()


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=span, adjust=False).mean()


def ma_crossover_signal(
    prices: pd.Series,
    fast_window: int = 20,
    slow_window: int = 50,
    ma_type: str = 'sma'
) -> pd.Series:
    """
    Generate signal from MA crossover.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    fast_window : int
        Fast MA window
    slow_window : int
        Slow MA window
    ma_type : str
        'sma' or 'ema'
        
    Returns
    -------
    pd.Series
        Signal: 1 (bullish), -1 (bearish)
    """
    if ma_type == 'ema':
        fast_ma = calculate_ema(prices, fast_window)
        slow_ma = calculate_ema(prices, slow_window)
    else:
        fast_ma = calculate_sma(prices, fast_window)
        slow_ma = calculate_sma(prices, slow_window)
    
    signal = pd.Series(0, index=prices.index)
    signal[fast_ma > slow_ma] = 1
    signal[fast_ma < slow_ma] = -1
    
    return signal


# =============================================================================
# ATR - Average True Range
# =============================================================================

def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Calculate Average True Range.
    
    True Range = max(H-L, |H-Cp|, |L-Cp|)
    ATR = SMA(TR, window)
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    window : int
        ATR period
        
    Returns
    -------
    pd.Series
        ATR values
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr


def calculate_atr_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Calculate ATR as ratio of price (normalized).
    
    Returns
    -------
    pd.Series
        ATR / Close (percentage volatility)
    """
    atr = calculate_atr(high, low, close, window)
    return atr / close


# =============================================================================
# MACD
# =============================================================================

@dataclass
class MACDResult:
    """MACD calculation result."""
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> MACDResult:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    fast_period : int
        Fast EMA period
    slow_period : int
        Slow EMA period
    signal_period : int
        Signal line EMA period
        
    Returns
    -------
    MACDResult
        MACD components
    """
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram
    )


def macd_signal(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.Series:
    """
    Generate trading signal from MACD.
    
    Returns
    -------
    pd.Series
        Signal: 1 (bullish), -1 (bearish)
    """
    macd = calculate_macd(prices, fast_period, slow_period, signal_period)
    
    signal = pd.Series(0, index=prices.index)
    signal[macd.histogram > 0] = 1
    signal[macd.histogram < 0] = -1
    
    return signal



# =============================================================================
# Technical Feature Generator (Combined)
# =============================================================================

class TechnicalFeatureGenerator(FeatureGenerator):
    """
    Generate multiple technical indicator features.
    
    Combines RSI, Bollinger, Z-Score, MA, ATR, MACD into
    a single feature set.
    
    Examples
    --------
    >>> gen = TechnicalFeatureGenerator()
    >>> result = gen.fit_transform(ohlcv_df)
    >>> print(result.feature_names)
    """
    
    DEFAULT_INDICATORS = [
        'rsi', 'bollinger', 'zscore', 'ma_cross', 'atr', 'macd'
    ]
    
    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        rsi_period: int = 14,
        bb_window: int = 20,
        bb_std: float = 2.0,
        zscore_window: int = 20,
        ma_fast: int = 20,
        ma_slow: int = 50,
        atr_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        """
        Initialize technical feature generator.
        
        Parameters
        ----------
        indicators : List[str], optional
            List of indicators to generate. Default: all
        rsi_period : int
            RSI lookback period
        bb_window : int
            Bollinger Bands window
        bb_std : float
            Bollinger Bands std multiplier
        zscore_window : int
            Z-Score window
        ma_fast : int
            Fast MA period
        ma_slow : int
            Slow MA period
        atr_period : int
            ATR period
        macd_fast : int
            MACD fast EMA
        macd_slow : int
            MACD slow EMA
        macd_signal : int
            MACD signal line EMA
        """
        config = FeatureConfig(
            name='technical_features',
            feature_type=FeatureType.TECHNICAL,
            params={
                'indicators': indicators or self.DEFAULT_INDICATORS,
                'rsi_period': rsi_period,
                'bb_window': bb_window,
            }
        )
        super().__init__(config)
        
        self.indicators = indicators or self.DEFAULT_INDICATORS
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.zscore_window = zscore_window
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.atr_period = atr_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
    
    def fit(self, data: pd.DataFrame) -> 'TechnicalFeatureGenerator':
        """Fit (no-op for technical indicators)."""
        self._validate_data(data)
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """
        Generate technical features.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
            
        Returns
        -------
        FeatureResult
            Technical indicator features
        """
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # Get price columns
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        high = data['high'] if 'high' in data.columns else close
        low = data['low'] if 'low' in data.columns else close
        
        # RSI
        if 'rsi' in self.indicators:
            features['rsi'] = calculate_rsi(close, self.rsi_period)
            features['rsi_signal'] = rsi_signal(close, self.rsi_period)
        
        # Bollinger Bands
        if 'bollinger' in self.indicators:
            bb = calculate_bollinger_bands(close, self.bb_window, self.bb_std)
            features['bb_percent_b'] = bb.percent_b
            features['bb_bandwidth'] = bb.bandwidth
            features['bb_signal'] = bollinger_signal(close, self.bb_window, self.bb_std)
        
        # Z-Score
        if 'zscore' in self.indicators:
            features['zscore'] = calculate_zscore(close, self.zscore_window)
            features['zscore_signal'] = zscore_signal(close, self.zscore_window)
        
        # MA Crossover
        if 'ma_cross' in self.indicators:
            features['sma_fast'] = calculate_sma(close, self.ma_fast)
            features['sma_slow'] = calculate_sma(close, self.ma_slow)
            features['ma_cross_signal'] = ma_crossover_signal(
                close, self.ma_fast, self.ma_slow
            )
        
        # ATR
        if 'atr' in self.indicators:
            features['atr'] = calculate_atr(high, low, close, self.atr_period)
            features['atr_ratio'] = calculate_atr_ratio(high, low, close, self.atr_period)
        
        # MACD
        if 'macd' in self.indicators:
            macd = calculate_macd(
                close, self.macd_fast, self.macd_slow, self.macd_signal_period
            )
            features['macd_line'] = macd.macd_line
            features['macd_signal'] = macd.signal_line
            features['macd_histogram'] = macd.histogram
            features['macd_trade_signal'] = macd_signal(
                close, self.macd_fast, self.macd_slow, self.macd_signal_period
            )
        
        # Handle NaN
        features = self._handle_nan(features, 'ffill')
        
        return FeatureResult(
            features=features,
            feature_names=list(features.columns),
            config=self.config,
            metadata={
                'indicators': self.indicators,
                'n_features': len(features.columns)
            }
        )
