"""
TA-Lib Integration Module - High-performance technical analysis.

TA-Lib provides 150+ technical indicators with C-optimized performance.
This module wraps TA-Lib functions into QuantLab's feature engineering pipeline.

Categories:
- Overlap Studies (MA, BBANDS, SAR, etc.)
- Momentum Indicators (RSI, MACD, Stochastic, etc.)
- Volume Indicators (OBV, AD, etc.)
- Volatility Indicators (ATR, NATR, TRANGE)
- Cycle Indicators (HT_DCPERIOD, HT_SINE, etc.)
- Pattern Recognition (CDL* - 61 candlestick patterns)
- Statistic Functions (STDDEV, VAR, CORREL, etc.)

Installation:
    # Windows (recommended)
    pip install TA-Lib-Precompiled
    
    # Or with conda
    conda install -c conda-forge ta-lib

Usage:
    from core.feature_engine.talib import TALibFeatureGenerator, check_talib
    
    # Check if TA-Lib is available
    if check_talib():
        gen = TALibFeatureGenerator()
        features = gen.fit_transform(ohlcv_df)
    
    # Or use individual functions
    from core.feature_engine.talib import momentum, overlap, volatility
    rsi = momentum.RSI(close, timeperiod=14)
    bbands = overlap.BBANDS(close, timeperiod=20)

Reference: https://ta-lib.org/
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Check TA-Lib availability
TALIB_AVAILABLE = False
try:
    import talib
    TALIB_AVAILABLE = True
    TALIB_VERSION = talib.__version__
    logger.info(f"[OK] TA-Lib {TALIB_VERSION} available")
except ImportError:
    TALIB_VERSION = None
    logger.warning(
        "[WARN] TA-Lib not installed. "
        "Install with: pip install TA-Lib-Precompiled (Windows) "
        "or conda install -c conda-forge ta-lib"
    )


def check_talib() -> bool:
    """Check if TA-Lib is available."""
    return TALIB_AVAILABLE


def get_talib_version() -> Optional[str]:
    """Get TA-Lib version if available."""
    return TALIB_VERSION


# Import submodules only if TA-Lib is available
if TALIB_AVAILABLE:
    from .overlap import (
        SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA,
        BBANDS, SAR, MIDPOINT, MIDPRICE,
        OverlapFeatureGenerator,
    )
    from .momentum import (
        RSI, STOCH, STOCHF, STOCHRSI,
        MACD, MACDEXT, MACDFIX,
        ADX, ADXR, APO, PPO, MOM, ROC, ROCP, ROCR,
        CCI, CMO, MFI, WILLR, ULTOSC, TRIX,
        AROON, AROONOSC, BOP, DX, MINUS_DI, PLUS_DI,
        MomentumFeatureGenerator,
    )
    from .volatility import (
        ATR, NATR, TRANGE,
        VolatilityFeatureGenerator,
    )
    from .volume import (
        OBV, AD, ADOSC,
        VolumeFeatureGenerator,
    )
    from .pattern import (
        get_all_patterns, get_bullish_patterns, get_bearish_patterns,
        scan_patterns, PatternScanner,
    )
    from .generator import (
        TALibFeatureGenerator,
        TALibConfig,
    )
    
    __all__ = [
        # Availability check
        'check_talib',
        'get_talib_version',
        'TALIB_AVAILABLE',
        'TALIB_VERSION',
        # Overlap
        'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA',
        'BBANDS', 'SAR', 'MIDPOINT', 'MIDPRICE',
        'OverlapFeatureGenerator',
        # Momentum
        'RSI', 'STOCH', 'STOCHF', 'STOCHRSI',
        'MACD', 'MACDEXT', 'MACDFIX',
        'ADX', 'ADXR', 'APO', 'PPO', 'MOM', 'ROC', 'ROCP', 'ROCR',
        'CCI', 'CMO', 'MFI', 'WILLR', 'ULTOSC', 'TRIX',
        'AROON', 'AROONOSC', 'BOP', 'DX', 'MINUS_DI', 'PLUS_DI',
        'MomentumFeatureGenerator',
        # Volatility
        'ATR', 'NATR', 'TRANGE',
        'VolatilityFeatureGenerator',
        # Volume
        'OBV', 'AD', 'ADOSC',
        'VolumeFeatureGenerator',
        # Pattern
        'get_all_patterns', 'get_bullish_patterns', 'get_bearish_patterns',
        'scan_patterns', 'PatternScanner',
        # Generator
        'TALibFeatureGenerator',
        'TALibConfig',
    ]
else:
    __all__ = [
        'check_talib',
        'get_talib_version',
        'TALIB_AVAILABLE',
        'TALIB_VERSION',
    ]
