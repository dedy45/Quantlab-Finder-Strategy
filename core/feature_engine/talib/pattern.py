"""
TA-Lib Candlestick Pattern Recognition.

TA-Lib provides 61 candlestick pattern recognition functions.
This module wraps them for easy pattern scanning.

Pattern Categories:
- Bullish Reversal (Hammer, Morning Star, Engulfing, etc.)
- Bearish Reversal (Shooting Star, Evening Star, etc.)
- Continuation (Doji, Spinning Top, etc.)

Reference: https://ta-lib.org/functions/
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib

from ..base import FeatureGenerator, FeatureConfig, FeatureResult, FeatureType

logger = logging.getLogger(__name__)


# =============================================================================
# Pattern Definitions
# =============================================================================

# All TA-Lib candlestick patterns
ALL_PATTERNS = {
    # Bullish Patterns
    'CDL2CROWS': 'Two Crows',
    'CDL3BLACKCROWS': 'Three Black Crows',
    'CDL3INSIDE': 'Three Inside Up/Down',
    'CDL3LINESTRIKE': 'Three-Line Strike',
    'CDL3OUTSIDE': 'Three Outside Up/Down',
    'CDL3STARSINSOUTH': 'Three Stars In The South',
    'CDL3WHITESOLDIERS': 'Three Advancing White Soldiers',
    'CDLABANDONEDBABY': 'Abandoned Baby',
    'CDLADVANCEBLOCK': 'Advance Block',
    'CDLBELTHOLD': 'Belt-hold',
    'CDLBREAKAWAY': 'Breakaway',
    'CDLCLOSINGMARUBOZU': 'Closing Marubozu',
    'CDLCONCEALBABYSWALL': 'Concealing Baby Swallow',
    'CDLCOUNTERATTACK': 'Counterattack',
    'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
    'CDLDOJI': 'Doji',
    'CDLDOJISTAR': 'Doji Star',
    'CDLDRAGONFLYDOJI': 'Dragonfly Doji',
    'CDLENGULFING': 'Engulfing Pattern',
    'CDLEVENINGDOJISTAR': 'Evening Doji Star',
    'CDLEVENINGSTAR': 'Evening Star',
    'CDLGAPSIDESIDEWHITE': 'Up/Down-gap side-by-side white lines',
    'CDLGRAVESTONEDOJI': 'Gravestone Doji',
    'CDLHAMMER': 'Hammer',
    'CDLHANGINGMAN': 'Hanging Man',
    'CDLHARAMI': 'Harami Pattern',
    'CDLHARAMICROSS': 'Harami Cross Pattern',
    'CDLHIGHWAVE': 'High-Wave Candle',
    'CDLHIKKAKE': 'Hikkake Pattern',
    'CDLHIKKAKEMOD': 'Modified Hikkake Pattern',
    'CDLHOMINGPIGEON': 'Homing Pigeon',
    'CDLIDENTICAL3CROWS': 'Identical Three Crows',
    'CDLINNECK': 'In-Neck Pattern',
    'CDLINVERTEDHAMMER': 'Inverted Hammer',
    'CDLKICKING': 'Kicking',
    'CDLKICKINGBYLENGTH': 'Kicking - bull/bear determined by the longer marubozu',
    'CDLLADDERBOTTOM': 'Ladder Bottom',
    'CDLLONGLEGGEDDOJI': 'Long Legged Doji',
    'CDLLONGLINE': 'Long Line Candle',
    'CDLMARUBOZU': 'Marubozu',
    'CDLMATCHINGLOW': 'Matching Low',
    'CDLMATHOLD': 'Mat Hold',
    'CDLMORNINGDOJISTAR': 'Morning Doji Star',
    'CDLMORNINGSTAR': 'Morning Star',
    'CDLONNECK': 'On-Neck Pattern',
    'CDLPIERCING': 'Piercing Pattern',
    'CDLRICKSHAWMAN': 'Rickshaw Man',
    'CDLRISEFALL3METHODS': 'Rising/Falling Three Methods',
    'CDLSEPARATINGLINES': 'Separating Lines',
    'CDLSHOOTINGSTAR': 'Shooting Star',
    'CDLSHORTLINE': 'Short Line Candle',
    'CDLSPINNINGTOP': 'Spinning Top',
    'CDLSTALLEDPATTERN': 'Stalled Pattern',
    'CDLSTICKSANDWICH': 'Stick Sandwich',
    'CDLTAKURI': 'Takuri (Dragonfly Doji with very long lower shadow)',
    'CDLTASUKIGAP': 'Tasuki Gap',
    'CDLTHRUSTING': 'Thrusting Pattern',
    'CDLTRISTAR': 'Tristar Pattern',
    'CDLUNIQUE3RIVER': 'Unique 3 River',
    'CDLUPSIDEGAP2CROWS': 'Upside Gap Two Crows',
    'CDLXSIDEGAP3METHODS': 'Upside/Downside Gap Three Methods',
}

# Bullish patterns (positive signal)
BULLISH_PATTERNS = [
    'CDL3WHITESOLDIERS', 'CDLHAMMER', 'CDLINVERTEDHAMMER',
    'CDLMORNINGSTAR', 'CDLMORNINGDOJISTAR', 'CDLPIERCING',
    'CDLENGULFING', 'CDLHARAMI', 'CDLDRAGONFLYDOJI',
    'CDL3INSIDE', 'CDL3OUTSIDE', 'CDLABANDONEDBABY',
]

# Bearish patterns (negative signal)
BEARISH_PATTERNS = [
    'CDL3BLACKCROWS', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
    'CDLEVENINGSTAR', 'CDLEVENINGDOJISTAR', 'CDLDARKCLOUDCOVER',
    'CDLENGULFING', 'CDLHARAMI', 'CDLGRAVESTONEDOJI',
    'CDL3INSIDE', 'CDL3OUTSIDE', 'CDLABANDONEDBABY',
]


def get_all_patterns() -> Dict[str, str]:
    """Get all available candlestick patterns."""
    return ALL_PATTERNS.copy()


def get_bullish_patterns() -> List[str]:
    """Get bullish pattern names."""
    return BULLISH_PATTERNS.copy()


def get_bearish_patterns() -> List[str]:
    """Get bearish pattern names."""
    return BEARISH_PATTERNS.copy()


# =============================================================================
# Pattern Scanning
# =============================================================================

def scan_patterns(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    patterns: List[str] = None
) -> pd.DataFrame:
    """
    Scan for candlestick patterns.
    
    Parameters
    ----------
    open_ : pd.Series
        Open prices
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    patterns : List[str], optional
        List of pattern names to scan. Default: all patterns
        
    Returns
    -------
    pd.DataFrame
        Pattern signals (-100 to 100, 0 = no pattern)
    """
    patterns = patterns or list(ALL_PATTERNS.keys())
    results = pd.DataFrame(index=close.index)
    
    for pattern in patterns:
        if hasattr(talib, pattern):
            func = getattr(talib, pattern)
            try:
                result = func(open_.values, high.values, low.values, close.values)
                results[pattern.lower()] = result
            except Exception as e:
                logger.warning(f"Error scanning {pattern}: {e}")
    
    return results


@dataclass
class PatternSignal:
    """Pattern detection result."""
    pattern: str
    name: str
    signal: int  # -100 to 100
    timestamp: pd.Timestamp


class PatternScanner(FeatureGenerator):
    """
    Scan for candlestick patterns using TA-Lib.
    
    Features:
    - Individual pattern signals
    - Aggregated bullish/bearish score
    - Pattern count
    
    Examples
    --------
    >>> scanner = PatternScanner(patterns=['CDLHAMMER', 'CDLENGULFING'])
    >>> result = scanner.fit_transform(ohlcv_df)
    """
    
    def __init__(
        self,
        patterns: List[str] = None,
        include_aggregate: bool = True,
        min_signal_strength: int = 0,
    ):
        """
        Initialize pattern scanner.
        
        Parameters
        ----------
        patterns : List[str], optional
            Patterns to scan. Default: all patterns
        include_aggregate : bool
            Include aggregated bullish/bearish scores
        min_signal_strength : int
            Minimum signal strength to include (0-100)
        """
        config = FeatureConfig(
            name='talib_patterns',
            feature_type=FeatureType.TECHNICAL,
            params={'patterns': patterns or 'all'}
        )
        super().__init__(config)
        
        self.patterns = patterns or list(ALL_PATTERNS.keys())
        self.include_aggregate = include_aggregate
        self.min_signal_strength = min_signal_strength
    
    def fit(self, data: pd.DataFrame) -> 'PatternScanner':
        """Fit (no-op for pattern recognition)."""
        self._validate_data(data)
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """Scan for patterns."""
        if not self._is_fitted:
            raise ValueError("Scanner not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        
        open_ = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Scan patterns
        pattern_signals = scan_patterns(open_, high, low, close, self.patterns)
        
        features = pattern_signals.copy()
        
        # Aggregate scores
        if self.include_aggregate:
            # Bullish score (sum of positive signals)
            bullish_cols = [c for c in features.columns 
                          if c.upper() in BULLISH_PATTERNS]
            if bullish_cols:
                features['bullish_score'] = features[bullish_cols].clip(lower=0).sum(axis=1)
            
            # Bearish score (sum of negative signals)
            bearish_cols = [c for c in features.columns 
                          if c.upper() in BEARISH_PATTERNS]
            if bearish_cols:
                features['bearish_score'] = features[bearish_cols].clip(upper=0).abs().sum(axis=1)
            
            # Net score
            features['pattern_net_score'] = (
                features.get('bullish_score', 0) - features.get('bearish_score', 0)
            )
            
            # Pattern count
            features['pattern_count'] = (features.abs() > self.min_signal_strength).sum(axis=1)
        
        return FeatureResult(
            features=features,
            feature_names=list(features.columns),
            config=self.config,
            metadata={
                'patterns_scanned': len(self.patterns),
                'n_features': len(features.columns),
            }
        )
    
    def get_active_patterns(
        self,
        data: pd.DataFrame,
        min_strength: int = 50
    ) -> List[PatternSignal]:
        """
        Get list of active patterns at the last bar.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
        min_strength : int
            Minimum signal strength
            
        Returns
        -------
        List[PatternSignal]
            Active patterns
        """
        result = self.fit_transform(data)
        last_row = result.features.iloc[-1]
        timestamp = result.features.index[-1]
        
        active = []
        for col in result.features.columns:
            if col.startswith('cdl') and abs(last_row[col]) >= min_strength:
                pattern_name = col.upper()
                active.append(PatternSignal(
                    pattern=pattern_name,
                    name=ALL_PATTERNS.get(pattern_name, pattern_name),
                    signal=int(last_row[col]),
                    timestamp=timestamp
                ))
        
        return active
