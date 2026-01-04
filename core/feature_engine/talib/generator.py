"""
TA-Lib Unified Feature Generator.

Combines all TA-Lib indicator categories into a single generator
for comprehensive technical analysis feature engineering.

Categories:
- Overlap (MA, BBANDS, SAR)
- Momentum (RSI, MACD, Stochastic, ADX)
- Volatility (ATR, NATR)
- Volume (OBV, AD)
- Patterns (61 candlestick patterns)

Reference: https://ta-lib.org/
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from ..base import FeatureGenerator, FeatureConfig, FeatureResult, FeatureType
from .overlap import OverlapFeatureGenerator
from .momentum import MomentumFeatureGenerator
from .volatility import VolatilityFeatureGenerator
from .volume import VolumeFeatureGenerator
from .pattern import PatternScanner

logger = logging.getLogger(__name__)


class TALibCategory(Enum):
    """TA-Lib indicator categories."""
    OVERLAP = 'overlap'
    MOMENTUM = 'momentum'
    VOLATILITY = 'volatility'
    VOLUME = 'volume'
    PATTERN = 'pattern'


@dataclass
class TALibConfig:
    """
    Configuration for TA-Lib feature generation.
    
    Attributes
    ----------
    categories : List[str]
        Categories to include: overlap, momentum, volatility, volume, pattern
    ma_periods : List[int]
        Moving average periods
    rsi_periods : List[int]
        RSI periods
    atr_periods : List[int]
        ATR periods
    include_patterns : bool
        Include candlestick patterns
    pattern_list : List[str]
        Specific patterns to scan (None = all)
    """
    categories: List[str] = field(default_factory=lambda: [
        'overlap', 'momentum', 'volatility'
    ])
    ma_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    ma_types: List[str] = field(default_factory=lambda: ['sma', 'ema'])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    atr_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    include_patterns: bool = False
    pattern_list: Optional[List[str]] = None
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Stochastic
    include_stoch: bool = True
    
    # MACD
    include_macd: bool = True
    
    # ADX
    include_adx: bool = True


class TALibFeatureGenerator(FeatureGenerator):
    """
    Unified TA-Lib feature generator.
    
    Combines multiple TA-Lib indicator categories into a single
    feature generation pipeline.
    
    Examples
    --------
    >>> # Default configuration
    >>> gen = TALibFeatureGenerator()
    >>> result = gen.fit_transform(ohlcv_df)
    >>> print(f"Generated {len(result.feature_names)} features")
    
    >>> # Custom configuration
    >>> config = TALibConfig(
    ...     categories=['overlap', 'momentum'],
    ...     ma_periods=[10, 20, 50, 200],
    ...     rsi_periods=[14],
    ... )
    >>> gen = TALibFeatureGenerator(config)
    >>> result = gen.fit_transform(ohlcv_df)
    
    >>> # With patterns
    >>> config = TALibConfig(
    ...     categories=['momentum', 'pattern'],
    ...     include_patterns=True,
    ... )
    >>> gen = TALibFeatureGenerator(config)
    """
    
    def __init__(self, config: TALibConfig = None):
        """
        Initialize TA-Lib feature generator.
        
        Parameters
        ----------
        config : TALibConfig, optional
            Configuration. Default: TALibConfig()
        """
        self.talib_config = config or TALibConfig()
        
        base_config = FeatureConfig(
            name='talib_features',
            feature_type=FeatureType.TECHNICAL,
            params={
                'categories': self.talib_config.categories,
                'ma_periods': self.talib_config.ma_periods,
                'rsi_periods': self.talib_config.rsi_periods,
            }
        )
        super().__init__(base_config)
        
        # Initialize sub-generators
        self._generators: Dict[str, FeatureGenerator] = {}
        self._init_generators()
    
    def _init_generators(self) -> None:
        """Initialize sub-generators based on config."""
        cfg = self.talib_config
        
        if 'overlap' in cfg.categories:
            self._generators['overlap'] = OverlapFeatureGenerator(
                ma_periods=cfg.ma_periods,
                ma_types=cfg.ma_types,
                bb_period=cfg.bb_period,
                bb_std=cfg.bb_std,
            )
        
        if 'momentum' in cfg.categories:
            self._generators['momentum'] = MomentumFeatureGenerator(
                rsi_periods=cfg.rsi_periods,
                include_stoch=cfg.include_stoch,
                include_macd=cfg.include_macd,
                include_adx=cfg.include_adx,
            )
        
        if 'volatility' in cfg.categories:
            self._generators['volatility'] = VolatilityFeatureGenerator(
                atr_periods=cfg.atr_periods,
            )
        
        if 'volume' in cfg.categories:
            self._generators['volume'] = VolumeFeatureGenerator()
        
        if 'pattern' in cfg.categories or cfg.include_patterns:
            self._generators['pattern'] = PatternScanner(
                patterns=cfg.pattern_list,
            )
    
    def fit(self, data: pd.DataFrame) -> 'TALibFeatureGenerator':
        """
        Fit all sub-generators.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        self
        """
        self._validate_data(data)
        
        for name, gen in self._generators.items():
            try:
                gen.fit(data)
            except Exception as e:
                logger.warning(f"Error fitting {name} generator: {e}")
        
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """
        Generate all TA-Lib features.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        FeatureResult
            Combined features from all categories
        """
        if not self._is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        all_features = pd.DataFrame(index=data.index)
        metadata = {
            'categories': list(self._generators.keys()),
            'features_per_category': {},
        }
        
        for name, gen in self._generators.items():
            try:
                result = gen.transform(data)
                
                # Prefix features with category
                for col in result.features.columns:
                    all_features[f'{name}_{col}'] = result.features[col]
                
                metadata['features_per_category'][name] = len(result.features.columns)
                
            except Exception as e:
                logger.warning(f"Error transforming {name}: {e}")
                metadata['features_per_category'][name] = 0
        
        # Handle NaN
        all_features = self._handle_nan(all_features, 'ffill')
        
        metadata['n_features'] = len(all_features.columns)
        
        return FeatureResult(
            features=all_features,
            feature_names=list(all_features.columns),
            config=self.config,
            metadata=metadata
        )
    
    def get_feature_importance(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        method: str = 'correlation'
    ) -> pd.Series:
        """
        Calculate feature importance.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
        target : pd.Series
            Target variable (e.g., returns)
        method : str
            'correlation' or 'mutual_info'
            
        Returns
        -------
        pd.Series
            Feature importance scores
        """
        result = self.fit_transform(data)
        features = result.features
        
        # Align target with features
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        if method == 'correlation':
            importance = features.corrwith(target).abs()
        else:
            # Mutual information (requires sklearn)
            from sklearn.feature_selection import mutual_info_regression
            importance = pd.Series(
                mutual_info_regression(features.fillna(0), target),
                index=features.columns
            )
        
        return importance.sort_values(ascending=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get generator summary."""
        return {
            'categories': list(self._generators.keys()),
            'config': {
                'ma_periods': self.talib_config.ma_periods,
                'rsi_periods': self.talib_config.rsi_periods,
                'atr_periods': self.talib_config.atr_periods,
                'include_patterns': self.talib_config.include_patterns,
            },
            'is_fitted': self._is_fitted,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_talib_features(
    data: pd.DataFrame,
    categories: List[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Quick function to generate TA-Lib features.
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data
    categories : List[str], optional
        Categories to include
    **kwargs
        Additional config parameters
        
    Returns
    -------
    pd.DataFrame
        Feature DataFrame
    """
    config = TALibConfig(
        categories=categories or ['overlap', 'momentum', 'volatility'],
        **kwargs
    )
    gen = TALibFeatureGenerator(config)
    result = gen.fit_transform(data)
    return result.features
