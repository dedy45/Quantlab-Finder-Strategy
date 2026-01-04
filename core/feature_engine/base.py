"""
Base classes for Feature Engineering.

Feature engineering transforms raw price data into meaningful
features for ML models while preserving statistical properties.

Reference: Protokol Kausalitas - Fase 2 (Sebab Statistik)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class FeatureType(Enum):
    """Types of features."""
    PRICE = "price"           # Raw price-based
    RETURN = "return"         # Return-based
    VOLATILITY = "volatility" # Volatility-based
    MOMENTUM = "momentum"     # Momentum indicators
    MEAN_REVERSION = "mean_reversion"  # Mean reversion indicators
    VOLUME = "volume"         # Volume-based
    TECHNICAL = "technical"   # Technical indicators
    STATISTICAL = "statistical"  # Statistical features
    CUSTOM = "custom"         # Custom features


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    name: str
    feature_type: FeatureType
    params: Dict[str, Any] = field(default_factory=dict)
    normalize: bool = True
    fillna_method: str = 'ffill'  # 'ffill', 'bfill', 'zero', 'mean'


@dataclass
class FeatureResult:
    """Result container for feature generation."""
    features: pd.DataFrame
    feature_names: List[str]
    config: FeatureConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)
    
    @property
    def n_samples(self) -> int:
        return len(self.features)
    
    def __str__(self) -> str:
        return (
            f"FeatureResult:\n"
            f"  Features: {self.n_features}\n"
            f"  Samples: {self.n_samples}\n"
            f"  Type: {self.config.feature_type.value}"
        )


class FeatureGenerator(ABC):
    """Abstract base class for feature generators."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'FeatureGenerator':
        """Fit the feature generator to data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """Transform data to features."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> FeatureResult:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data = data.set_index('date')
        
        return data.sort_index()
    
    def _handle_nan(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """Handle NaN values."""
        if method == 'ffill':
            return df.ffill().bfill()
        elif method == 'bfill':
            return df.bfill().ffill()
        elif method == 'zero':
            return df.fillna(0)
        elif method == 'mean':
            return df.fillna(df.mean())
        else:
            return df.ffill().bfill()


def get_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = 'simple'
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from prices.
    
    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price data
    method : str
        'simple' for arithmetic returns, 'log' for log returns
        
    Returns
    -------
    pd.Series or pd.DataFrame
        Returns
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def normalize_features(
    features: pd.DataFrame,
    method: str = 'zscore',
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Normalize features.
    
    Parameters
    ----------
    features : pd.DataFrame
        Features to normalize
    method : str
        'zscore', 'minmax', 'rank'
    window : int, optional
        Rolling window for normalization
        
    Returns
    -------
    pd.DataFrame
        Normalized features
    """
    if method == 'zscore':
        if window:
            mean = features.rolling(window).mean()
            std = features.rolling(window).std()
            return (features - mean) / std
        else:
            return (features - features.mean()) / features.std()
    
    elif method == 'minmax':
        if window:
            min_val = features.rolling(window).min()
            max_val = features.rolling(window).max()
            return (features - min_val) / (max_val - min_val)
        else:
            return (features - features.min()) / (features.max() - features.min())
    
    elif method == 'rank':
        if window:
            return features.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        else:
            return features.rank(pct=True)
    
    else:
        return features
