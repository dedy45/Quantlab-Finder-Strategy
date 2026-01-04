"""
Forecast Scaling - Robert Carver's Systematic Trading.

Converts raw trading signals to scaled forecasts in range -20 to +20.
Average absolute forecast = 10 (represents full position).

Reference: Systematic Trading, Chapter 5
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Carver's constants
FORECAST_CAP = 20.0
TARGET_ABS_FORECAST = 10.0


@dataclass
class ForecastConfig:
    """Configuration for forecast scaling."""
    target_abs_forecast: float = TARGET_ABS_FORECAST
    forecast_cap: float = FORECAST_CAP
    min_periods: int = 20


@dataclass
class ForecastResult:
    """Result of forecast scaling."""
    scaled_forecast: pd.Series
    raw_forecast: pd.Series
    scalar: float
    metadata: Dict[str, Any]


class ForecastScaler:
    """
    Scale raw signals to Carver's forecast range (-20 to +20).
    
    Carver's approach:
    - Raw signals can be any range (e.g., -1 to +1, or unbounded)
    - Scale so average absolute value = 10
    - Cap at ±20 to prevent extreme positions
    
    Examples
    --------
    >>> scaler = ForecastScaler()
    >>> raw = pd.Series([1, -1, 1, -1, 0.5])  # MA crossover signals
    >>> result = scaler.scale(raw)
    >>> print(result.scaled_forecast)  # Will be around ±10
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """Initialize forecast scaler."""
        self.config = config or ForecastConfig()
        self._scalar = None
    
    def fit(self, raw_forecast: pd.Series) -> 'ForecastScaler':
        """
        Fit scaler to historical forecasts.
        
        Parameters
        ----------
        raw_forecast : pd.Series
            Historical raw forecast values
            
        Returns
        -------
        ForecastScaler
            Self for method chaining
        """
        # Calculate average absolute forecast
        clean = raw_forecast.dropna()
        
        if len(clean) < self.config.min_periods:
            logger.warning(f"Insufficient data ({len(clean)} < {self.config.min_periods})")
            self._scalar = 1.0
            return self
        
        avg_abs = clean.abs().mean()
        
        if avg_abs == 0 or np.isnan(avg_abs):
            logger.warning("Average absolute forecast is zero, using scalar=1.0")
            self._scalar = 1.0
        else:
            self._scalar = self.config.target_abs_forecast / avg_abs
        
        logger.debug(f"Forecast scalar: {self._scalar:.4f}")
        return self
    
    def transform(self, raw_forecast: pd.Series) -> ForecastResult:
        """
        Transform raw forecast to scaled forecast.
        
        Parameters
        ----------
        raw_forecast : pd.Series
            Raw forecast values
            
        Returns
        -------
        ForecastResult
            Scaled forecast with metadata
        """
        if self._scalar is None:
            self.fit(raw_forecast)
        
        # Scale
        scaled = raw_forecast * self._scalar
        
        # Cap at ±20
        capped = scaled.clip(-self.config.forecast_cap, self.config.forecast_cap)
        
        return ForecastResult(
            scaled_forecast=capped,
            raw_forecast=raw_forecast,
            scalar=self._scalar,
            metadata={
                'target_abs': self.config.target_abs_forecast,
                'cap': self.config.forecast_cap,
                'actual_avg_abs': capped.abs().mean(),
            }
        )
    
    def scale(self, raw_forecast: pd.Series) -> ForecastResult:
        """Fit and transform in one step."""
        return self.fit(raw_forecast).transform(raw_forecast)


class ForecastCombiner:
    """
    Combine multiple forecasts with Forecast Diversification Multiplier (FDM).
    
    Carver's approach:
    - Weight forecasts by expected Sharpe Ratio
    - Apply FDM to account for diversification benefit
    - FDM = 1 / sqrt(sum of weighted correlations)
    
    Reference: Systematic Trading, Chapter 6
    
    Examples
    --------
    >>> combiner = ForecastCombiner()
    >>> forecasts = {
    ...     'ewmac_fast': pd.Series([10, -10, 5]),
    ...     'ewmac_slow': pd.Series([8, -5, 10]),
    ... }
    >>> weights = {'ewmac_fast': 0.5, 'ewmac_slow': 0.5}
    >>> result = combiner.combine(forecasts, weights)
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """Initialize forecast combiner."""
        self.config = config or ForecastConfig()
    
    def calculate_fdm(
        self,
        weights: Dict[str, float],
        correlations: pd.DataFrame
    ) -> float:
        """
        Calculate Forecast Diversification Multiplier.
        
        FDM = 1 / sqrt(sum(w_i * w_j * rho_ij))
        
        Parameters
        ----------
        weights : dict
            Forecast weights (must sum to 1)
        correlations : pd.DataFrame
            Correlation matrix of forecasts
            
        Returns
        -------
        float
            FDM value (typically 1.0 to 2.5)
        """
        names = list(weights.keys())
        n = len(names)
        
        if n == 1:
            return 1.0
        
        # Calculate weighted sum of correlations
        weighted_sum = 0.0
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                w_i = weights[name_i]
                w_j = weights[name_j]
                
                if name_i in correlations.index and name_j in correlations.columns:
                    rho = correlations.loc[name_i, name_j]
                else:
                    rho = 1.0 if i == j else 0.0
                
                weighted_sum += w_i * w_j * rho
        
        if weighted_sum <= 0:
            return 1.0
        
        fdm = 1.0 / np.sqrt(weighted_sum)
        
        # Cap FDM at reasonable values
        fdm = min(fdm, 2.5)
        
        logger.debug(f"FDM: {fdm:.4f}")
        return fdm
    
    def combine(
        self,
        forecasts: Dict[str, pd.Series],
        weights: Dict[str, float],
        correlations: Optional[pd.DataFrame] = None
    ) -> ForecastResult:
        """
        Combine multiple forecasts.
        
        Parameters
        ----------
        forecasts : dict
            Dictionary of forecast name -> pd.Series
        weights : dict
            Dictionary of forecast name -> weight (should sum to 1)
        correlations : pd.DataFrame, optional
            Correlation matrix. If None, estimated from data.
            
        Returns
        -------
        ForecastResult
            Combined forecast
        """
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Align all forecasts
        df = pd.DataFrame(forecasts)
        
        # Calculate correlations if not provided
        if correlations is None:
            correlations = df.corr()
        
        # Calculate FDM
        fdm = self.calculate_fdm(weights, correlations)
        
        # Weighted sum
        combined = pd.Series(0.0, index=df.index)
        for name, weight in weights.items():
            if name in df.columns:
                combined += weight * df[name].fillna(0)
        
        # Apply FDM
        combined = combined * fdm
        
        # Cap
        combined = combined.clip(-self.config.forecast_cap, self.config.forecast_cap)
        
        return ForecastResult(
            scaled_forecast=combined,
            raw_forecast=combined / fdm,  # Before FDM
            scalar=fdm,
            metadata={
                'fdm': fdm,
                'weights': weights,
                'n_forecasts': len(forecasts),
            }
        )


def scale_forecast(raw: pd.Series, target: float = 10.0, cap: float = 20.0) -> pd.Series:
    """
    Quick function to scale forecast.
    
    Parameters
    ----------
    raw : pd.Series
        Raw forecast values
    target : float
        Target average absolute forecast
    cap : float
        Maximum absolute forecast
        
    Returns
    -------
    pd.Series
        Scaled forecast
    """
    config = ForecastConfig(target_abs_forecast=target, forecast_cap=cap)
    scaler = ForecastScaler(config)
    result = scaler.scale(raw)
    return result.scaled_forecast
