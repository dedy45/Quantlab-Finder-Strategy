"""
Carver Position Sizing - Robert Carver's Systematic Trading.

Position sizing based on:
- Target volatility (25% annual default)
- Scaled forecast (-20 to +20)
- Instrument volatility

Defaults loaded from config module (no hardcoded values).

Reference: Systematic Trading, Chapter 7

Formula:
    Position = (Capital × Daily Vol Target × Forecast) / (10 × Instrument Vol × Price)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_carver_defaults() -> Dict[str, Any]:
    """Get Carver position sizing defaults from config module."""
    try:
        from config import get_config
        cfg = get_config()
        return {
            'capital': cfg.backtest.initial_capital,
            'target_vol': cfg.backtest.target_volatility_pct,
            'max_leverage': cfg.backtest.max_leverage,
        }
    except Exception as e:
        logger.warning(f"[WARN] Could not load config, using fallback defaults: {e}")
        return {
            'capital': 100000.0,
            'target_vol': 0.25,
            'max_leverage': 2.0,
        }


# Load defaults from config
_DEFAULTS = _get_carver_defaults()
DEFAULT_TARGET_VOL = _DEFAULTS['target_vol']
DEFAULT_FORECAST_SCALAR = 10.0  # Carver's standard
TRADING_DAYS = 252


@dataclass
class CarverPositionConfig:
    """
    Configuration for Carver position sizing.
    
    Defaults loaded from config module (config/default.yaml).
    """
    capital: float = None  # Will be set from config
    target_vol: float = DEFAULT_TARGET_VOL
    forecast_scalar: float = DEFAULT_FORECAST_SCALAR
    trading_days: int = TRADING_DAYS
    min_position: float = 0.0
    max_leverage: float = None  # Will be set from config
    
    def __post_init__(self):
        """Load defaults from config if not set."""
        defaults = _get_carver_defaults()
        if self.capital is None:
            self.capital = defaults['capital']
        if self.max_leverage is None:
            self.max_leverage = defaults['max_leverage']


@dataclass
class PositionResult:
    """Result of position calculation."""
    position: float
    position_value: float
    leverage: float
    metadata: Dict[str, Any]


class CarverPositionSizer:
    """
    Position sizing ala Robert Carver.
    
    Key principles:
    1. Target consistent volatility, not position size
    2. Scale position by forecast strength
    3. Account for instrument volatility
    
    Examples
    --------
    >>> sizer = CarverPositionSizer(capital=100000, target_vol=0.25)
    >>> 
    >>> # Full position (forecast=10)
    >>> result = sizer.calculate(
    ...     forecast=10,
    ...     instrument_vol=0.15,
    ...     price=1900
    ... )
    >>> print(f"Position: {result.position:.2f} units")
    >>> 
    >>> # Half position (forecast=5)
    >>> result = sizer.calculate(forecast=5, instrument_vol=0.15, price=1900)
    """
    
    def __init__(self, config: Optional[CarverPositionConfig] = None, **kwargs):
        """
        Initialize position sizer.
        
        Parameters
        ----------
        config : CarverPositionConfig, optional
            Configuration object
        **kwargs
            Override config values (capital, target_vol, etc.)
        """
        self.config = config or CarverPositionConfig()
        
        # Allow kwargs to override config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    @property
    def daily_vol_target(self) -> float:
        """Daily volatility target."""
        return self.config.target_vol / np.sqrt(self.config.trading_days)
    
    def calculate(
        self,
        forecast: float,
        instrument_vol: float,
        price: float,
        fx_rate: float = 1.0
    ) -> PositionResult:
        """
        Calculate position size.
        
        Parameters
        ----------
        forecast : float
            Scaled forecast (-20 to +20, average 10)
        instrument_vol : float
            Daily volatility of instrument (e.g., 0.01 = 1%)
        price : float
            Current price of instrument
        fx_rate : float
            FX rate if instrument in different currency
            
        Returns
        -------
        PositionResult
            Position size and metadata
        """
        # Carver's formula
        numerator = self.config.capital * self.daily_vol_target * forecast
        denominator = self.config.forecast_scalar * instrument_vol * price * fx_rate
        
        if denominator == 0:
            logger.warning("Denominator is zero, returning 0 position")
            position = 0.0
        else:
            position = numerator / denominator
        
        # Calculate position value and leverage
        position_value = abs(position) * price * fx_rate
        leverage = position_value / self.config.capital
        
        # Check leverage limit
        if leverage > self.config.max_leverage:
            scale_factor = self.config.max_leverage / leverage
            position *= scale_factor
            position_value *= scale_factor
            leverage = self.config.max_leverage
            logger.warning(f"Position scaled down due to leverage limit: {leverage:.2f}x")
        
        return PositionResult(
            position=position,
            position_value=position_value,
            leverage=leverage,
            metadata={
                'forecast': forecast,
                'instrument_vol': instrument_vol,
                'price': price,
                'daily_vol_target': self.daily_vol_target,
                'capital': self.config.capital,
            }
        )
    
    def calculate_from_returns(
        self,
        forecast: float,
        returns: pd.Series,
        price: float,
        vol_lookback: int = 36
    ) -> PositionResult:
        """
        Calculate position with volatility estimated from returns.
        
        Parameters
        ----------
        forecast : float
            Scaled forecast
        returns : pd.Series
            Historical returns
        price : float
            Current price
        vol_lookback : int
            Lookback for volatility calculation
            
        Returns
        -------
        PositionResult
            Position size
        """
        # Estimate daily volatility
        instrument_vol = returns.rolling(vol_lookback).std().iloc[-1]
        
        if np.isnan(instrument_vol) or instrument_vol == 0:
            instrument_vol = returns.std()
        
        return self.calculate(forecast, instrument_vol, price)


class PositionInertia:
    """
    Reduce trading frequency with buffer zones.
    
    Carver's approach:
    - Don't trade for small position changes
    - Use buffer zone (e.g., 10%) around target
    - Reduces transaction costs
    
    Reference: Systematic Trading, Chapter 12
    
    Examples
    --------
    >>> inertia = PositionInertia(buffer_fraction=0.10)
    >>> 
    >>> # Small change - don't trade
    >>> inertia.should_trade(current=100, target=105)  # False
    >>> 
    >>> # Large change - trade
    >>> inertia.should_trade(current=100, target=120)  # True
    """
    
    def __init__(self, buffer_fraction: float = 0.10):
        """
        Initialize position inertia.
        
        Parameters
        ----------
        buffer_fraction : float
            Minimum change required to trade (e.g., 0.10 = 10%)
        """
        self.buffer = buffer_fraction
    
    def should_trade(
        self,
        current_position: float,
        target_position: float
    ) -> bool:
        """
        Determine if position change is large enough to trade.
        
        Parameters
        ----------
        current_position : float
            Current position size
        target_position : float
            Target position size
            
        Returns
        -------
        bool
            True if should trade
        """
        # Always trade if going from flat to position
        if current_position == 0:
            return target_position != 0
        
        # Always trade if going to flat
        if target_position == 0:
            return True
        
        # Check if change exceeds buffer
        change_pct = abs(target_position - current_position) / abs(current_position)
        return change_pct > self.buffer
    
    def adjust_target(
        self,
        current_position: float,
        target_position: float
    ) -> float:
        """
        Adjust target position based on inertia.
        
        Returns current position if change is too small.
        """
        if self.should_trade(current_position, target_position):
            return target_position
        return current_position


class IDMCalculator:
    """
    Instrument Diversification Multiplier Calculator.
    
    IDM accounts for diversification benefit across instruments.
    
    Formula:
        IDM = 1 / sqrt(sum(w_i * w_j * rho_ij))
    
    Typical values:
    - 1 instrument: 1.0
    - 2 instruments: 1.2-1.4
    - 5 instruments: 1.5-2.0
    - 10+ instruments: 2.0-2.5
    
    Reference: Systematic Trading, Chapter 11
    """
    
    def __init__(self, max_idm: float = 2.5):
        """Initialize IDM calculator."""
        self.max_idm = max_idm
    
    def calculate(
        self,
        weights: Dict[str, float],
        correlations: pd.DataFrame
    ) -> float:
        """
        Calculate IDM.
        
        Parameters
        ----------
        weights : dict
            Instrument weights (should sum to 1)
        correlations : pd.DataFrame
            Correlation matrix
            
        Returns
        -------
        float
            IDM value
        """
        names = list(weights.keys())
        n = len(names)
        
        if n == 1:
            return 1.0
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        # Calculate weighted correlation sum
        weighted_sum = 0.0
        for name_i in names:
            for name_j in names:
                w_i = weights.get(name_i, 0)
                w_j = weights.get(name_j, 0)
                
                if name_i in correlations.index and name_j in correlations.columns:
                    rho = correlations.loc[name_i, name_j]
                else:
                    rho = 1.0 if name_i == name_j else 0.5
                
                weighted_sum += w_i * w_j * rho
        
        if weighted_sum <= 0:
            return 1.0
        
        idm = 1.0 / np.sqrt(weighted_sum)
        return min(idm, self.max_idm)
    
    def estimate_from_n(self, n_instruments: int, avg_correlation: float = 0.3) -> float:
        """
        Estimate IDM from number of instruments.
        
        Quick approximation when full correlation matrix not available.
        """
        if n_instruments <= 1:
            return 1.0
        
        # Equal weights assumed
        w = 1.0 / n_instruments
        
        # Weighted sum = n*w^2 + n*(n-1)*w^2*rho
        weighted_sum = n_instruments * w**2 + n_instruments * (n_instruments - 1) * w**2 * avg_correlation
        
        idm = 1.0 / np.sqrt(weighted_sum)
        return min(idm, self.max_idm)
