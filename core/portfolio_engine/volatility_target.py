"""
Volatility Targeting.

Scales portfolio exposure to maintain constant volatility.
This helps with risk management and position sizing.

Reference: Kompendium #14
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_TARGET_VOL = 0.15  # 15% annual


class VolatilityTargeter:
    """
    Volatility targeting for position sizing.
    
    Scales positions to maintain target portfolio volatility.
    
    Parameters
    ----------
    target_vol : float, default=0.15
        Target annual volatility (e.g., 0.15 = 15%)
    lookback : int, default=20
        Lookback period for volatility estimation
    vol_floor : float, default=0.05
        Minimum volatility to prevent extreme leverage
    vol_cap : float, default=0.50
        Maximum volatility to prevent extreme deleveraging
    max_leverage : float, default=2.0
        Maximum leverage allowed
    """
    
    def __init__(
        self,
        target_vol: float = DEFAULT_TARGET_VOL,
        lookback: int = 20,
        vol_floor: float = 0.05,
        vol_cap: float = 0.50,
        max_leverage: float = 2.0
    ):
        assert target_vol > 0, "target_vol must be positive"
        assert lookback > 0, "lookback must be positive"
        assert vol_floor > 0, "vol_floor must be positive"
        assert vol_cap > vol_floor, "vol_cap must be > vol_floor"
        assert max_leverage > 0, "max_leverage must be positive"
        
        self.target_vol = target_vol
        self.lookback = lookback
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
        self.max_leverage = max_leverage
        
        self._current_vol: Optional[float] = None
        self._scalar: Optional[float] = None
        
    def calculate_scalar(
        self, 
        returns: Union[pd.Series, pd.DataFrame]
    ) -> float:
        """
        Calculate volatility scalar for position sizing.
        
        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Historical returns
            
        Returns
        -------
        float
            Scalar to multiply positions by
        """
        assert returns is not None, "Returns cannot be None"
        assert len(returns) >= self.lookback, f"Need at least {self.lookback} observations"
        
        try:
            # Calculate realized volatility
            if isinstance(returns, pd.DataFrame):
                # Portfolio returns (assume equal weight if multiple columns)
                returns = returns.mean(axis=1)
            
            # Use recent returns for volatility estimate
            recent_returns = returns.iloc[-self.lookback:]
            daily_vol = recent_returns.std()
            
            # Annualize
            annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            # Apply floor and cap
            annual_vol = np.clip(annual_vol, self.vol_floor, self.vol_cap)
            
            # Calculate scalar
            scalar = self.target_vol / annual_vol
            
            # Apply leverage cap
            scalar = min(scalar, self.max_leverage)
            
            self._current_vol = annual_vol
            self._scalar = scalar
            
            logger.info(
                f"Vol targeting: current={annual_vol:.2%}, "
                f"target={self.target_vol:.2%}, scalar={scalar:.2f}"
            )
            
            return scalar
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            raise
    
    def scale_weights(
        self, 
        weights: Dict[str, float],
        returns: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Scale portfolio weights to target volatility.
        
        Parameters
        ----------
        weights : Dict[str, float]
            Original portfolio weights
        returns : pd.Series or pd.DataFrame
            Historical returns for volatility estimation
            
        Returns
        -------
        Dict[str, float]
            Scaled weights
        """
        scalar = self.calculate_scalar(returns)
        
        scaled = {asset: weight * scalar for asset, weight in weights.items()}
        
        return scaled
    
    def get_info(self) -> Dict[str, float]:
        """Get current volatility targeting info."""
        return {
            'target_vol': self.target_vol,
            'current_vol': self._current_vol,
            'scalar': self._scalar,
            'lookback': self.lookback,
        }


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int, default=20
        Rolling window size
    annualize : bool, default=True
        Whether to annualize volatility
        
    Returns
    -------
    pd.Series
        Rolling volatility
    """
    assert returns is not None, "Returns cannot be None"
    assert window > 0, "Window must be positive"
    
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return vol
