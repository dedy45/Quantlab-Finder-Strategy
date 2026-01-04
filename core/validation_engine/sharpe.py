"""
Sharpe Ratio Calculator.

The Sharpe Ratio measures risk-adjusted returns:
SR = (Mean Return - Risk Free Rate) / Std Dev of Returns

Annualized SR = SR * sqrt(periods_per_year)

Reference: Kompendium #1
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Union, List
from dataclasses import dataclass

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class SharpeResult:
    """Result container for Sharpe calculations."""
    sharpe_ratio: float
    annualized_sharpe: float
    mean_return: float
    std_return: float
    n_observations: int
    periods_per_year: int
    risk_free_rate: float
    
    def __str__(self) -> str:
        return (
            f"Sharpe Ratio: {self.sharpe_ratio:.4f}\n"
            f"Annualized: {self.annualized_sharpe:.4f}\n"
            f"Mean Return: {self.mean_return:.6f}\n"
            f"Std Return: {self.std_return:.6f}\n"
            f"Observations: {self.n_observations}"
        )
    
    def __float__(self) -> float:
        """Allow float conversion for formatting."""
        return self.annualized_sharpe
    
    def __format__(self, format_spec: str) -> str:
        """Allow format string usage."""
        return format(self.annualized_sharpe, format_spec)


class SharpeCalculator:
    """
    Calculate Sharpe Ratio and related metrics.
    
    Examples
    --------
    >>> calc = SharpeCalculator(periods_per_year=252)
    >>> result = calc.calculate(returns)
    >>> print(f"Annualized Sharpe: {result.annualized_sharpe:.2f}")
    """
    
    def __init__(
        self,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        """
        Initialize calculator.
        
        Parameters
        ----------
        periods_per_year : int
            Number of periods per year (252 for daily, 12 for monthly)
        risk_free_rate : float
            Annual risk-free rate (default 0)
            
        Raises
        ------
        ValueError
            If periods_per_year <= 0 or risk_free_rate < 0
        """
        # Input validation
        assert periods_per_year > 0, "periods_per_year must be positive"
        assert risk_free_rate >= 0, "risk_free_rate cannot be negative"
        
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        
        logger.debug(
            f"SharpeCalculator initialized: periods={periods_per_year}, rf={risk_free_rate}"
        )
    
    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
    ) -> SharpeResult:
        """
        Calculate Sharpe Ratio.
        
        Parameters
        ----------
        returns : pd.Series, np.ndarray, or List[float]
            Period returns (not cumulative)
            
        Returns
        -------
        SharpeResult
            Sharpe ratio and related metrics
            
        Raises
        ------
        ValueError
            If returns is empty or all NaN
        """
        try:
            # Convert to numpy array (vectorized)
            returns_arr = np.asarray(returns, dtype=np.float64)
            
            # Remove NaN values (vectorized)
            valid_mask = ~np.isnan(returns_arr)
            returns_clean = returns_arr[valid_mask]
            
            n = len(returns_clean)
            
            # Edge case: insufficient data
            if n < 2:
                logger.warning(f"Insufficient data for Sharpe calculation: n={n}")
                return SharpeResult(
                    sharpe_ratio=0.0,
                    annualized_sharpe=0.0,
                    mean_return=0.0,
                    std_return=0.0,
                    n_observations=n,
                    periods_per_year=self.periods_per_year,
                    risk_free_rate=self.risk_free_rate,
                )
            
            # Convert annual risk-free to period risk-free
            rf_period = self.risk_free_rate / self.periods_per_year
            
            # Calculate excess returns (vectorized)
            excess_returns = returns_clean - rf_period
            
            # Calculate statistics (vectorized numpy operations)
            mean_return = np.mean(excess_returns)
            std_return = np.std(excess_returns, ddof=1)
            
            # Sharpe Ratio with div-by-zero protection
            if std_return == 0 or np.isnan(std_return):
                logger.warning("Zero or NaN standard deviation, returning Sharpe=0")
                sharpe = 0.0
            else:
                sharpe = mean_return / std_return
            
            # Annualized Sharpe
            annualized = sharpe * np.sqrt(self.periods_per_year)
            
            logger.debug(f"Sharpe calculated: SR={sharpe:.4f}, Ann={annualized:.4f}, n={n}")
            
            return SharpeResult(
                sharpe_ratio=float(sharpe),
                annualized_sharpe=float(annualized),
                mean_return=float(mean_return),
                std_return=float(std_return),
                n_observations=n,
                periods_per_year=self.periods_per_year,
                risk_free_rate=self.risk_free_rate,
            )
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe Ratio: {e}")
            raise
    
    def calculate_rolling(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """
        Calculate rolling Sharpe Ratio.
        
        Parameters
        ----------
        returns : pd.Series
            Period returns
        window : int
            Rolling window size
            
        Returns
        -------
        pd.Series
            Rolling Sharpe Ratio
            
        Raises
        ------
        ValueError
            If window <= 0 or window > len(returns)
        """
        # Input validation
        assert window > 0, "window must be positive"
        assert isinstance(returns, pd.Series), "returns must be pd.Series"
        
        if window > len(returns):
            logger.warning(f"Window ({window}) > data length ({len(returns)})")
        
        try:
            rf_period = self.risk_free_rate / self.periods_per_year
            excess = returns - rf_period
            
            # Vectorized rolling calculations
            rolling_mean = excess.rolling(window, min_periods=window//2).mean()
            rolling_std = excess.rolling(window, min_periods=window//2).std()
            
            # Avoid division by zero (vectorized)
            rolling_sharpe = np.where(
                rolling_std != 0,
                rolling_mean / rolling_std * np.sqrt(self.periods_per_year),
                0.0
            )
            
            return pd.Series(rolling_sharpe, index=returns.index, name='rolling_sharpe')
            
        except Exception as e:
            logger.error(f"Error calculating rolling Sharpe: {e}")
            raise


def calculate_sharpe(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Quick Sharpe Ratio calculation.
    
    Parameters
    ----------
    returns : pd.Series, np.ndarray, or List[float]
        Period returns
    risk_free_rate : float
        Risk-free rate per period
        
    Returns
    -------
    float
        Sharpe Ratio (not annualized)
    """
    try:
        returns_arr = np.asarray(returns, dtype=np.float64)
        returns_clean = returns_arr[~np.isnan(returns_arr)]
        
        if len(returns_clean) < 2:
            return 0.0
        
        excess = returns_clean - risk_free_rate
        std = np.std(excess, ddof=1)
        
        if std == 0 or np.isnan(std):
            return 0.0
        
        return float(np.mean(excess) / std)
        
    except Exception as e:
        logger.error(f"Error in calculate_sharpe: {e}")
        return 0.0


def calculate_annualized_sharpe(
    returns: Union[pd.Series, np.ndarray, List[float]],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Quick annualized Sharpe Ratio calculation.
    
    Parameters
    ----------
    returns : pd.Series, np.ndarray, or List[float]
        Period returns
    periods_per_year : int
        Periods per year (252 for daily)
    risk_free_rate : float
        Annual risk-free rate
        
    Returns
    -------
    float
        Annualized Sharpe Ratio
    """
    calc = SharpeCalculator(
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    result = calc.calculate(returns)
    return result.annualized_sharpe


def calculate_sortino(
    returns: Union[pd.Series, np.ndarray],
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino Ratio (downside risk only).
    
    Sortino = (Mean Return - Target) / Downside Deviation
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Period returns
    target_return : float
        Target return (default 0)
    periods_per_year : int
        Periods per year for annualization
        
    Returns
    -------
    float
        Annualized Sortino Ratio
    """
    try:
        returns_arr = np.asarray(returns, dtype=np.float64)
        returns_clean = returns_arr[~np.isnan(returns_arr)]
        
        if len(returns_clean) < 2:
            return 0.0
        
        # Downside returns only (vectorized)
        downside = np.minimum(returns_clean - target_return, 0)
        downside_std = np.sqrt(np.mean(downside ** 2))
        
        if downside_std == 0:
            return 0.0
        
        mean_excess = np.mean(returns_clean) - target_return
        sortino = mean_excess / downside_std * np.sqrt(periods_per_year)
        
        return float(sortino)
        
    except Exception as e:
        logger.error(f"Error calculating Sortino: {e}")
        return 0.0


def calculate_calmar(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar = Annualized Return / Max Drawdown
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Period returns
    periods_per_year : int
        Periods per year
        
    Returns
    -------
    float
        Calmar Ratio
    """
    try:
        returns_arr = np.asarray(returns, dtype=np.float64)
        returns_clean = returns_arr[~np.isnan(returns_arr)]
        
        if len(returns_clean) < 2:
            return 0.0
        
        # Annualized return
        total_return = np.prod(1 + returns_clean) - 1
        n_years = len(returns_clean) / periods_per_year
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Max drawdown (vectorized)
        cumulative = np.cumprod(1 + returns_clean)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns)
        
        if max_dd == 0:
            return 0.0
        
        calmar = ann_return / abs(max_dd)
        
        return float(calmar)
        
    except Exception as e:
        logger.error(f"Error calculating Calmar: {e}")
        return 0.0
