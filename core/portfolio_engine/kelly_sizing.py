"""
Kelly Criterion Position Sizing.

Optimal bet sizing based on edge and variance.
Half-Kelly is recommended for practical use.

Reference: Kompendium #13
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
DEFAULT_KELLY_FRACTION = 0.5  # Half-Kelly


@dataclass
class KellyResult:
    """Result of Kelly calculation."""
    
    full_kelly: float
    fraction: float
    recommended_size: float
    win_rate: float
    avg_win: float
    avg_loss: float
    edge: float
    
    @property
    def is_positive_edge(self) -> bool:
        """Check if strategy has positive edge."""
        return self.edge > 0


class KellySizer:
    """
    Kelly Criterion position sizer.
    
    Calculates optimal position size based on win rate and
    win/loss ratio. Half-Kelly is recommended for safety.
    
    Parameters
    ----------
    fraction : float, default=0.5
        Kelly fraction (0.5 = Half-Kelly)
    max_size : float, default=1.0
        Maximum position size
    min_trades : int, default=30
        Minimum trades required for reliable estimate
    """
    
    def __init__(
        self,
        fraction: float = DEFAULT_KELLY_FRACTION,
        max_size: float = 1.0,
        min_trades: int = 30
    ):
        assert 0 < fraction <= 1.0, "fraction must be in (0, 1]"
        assert max_size > 0, "max_size must be positive"
        assert min_trades > 0, "min_trades must be positive"
        
        self.fraction = fraction
        self.max_size = max_size
        self.min_trades = min_trades
        
    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> KellyResult:
        """
        Calculate Kelly position size from win rate and payoffs.
        
        Parameters
        ----------
        win_rate : float
            Probability of winning (0 to 1)
        avg_win : float
            Average winning trade return (positive)
        avg_loss : float
            Average losing trade return (positive, will be negated)
            
        Returns
        -------
        KellyResult
            Kelly calculation result
        """
        assert 0 <= win_rate <= 1, "win_rate must be in [0, 1]"
        assert avg_win > 0, "avg_win must be positive"
        assert avg_loss > 0, "avg_loss must be positive"
        
        try:
            # Kelly formula: f* = (bp - q) / b
            # where b = odds (avg_win / avg_loss)
            # p = win probability
            # q = loss probability (1 - p)
            
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - p
            
            # Full Kelly
            full_kelly = (b * p - q) / b
            
            # Edge = expected value per unit bet
            edge = p * avg_win - q * avg_loss
            
            # Apply fraction and constraints
            recommended = full_kelly * self.fraction
            recommended = np.clip(recommended, 0, self.max_size)
            
            result = KellyResult(
                full_kelly=full_kelly,
                fraction=self.fraction,
                recommended_size=recommended,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                edge=edge
            )
            
            logger.info(
                f"Kelly: full={full_kelly:.2%}, "
                f"recommended={recommended:.2%}, edge={edge:.4f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Kelly calculation failed: {e}")
            raise
    
    def calculate_from_returns(
        self, 
        returns: pd.Series
    ) -> KellyResult:
        """
        Calculate Kelly from historical returns.
        
        Parameters
        ----------
        returns : pd.Series
            Historical trade returns
            
        Returns
        -------
        KellyResult
            Kelly calculation result
        """
        assert returns is not None, "Returns cannot be None"
        assert len(returns) >= self.min_trades, \
            f"Need at least {self.min_trades} trades"
        
        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            logger.warning("No wins or no losses, cannot calculate Kelly")
            return KellyResult(
                full_kelly=0.0,
                fraction=self.fraction,
                recommended_size=0.0,
                win_rate=len(wins) / len(returns),
                avg_win=wins.mean() if len(wins) > 0 else 0.0,
                avg_loss=abs(losses.mean()) if len(losses) > 0 else 0.0,
                edge=returns.mean()
            )
        
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        return self.calculate(win_rate, avg_win, avg_loss)
    
    def calculate_gaussian(
        self,
        mean_return: float,
        std_return: float
    ) -> float:
        """
        Calculate Kelly for Gaussian returns.
        
        For normally distributed returns:
        f* = μ / σ²
        
        Parameters
        ----------
        mean_return : float
            Expected return
        std_return : float
            Standard deviation of returns
            
        Returns
        -------
        float
            Recommended position size
        """
        assert std_return > 0, "std_return must be positive"
        
        try:
            # Gaussian Kelly: f* = μ / σ²
            full_kelly = mean_return / (std_return ** 2)
            
            # Apply fraction and constraints
            recommended = full_kelly * self.fraction
            recommended = np.clip(recommended, 0, self.max_size)
            
            logger.info(
                f"Gaussian Kelly: μ={mean_return:.4f}, σ={std_return:.4f}, "
                f"f*={full_kelly:.2%}, recommended={recommended:.2%}"
            )
            
            return recommended
            
        except Exception as e:
            logger.error(f"Gaussian Kelly failed: {e}")
            raise


def calculate_kelly_simple(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.5
) -> float:
    """
    Simple Kelly calculation.
    
    Parameters
    ----------
    win_rate : float
        Probability of winning
    win_loss_ratio : float
        Ratio of average win to average loss
    fraction : float, default=0.5
        Kelly fraction (0.5 = Half-Kelly)
        
    Returns
    -------
    float
        Recommended position size
    """
    b = win_loss_ratio
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    return max(0, kelly * fraction)
