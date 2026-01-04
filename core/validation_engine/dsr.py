"""
Deflated Sharpe Ratio (DSR) Calculator.

DSR adjusts the Sharpe Ratio for multiple testing bias.
When you try many strategies, some will look good by chance.

Key insight: E[max{SR_k}] = sigma_SR * sqrt(2 * ln(K))
Where K = number of strategies tried

Reference: Bailey & Lopez de Prado (2014)
Kompendium #4, #5
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Optional
from dataclasses import dataclass

from .psr import PSRCalculator

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class DSRResult:
    """Result container for DSR calculations."""
    dsr: float  # Deflated Sharpe Ratio
    observed_sr: float
    expected_max_sr: float  # E[max{SR}] under null
    haircut: float  # How much SR was deflated
    n_trials: int
    n_observations: int
    psr_deflated: float  # PSR using deflated benchmark
    
    @property
    def is_significant(self) -> bool:
        """Check if DSR > 1.0 (beats expected maximum)."""
        return self.dsr > 1.0
    
    def __str__(self) -> str:
        status = "SIGNIFICANT" if self.is_significant else "NOT SIGNIFICANT"
        return (
            f"DSR: {self.dsr:.4f} [{status}]\n"
            f"Observed SR: {self.observed_sr:.4f}\n"
            f"Expected Max SR: {self.expected_max_sr:.4f}\n"
            f"Haircut: {self.haircut:.4f}\n"
            f"Trials: {self.n_trials}\n"
            f"PSR (deflated): {self.psr_deflated:.4f}"
        )
    
    def __float__(self) -> float:
        """Allow float conversion."""
        return self.dsr
    
    def __format__(self, format_spec: str) -> str:
        """Allow format string usage."""
        return format(self.dsr, format_spec)


class DSRCalculator:
    """
    Calculate Deflated Sharpe Ratio.
    
    DSR accounts for the fact that when you try K strategies,
    the best one will have inflated SR even if all are random.
    
    Examples
    --------
    >>> calc = DSRCalculator(n_trials=100)
    >>> result = calc.calculate(returns)
    >>> if result.dsr > 1.0:
    ...     print("Strategy beats expected maximum!")
    """
    
    def __init__(
        self,
        n_trials: int = 1,
        periods_per_year: int = 252,
    ):
        """
        Initialize DSR calculator.
        
        Parameters
        ----------
        n_trials : int
            Number of strategies/parameters tried
        periods_per_year : int
            Number of periods per year
        """
        self.n_trials = max(1, n_trials)
        self.periods_per_year = periods_per_year
    
    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_trials: Optional[int] = None,
    ) -> DSRResult:
        """
        Calculate DSR from returns.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Period returns
        n_trials : int, optional
            Override number of trials
            
        Returns
        -------
        DSRResult
            DSR and related statistics
            
        Raises
        ------
        ValueError
            If returns is empty or invalid
        """
        # Input validation
        assert returns is not None, "Returns cannot be None"
        
        try:
            returns = np.asarray(returns, dtype=np.float64)
            returns = returns[~np.isnan(returns)]
            
            n = len(returns)
            k = n_trials if n_trials is not None else self.n_trials
            
            if n < 10:
                logger.warning(f"Insufficient observations ({n}), returning default DSR=0.0")
                return DSRResult(
                    dsr=0.0,
                    observed_sr=0.0,
                    expected_max_sr=0.0,
                    haircut=0.0,
                    n_trials=k,
                    n_observations=n,
                    psr_deflated=0.5,
                )
            
            # Calculate observed SR (vectorized)
            mean_r = np.mean(returns)
            std_r = np.std(returns, ddof=1)
            
            if std_r == 0 or np.isnan(std_r):
                logger.warning("Zero or NaN standard deviation")
                observed_sr = 0.0
            else:
                observed_sr = mean_r / std_r
            
            # Calculate expected maximum SR under null hypothesis
            # E[max{SR_k}] = sigma_SR * sqrt(2 * ln(K))
            sigma_sr = 1.0 / np.sqrt(n)
            expected_max_sr = self._expected_max_sr(k, sigma_sr)
            
            # Haircut = how much we deflate
            haircut = expected_max_sr
            
            # DSR = observed SR / expected max SR (with division safety)
            dsr = np.divide(
                observed_sr,
                expected_max_sr,
                out=np.array(observed_sr),
                where=expected_max_sr != 0
            )
            if isinstance(dsr, np.ndarray):
                dsr = float(dsr)
            
            # Calculate PSR with deflated benchmark
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns, fisher=False)
            
            psr_calc = PSRCalculator(benchmark_sr=expected_max_sr)
            psr_result = psr_calc.calculate_from_sr(
                observed_sr=observed_sr,
                n_observations=n,
                skewness=skewness,
                kurtosis=kurtosis,
            )
            
            logger.debug(f"DSR calculated: {dsr:.4f} (SR={observed_sr:.4f}, trials={k})")
            
            return DSRResult(
                dsr=dsr,
                observed_sr=observed_sr,
                expected_max_sr=expected_max_sr,
                haircut=haircut,
                n_trials=k,
                n_observations=n,
                psr_deflated=psr_result.psr,
            )
            
        except Exception as e:
            logger.error(f"Error calculating DSR: {e}")
            raise
    
    def _expected_max_sr(self, k: int, sigma_sr: float) -> float:
        """
        Calculate expected maximum SR under null.
        
        E[max{SR_k}] ≈ σ_SR * [(1 - γ) * Φ^(-1)(1 - 1/K) + γ * Φ^(-1)(1 - 1/(K*e))]
        
        Simplified: E[max{SR_k}] ≈ σ_SR * sqrt(2 * ln(K))
        """
        if k <= 1:
            return 0.0
        
        # Simplified formula
        return sigma_sr * np.sqrt(2 * np.log(k))
    
    def calculate_from_sr(
        self,
        observed_sr: float,
        n_observations: int,
        n_trials: Optional[int] = None,
    ) -> DSRResult:
        """
        Calculate DSR from pre-computed Sharpe Ratio.
        
        Parameters
        ----------
        observed_sr : float
            Observed Sharpe Ratio
        n_observations : int
            Number of observations
        n_trials : int, optional
            Number of trials (overrides init value)
            
        Returns
        -------
        DSRResult
            DSR and related statistics
        """
        k = n_trials if n_trials is not None else self.n_trials
        
        sigma_sr = 1 / np.sqrt(n_observations)
        expected_max_sr = self._expected_max_sr(k, sigma_sr)
        
        if expected_max_sr == 0:
            dsr = observed_sr
        else:
            dsr = observed_sr / expected_max_sr
        
        # PSR with deflated benchmark
        psr_calc = PSRCalculator(benchmark_sr=expected_max_sr)
        psr_result = psr_calc.calculate_from_sr(
            observed_sr=observed_sr,
            n_observations=n_observations,
        )
        
        return DSRResult(
            dsr=dsr,
            observed_sr=observed_sr,
            expected_max_sr=expected_max_sr,
            haircut=expected_max_sr,
            n_trials=k,
            n_observations=n_observations,
            psr_deflated=psr_result.psr,
        )


def calculate_dsr(
    returns: Union[pd.Series, np.ndarray],
    n_trials: int = 1,
) -> float:
    """
    Quick DSR calculation.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Period returns
    n_trials : int
        Number of strategies tried
        
    Returns
    -------
    float
        DSR value
    """
    calc = DSRCalculator(n_trials=n_trials)
    result = calc.calculate(returns)
    return result.dsr


def estimate_trials_from_parameters(
    n_parameters: int,
    values_per_parameter: int = 10,
) -> int:
    """
    Estimate number of trials from parameter grid.
    
    Parameters
    ----------
    n_parameters : int
        Number of parameters optimized
    values_per_parameter : int
        Average values tested per parameter
        
    Returns
    -------
    int
        Estimated number of trials
    """
    return values_per_parameter ** n_parameters


def false_strategy_theorem(
    n_trials: int,
    n_observations: int = 252,
) -> float:
    """
    Calculate expected maximum SR from random strategies.
    
    This shows how high SR can be just by chance.
    
    Parameters
    ----------
    n_trials : int
        Number of strategies tried
    n_observations : int
        Number of observations per strategy
        
    Returns
    -------
    float
        Expected maximum SR under null hypothesis
        
    Examples
    --------
    >>> # If you try 100 random strategies with 1 year of daily data
    >>> expected_sr = false_strategy_theorem(100, 252)
    >>> print(f"Expected max SR by chance: {expected_sr:.2f}")
    Expected max SR by chance: 0.19
    """
    sigma_sr = 1 / np.sqrt(n_observations)
    return sigma_sr * np.sqrt(2 * np.log(n_trials))
