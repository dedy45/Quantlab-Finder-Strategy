"""
Bootstrap Engine - Non-parametric confidence intervals.

Bootstrap resampling allows us to estimate confidence intervals
without assuming a specific distribution for returns.

Key use: Get confidence interval for Sharpe Ratio
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Callable, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .sharpe import calculate_sharpe


@dataclass
class BootstrapResult:
    """Result container for bootstrap analysis."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    bootstrap_distribution: np.ndarray
    
    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower
    
    @property
    def is_significant(self) -> bool:
        """Check if CI excludes zero."""
        return self.ci_lower > 0 or self.ci_upper < 0
    
    def __str__(self) -> str:
        status = "SIGNIFICANT" if self.is_significant else "NOT SIGNIFICANT"
        return (
            f"Point Estimate: {self.point_estimate:.4f}\n"
            f"{self.confidence_level*100:.0f}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"CI Width: {self.ci_width:.4f}\n"
            f"Status: [{status}]"
        )


class BootstrapEngine:
    """
    Bootstrap resampling for confidence intervals.
    
    Supports:
    - Basic bootstrap (with replacement)
    - Block bootstrap (for time series)
    - Stationary bootstrap (random block lengths)
    
    Examples
    --------
    >>> engine = BootstrapEngine(n_bootstrap=5000)
    >>> result = engine.sharpe_ci(returns)
    >>> print(f"95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    """
    
    def __init__(
        self,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ):
        """
        Initialize bootstrap engine.
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals (default 95%)
        random_state : int, optional
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def sharpe_ci(
        self,
        returns: Union[pd.Series, np.ndarray],
        method: str = 'basic',
        block_size: int = 20,
    ) -> BootstrapResult:
        """
        Calculate bootstrap confidence interval for Sharpe Ratio.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Period returns
        method : str
            'basic', 'block', or 'stationary'
        block_size : int
            Block size for block bootstrap
            
        Returns
        -------
        BootstrapResult
            Bootstrap CI and distribution
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        # Point estimate
        point_estimate = calculate_sharpe(returns)
        
        # Generate bootstrap samples
        if method == 'basic':
            bootstrap_stats = self._basic_bootstrap(returns, calculate_sharpe)
        elif method == 'block':
            bootstrap_stats = self._block_bootstrap(returns, calculate_sharpe, block_size)
        elif method == 'stationary':
            bootstrap_stats = self._stationary_bootstrap(returns, calculate_sharpe, block_size)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        
        return BootstrapResult(
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            bootstrap_distribution=bootstrap_stats,
        )
    
    def generic_ci(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        method: str = 'basic',
        block_size: int = 20,
    ) -> BootstrapResult:
        """
        Calculate bootstrap CI for any statistic.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        statistic_func : Callable
            Function that computes the statistic
        method : str
            Bootstrap method
        block_size : int
            Block size for block methods
            
        Returns
        -------
        BootstrapResult
            Bootstrap CI and distribution
        """
        point_estimate = statistic_func(data)
        
        if method == 'basic':
            bootstrap_stats = self._basic_bootstrap(data, statistic_func)
        elif method == 'block':
            bootstrap_stats = self._block_bootstrap(data, statistic_func, block_size)
        else:
            bootstrap_stats = self._basic_bootstrap(data, statistic_func)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        
        return BootstrapResult(
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            bootstrap_distribution=bootstrap_stats,
        )
    
    def _basic_bootstrap(
        self,
        data: np.ndarray,
        statistic_func: Callable,
    ) -> np.ndarray:
        """Basic bootstrap with replacement."""
        n = len(data)
        bootstrap_stats = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n, size=n)
            sample = data[indices]
            bootstrap_stats[i] = statistic_func(sample)
        
        return bootstrap_stats
    
    def _block_bootstrap(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        block_size: int,
    ) -> np.ndarray:
        """
        Block bootstrap for time series.
        
        Preserves autocorrelation structure by resampling blocks.
        """
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))
        bootstrap_stats = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            # Sample block starting positions
            block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
            
            # Construct bootstrap sample from blocks
            sample = []
            for start in block_starts:
                sample.extend(data[start:start + block_size])
            
            sample = np.array(sample[:n])  # Trim to original length
            bootstrap_stats[i] = statistic_func(sample)
        
        return bootstrap_stats
    
    def _stationary_bootstrap(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        avg_block_size: int,
    ) -> np.ndarray:
        """
        Stationary bootstrap with random block lengths.
        
        Block lengths follow geometric distribution.
        """
        n = len(data)
        p = 1 / avg_block_size  # Probability of ending block
        bootstrap_stats = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            sample = []
            pos = np.random.randint(0, n)
            
            while len(sample) < n:
                sample.append(data[pos])
                
                # Decide whether to continue block or start new
                if np.random.random() < p:
                    pos = np.random.randint(0, n)  # New random position
                else:
                    pos = (pos + 1) % n  # Continue (wrap around)
            
            sample = np.array(sample[:n])
            bootstrap_stats[i] = statistic_func(sample)
        
        return bootstrap_stats


def bootstrap_sharpe_ci(
    returns: Union[pd.Series, np.ndarray],
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Quick bootstrap CI for Sharpe Ratio.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Period returns
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level
        
    Returns
    -------
    Tuple[float, float, float]
        (lower_bound, point_estimate, upper_bound)
    """
    engine = BootstrapEngine(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )
    result = engine.sharpe_ci(returns)
    return (result.ci_lower, result.point_estimate, result.ci_upper)
