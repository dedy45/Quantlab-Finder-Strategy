"""
Monte Carlo Engine - Permutation tests for strategy validation.

Monte Carlo permutation tests answer:
"What is the probability of achieving this performance by chance?"

Key test: Shuffle signals, keep market returns, compare performance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Callable, List
from dataclasses import dataclass
from tqdm import tqdm

from .sharpe import calculate_sharpe


@dataclass
class MonteCarloResult:
    """Result container for Monte Carlo tests."""
    p_value: float
    observed_statistic: float
    null_distribution: np.ndarray
    n_simulations: int
    n_better: int  # How many random strategies beat actual
    
    @property
    def is_significant(self) -> bool:
        """Check if p-value < 0.05."""
        return self.p_value < 0.05
    
    @property
    def percentile(self) -> float:
        """Percentile of observed statistic in null distribution."""
        return (1 - self.p_value) * 100
    
    def __str__(self) -> str:
        status = "SIGNIFICANT" if self.is_significant else "NOT SIGNIFICANT"
        return (
            f"p-value: {self.p_value:.4f} [{status}]\n"
            f"Observed: {self.observed_statistic:.4f}\n"
            f"Percentile: {self.percentile:.1f}%\n"
            f"Random strategies that beat actual: {self.n_better}/{self.n_simulations}"
        )


class MonteCarloEngine:
    """
    Monte Carlo permutation tests for strategy validation.
    
    Tests whether a strategy has genuine predictive power
    or is just lucky.
    
    Examples
    --------
    >>> engine = MonteCarloEngine(n_simulations=1000)
    >>> result = engine.permutation_test(signals, market_returns)
    >>> if result.p_value < 0.05:
    ...     print("Strategy has significant predictive power!")
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        random_state: Optional[int] = None,
        show_progress: bool = True,
    ):
        """
        Initialize Monte Carlo engine.
        
        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations
        random_state : int, optional
            Random seed for reproducibility
        show_progress : bool
            Show progress bar
        """
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.show_progress = show_progress
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def permutation_test(
        self,
        signals: Union[pd.Series, np.ndarray],
        market_returns: Union[pd.Series, np.ndarray],
        statistic_func: Optional[Callable] = None,
    ) -> MonteCarloResult:
        """
        Test if signals have predictive power.
        
        H0: Signals have no predictive power (random)
        H1: Signals predict market direction
        
        Parameters
        ----------
        signals : pd.Series or np.ndarray
            Trading signals (-1, 0, 1)
        market_returns : pd.Series or np.ndarray
            Market returns
        statistic_func : Callable, optional
            Function to compute test statistic (default: Sharpe)
            
        Returns
        -------
        MonteCarloResult
            Test results with p-value
        """
        signals = np.asarray(signals)
        market_returns = np.asarray(market_returns)
        
        # Align lengths
        min_len = min(len(signals), len(market_returns))
        signals = signals[:min_len]
        market_returns = market_returns[:min_len]
        
        # Remove NaN
        mask = ~(np.isnan(signals) | np.isnan(market_returns))
        signals = signals[mask]
        market_returns = market_returns[mask]
        
        # Default statistic: Sharpe Ratio
        if statistic_func is None:
            statistic_func = calculate_sharpe
        
        # Calculate actual strategy returns
        strategy_returns = signals * market_returns
        observed_stat = statistic_func(strategy_returns)
        
        # Monte Carlo simulation
        null_distribution = np.zeros(self.n_simulations)
        
        iterator = range(self.n_simulations)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Monte Carlo")
        
        for i in iterator:
            # Shuffle signals (break temporal relationship)
            shuffled_signals = np.random.permutation(signals)
            
            # Calculate returns with shuffled signals
            random_returns = shuffled_signals * market_returns
            null_distribution[i] = statistic_func(random_returns)
        
        # Calculate p-value (proportion of random >= actual)
        n_better = np.sum(null_distribution >= observed_stat)
        p_value = (n_better + 1) / (self.n_simulations + 1)  # +1 for continuity correction
        
        return MonteCarloResult(
            p_value=p_value,
            observed_statistic=observed_stat,
            null_distribution=null_distribution,
            n_simulations=self.n_simulations,
            n_better=n_better,
        )
    
    def returns_test(
        self,
        strategy_returns: Union[pd.Series, np.ndarray],
        statistic_func: Optional[Callable] = None,
    ) -> MonteCarloResult:
        """
        Test if returns are significantly different from random.
        
        Shuffles the order of returns to test if sequence matters.
        
        Parameters
        ----------
        strategy_returns : pd.Series or np.ndarray
            Strategy returns
        statistic_func : Callable, optional
            Function to compute test statistic
            
        Returns
        -------
        MonteCarloResult
            Test results
        """
        returns = np.asarray(strategy_returns)
        returns = returns[~np.isnan(returns)]
        
        if statistic_func is None:
            statistic_func = calculate_sharpe
        
        observed_stat = statistic_func(returns)
        
        # Shuffle returns (tests if order matters)
        null_distribution = np.zeros(self.n_simulations)
        
        iterator = range(self.n_simulations)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Monte Carlo")
        
        for i in iterator:
            shuffled = np.random.permutation(returns)
            null_distribution[i] = statistic_func(shuffled)
        
        n_better = np.sum(null_distribution >= observed_stat)
        p_value = (n_better + 1) / (self.n_simulations + 1)
        
        return MonteCarloResult(
            p_value=p_value,
            observed_statistic=observed_stat,
            null_distribution=null_distribution,
            n_simulations=self.n_simulations,
            n_better=n_better,
        )
    
    def path_simulation(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_paths: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate equity paths by resampling returns.
        
        Useful for visualizing range of possible outcomes.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Historical returns
        n_paths : int, optional
            Number of paths to simulate
            
        Returns
        -------
        np.ndarray
            Simulated equity paths (n_paths x n_periods)
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        n_paths = n_paths or self.n_simulations
        n_periods = len(returns)
        
        # Simulate paths by resampling
        paths = np.zeros((n_paths, n_periods))
        
        for i in range(n_paths):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns, size=n_periods, replace=True)
            # Calculate cumulative equity
            paths[i] = np.cumprod(1 + sampled_returns)
        
        return paths
    
    def drawdown_distribution(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_paths: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate distribution of maximum drawdowns.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Historical returns
        n_paths : int, optional
            Number of paths to simulate
            
        Returns
        -------
        np.ndarray
            Distribution of max drawdowns
        """
        paths = self.path_simulation(returns, n_paths)
        
        max_drawdowns = np.zeros(len(paths))
        
        for i, path in enumerate(paths):
            # Calculate drawdown
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns[i] = np.min(drawdown)
        
        return max_drawdowns


def permutation_test(
    signals: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    n_simulations: int = 1000,
) -> float:
    """
    Quick permutation test.
    
    Parameters
    ----------
    signals : pd.Series or np.ndarray
        Trading signals
    market_returns : pd.Series or np.ndarray
        Market returns
    n_simulations : int
        Number of simulations
        
    Returns
    -------
    float
        p-value
    """
    engine = MonteCarloEngine(n_simulations=n_simulations, show_progress=False)
    result = engine.permutation_test(signals, market_returns)
    return result.p_value


def simulate_equity_paths(
    returns: Union[pd.Series, np.ndarray],
    n_paths: int = 1000,
) -> np.ndarray:
    """
    Quick equity path simulation.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical returns
    n_paths : int
        Number of paths
        
    Returns
    -------
    np.ndarray
        Simulated paths
    """
    engine = MonteCarloEngine(n_simulations=n_paths, show_progress=False)
    return engine.path_simulation(returns, n_paths)
