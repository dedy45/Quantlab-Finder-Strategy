"""
Validation Engine - Statistical validation for trading strategies.

This module provides tools to validate that trading strategies
have genuine predictive power and are not just lucky outcomes.

Key metrics:
- Sharpe Ratio (SR): Risk-adjusted returns
- Deflated Sharpe Ratio (DSR): SR adjusted for multiple testing
- Probabilistic Sharpe Ratio (PSR): Probability SR > benchmark
- Bootstrap Confidence Intervals
- Monte Carlo Permutation Tests
- Walk-Forward Validation
- Purged K-Fold Cross-Validation
- Robustness Testing

Reference: Protokol Kausalitas - Validasi Strategi
"""

from .sharpe import (
    SharpeCalculator, 
    calculate_sharpe, 
    calculate_annualized_sharpe,
    calculate_sortino,
    calculate_calmar,
)
from .psr import PSRCalculator, calculate_psr
from .dsr import DSRCalculator, calculate_dsr
from .bootstrap import BootstrapEngine, bootstrap_sharpe_ci
from .monte_carlo import MonteCarloEngine, permutation_test
from .walk_forward import WalkForwardValidator, WalkForwardResult, walk_forward_validate
from .purged_kfold import PurgedKFold, TimeSeriesSplit, PurgedCVResult, cross_val_score
from .robustness_tester import RobustnessTester, RobustnessResult

__all__ = [
    # Sharpe
    'SharpeCalculator',
    'calculate_sharpe',
    'calculate_annualized_sharpe',
    'calculate_sortino',
    'calculate_calmar',
    # PSR
    'PSRCalculator',
    'calculate_psr',
    # DSR
    'DSRCalculator',
    'calculate_dsr',
    # Bootstrap
    'BootstrapEngine',
    'bootstrap_sharpe_ci',
    # Monte Carlo
    'MonteCarloEngine',
    'permutation_test',
    # Walk-Forward
    'WalkForwardValidator',
    'WalkForwardResult',
    'walk_forward_validate',
    # Purged K-Fold
    'PurgedKFold',
    'TimeSeriesSplit',
    'PurgedCVResult',
    'cross_val_score',
    # Robustness
    'RobustnessTester',
    'RobustnessResult',
]
