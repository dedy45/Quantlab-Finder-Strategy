"""
Research Pipeline for FASE 5: Production.

Modules for systematic strategy research and validation.
"""

from .strategy_scanner import StrategyScanner, ScanResult
from .candidate_validator import CandidateValidator, ValidationResult
from .run_research import ResearchPipeline
from .data_validator import (
    ValidationReport,
    validate_dataframe,
    validate_returns,
    validate_dimensionality,
    validate_train_test_split,
    safe_divide,
    clean_returns
)
from .parameter_optimizer import (
    WalkForwardOptimizer,
    ParameterSpaces,
    OptimizationResult,
    GridSearchResult,
    optimize_strategy
)

__all__ = [
    # Strategy Scanner
    'StrategyScanner',
    'ScanResult',
    # Candidate Validator
    'CandidateValidator',
    'ValidationResult',
    # Research Pipeline
    'ResearchPipeline',
    # Data Validator
    'ValidationReport',
    'validate_dataframe',
    'validate_returns',
    'validate_dimensionality',
    'validate_train_test_split',
    'safe_divide',
    'clean_returns',
    # Parameter Optimizer
    'WalkForwardOptimizer',
    'ParameterSpaces',
    'OptimizationResult',
    'GridSearchResult',
    'optimize_strategy',
]
