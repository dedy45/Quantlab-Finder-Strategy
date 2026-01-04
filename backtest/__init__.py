"""
Backtest - Multi-engine backtesting framework.

Supports:
- VectorBT: Fast vectorized screening (1000+ ideas)
- Nautilus: Realistic event-driven validation (Top 50)
- LEAN: Alpha Streams submission

4-Phase Workflow:
1. Screening (VectorBT) → Top 50
2. Validation (Nautilus) → Top 10
3. Deep Analysis (Quant Lab) → Top 3
4. Paper Trading → Best 1
"""

from .base import (
    BacktestConfig,
    BacktestEngine,
    BacktestMetrics,
    BacktestResult,
    BaseBacktestAdapter
)

from .vectorbt import (
    VectorBTAdapter,
    VectorBTConfig,
    StrategyScreener,
    ScreeningResult
)

from .nautilus import (
    NautilusAdapter,
    NautilusConfig,
    CandidateValidator,
    ValidationResult
)

__all__ = [
    # Base
    'BacktestConfig',
    'BacktestEngine',
    'BacktestMetrics',
    'BacktestResult',
    'BaseBacktestAdapter',
    # VectorBT
    'VectorBTAdapter',
    'VectorBTConfig',
    'StrategyScreener',
    'ScreeningResult',
    # Nautilus
    'NautilusAdapter',
    'NautilusConfig',
    'CandidateValidator',
    'ValidationResult'
]
