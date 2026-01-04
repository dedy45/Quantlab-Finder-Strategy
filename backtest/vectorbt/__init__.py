"""
VectorBT Adapter - Fast vectorized backtesting for screening.

Use for:
- Screening 1000+ strategy combinations
- Parameter optimization
- Quick prototyping

Accuracy: ~80-90% (simplified fill model)
Speed: Very fast (vectorized NumPy/Pandas)
"""

from .adapter import VectorBTAdapter, VectorBTConfig
from .screener import StrategyScreener, ScreeningResult

__all__ = [
    'VectorBTAdapter',
    'VectorBTConfig',
    'StrategyScreener',
    'ScreeningResult'
]
