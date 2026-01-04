"""
Portfolio Engine - Portfolio construction and allocation methods.

FASE 3: Portfolio Construction
- HRP (Hierarchical Risk Parity)
- Equal Weight allocation
- Volatility Targeting
- Kelly Criterion sizing
- Carver Position Sizing (Systematic Trading)
"""

from .base import (
    BaseAllocator,
    AllocationConfig,
    AllocationResult,
)
from .hrp_allocator import HRPAllocator
from .equal_weight import EqualWeightAllocator
from .volatility_target import VolatilityTargeter
from .kelly_sizing import KellySizer
from .carver_position import (
    CarverPositionSizer,
    CarverPositionConfig,
    PositionInertia,
    IDMCalculator,
)

__all__ = [
    'BaseAllocator',
    'AllocationConfig',
    'AllocationResult',
    'HRPAllocator',
    'EqualWeightAllocator',
    'VolatilityTargeter',
    'KellySizer',
    # Carver
    'CarverPositionSizer',
    'CarverPositionConfig',
    'PositionInertia',
    'IDMCalculator',
]
