"""
Labeling Module - Generate labels for ML models.

FASE 1B: Labeling
- triple_barrier.py - Triple-barrier labeling method
- trend_scanning.py - T-value based trend labels (TODO)
- meta_labeling.py - Secondary model labels

Reference: Protokol Kausalitas - Fase 1 (Sebab Fundamental)
"""

from .triple_barrier import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    TripleBarrierResult,
    BarrierType,
    apply_triple_barrier,
    get_vertical_barrier,
    get_horizontal_barriers,
    get_daily_volatility,
)

from .meta_labeling import (
    MetaLabeler,
    MetaLabelResult,
    generate_meta_labels,
    calculate_bet_size,
)

__all__ = [
    # Triple Barrier
    'TripleBarrierLabeler',
    'TripleBarrierConfig',
    'TripleBarrierResult',
    'BarrierType',
    'apply_triple_barrier',
    'get_vertical_barrier',
    'get_horizontal_barriers',
    'get_daily_volatility',
    # Meta Labeling
    'MetaLabeler',
    'MetaLabelResult',
    'generate_meta_labels',
    'calculate_bet_size',
]
