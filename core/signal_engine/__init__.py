"""
Signal Engine - Signal generation and regime detection.

This module handles:
- Market regime detection (HMM, volatility clustering)
- Signal generation from various sources
- Signal filtering based on regime
"""

from .regime import RegimeDetector, VolatilityRegime, HMMRegimeDetector

__all__ = [
    'RegimeDetector',
    'VolatilityRegime', 
    'HMMRegimeDetector',
]
