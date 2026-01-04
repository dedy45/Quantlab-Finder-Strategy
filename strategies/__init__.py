"""
Strategy Library.

Collection of trading strategies for Quant Lab.
All strategies follow the fit/predict pattern.

Available Strategies:
- MomentumStrategy: Trend following based on MA crossover or momentum
- MeanReversionStrategy: Z-score, RSI, Bollinger based mean reversion
- MLStrategy: Machine learning based alpha generation
- EWMACStrategy: Robert Carver's EWMAC (Systematic Trading)
- MultiEWMACStrategy: Multiple EWMAC timeframes combined

Reference: 
- Protokol Kausalitas - FASE 2 (Strategy Development)
- Systematic Trading by Robert Carver
"""

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    StrategySignal,
    BacktestResult,
)

from .momentum_strategy import (
    MomentumStrategy,
    MomentumConfig,
    time_series_momentum,
    cross_sectional_momentum,
)

from .mean_reversion import (
    MeanReversionStrategy,
    MeanReversionConfig,
    calculate_zscore,
    calculate_half_life,
)

from .ml_strategy import (
    MLStrategy,
    MLStrategyConfig,
    ModelType,
)

from .ewmac_strategy import (
    EWMACStrategy,
    EWMACConfig,
    MultiEWMACStrategy,
    ewmac_forecast,
    EWMAC_VARIATIONS,
)


__all__ = [
    # Base
    'BaseStrategy',
    'StrategyConfig',
    'StrategyType',
    'StrategySignal',
    'BacktestResult',
    
    # Momentum
    'MomentumStrategy',
    'MomentumConfig',
    'time_series_momentum',
    'cross_sectional_momentum',
    
    # Mean Reversion
    'MeanReversionStrategy',
    'MeanReversionConfig',
    'calculate_zscore',
    'calculate_half_life',
    
    # ML
    'MLStrategy',
    'MLStrategyConfig',
    'ModelType',
    
    # EWMAC (Carver)
    'EWMACStrategy',
    'EWMACConfig',
    'MultiEWMACStrategy',
    'ewmac_forecast',
    'EWMAC_VARIATIONS',
]
