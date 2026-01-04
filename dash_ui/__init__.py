"""
QuantLab Dash UI - Trading Dashboard.

A Plotly Dash application for quantitative trading analysis.

Run with: python -m dash_ui.app
Access at: http://localhost:8050

Pages:
- / (Dashboard) - Overview & KPIs
- /data-studio - ArcticDB data explorer
- /backtest - Backtest Arena
- /risk-lab - Risk analysis
- /settings - Configuration

Key Modules:
- theme: CYBORG color palette and Plotly layout
- cache: Server-side caching with Flask-Caching
- data_loader: Cached data loading from ArcticDB
- error_handler: Centralized error handling
- utils: LTTB downsampling, validators
"""

from .theme import COLORS, PLOTLY_LAYOUT, CHART_HEIGHT
from .cache import (
    init_cache,
    generate_cache_id,
    generate_data_key,
    save_to_cache,
    load_from_cache,
)
from .data_loader import (
    get_available_symbols,
    get_available_timeframes,
    load_ohlcv_cached,
)
from .error_handler import (
    create_error_alert,
    create_validation_error_alert,
    create_no_data_alert,
    safe_callback,
    log_error,
)

__all__ = [
    # Theme
    'COLORS',
    'PLOTLY_LAYOUT',
    'CHART_HEIGHT',
    # Cache
    'init_cache',
    'generate_cache_id',
    'generate_data_key',
    'save_to_cache',
    'load_from_cache',
    # Data
    'get_available_symbols',
    'get_available_timeframes',
    'load_ohlcv_cached',
    # Error handling
    'create_error_alert',
    'create_validation_error_alert',
    'create_no_data_alert',
    'safe_callback',
    'log_error',
]
