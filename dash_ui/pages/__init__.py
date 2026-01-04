"""
QuantLab Dash UI - Pages.

Pages are imported at startup by index.py to register callbacks.
Each page exports a `layout` variable.

Pages:
- data_quality_gate: Data validation (FIRST PAGE - must pass before others)
- data_studio: ArcticDB data explorer
- backtest_arena: Backtest runner
- risk_lab: Risk analysis
- settings: Configuration management
- dashboard: (Legacy - redirects to data_quality_gate)
"""

# Pages are imported at startup in index.py to register callbacks
# This is CRITICAL - without startup imports, callbacks won't work!

__all__ = [
    'data_quality_gate',
    'data_studio',
    'backtest_arena',
    'risk_lab',
    'settings',
    'dashboard',  # Legacy
]
