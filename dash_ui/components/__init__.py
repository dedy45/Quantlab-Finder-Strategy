"""
QuantLab Dash UI - Reusable Components.

Components:
- sidebar: Navigation sidebar
- cards: KPI and metric cards
- charts: Plotly chart wrappers
- tables: Data tables with pagination
- toast: Toast notifications
"""
from .sidebar import create_sidebar
from .navbar import create_navbar  # Legacy alias
from .cards import kpi_card, kpi_row, metric_card, stat_card, info_alert
from .charts import (
    candlestick_chart,
    equity_curve_chart,
    drawdown_chart,
    returns_distribution_chart,
    rolling_metric_chart,
    empty_chart,
    correlation_heatmap,
)
from .tables import paginated_table, simple_table, stats_table
from .toast import (
    create_toast,
    success_toast,
    error_toast,
    warning_toast,
    info_toast,
    toast_container,
)

__all__ = [
    # Navigation
    'create_sidebar',
    'create_navbar',
    # Cards
    'kpi_card',
    'kpi_row',
    'metric_card',
    'stat_card',
    'info_alert',
    # Charts
    'candlestick_chart',
    'equity_curve_chart',
    'drawdown_chart',
    'returns_distribution_chart',
    'rolling_metric_chart',
    'empty_chart',
    'correlation_heatmap',
    # Tables
    'paginated_table',
    'simple_table',
    'stats_table',
    # Toast
    'create_toast',
    'success_toast',
    'error_toast',
    'warning_toast',
    'info_toast',
    'toast_container',
]
