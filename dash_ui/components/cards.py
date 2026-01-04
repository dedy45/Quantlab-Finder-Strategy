"""
Card Components for Dashboard.

KPI cards, metric cards, and card rows for consistent styling.
"""
from typing import List, Dict, Any, Optional
from dash import html
import dash_bootstrap_components as dbc

from ..theme import COLORS, KPI_CARD_STYLE, get_bootstrap_color


def kpi_card(
    title: str,
    value: str,
    color: str = None,
    subtitle: str = None,
    trend: str = None,
) -> dbc.Card:
    """
    Create a KPI card.
    
    Parameters
    ----------
    title : str
        KPI title/label
    value : str
        KPI value to display
    color : str, optional
        Bootstrap color class ('success', 'danger', 'warning', 'info', 'primary')
        or hex color
    subtitle : str, optional
        Subtitle text below value
    trend : str, optional
        Trend indicator ('up', 'down', 'neutral')
        
    Returns
    -------
    dbc.Card
        KPI card component
    """
    # Determine color class
    if color in ('success', 'danger', 'warning', 'info', 'primary', 'secondary'):
        value_class = f"text-{color}"
        value_style = {}
    elif color:
        value_class = ""
        value_style = {'color': color}
    else:
        value_class = ""
        value_style = {'color': COLORS['text_bright']}
    
    # Trend icon
    trend_icon = None
    if trend == 'up':
        trend_icon = html.Span("↑", className="text-success ms-2")
    elif trend == 'down':
        trend_icon = html.Span("↓", className="text-danger ms-2")
    
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-2", style={'fontSize': '0.85rem'}),
            html.Div([
                html.H3(
                    value,
                    className=f"mb-0 {value_class}",
                    style=value_style,
                ),
                trend_icon,
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
            html.Small(subtitle, className="text-muted") if subtitle else None,
        ], className="text-center py-3"),
    ], className="kpi-card h-100")


def kpi_row(metrics: List[Dict[str, Any]]) -> dbc.Row:
    """
    Create a row of KPI cards.
    
    Parameters
    ----------
    metrics : list of dict
        List of metric dictionaries with keys:
        - title: str
        - value: str
        - color: str (optional)
        - subtitle: str (optional)
        - trend: str (optional)
        
    Returns
    -------
    dbc.Row
        Row of KPI cards
        
    Examples
    --------
    >>> metrics = [
    ...     {'title': 'PSR', 'value': '97.2%', 'color': 'success'},
    ...     {'title': 'Sharpe', 'value': '1.85', 'color': 'info'},
    ...     {'title': 'Max DD', 'value': '-12.3%', 'color': 'warning'},
    ... ]
    >>> row = kpi_row(metrics)
    """
    n_cols = len(metrics)
    col_width = 12 // n_cols if n_cols <= 6 else 2
    
    return dbc.Row([
        dbc.Col(
            kpi_card(
                title=m.get('title', ''),
                value=m.get('value', '-'),
                color=m.get('color'),
                subtitle=m.get('subtitle'),
                trend=m.get('trend'),
            ),
            width=col_width,
        )
        for m in metrics
    ], className="g-3 mb-4")


def metric_card(
    title: str,
    children,
    icon: str = None,
    color: str = None,
) -> dbc.Card:
    """
    Create a metric card with custom content.
    
    Parameters
    ----------
    title : str
        Card title
    children : any
        Card body content
    icon : str, optional
        Icon emoji or character
    color : str, optional
        Header accent color
        
    Returns
    -------
    dbc.Card
        Metric card component
    """
    header_style = {}
    if color:
        header_style['borderLeft'] = f"3px solid {COLORS.get(color, color)}"
    
    return dbc.Card([
        dbc.CardHeader([
            html.Span(icon, className="me-2") if icon else None,
            html.Span(title, style={'fontWeight': '500'}),
        ], style=header_style),
        dbc.CardBody(children),
    ], className="mb-3")


def stat_card(
    title: str,
    stats: List[Dict[str, str]],
    icon: str = None,
) -> dbc.Card:
    """
    Create a statistics card with multiple key-value pairs.
    
    Parameters
    ----------
    title : str
        Card title
    stats : list of dict
        List of {'label': str, 'value': str} dictionaries
    icon : str, optional
        Icon emoji
        
    Returns
    -------
    dbc.Card
        Statistics card component
    """
    stat_rows = [
        dbc.Row([
            dbc.Col(html.Span(s['label'], className="text-muted"), width=6),
            dbc.Col(html.Span(s['value'], className="text-end"), width=6),
        ], className="mb-2")
        for s in stats
    ]
    
    return metric_card(title, stat_rows, icon=icon)


def info_alert(
    title: str,
    message: str,
    color: str = 'info',
    icon: str = None,
) -> dbc.Alert:
    """
    Create an info alert card.
    
    Parameters
    ----------
    title : str
        Alert title
    message : str
        Alert message
    color : str, default='info'
        Bootstrap color class
    icon : str, optional
        Icon emoji
        
    Returns
    -------
    dbc.Alert
        Alert component
    """
    return dbc.Alert([
        html.Div([
            html.Span(icon, className="me-2") if icon else None,
            html.Strong(title),
        ]),
        html.P(message, className="mb-0 mt-2"),
    ], color=color, className="mb-3")
