"""
Dashboard Page - Main overview with KPIs.

URL: /
Displays portfolio overview, KPIs, and equity curve.

CRITICAL: Uses REAL data from ArcticDB only.
NO synthetic/dummy data allowed.
"""
import logging
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from ..theme import COLORS, PLOTLY_LAYOUT, CHART_HEIGHT
from ..components.cards import kpi_card, kpi_row, metric_card
from ..components.charts import equity_curve_chart, empty_chart
from ..error_handler import create_error_alert, create_no_data_alert

logger = logging.getLogger(__name__)

# =============================================================================
# LAYOUT
# =============================================================================

layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("Dashboard", style={'color': COLORS['text_bright']}),
                dbc.Badge("Live", color="success", className="ms-2"),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.P(
                "Portfolio overview and key performance indicators",
                style={'color': COLORS['text_muted']}
            ),
        ]),
    ], className='mb-4'),
    
    # Alert container for errors
    html.Div(id='dashboard-alert-container'),
    
    # KPI Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("PSR", className="text-muted"),
                    html.H3(id='dashboard-psr', children="-", className='text-success'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Sharpe Ratio", className="text-muted"),
                    html.H3(id='dashboard-sharpe', children="-", className='text-info'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Max Drawdown", className="text-muted"),
                    html.H3(id='dashboard-maxdd', children="-", className='text-warning'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Volatility", className="text-muted"),
                    html.H3(id='dashboard-vol', children="-", className='text-info'),
                ], className='text-center'),
            ]),
        ], width=3),
    ], className='mb-4'),
    
    # Charts Row
    dbc.Row([
        # Equity Curve
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("📈", className="me-2"),
                    "Equity Curve (XAUUSD)",
                ]),
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(
                            id='dashboard-equity-chart',
                            config={'displayModeBar': False},
                            style={'height': CHART_HEIGHT['medium']},
                        ),
                    ]),
                ]),
            ]),
        ], width=8),
        
        # Data Status
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("📊", className="me-2"),
                    "Data Status",
                ]),
                dbc.CardBody([
                    html.Div(id='dashboard-data-status', children=[
                        html.P("Loading...", className='text-muted'),
                    ]),
                ]),
            ]),
        ], width=4),
    ], className='mb-4'),
    
    # Info Alert
    dbc.Alert([
        html.Strong("QuantLab Dashboard: "),
        "Displaying real data from ArcticDB. ",
        "Navigate to Data Studio for detailed exploration.",
    ], color="info"),
    
    # Hidden interval for auto-refresh (enabled for initial load)
    dcc.Interval(
        id='dashboard-refresh-interval',
        interval=60000,  # 1 minute
        n_intervals=0,
        max_intervals=1,  # Only trigger once on load
        disabled=False,  # Enabled for initial load
    ),
    
], fluid=True, style={'backgroundColor': COLORS['bg_primary'], 'minHeight': '100vh'})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    [
        Output('dashboard-psr', 'children'),
        Output('dashboard-sharpe', 'children'),
        Output('dashboard-maxdd', 'children'),
        Output('dashboard-vol', 'children'),
        Output('dashboard-equity-chart', 'figure'),
        Output('dashboard-data-status', 'children'),
        Output('dashboard-alert-container', 'children'),
    ],
    Input('dashboard-refresh-interval', 'n_intervals'),
)
def update_dashboard(_):
    """
    Update dashboard with real data from ArcticDB.
    
    CRITICAL: Uses REAL data only, NO synthetic data.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    
    # Default empty chart
    empty_fig = empty_chart("Loading data...", height=CHART_HEIGHT['medium'])
    
    try:
        # Load REAL data from ArcticDB
        from ..data_loader import load_ohlcv_cached, get_available_symbols
        
        # Get available symbols
        symbols = get_available_symbols()
        
        if not symbols:
            return (
                "-", "-", "-", "-",
                empty_chart("No data in ArcticDB"),
                html.P("No symbols available", className='text-warning'),
                create_no_data_alert(),
            )
        
        # Use first available symbol (typically XAUUSD)
        symbol = symbols[0] if 'XAUUSD' not in symbols else 'XAUUSD'
        
        # Load 1 year of daily data
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        df = load_ohlcv_cached(symbol, start_date, end_date, '1D')
        
        if df is None or len(df) == 0:
            # Try hourly data
            df = load_ohlcv_cached(symbol, start_date, end_date, '1H')
        
        if df is None or len(df) == 0:
            return (
                "-", "-", "-", "-",
                empty_chart(f"No data for {symbol}"),
                html.P(f"No data available for {symbol}", className='text-warning'),
                create_no_data_alert(symbol),
            )
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 30:
            return (
                "-", "-", "-", "-",
                empty_chart("Insufficient data"),
                html.P("Need at least 30 data points", className='text-warning'),
                create_error_alert("Insufficient Data", "Need at least 30 data points for analysis"),
            )
        
        # Calculate metrics
        # Annualization factor
        if len(df) > 252:
            ann_factor = 252  # Daily
        else:
            ann_factor = 252 * 24  # Hourly
        
        # Sharpe Ratio
        sharpe = (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(ann_factor)
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # PSR (simplified)
        n = len(returns)
        se_sr = np.sqrt((1 + 0.5 * sharpe**2) / (n - 1)) if n > 1 else 1
        from scipy import stats
        psr = stats.norm.cdf(sharpe / se_sr) if se_sr > 0 else 0.5
        
        # Create equity chart
        equity = 100 * cumulative
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Equity',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba(42, 159, 214, 0.1)",
        ))
        
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=CHART_HEIGHT['medium'],
            showlegend=False,
            yaxis_title="Equity (normalized)",
            xaxis_title="",
        )
        
        # Data status
        data_status = html.Div([
            html.Div([
                html.Strong("Symbol: "),
                html.Span(symbol),
            ], className='mb-2'),
            html.Div([
                html.Strong("Rows: "),
                html.Span(f"{len(df):,}"),
            ], className='mb-2'),
            html.Div([
                html.Strong("Period: "),
                html.Span(f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"),
            ], className='mb-2'),
            html.Hr(),
            html.Div([
                html.Strong("Available Symbols: "),
                html.Span(", ".join(symbols[:5])),
                html.Span("..." if len(symbols) > 5 else ""),
            ]),
        ])
        
        return (
            f"{psr:.1%}",
            f"{sharpe:.2f}",
            f"{max_dd:.1%}",
            f"{volatility:.1%}",
            fig,
            data_status,
            None,  # No error
        )
        
    except Exception as e:
        logger.error(f"Dashboard update failed: {e}")
        return (
            "-", "-", "-", "-",
            empty_chart(f"Error: {str(e)[:50]}"),
            html.P(f"Error: {str(e)}", className='text-danger'),
            create_error_alert("Dashboard Error", str(e)),
        )
