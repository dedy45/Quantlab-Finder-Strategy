"""
Data Studio Page - ArcticDB Data Explorer.

URL: /data-studio
Explore OHLCV data from ArcticDB with candlestick charts and data tables.

CRITICAL: Uses REAL data from ArcticDB only.
NO synthetic/dummy data allowed.
"""
import logging
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from ..theme import COLORS, PLOTLY_LAYOUT, CHART_HEIGHT
from ..components.charts import candlestick_chart, empty_chart
from ..components.tables import paginated_table
from ..components.cards import metric_card
from ..data_loader import (
    get_available_symbols,
    get_available_timeframes,
    load_ohlcv_cached,
)
from ..error_handler import (
    create_error_alert,
    create_no_data_alert,
    create_validation_error_alert,
)
from ..utils.validators import validate_symbol, validate_date_range

logger = logging.getLogger(__name__)

# =============================================================================
# LAYOUT
# =============================================================================

# Default date range - use static defaults, actual range loaded via callback
# This avoids ArcticDB read at import time for fast startup
DEFAULT_START = '2024-01-01'
DEFAULT_END = '2024-12-31'

layout = dbc.Container([
    # Initial load trigger - fires once when page loads
    dcc.Interval(
        id='ds-init-interval',
        interval=500,  # 500ms delay for component mounting
        n_intervals=0,
        max_intervals=1,  # Only fire once
    ),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("Data Studio", style={'color': COLORS['text_bright']}),
            html.P(
                "Explore OHLCV data from ArcticDB",
                style={'color': COLORS['text_muted']}
            ),
        ]),
    ], className='mb-4'),
    
    # Control Panel
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Symbol", style={'color': COLORS['text']}),
                    dcc.Dropdown(
                        id='ds-symbol-dropdown',
                        placeholder='Select symbol...',
                        style={'backgroundColor': COLORS['bg_tertiary']},
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label("Timeframe", style={'color': COLORS['text']}),
                    dcc.Dropdown(
                        id='ds-timeframe-dropdown',
                        options=[
                            {'label': tf, 'value': tf}
                            for tf in get_available_timeframes()
                        ],
                        value='1H',
                        style={'backgroundColor': COLORS['bg_tertiary']},
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label("Start Date", style={'color': COLORS['text']}),
                    dbc.Input(
                        id='ds-start-date',
                        type='date',
                        value=DEFAULT_START,
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label("End Date", style={'color': COLORS['text']}),
                    dbc.Input(
                        id='ds-end-date',
                        type='date',
                        value=DEFAULT_END,
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label(" ", style={'visibility': 'hidden'}),
                    dbc.Button(
                        "Load Data",
                        id='ds-load-btn',
                        color='primary',
                        className='w-100',
                    ),
                ], width=2),
            ], className='align-items-end'),
        ]),
    ], className='mb-4'),
    
    # Alert container
    html.Div(id='ds-alert-container'),
    
    # Data Info
    html.Div(id='ds-data-info', className='mb-3'),
    
    # Chart
    dbc.Card([
        dbc.CardHeader([
            html.Span("📊", className="me-2"),
            "Price Chart",
        ]),
        dbc.CardBody([
            dcc.Loading([
                dcc.Graph(
                    id='ds-price-chart',
                    config={'displayModeBar': True},
                    style={'height': CHART_HEIGHT['large']},
                ),
            ]),
        ]),
    ], className='mb-4'),
    
    # Data Table
    dbc.Card([
        dbc.CardHeader([
            html.Span("📋", className="me-2"),
            "OHLCV Data",
            html.Small(
                " (showing last 100 rows)",
                className='text-muted ms-2'
            ),
        ]),
        dbc.CardBody([
            dcc.Loading([
                html.Div(id='ds-data-table'),
            ]),
        ]),
    ]),
    
], fluid=True, style={'backgroundColor': COLORS['bg_primary'], 'minHeight': '100vh'})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output('ds-symbol-dropdown', 'options'),
    Output('ds-symbol-dropdown', 'value'),
    Input('ds-init-interval', 'n_intervals'),  # Trigger on page load
)
def load_symbols(n_intervals):
    """Load available symbols from ArcticDB."""
    try:
        symbols = get_available_symbols()
        options = [{'label': s, 'value': s} for s in symbols]
        
        # Default to XAUUSD if available
        default = 'XAUUSD' if 'XAUUSD' in symbols else (symbols[0] if symbols else None)
        
        logger.info(f"Data Studio: Loaded {len(symbols)} symbols")
        return options, default
    except Exception as e:
        logger.error(f"Failed to load symbols: {e}")
        return [], None


@callback(
    [
        Output('ds-price-chart', 'figure'),
        Output('ds-data-info', 'children'),
        Output('ds-data-table', 'children'),
        Output('ds-alert-container', 'children'),
    ],
    [
        Input('ds-load-btn', 'n_clicks'),
        Input('ds-symbol-dropdown', 'value'),  # Auto-load when symbol changes
    ],
    [
        State('ds-timeframe-dropdown', 'value'),
        State('ds-start-date', 'value'),
        State('ds-end-date', 'value'),
    ],
)
def load_data(n_clicks, symbol, timeframe, start_date, end_date):
    """
    Load OHLCV data from ArcticDB.
    
    CRITICAL: Uses REAL data only, NO synthetic data.
    """
    # Get trigger context
    from dash import ctx
    trigger_id = ctx.triggered_id if ctx.triggered_id else None
    # Empty chart for initial state
    empty_fig = empty_chart("Select symbol and click 'Load Data'", height=CHART_HEIGHT['large'])
    
    # Validate inputs
    if not symbol:
        return (
            empty_fig,
            None,
            html.P("Please select a symbol", className='text-muted'),
            None,
        )
    
    is_valid, error = validate_symbol(symbol)
    if not is_valid:
        return (
            empty_fig,
            None,
            html.P(error, className='text-warning'),
            create_validation_error_alert("Symbol", error),
        )
    
    is_valid, error = validate_date_range(start_date, end_date)
    if not is_valid:
        return (
            empty_fig,
            None,
            html.P(error, className='text-warning'),
            create_validation_error_alert("Date Range", error),
        )
    
    try:
        # Load REAL data from ArcticDB
        df = load_ohlcv_cached(symbol, start_date, end_date, timeframe)
        
        if df is None or len(df) == 0:
            return (
                empty_chart(f"No data for {symbol} ({timeframe})", height=CHART_HEIGHT['large']),
                dbc.Alert(f"No data found for {symbol} ({timeframe})", color='warning'),
                html.P("No data available", className='text-muted'),
                create_no_data_alert(symbol, timeframe),
            )
        
        # Create candlestick chart
        fig = candlestick_chart(
            df,
            title=f"{symbol} / {timeframe}",
            height=CHART_HEIGHT['large'],
            show_volume='volume' in df.columns,
        )
        
        # Data info alert
        info = dbc.Alert([
            html.Strong(f"{len(df):,} rows"),
            f" | {df.index.min().strftime('%Y-%m-%d %H:%M')} → {df.index.max().strftime('%Y-%m-%d %H:%M')}",
            f" | Source: ArcticDB",
        ], color='success', className='mb-0')
        
        # Data table (last 100 rows)
        table = paginated_table(
            df.tail(100),
            id='ds-ohlcv-table',
            page_size=20,
            columns=['open', 'high', 'low', 'close', 'volume'] if 'volume' in df.columns else ['open', 'high', 'low', 'close'],
        )
        
        logger.info(f"Loaded {len(df)} rows for {symbol} ({timeframe})")
        
        return fig, info, table, None
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return (
            empty_chart(f"Error: {str(e)[:50]}", height=CHART_HEIGHT['large']),
            dbc.Alert(f"Error: {str(e)}", color='danger'),
            html.P(f"Error loading data", className='text-danger'),
            create_error_alert("Data Load Error", str(e)),
        )
