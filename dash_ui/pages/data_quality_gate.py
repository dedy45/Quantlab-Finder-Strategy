"""
Data Quality Gate Page - First Page / Foundation.

URL: /
Philosophy: "Garbage In = Garbage Out"

This page MUST PASS before other features can be used.
Validates that data is "statistically sound" for analysis.

CRITICAL: Uses REAL data from ArcticDB only.
NO synthetic/dummy data allowed.
"""
import logging
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from ..theme import COLORS, PLOTLY_LAYOUT, CHART_HEIGHT
from ..components.charts import empty_chart
from ..error_handler import create_error_alert, create_no_data_alert

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

GRADE_COLORS = {
    'A': '#77b300',  # Green
    'B': '#2a9fd6',  # Blue
    'C': '#ff8800',  # Orange
    'D': '#cc0000',  # Red
    'F': '#880000',  # Dark Red
}

GRADE_DESCRIPTIONS = {
    'A': 'Excellent - Ready for any analysis',
    'B': 'Good - Minor issues, proceed with caution',
    'C': 'Fair - Significant issues, review recommended',
    'D': 'Poor - Major issues, not recommended',
    'F': 'Fail - Data not suitable for analysis',
}


# =============================================================================
# HELPER FUNCTIONS (must be defined before layout)
# =============================================================================

def _create_checklist_placeholder():
    """Create placeholder checklist items."""
    categories = [
        ('Completeness', '20%', 'Checks for missing values, data gaps, and bar completeness'),
        ('Distribution', '15%', 'Analyzes returns normality, skewness, and kurtosis'),
        ('Stationarity', '20%', 'Tests if returns are stationary (ADF test)'),
        ('Autocorrelation', '10%', 'Detects serial correlation in returns'),
        ('Outliers', '15%', 'Identifies extreme values using Z-score and IQR'),
        ('Sample Size', '10%', 'Verifies sufficient observations for inference'),
        ('OHLC Integrity', '10%', 'Validates OHLC price relationships'),
    ]
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span("⬜", className="me-2", style={'fontSize': '16px'}),
                html.Span(name, style={'color': COLORS['text_muted']}),
                html.Span(f" ({weight})", style={'color': COLORS['text_muted'], 'fontSize': '12px'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Small(desc, style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginLeft': '28px'}),
            dbc.Progress(value=0, color='secondary', style={'height': '8px', 'marginTop': '5px'}),
        ], className='mb-3')
        for name, weight, desc in categories
    ])


def _create_checklist_item(name: str, score: float, passed: bool, weight: str, status: str = "", advice: str = ""):
    """Create a single checklist item with progress bar and advice."""
    icon = "✅" if passed else ("⚠️" if score >= 50 else "❌")
    color = "success" if passed else ("warning" if score >= 50 else "danger")
    
    # Status label
    status_text = status if status else ("PASS" if passed else ("WARNING" if score >= 50 else "FAIL"))
    status_color = COLORS['success'] if passed else (COLORS['warning'] if score >= 50 else COLORS['danger'])
    
    children = [
        html.Div([
            html.Span(icon, className="me-2", style={'fontSize': '16px'}),
            html.Span(name, style={'color': COLORS['text'], 'fontWeight': '500'}),
            html.Span(f" ({weight})", style={'color': COLORS['text_muted'], 'fontSize': '12px'}),
            html.Span(f" [{status_text}]", style={'color': status_color, 'fontSize': '11px', 'marginLeft': '8px'}),
            html.Span(f" - {score:.0f}%", style={'color': COLORS['text'], 'marginLeft': 'auto', 'fontWeight': 'bold'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        dbc.Progress(value=score, color=color, style={'height': '8px', 'marginTop': '5px'}),
    ]
    
    # Add advice if score < 80% and advice is provided
    if score < 80 and advice:
        children.append(
            html.Div([
                html.Small(f"💡 {advice}", style={
                    'color': COLORS['warning'] if score >= 50 else COLORS['danger'],
                    'fontSize': '11px',
                    'fontStyle': 'italic',
                    'marginTop': '4px',
                    'display': 'block',
                }),
            ])
        )
    
    return html.Div(children, className='mb-3')


def _create_quality_score_display(score: float, grade: str, can_proceed: bool = None, status_label: str = ""):
    """Create quality score display with gauge and proceed indicator."""
    color = GRADE_COLORS.get(grade, COLORS['text_muted'])
    description = GRADE_DESCRIPTIONS.get(grade, '')
    
    # Determine proceed status
    if can_proceed is None:
        can_proceed = grade in ['A', 'B', 'C']
    
    proceed_text = "✅ Safe to Proceed" if can_proceed else "❌ Do NOT Proceed"
    proceed_color = COLORS['success'] if can_proceed else COLORS['danger']
    
    return html.Div([
        html.Div([
            html.H1(f"{score:.0f}%", style={
                'fontSize': '72px',
                'fontWeight': 'bold',
                'color': color,
                'marginBottom': '0',
            }),
            html.H2(f"Grade {grade}", style={
                'fontSize': '36px',
                'color': color,
                'marginTop': '0',
            }),
        ], style={'textAlign': 'center'}),
        html.P(description, style={
            'color': COLORS['text_muted'],
            'textAlign': 'center',
            'marginTop': '10px',
        }),
        # Proceed indicator
        html.Div([
            html.Span(proceed_text, style={
                'color': proceed_color,
                'fontWeight': 'bold',
                'fontSize': '16px',
            }),
        ], style={'textAlign': 'center', 'marginTop': '15px'}),
        # Threshold info
        html.P(
            "Minimum threshold: 70% (Grade C)" if not can_proceed else "",
            style={
                'color': COLORS['text_muted'],
                'textAlign': 'center',
                'fontSize': '12px',
                'marginTop': '5px',
            }
        ),
        dbc.Progress(
            value=score,
            color='success' if grade in ['A', 'B'] else ('warning' if grade == 'C' else 'danger'),
            style={'height': '20px', 'marginTop': '15px'},
        ),
    ], style={'padding': '20px'})


# =============================================================================
# LAYOUT
# =============================================================================

layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2([
                    html.Span("🔬", className="me-2"),
                    "Data Quality Gate"
                ], style={'color': COLORS['text_bright']}),
                html.Div(id='quality-status-badge'),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}),
            html.P(
                '"Garbage In = Garbage Out" - Data MUST pass validation before analysis',
                style={'color': COLORS['text_muted'], 'fontStyle': 'italic'}
            ),
        ]),
    ], className='mb-4'),
    
    # Alert container
    html.Div(id='quality-alert-container'),
    
    # Control Panel
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Symbol", style={'color': COLORS['text']}),
                    dcc.Dropdown(
                        id='quality-symbol-dropdown',
                        placeholder="Select symbol...",
                        style={'backgroundColor': COLORS['bg_secondary']},
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Label("Timeframe", style={'color': COLORS['text']}),
                    dcc.Dropdown(
                        id='quality-timeframe-dropdown',
                        options=[
                            {'label': '15 Minutes', 'value': '15T'},
                            {'label': '1 Hour', 'value': '1H'},
                            {'label': '4 Hours', 'value': '4H'},
                            {'label': 'Daily', 'value': '1D'},
                        ],
                        value='1H',
                        style={'backgroundColor': COLORS['bg_secondary']},
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label(" ", style={'color': 'transparent'}),
                    dbc.Button(
                        [html.Span( className="me-2"), "All Data"],
                        id='quality-alldata-btn',
                        color='info',
                        outline=True,
                        className='w-100',
                        title="Load all available data and auto-fill date range",
                    ),
                ], width=1),
                dbc.Col([
                    dbc.Label("Start Date", style={'color': COLORS['text']}),
                    dbc.Input(
                        id='quality-start-date',
                        type='date',
                        value='2024-01-01',
                        style={'backgroundColor': COLORS['bg_secondary'], 'color': COLORS['text']},
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label("End Date", style={'color': COLORS['text']}),
                    dbc.Input(
                        id='quality-end-date',
                        type='date',
                        value='2024-12-31',
                        style={'backgroundColor': COLORS['bg_secondary'], 'color': COLORS['text']},
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label(" ", style={'color': 'transparent'}),
                    dbc.Button(
                        [html.Span("🔍", className="me-2"), "Validate"],
                        id='quality-validate-btn',
                        color='primary',
                        className='w-100',
                    ),
                ], width=2),
            ]),
        ]),
    ], className='mb-4', style={'backgroundColor': COLORS['bg_secondary']}),
    
    # Quality Score Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Quality Score"),
                dbc.CardBody([
                    html.Div(id='quality-score-display', children=[
                        html.Div([
                            html.H1("-", style={'fontSize': '72px', 'fontWeight': 'bold', 'color': COLORS['text_muted']}),
                            html.P("Run validation to see score", style={'color': COLORS['text_muted']}),
                        ], style={'textAlign': 'center', 'padding': '30px'}),
                    ]),
                ]),
            ], style={'backgroundColor': COLORS['bg_secondary'], 'height': '100%'}),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("✅ Validation Checklist"),
                dbc.CardBody([
                    html.Div(id='quality-checklist', children=[
                        _create_checklist_placeholder(),
                    ]),
                ]),
            ], style={'backgroundColor': COLORS['bg_secondary'], 'height': '100%'}),
        ], width=8),
    ], className='mb-4'),
    
    # Diagnostic Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Returns Distribution"),
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(
                            id='quality-distribution-chart',
                            config={'displayModeBar': False},
                            style={'height': CHART_HEIGHT['medium']},
                        ),
                    ]),
                ]),
            ], style={'backgroundColor': COLORS['bg_secondary']}),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📉 Autocorrelation (ACF)"),
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(
                            id='quality-acf-chart',
                            config={'displayModeBar': False},
                            style={'height': CHART_HEIGHT['medium']},
                        ),
                    ]),
                ]),
            ], style={'backgroundColor': COLORS['bg_secondary']}),
        ], width=6),
    ], className='mb-4'),
    
    # Detailed Report
    dbc.Card([
        dbc.CardHeader("📋 Detailed Report & Recommendations"),
        dbc.CardBody([
            html.Div(id='quality-detailed-report', children=[
                html.P("Run validation to see detailed report", 
                      style={'color': COLORS['text_muted'], 'textAlign': 'center', 'padding': '20px'}),
            ]),
        ]),
    ], className='mb-4', style={'backgroundColor': COLORS['bg_secondary']}),
    
    # Action Buttons
    dbc.Row([
        dbc.Col([
            dbc.Button(
                [html.Span("➡️", className="me-2"), "Proceed to Data Studio"],
                id='quality-proceed-btn',
                color='success',
                size='lg',
                disabled=True,
                className='me-2',
            ),
            dbc.Button(
                [html.Span("📄", className="me-2"), "Export Report"],
                id='quality-export-btn',
                color='secondary',
                size='lg',
                disabled=True,
            ),
        ], className='text-center'),
    ], className='mb-4'),
    
    # Session store for quality status
    dcc.Store(id='quality-status-store', storage_type='session', data={
        'validated': False,
        'symbol': None,
        'timeframe': None,
        'quality_score': 0,
        'grade': 'F',
        'passed': False,
    }),
    
    # Hidden link for navigation
    dcc.Location(id='quality-redirect', refresh=True),
    
], fluid=True, style={'backgroundColor': COLORS['bg_primary'], 'minHeight': '100vh'})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output('quality-symbol-dropdown', 'options'),
    Output('quality-symbol-dropdown', 'value'),
    Input('quality-symbol-dropdown', 'id'),
)
def load_symbols(_):
    """Load available symbols from ArcticDB."""
    try:
        from ..data_loader import get_available_symbols, get_default_symbol
        
        symbols = get_available_symbols()
        options = [{'label': s, 'value': s} for s in symbols]
        default = get_default_symbol()
        
        return options, default
    except Exception as e:
        logger.error(f"Failed to load symbols: {e}")
        return [], None


@callback(
    [
        Output('quality-start-date', 'value'),
        Output('quality-end-date', 'value'),
        Output('quality-alert-container', 'children', allow_duplicate=True),
    ],
    Input('quality-alldata-btn', 'n_clicks'),
    State('quality-symbol-dropdown', 'value'),
    State('quality-timeframe-dropdown', 'value'),
    prevent_initial_call=True,
)
def load_all_data_range(n_clicks, symbol, current_timeframe):
    """Load all available data range for selected symbol+timeframe and auto-fill dates."""
    from dash import no_update
    import pandas as pd
    
    if not symbol:
        return no_update, no_update, dbc.Alert(
            "Please select a symbol first",
            color="warning",
            dismissable=True,
            duration=3000,
        )
    
    if not current_timeframe:
        return no_update, no_update, dbc.Alert(
            "Please select a timeframe first",
            color="warning",
            dismissable=True,
            duration=3000,
        )
    
    try:
        from ..data_loader import load_all_data_for_symbol
        
        # Load data for the SELECTED timeframe (not default)
        logger.info(f"[All Data] Loading data for {symbol} at {current_timeframe}")
        df, actual_timeframe = load_all_data_for_symbol(symbol, current_timeframe)
        
        if df is None or len(df) == 0:
            return no_update, no_update, dbc.Alert(
                f"No data found for {symbol} at {current_timeframe}",
                color="danger",
                dismissable=True,
                duration=3000,
            )
        
        # Debug: Log index info
        logger.info(f"[All Data] DataFrame index type: {type(df.index)}")
        logger.info(f"[All Data] DataFrame columns: {df.columns.tolist()}")
        
        # Get date range from data
        if isinstance(df.index, pd.DatetimeIndex):
            start_date = df.index.min()
            end_date = df.index.max()
            logger.info(f"[All Data] Using DatetimeIndex: {start_date} to {end_date}")
        elif 'timestamp' in df.columns:
            # Fallback: timestamp is still a column
            start_date = pd.to_datetime(df['timestamp'].min())
            end_date = pd.to_datetime(df['timestamp'].max())
            logger.info(f"[All Data] Using timestamp column: {start_date} to {end_date}")
        else:
            logger.error(f"[All Data] Cannot determine date range - no timestamp found")
            return no_update, no_update, dbc.Alert(
                f"Cannot determine date range for {symbol} - no timestamp found",
                color="danger",
                dismissable=True,
                duration=5000,
            )
        
        # Convert to string format for date input
        if hasattr(start_date, 'strftime'):
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            # Fallback for numpy datetime64 or similar
            start_str = str(start_date)[:10]
            end_str = str(end_date)[:10]
        
        logger.info(f"[All Data] Final date range: {start_str} to {end_str} ({len(df):,} bars at {current_timeframe})")
        
        return start_str, end_str, dbc.Alert(
            [
                html.Strong(f"✅ Loaded {symbol} {current_timeframe} data range: "),
                f"{start_str} to {end_str} ({len(df):,} bars)"
            ],
            color="success",
            dismissable=True,
            duration=5000,
        )
        
    except Exception as e:
        logger.error(f"[All Data] Failed to load all data range: {e}")
        import traceback
        traceback.print_exc()
        return no_update, no_update, dbc.Alert(
            f"Error loading data: {str(e)}",
            color="danger",
            dismissable=True,
            duration=5000,
        )


@callback(
    [
        Output('quality-score-display', 'children'),
        Output('quality-checklist', 'children'),
        Output('quality-distribution-chart', 'figure'),
        Output('quality-acf-chart', 'figure'),
        Output('quality-detailed-report', 'children'),
        Output('quality-status-badge', 'children'),
        Output('quality-proceed-btn', 'disabled'),
        Output('quality-export-btn', 'disabled'),
        Output('quality-status-store', 'data'),
        Output('quality-alert-container', 'children', allow_duplicate=True),
    ],
    Input('quality-validate-btn', 'n_clicks'),
    [
        State('quality-symbol-dropdown', 'value'),
        State('quality-timeframe-dropdown', 'value'),
        State('quality-start-date', 'value'),
        State('quality-end-date', 'value'),
    ],
    prevent_initial_call=True,
)
def validate_data(n_clicks, symbol, timeframe, start_date, end_date):
    """Run comprehensive data quality validation."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    empty_fig = empty_chart("Run validation to see chart", height=CHART_HEIGHT['medium'])
    default_status = {
        'validated': False, 'symbol': None, 'timeframe': None,
        'quality_score': 0, 'grade': 'F', 'passed': False,
    }
    
    if not symbol:
        return (
            _create_quality_score_display(0, 'F'),
            _create_checklist_placeholder(),
            empty_fig, empty_fig,
            html.P("Please select a symbol", style={'color': COLORS['warning']}),
            dbc.Badge("Not Validated", color="secondary"),
            True, True, default_status,
            create_error_alert("Validation Error", "Please select a symbol"),
        )
    
    try:
        from ..data_loader import load_ohlcv_cached
        from ..utils.data_quality import validate_data_quality
        
        logger.info(f"Validating {symbol} {timeframe} from {start_date} to {end_date}")
        df = load_ohlcv_cached(symbol, start_date, end_date, timeframe)
        
        if df is None or len(df) == 0:
            return (
                _create_quality_score_display(0, 'F'),
                _create_checklist_placeholder(),
                empty_chart(f"No data for {symbol}"), empty_fig,
                html.P(f"No data available for {symbol}", style={'color': COLORS['danger']}),
                dbc.Badge("No Data", color="danger"),
                True, True, default_status,
                create_no_data_alert(symbol),
            )
        
        # Log data size for debugging
        logger.info(f"Loaded {len(df):,} bars for validation")
        
        report = validate_data_quality(df, symbol, timeframe, start_date, end_date)
        
        # Correct weight labels matching CATEGORY_WEIGHTS in data_quality.py
        weight_labels = {
            'completeness': '20%', 'distribution': '15%', 'stationarity': '20%',
            'autocorrelation': '10%', 'outliers': '15%', 'sample_size': '10%', 'ohlc_integrity': '10%',
        }
        
        checklist_items = []
        for cat_name, cat_result in report.categories.items():
            weight = weight_labels.get(cat_name, '')
            checklist_items.append(_create_checklist_item(
                cat_result.name, 
                cat_result.score, 
                cat_result.passed, 
                weight,
                status=cat_result.status,
                advice=cat_result.advice
            ))
        
        returns = df['close'].pct_change().dropna()
        
        # =====================================================================
        # PERFORMANCE OPTIMIZATION: Downsample for charts
        # Browser can't handle 50k+ points efficiently
        # =====================================================================
        MAX_CHART_POINTS = 5000  # Max points for histogram/QQ plot
        
        if len(returns) > MAX_CHART_POINTS:
            logger.info(f"Downsampling returns for charts: {len(returns):,} -> {MAX_CHART_POINTS}")
            # Random sample for histogram (preserves distribution)
            sample_indices = np.random.choice(len(returns), MAX_CHART_POINTS, replace=False)
            returns_chart = returns.iloc[np.sort(sample_indices)]
        else:
            returns_chart = returns
        
        # Distribution chart with downsampled data
        dist_fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Q-Q Plot'))
        dist_fig.add_trace(go.Histogram(
            x=returns_chart, 
            nbinsx=50, 
            name='Returns', 
            marker_color=COLORS['primary'], 
            opacity=0.7
        ), row=1, col=1)
        
        from scipy import stats
        x_range = np.linspace(returns_chart.min(), returns_chart.max(), 100)
        normal_pdf = stats.norm.pdf(x_range, returns_chart.mean(), returns_chart.std())
        normal_pdf = normal_pdf * len(returns_chart) * (returns_chart.max() - returns_chart.min()) / 50
        dist_fig.add_trace(go.Scatter(
            x=x_range, 
            y=normal_pdf, 
            mode='lines', 
            name='Normal', 
            line=dict(color=COLORS['warning'], width=2)
        ), row=1, col=1)
        
        # Q-Q Plot with LIMITED points (max 1000 for smooth rendering)
        QQ_MAX_POINTS = 1000
        if len(returns_chart) > QQ_MAX_POINTS:
            qq_sample = np.random.choice(len(returns_chart), QQ_MAX_POINTS, replace=False)
            qq_returns = returns_chart.iloc[np.sort(qq_sample)]
        else:
            qq_returns = returns_chart
        
        sorted_returns = np.sort(qq_returns.values)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
        
        dist_fig.add_trace(go.Scatter(
            x=theoretical_quantiles, 
            y=sorted_returns, 
            mode='markers', 
            name='Q-Q', 
            marker=dict(color=COLORS['primary'], size=4)
        ), row=1, col=2)
        dist_fig.add_trace(go.Scatter(
            x=[theoretical_quantiles.min(), theoretical_quantiles.max()], 
            y=[sorted_returns.min(), sorted_returns.max()], 
            mode='lines', 
            name='Reference', 
            line=dict(color=COLORS['danger'], dash='dash')
        ), row=1, col=2)
        dist_fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT['medium'], showlegend=False)
        
        # ACF chart (already efficient - only 20 lags)
        from statsmodels.tsa.stattools import acf
        acf_values = acf(returns.dropna(), nlags=20)
        acf_fig = go.Figure()
        acf_fig.add_trace(go.Bar(
            x=list(range(len(acf_values))), 
            y=acf_values, 
            name='ACF', 
            marker_color=COLORS['primary']
        ))
        conf_bound = 1.96 / np.sqrt(len(returns))
        acf_fig.add_hline(y=conf_bound, line_dash="dash", line_color=COLORS['danger'])
        acf_fig.add_hline(y=-conf_bound, line_dash="dash", line_color=COLORS['danger'])
        acf_fig.add_hline(y=0, line_color=COLORS['text_muted'])
        acf_fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT['medium'], xaxis_title="Lag", yaxis_title="ACF", showlegend=False)
        
        # Detailed report with English summary
        detailed_report = html.Div([
            # English Summary Section (NEW)
            html.Div([
                html.H5("📝 Assessment Summary", style={'color': COLORS['text'], 'marginBottom': '15px'}),
                html.Pre(
                    report.english_summary,
                    style={
                        'backgroundColor': COLORS['bg_primary'],
                        'color': COLORS['text'],
                        'padding': '15px',
                        'borderRadius': '5px',
                        'border': f"1px solid {COLORS['border']}",
                        'whiteSpace': 'pre-wrap',
                        'fontFamily': 'monospace',
                        'fontSize': '13px',
                    }
                ),
            ], className='mb-4'),
            
            html.Hr(style={'borderColor': COLORS['border']}),
            
            # Recommendations
            html.H5("💡 Recommendations", style={'color': COLORS['text'], 'marginBottom': '15px'}),
            html.Div([
                html.P(rec, style={
                    'color': COLORS['text'], 
                    'marginBottom': '8px',
                    'padding': '8px',
                    'backgroundColor': COLORS['bg_primary'],
                    'borderRadius': '4px',
                    'borderLeft': f"3px solid {COLORS['warning'] if '⚠️' in rec or '📊' in rec else (COLORS['danger'] if '🚫' in rec or '❌' in rec else COLORS['success'])}",
                }) for rec in report.recommendations
            ], className='mb-4'),
            
            html.Hr(style={'borderColor': COLORS['border']}),
            
            # Test Details
            html.H5("🔬 Test Details", style={'color': COLORS['text'], 'marginBottom': '15px'}),
            dbc.Accordion([
                dbc.AccordionItem([
                    # Category advice at top
                    html.Div([
                        html.Strong("Status: ", style={'color': COLORS['text']}),
                        html.Span(
                            cat_result.status,
                            style={
                                'color': COLORS['success'] if cat_result.status == 'PASS' else (COLORS['warning'] if cat_result.status == 'WARNING' else COLORS['danger']),
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Br(),
                        html.Strong("Advice: ", style={'color': COLORS['text']}),
                        html.Span(cat_result.advice, style={'color': COLORS['text_muted'], 'fontStyle': 'italic'}),
                    ], className='mb-3', style={'padding': '10px', 'backgroundColor': COLORS['bg_primary'], 'borderRadius': '4px'}),
                    # Individual tests
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Strong(test.name, style={'color': COLORS['text']}),
                                html.Span(
                                    " ✅ PASS" if test.passed else " ❌ FAIL", 
                                    style={'color': COLORS['success'] if test.passed else COLORS['danger']}
                                ),
                            ]),
                            html.P([
                                f"Value: {test.value:.4f}" if isinstance(test.value, float) else f"Value: {test.value}",
                                f" | Threshold: {test.threshold}",
                                f" | {test.interpretation}"
                            ], style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginBottom': '2px'}),
                            # Show test advice if failed
                            html.Small(
                                f"💡 {test.advice}" if test.advice and not test.passed else "",
                                style={'color': COLORS['warning'], 'fontStyle': 'italic'}
                            ) if test.advice and not test.passed else None,
                        ], className='mb-2')
                        for test in cat_result.tests
                    ]),
                ], title=f"{cat_result.name} - {cat_result.score:.0f}% [{cat_result.status}]")
                for cat_name, cat_result in report.categories.items()
            ], start_collapsed=True),
            
            html.Hr(style={'borderColor': COLORS['border']}),
            
            # Data Summary
            html.H5("📊 Data Summary", style={'color': COLORS['text'], 'marginBottom': '15px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([html.Strong("Symbol: ", style={'color': COLORS['text_muted']}), html.Span(report.symbol, style={'color': COLORS['text']})]),
                    html.Div([html.Strong("Timeframe: ", style={'color': COLORS['text_muted']}), html.Span(report.timeframe, style={'color': COLORS['text']})])
                ], width=4),
                dbc.Col([
                    html.Div([html.Strong("Period: ", style={'color': COLORS['text_muted']}), html.Span(f"{report.start_date} to {report.end_date}", style={'color': COLORS['text']})]),
                    html.Div([html.Strong("Total Rows: ", style={'color': COLORS['text_muted']}), html.Span(f"{report.total_rows:,}", style={'color': COLORS['text']})])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Strong("Quality Score: ", style={'color': COLORS['text_muted']}), 
                        html.Span(f"{report.quality_score:.1f}%", style={'color': GRADE_COLORS.get(report.grade), 'fontWeight': 'bold'})
                    ]),
                    html.Div([
                        html.Strong("Grade: ", style={'color': COLORS['text_muted']}), 
                        html.Span(report.grade, style={'color': GRADE_COLORS.get(report.grade), 'fontWeight': 'bold', 'fontSize': '18px'})
                    ]),
                    html.Div([
                        html.Strong("Can Proceed: ", style={'color': COLORS['text_muted']}), 
                        html.Span(
                            "YES ✅" if report.can_proceed else "NO ❌", 
                            style={'color': COLORS['success'] if report.can_proceed else COLORS['danger'], 'fontWeight': 'bold'}
                        )
                    ])
                ], width=4),
            ]),
        ])
        
        badge_color = 'success' if report.can_proceed else ('warning' if report.grade == 'C' else 'danger')
        status_badge = dbc.Badge(
            f"{'PASS' if report.can_proceed else 'FAIL'} - Grade {report.grade}", 
            color=badge_color, 
            className='ms-2', 
            style={'fontSize': '14px'}
        )
        
        status_data = {
            'validated': True, 
            'symbol': symbol, 
            'timeframe': timeframe, 
            'quality_score': report.quality_score, 
            'grade': report.grade, 
            'passed': report.passed,
            'can_proceed': report.can_proceed,
            'timestamp': report.timestamp
        }
        
        return (
            _create_quality_score_display(report.quality_score, report.grade, report.can_proceed, report.status_label),
            html.Div(checklist_items),
            dist_fig, acf_fig, detailed_report, status_badge,
            not report.can_proceed, False, status_data, None,
        )
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return (
            _create_quality_score_display(0, 'F'),
            _create_checklist_placeholder(),
            empty_chart(f"Error: {str(e)[:30]}"), empty_chart("Error"),
            html.P(f"Error: {str(e)}", style={'color': COLORS['danger']}),
            dbc.Badge("Error", color="danger"),
            True, True, default_status,
            create_error_alert("Validation Error", str(e)),
        )


@callback(
    Output('quality-redirect', 'pathname'),
    Input('quality-proceed-btn', 'n_clicks'),
    State('quality-status-store', 'data'),
    prevent_initial_call=True,
)
def proceed_to_data_studio(n_clicks, status_data):
    """Navigate to Data Studio if validation passed."""
    if status_data and status_data.get('passed'):
        return '/data-studio'
    return None
