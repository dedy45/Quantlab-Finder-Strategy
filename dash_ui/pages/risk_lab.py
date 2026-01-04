"""
Risk Lab Page - Comprehensive Risk Analysis.

Provides:
- VaR calculations
- Drawdown analysis
- Rolling volatility
- Correlation heatmap
- Monte Carlo simulation
- Risk alerts
- Export to PDF/CSV

CRITICAL: All data MUST come from ArcticDB (REAL DATA).
NO synthetic/dummy data allowed.
"""
import logging
import io
from datetime import datetime
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

from ..theme import COLORS, PLOTLY_LAYOUT, CHART_HEIGHT
from ..error_handler import create_error_alert, create_no_data_alert

logger = logging.getLogger(__name__)

# =============================================================================
# LAYOUT
# =============================================================================

layout = dbc.Container([
    # Initial load trigger - fires once when page loads
    dcc.Interval(
        id='risk-init-interval',
        interval=500,
        n_intervals=0,
        max_intervals=1,
    ),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("Risk Lab", style={'color': COLORS['text_bright']}),
            html.P(
                "Comprehensive risk analysis and monitoring",
                style={'color': COLORS['text_muted']}
            ),
        ]),
    ], className='mb-4'),
    
    # Controls row
    dbc.Row([
        dbc.Col([
            dbc.Label("Symbol", style={'color': COLORS['text']}),
            dcc.Dropdown(
                id='risk-symbol-dropdown',
                placeholder='Select symbol...',
                style={'backgroundColor': COLORS['bg_tertiary']},
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("Timeframe", style={'color': COLORS['text']}),
            dcc.Dropdown(
                id='risk-timeframe-dropdown',
                options=[
                    {'label': '1 Hour', 'value': '1H'},
                    {'label': '4 Hours', 'value': '4H'},
                    {'label': 'Daily', 'value': '1D'},
                ],
                value='1D',
                style={'backgroundColor': COLORS['bg_tertiary']},
            ),
        ], width=2),
        dbc.Col([
            dbc.Label("VaR Confidence", style={'color': COLORS['text']}),
            dcc.Dropdown(
                id='risk-var-confidence',
                options=[
                    {'label': '95%', 'value': 0.95},
                    {'label': '99%', 'value': 0.99},
                ],
                value=0.95,
                style={'backgroundColor': COLORS['bg_tertiary']},
            ),
        ], width=2),
        dbc.Col([
            dbc.Button(
                "Calculate Risk",
                id='risk-calculate-btn',
                color='primary',
                className='mt-4',
            ),
        ], width=2),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="bi bi-file-earmark-pdf me-1"), "PDF"],
                    id='risk-export-pdf-btn',
                    color='secondary',
                    size='sm',
                    disabled=True,
                    className='mt-4',
                ),
                dbc.Button(
                    [html.I(className="bi bi-file-earmark-spreadsheet me-1"), "CSV"],
                    id='risk-export-csv-btn',
                    color='secondary',
                    size='sm',
                    disabled=True,
                    className='mt-4',
                ),
            ]),
        ], width=3),
    ], className='mb-4'),
    
    # Alert container
    html.Div(id='risk-alert-container'),
    
    # KPI Cards Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("VaR (95%)", className='text-muted'),
                    html.H3(id='risk-var-value', children="-", className='text-danger'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Max Drawdown", className='text-muted'),
                    html.H3(id='risk-maxdd-value', children="-", className='text-warning'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Volatility (Ann.)", className='text-muted'),
                    html.H3(id='risk-vol-value', children="-", className='text-info'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Kelly Fraction", className='text-muted'),
                    html.H3(id='risk-kelly-value', children="-", className='text-success'),
                ], className='text-center'),
            ]),
        ], width=3),
    ], className='mb-4'),
    
    # KPI Cards Row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Sharpe Ratio", className='text-muted'),
                    html.H3(id='risk-sharpe-value', children="-", className='text-info'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Sortino Ratio", className='text-muted'),
                    html.H3(id='risk-sortino-value', children="-", className='text-info'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Calmar Ratio", className='text-muted'),
                    html.H3(id='risk-calmar-value', children="-", className='text-info'),
                ], className='text-center'),
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("PSR", className='text-muted'),
                    html.H3(id='risk-psr-value', children="-", className='text-success'),
                ], className='text-center'),
            ]),
        ], width=3),
    ], className='mb-4'),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Drawdown Chart"),
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(
                            id='risk-drawdown-chart',
                            config={'displayModeBar': False},
                            style={'height': CHART_HEIGHT['medium']},
                        ),
                    ]),
                ]),
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rolling Volatility"),
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(
                            id='risk-volatility-chart',
                            config={'displayModeBar': False},
                            style={'height': CHART_HEIGHT['medium']},
                        ),
                    ]),
                ]),
            ]),
        ], width=6),
    ], className='mb-4'),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Monte Carlo Simulation"),
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(
                            id='risk-montecarlo-chart',
                            config={'displayModeBar': False},
                            style={'height': CHART_HEIGHT['medium']},
                        ),
                    ]),
                ]),
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Alerts"),
                dbc.CardBody([
                    html.Div(id='risk-alerts-container', children=[
                        html.P(
                            "Click 'Calculate Risk' to analyze",
                            className='text-muted text-center'
                        ),
                    ]),
                ]),
            ]),
        ], width=6),
    ]),
    
    # Hidden stores for export
    dcc.Store(id='risk-metrics-store', data=None),
    dcc.Store(id='risk-symbol-store', data=None),
    
    # Download components
    dcc.Download(id='risk-download-pdf'),
    dcc.Download(id='risk-download-csv'),
    
], fluid=True, style={'backgroundColor': COLORS['bg_primary'], 'minHeight': '100vh'})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output('risk-symbol-dropdown', 'options'),
    Output('risk-symbol-dropdown', 'value'),
    Input('risk-init-interval', 'n_intervals'),  # Trigger on page load
)
def load_symbols(n_intervals):
    """Load available symbols from ArcticDB."""
    try:
        from ..data_loader import get_available_symbols
        symbols = get_available_symbols()
        options = [{'label': s, 'value': s} for s in symbols]
        default = 'XAUUSD' if 'XAUUSD' in symbols else (symbols[0] if symbols else None)
        logger.info(f"Risk Lab: Loaded {len(symbols)} symbols")
        return options, default
    except Exception as e:
        logger.error(f"Failed to load symbols: {e}")
        return [], None


# Placeholder callback - full implementation in Phase 7
@callback(
    [
        Output('risk-var-value', 'children'),
        Output('risk-maxdd-value', 'children'),
        Output('risk-vol-value', 'children'),
        Output('risk-kelly-value', 'children'),
        Output('risk-sharpe-value', 'children'),
        Output('risk-sortino-value', 'children'),
        Output('risk-calmar-value', 'children'),
        Output('risk-psr-value', 'children'),
        Output('risk-drawdown-chart', 'figure'),
        Output('risk-volatility-chart', 'figure'),
        Output('risk-montecarlo-chart', 'figure'),
        Output('risk-alerts-container', 'children'),
        Output('risk-alert-container', 'children'),
        Output('risk-metrics-store', 'data'),
        Output('risk-symbol-store', 'data'),
        Output('risk-export-pdf-btn', 'disabled'),
        Output('risk-export-csv-btn', 'disabled'),
    ],
    Input('risk-calculate-btn', 'n_clicks'),
    [
        State('risk-symbol-dropdown', 'value'),
        State('risk-timeframe-dropdown', 'value'),
        State('risk-var-confidence', 'value'),
    ],
    prevent_initial_call=True,
)
def calculate_risk(n_clicks, symbol, timeframe, var_confidence):
    """
    Calculate comprehensive risk metrics.
    
    CRITICAL: Uses REAL data from ArcticDB only.
    """
    import plotly.graph_objects as go
    
    # Empty chart template
    empty_fig = go.Figure()
    empty_fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT['medium'])
    empty_fig.add_annotation(
        text="Select symbol and click Calculate",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=COLORS['text_muted'])
    )
    
    # Validate inputs
    if not symbol:
        return (
            "-", "-", "-", "-", "-", "-", "-", "-",
            empty_fig, empty_fig, empty_fig,
            html.P("Please select a symbol", className='text-warning'),
            create_error_alert("Validation Error", "Please select a symbol"),
            None, None, True, True,  # No metrics, buttons disabled
        )
    
    try:
        # Load REAL data from ArcticDB
        from ..data_loader import load_ohlcv_cached, get_default_date_range
        
        start, end = get_default_date_range()
        df = load_ohlcv_cached(symbol, start, end, timeframe)
        
        if df is None or len(df) == 0:
            return (
                "-", "-", "-", "-", "-", "-", "-", "-",
                empty_fig, empty_fig, empty_fig,
                html.P(f"No data for {symbol}", className='text-warning'),
                create_no_data_alert(symbol, timeframe),
                None, None, True, True,
            )
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 30:
            return (
                "-", "-", "-", "-", "-", "-", "-", "-",
                empty_fig, empty_fig, empty_fig,
                html.P("Insufficient data for analysis", className='text-warning'),
                create_error_alert("Insufficient Data", "Need at least 30 data points"),
                None, None, True, True,
            )
        
        # Calculate metrics using core modules
        import numpy as np
        
        # Basic metrics
        ann_factor = 252 if timeframe == '1D' else (252 * 24 if timeframe == '1H' else 252 * 6)
        
        # VaR (Historical)
        var_value = np.percentile(returns, (1 - var_confidence) * 100)
        
        # Volatility
        volatility = returns.std() * np.sqrt(ann_factor)
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Sharpe (assuming 0 risk-free rate)
        sharpe = (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0
        
        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 0
        sortino = (returns.mean() * ann_factor) / downside_std if downside_std > 0 else 0
        
        # Calmar
        ann_return = returns.mean() * ann_factor
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        # Kelly (simplified)
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        kelly = (win_rate - (1 - win_rate) / (avg_win / avg_loss)) if avg_loss > 0 and avg_win > 0 else 0
        kelly = max(0, min(kelly, 1))  # Clamp to [0, 1]
        
        # PSR (simplified)
        try:
            from core.validation_engine import calculate_psr
            psr = calculate_psr(returns, benchmark_sr=0.0)
        except ImportError:
            # Fallback calculation
            n = len(returns)
            se_sr = np.sqrt((1 + 0.5 * sharpe**2) / (n - 1))
            from scipy import stats
            psr = stats.norm.cdf(sharpe / se_sr) if se_sr > 0 else 0.5
        
        # Create drawdown chart
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            fill='tozeroy',
            fillcolor='rgba(204, 0, 0, 0.3)',
            line=dict(color=COLORS['danger'], width=1),
            name='Drawdown'
        ))
        dd_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=CHART_HEIGHT['medium'],
            yaxis_title='Drawdown (%)',
            showlegend=False,
        )
        
        # Create volatility chart
        rolling_vol = returns.rolling(20).std() * np.sqrt(ann_factor) * 100
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            line=dict(color=COLORS['primary'], width=1),
            name='20-day Rolling Vol'
        ))
        vol_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=CHART_HEIGHT['medium'],
            yaxis_title='Volatility (%)',
            showlegend=False,
        )
        
        # Monte Carlo simulation (based on REAL returns distribution)
        mc_fig = go.Figure()
        n_simulations = 100
        n_days = 252
        
        # Use REAL mean and std from historical returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        for i in range(n_simulations):
            sim_returns = np.random.normal(mean_return, std_return, n_days)
            sim_equity = 100 * (1 + sim_returns).cumprod()
            mc_fig.add_trace(go.Scatter(
                x=list(range(n_days)),
                y=sim_equity,
                mode='lines',
                line=dict(width=0.5, color=COLORS['primary']),
                opacity=0.3,
                showlegend=False,
            ))
        
        mc_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=CHART_HEIGHT['medium'],
            xaxis_title='Days',
            yaxis_title='Equity',
        )
        
        # Risk alerts
        from config import get_config
        cfg = get_config()
        
        alerts = []
        if abs(max_dd) > cfg.risk.max_portfolio_drawdown:
            alerts.append(dbc.Alert(
                f"⚠️ Max DD ({max_dd:.1%}) exceeds limit ({cfg.risk.max_portfolio_drawdown:.0%})",
                color='danger',
                className='mb-2'
            ))
        if volatility > 0.30:
            alerts.append(dbc.Alert(
                f"⚠️ High volatility: {volatility:.1%} annualized",
                color='warning',
                className='mb-2'
            ))
        if psr < 0.95:
            alerts.append(dbc.Alert(
                f"⚠️ PSR ({psr:.1%}) below 95% threshold",
                color='warning',
                className='mb-2'
            ))
        
        if not alerts:
            alerts.append(dbc.Alert(
                "✅ All risk metrics within acceptable limits",
                color='success',
                className='mb-2'
            ))
        
        # Store metrics for export
        metrics_data = {
            'var': var_value,
            'max_dd': max_dd,
            'volatility': volatility,
            'kelly': kelly,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'psr': psr,
            'var_confidence': var_confidence,
            'timeframe': timeframe,
            'data_points': len(returns),
            'calculated_at': datetime.now().isoformat(),
        }
        
        return (
            f"{var_value:.2%}",
            f"{max_dd:.2%}",
            f"{volatility:.1%}",
            f"{kelly:.1%}",
            f"{sharpe:.2f}",
            f"{sortino:.2f}",
            f"{calmar:.2f}",
            f"{psr:.1%}",
            dd_fig,
            vol_fig,
            mc_fig,
            alerts,
            None,  # No error alert
            metrics_data,  # Store metrics
            symbol,  # Store symbol
            False,  # Enable PDF button
            False,  # Enable CSV button
        )
        
    except Exception as e:
        logger.error(f"Risk calculation failed: {e}")
        return (
            "-", "-", "-", "-", "-", "-", "-", "-",
            empty_fig, empty_fig, empty_fig,
            html.P(f"Error: {str(e)}", className='text-danger'),
            create_error_alert("Calculation Error", str(e)),
            None, None, True, True,
        )



# =============================================================================
# EXPORT CALLBACKS
# =============================================================================

@callback(
    Output('risk-download-csv', 'data'),
    Input('risk-export-csv-btn', 'n_clicks'),
    [
        State('risk-metrics-store', 'data'),
        State('risk-symbol-store', 'data'),
    ],
    prevent_initial_call=True,
)
def export_csv(n_clicks, metrics, symbol):
    """Export risk metrics as CSV."""
    if not metrics or not symbol:
        return None
    
    try:
        # Create DataFrame with metrics
        data = {
            'Metric': [
                'Symbol',
                'Timeframe',
                f'VaR ({metrics["var_confidence"]*100:.0f}%)',
                'Max Drawdown',
                'Volatility (Ann.)',
                'Kelly Fraction',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Calmar Ratio',
                'PSR',
                'Data Points',
                'Calculated At',
            ],
            'Value': [
                symbol,
                metrics['timeframe'],
                f"{metrics['var']*100:.2f}%",
                f"{metrics['max_dd']*100:.2f}%",
                f"{metrics['volatility']*100:.1f}%",
                f"{metrics['kelly']*100:.1f}%",
                f"{metrics['sharpe']:.4f}",
                f"{metrics['sortino']:.4f}",
                f"{metrics['calmar']:.4f}",
                f"{metrics['psr']*100:.2f}%",
                str(metrics['data_points']),
                metrics['calculated_at'],
            ],
        }
        
        df = pd.DataFrame(data)
        
        # Generate filename
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"risk_report_{symbol}_{date_str}.csv"
        
        return dcc.send_data_frame(df.to_csv, filename, index=False)
        
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return None


@callback(
    Output('risk-download-pdf', 'data'),
    Input('risk-export-pdf-btn', 'n_clicks'),
    [
        State('risk-metrics-store', 'data'),
        State('risk-symbol-store', 'data'),
    ],
    prevent_initial_call=True,
)
def export_pdf(n_clicks, metrics, symbol):
    """
    Export risk metrics as PDF-like text report.
    
    Note: For full PDF generation, consider using reportlab or weasyprint.
    This implementation generates a formatted text report.
    """
    if not metrics or not symbol:
        return None
    
    try:
        # Generate formatted text report
        report_lines = [
            "=" * 60,
            "QUANTLAB RISK ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Symbol: {symbol}",
            f"Timeframe: {metrics['timeframe']}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Points: {metrics['data_points']}",
            "",
            "-" * 60,
            "RISK METRICS",
            "-" * 60,
            "",
            f"Value at Risk ({metrics['var_confidence']*100:.0f}%): {metrics['var']*100:.2f}%",
            f"Maximum Drawdown: {metrics['max_dd']*100:.2f}%",
            f"Annualized Volatility: {metrics['volatility']*100:.1f}%",
            f"Kelly Fraction: {metrics['kelly']*100:.1f}%",
            "",
            "-" * 60,
            "PERFORMANCE METRICS",
            "-" * 60,
            "",
            f"Sharpe Ratio: {metrics['sharpe']:.4f}",
            f"Sortino Ratio: {metrics['sortino']:.4f}",
            f"Calmar Ratio: {metrics['calmar']:.4f}",
            f"Probabilistic Sharpe Ratio (PSR): {metrics['psr']*100:.2f}%",
            "",
            "-" * 60,
            "INTERPRETATION",
            "-" * 60,
            "",
        ]
        
        # Add interpretations
        if metrics['psr'] >= 0.95:
            report_lines.append("✓ PSR >= 95%: Strategy is statistically significant")
        else:
            report_lines.append("⚠ PSR < 95%: Strategy may not be statistically significant")
        
        if abs(metrics['max_dd']) <= 0.20:
            report_lines.append("✓ Max DD <= 20%: Drawdown within acceptable limits")
        else:
            report_lines.append("⚠ Max DD > 20%: Drawdown exceeds recommended limits")
        
        if metrics['sharpe'] >= 1.0:
            report_lines.append("✓ Sharpe >= 1.0: Good risk-adjusted returns")
        elif metrics['sharpe'] >= 0.5:
            report_lines.append("○ Sharpe 0.5-1.0: Moderate risk-adjusted returns")
        else:
            report_lines.append("⚠ Sharpe < 0.5: Poor risk-adjusted returns")
        
        report_lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])
        
        report_content = "\n".join(report_lines)
        
        # Generate filename
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"risk_report_{symbol}_{date_str}.txt"
        
        return dict(content=report_content, filename=filename)
        
    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        return None
