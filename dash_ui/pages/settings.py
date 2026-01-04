"""
Settings Page - Configuration Management.

URL: /settings
Manage backtest, trading, and risk settings.

CRITICAL: ALL settings loaded from config/ module.
NO hardcoded default values.
"""
import logging
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from ..theme import COLORS
from ..error_handler import create_error_alert, create_success_alert
from ..utils.validators import validate_numeric

logger = logging.getLogger(__name__)

# =============================================================================
# LAYOUT
# =============================================================================

layout = dbc.Container([
    # Initial load trigger - fires once when page loads
    dcc.Interval(
        id='settings-init-interval',
        interval=500,
        n_intervals=0,
        max_intervals=1,
    ),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("Settings", style={'color': COLORS['text_bright']}),
                dbc.Badge("v0.7.2", color="primary", className="ms-2"),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.P(
                "Configuration loaded from config/default.yaml and config/user.yaml",
                style={'color': COLORS['text_muted']}
            ),
        ]),
    ], className='mb-4'),
    
    # Alert container
    html.Div(id='settings-alert-container'),
    
    # Status message
    html.Div(id='settings-status', className='mb-3'),
    
    # Backtest Settings
    dbc.Card([
        dbc.CardHeader([
            html.Span("📊", className="me-2"),
            "Backtest Settings",
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Initial Capital ($)", className='text-muted'),
                    dbc.Input(
                        id='settings-bt-capital',
                        type='number',
                        min=1000,
                        step=1000,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Commission (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-bt-commission',
                        type='number',
                        min=0,
                        max=5,
                        step=0.01,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Slippage (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-bt-slippage',
                        type='number',
                        min=0,
                        max=5,
                        step=0.01,
                    ),
                ], width=4),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Position (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-bt-max-pos',
                        type='number',
                        min=1,
                        max=100,
                        step=1,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Max Drawdown (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-bt-max-dd',
                        type='number',
                        min=1,
                        max=100,
                        step=1,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Target Volatility (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-bt-target-vol',
                        type='number',
                        min=1,
                        max=100,
                        step=1,
                    ),
                ], width=4),
            ]),
        ]),
    ], className='mb-4'),
    
    # Risk Settings
    dbc.Card([
        dbc.CardHeader([
            html.Span("🛡️", className="me-2"),
            "Risk Settings",
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Stop Loss (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-risk-sl',
                        type='number',
                        min=0.1,
                        max=50,
                        step=0.1,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Take Profit (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-risk-tp',
                        type='number',
                        min=0.1,
                        max=100,
                        step=0.1,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Kelly Fraction (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-risk-kelly',
                        type='number',
                        min=0,
                        max=100,
                        step=5,
                    ),
                ], width=4),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Portfolio Drawdown (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-risk-max-dd',
                        type='number',
                        min=1,
                        max=100,
                        step=1,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Max Correlation", className='text-muted'),
                    dbc.Input(
                        id='settings-risk-max-corr',
                        type='number',
                        min=0,
                        max=1,
                        step=0.05,
                    ),
                ], width=4),
            ]),
        ]),
    ], className='mb-4'),
    
    # Validation Settings
    dbc.Card([
        dbc.CardHeader([
            html.Span("✅", className="me-2"),
            "Validation Settings",
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Min PSR", className='text-muted'),
                    dbc.Input(
                        id='settings-val-min-psr',
                        type='number',
                        min=0,
                        max=1,
                        step=0.01,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Min DSR", className='text-muted'),
                    dbc.Input(
                        id='settings-val-min-dsr',
                        type='number',
                        min=0,
                        max=5,
                        step=0.1,
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Max Sharpe Degradation (%)", className='text-muted'),
                    dbc.Input(
                        id='settings-val-max-deg',
                        type='number',
                        min=0,
                        max=100,
                        step=5,
                    ),
                ], width=4),
            ]),
        ]),
    ], className='mb-4'),
    
    # Action Buttons
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Save Settings",
                id='settings-save-btn',
                color='primary',
                className='me-2',
            ),
            dbc.Button(
                "Reset to Defaults",
                id='settings-reset-btn',
                color='secondary',
                outline=True,
            ),
        ]),
    ]),
    
], fluid=True, style={'backgroundColor': COLORS['bg_primary'], 'minHeight': '100vh'})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    [
        Output('settings-bt-capital', 'value'),
        Output('settings-bt-commission', 'value'),
        Output('settings-bt-slippage', 'value'),
        Output('settings-bt-max-pos', 'value'),
        Output('settings-bt-max-dd', 'value'),
        Output('settings-bt-target-vol', 'value'),
        Output('settings-risk-sl', 'value'),
        Output('settings-risk-tp', 'value'),
        Output('settings-risk-kelly', 'value'),
        Output('settings-risk-max-dd', 'value'),
        Output('settings-risk-max-corr', 'value'),
        Output('settings-val-min-psr', 'value'),
        Output('settings-val-min-dsr', 'value'),
        Output('settings-val-max-deg', 'value'),
    ],
    Input('settings-init-interval', 'n_intervals'),  # Trigger on page load
)
def load_settings(n_intervals):
    """
    Load settings from config module.
    
    CRITICAL: ALL values from config/, NO hardcoded defaults.
    """
    try:
        from config import get_config
        cfg = get_config()
        
        return (
            # Backtest
            cfg.backtest.initial_capital,
            cfg.backtest.commission_pct * 100,
            cfg.backtest.slippage_pct * 100,
            cfg.backtest.max_position_pct * 100,
            cfg.backtest.max_drawdown_pct * 100,
            cfg.backtest.target_volatility_pct * 100,
            # Risk
            cfg.risk.stop_loss_pct * 100,
            cfg.risk.take_profit_pct * 100,
            cfg.risk.kelly_fraction * 100,
            cfg.risk.max_portfolio_drawdown * 100,
            cfg.risk.max_correlation,
            # Validation
            cfg.validation.min_psr,
            cfg.validation.min_dsr,
            cfg.validation.max_sharpe_degradation * 100,
        )
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        # Return None for all fields - will show empty
        return tuple([None] * 14)


@callback(
    Output('settings-status', 'children'),
    Output('settings-alert-container', 'children'),
    Input('settings-save-btn', 'n_clicks'),
    [
        State('settings-bt-capital', 'value'),
        State('settings-bt-commission', 'value'),
        State('settings-bt-slippage', 'value'),
        State('settings-bt-max-pos', 'value'),
        State('settings-bt-max-dd', 'value'),
        State('settings-bt-target-vol', 'value'),
        State('settings-risk-sl', 'value'),
        State('settings-risk-tp', 'value'),
        State('settings-risk-kelly', 'value'),
        State('settings-risk-max-dd', 'value'),
        State('settings-risk-max-corr', 'value'),
        State('settings-val-min-psr', 'value'),
        State('settings-val-min-dsr', 'value'),
        State('settings-val-max-deg', 'value'),
    ],
    prevent_initial_call=True,
)
def save_settings(
    n_clicks,
    bt_capital, bt_commission, bt_slippage, bt_max_pos, bt_max_dd, bt_target_vol,
    risk_sl, risk_tp, risk_kelly, risk_max_dd, risk_max_corr,
    val_min_psr, val_min_dsr, val_max_deg,
):
    """
    Save settings to config/user.yaml.
    
    CRITICAL: Saves via config.save_config() to user.yaml.
    """
    try:
        from config import get_config, save_config
        
        # Validate inputs
        is_valid, error = validate_numeric(bt_capital, 'Initial Capital', min_value=1000)
        if not is_valid:
            return None, create_error_alert("Validation Error", error)
        
        cfg = get_config()
        
        # Update backtest settings
        cfg.backtest.initial_capital = float(bt_capital)
        cfg.backtest.commission_pct = float(bt_commission) / 100
        cfg.backtest.slippage_pct = float(bt_slippage) / 100
        cfg.backtest.max_position_pct = float(bt_max_pos) / 100
        cfg.backtest.max_drawdown_pct = float(bt_max_dd) / 100
        cfg.backtest.target_volatility_pct = float(bt_target_vol) / 100
        
        # Update risk settings
        cfg.risk.stop_loss_pct = float(risk_sl) / 100
        cfg.risk.take_profit_pct = float(risk_tp) / 100
        cfg.risk.kelly_fraction = float(risk_kelly) / 100
        cfg.risk.max_portfolio_drawdown = float(risk_max_dd) / 100
        cfg.risk.max_correlation = float(risk_max_corr)
        
        # Update validation settings
        cfg.validation.min_psr = float(val_min_psr)
        cfg.validation.min_dsr = float(val_min_dsr)
        cfg.validation.max_sharpe_degradation = float(val_max_deg) / 100
        
        # Save to user.yaml
        save_config(cfg)
        
        logger.info("Settings saved successfully")
        return (
            dbc.Alert("Settings saved successfully!", color='success', duration=5000),
            None,
        )
        
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return None, create_error_alert("Save Error", str(e))


@callback(
    Output('settings-status', 'children', allow_duplicate=True),
    Output('settings-alert-container', 'children', allow_duplicate=True),
    Input('settings-reset-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_settings(n_clicks):
    """
    Reset settings to defaults by removing user.yaml.
    """
    try:
        import os
        from pathlib import Path
        
        # Find user.yaml relative to QuantLab root
        user_config = Path(__file__).parent.parent.parent / 'config' / 'user.yaml'
        
        if user_config.exists():
            os.remove(user_config)
            logger.info("User config removed, reset to defaults")
            return (
                dbc.Alert(
                    "Settings reset to defaults! Refresh the page to see changes.",
                    color='info',
                    duration=5000,
                ),
                None,
            )
        else:
            return (
                dbc.Alert("Already using default settings.", color='info', duration=3000),
                None,
            )
            
    except Exception as e:
        logger.error(f"Failed to reset settings: {e}")
        return None, create_error_alert("Reset Error", str(e))
