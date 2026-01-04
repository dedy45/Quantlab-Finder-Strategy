"""
QuantLab Dash UI - Router and Main Layout.

Handles URL routing and main app layout with sidebar navigation.
Features:
- Dark/Light theme toggle
- Responsive mobile layout
- 3D shadow effects

CRITICAL FIX: All page modules are imported at startup to ensure
callbacks are registered with the Dash app instance.

Usage:
    from dash_ui.index import create_layout, register_callbacks
    
    app.layout = create_layout()
    register_callbacks(app)
"""
import logging
from dash import html, dcc, callback, Input, Output, clientside_callback, ClientsideFunction
import dash_bootstrap_components as dbc

from .theme import COLORS, SIDEBAR_STYLE, CONTENT_STYLE
from .components.sidebar import create_sidebar

# CRITICAL: Import all pages at startup to register callbacks
# Without this, callbacks defined in page modules won't work!
from .pages import data_quality_gate
from .pages import data_studio
from .pages import backtest_arena
from .pages import risk_lab
from .pages import settings

logger = logging.getLogger(__name__)


def create_layout():
    """
    Create the main app layout with sidebar and content area.
    
    Features:
    - Responsive layout (mobile/tablet/desktop)
    - Theme toggle (dark/light)
    - 3D shadow effects
    
    Returns
    -------
    html.Div
        Main layout container
    """
    return html.Div([
        # URL tracking
        dcc.Location(id='url', refresh=False),
        
        # Theme store (persists across sessions)
        dcc.Store(id='theme-store', storage_type='local', data='dark'),
        
        # Session store for metadata only (NOT large data!)
        dcc.Store(id='session-store', storage_type='session', data={
            'current_page': '/',
            'last_symbol': None,
            'last_timeframe': None,
        }),
        
        # Data quality gate status - shared across pages
        dcc.Store(id='global-quality-status', storage_type='session', data={
            'validated': False,
            'symbol': None,
            'timeframe': None,
            'quality_score': 0,
            'grade': 'F',
            'passed': False,
        }),
        
        # Toast container for notifications
        html.Div(id='toast-container', style={
            'position': 'fixed',
            'top': '70px',
            'right': '20px',
            'zIndex': 9999,
        }),
        
        # Main layout wrapper
        html.Div([
            # Desktop Sidebar (hidden on mobile via CSS)
            html.Div(
                create_sidebar(),
                className='sidebar-desktop',
            ),
            
            # Mobile Header with dropdown menu (hidden on desktop via CSS)
            _create_mobile_header(),
            
            # Main content area
            html.Div(
                html.Div(id='page-content'),
                className='content-container',
            ),
        ], id='main-wrapper', className='main-wrapper', **{'data-theme': 'dark'}),
        
    ], id='app-container')


def _create_mobile_header():
    """Create mobile header with dropdown menu."""
    return html.Div([
        # Fixed header bar
        html.Div([
            # Brand
            html.Div([
                html.Span("⚡", style={'fontSize': '1.2rem', 'marginRight': '6px'}),
                html.Span("QuantLab", style={
                    'color': 'var(--primary)',
                    'fontWeight': 'bold',
                    'fontSize': '1rem',
                }),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            # Right side: Theme toggle + Menu button
            html.Div([
                html.Button(
                    id='theme-toggle-mobile',
                    children='🌙',
                    style={
                        'background': 'none',
                        'border': 'none',
                        'fontSize': '1.2rem',
                        'cursor': 'pointer',
                        'padding': '8px',
                    }
                ),
                html.Button(
                    id='mobile-menu-btn',
                    children='☰',
                    style={
                        'background': 'none',
                        'border': 'none',
                        'fontSize': '1.4rem',
                        'cursor': 'pointer',
                        'padding': '8px',
                        'color': 'var(--text-primary)',
                    }
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '5px'}),
        ], className='mobile-header-bar'),
        
        # Dropdown menu (hidden by default, shown when menu button clicked)
        html.Div([
            dcc.Link([
                html.Span("🔬", style={'marginRight': '10px'}),
                "Data Quality"
            ], href="/", className='mobile-menu-item'),
            dcc.Link([
                html.Span("💾", style={'marginRight': '10px'}),
                "Data Studio"
            ], href="/data-studio", className='mobile-menu-item'),
            dcc.Link([
                html.Span("🎯", style={'marginRight': '10px'}),
                "Backtest Arena"
            ], href="/backtest", className='mobile-menu-item'),
            dcc.Link([
                html.Span("⚠️", style={'marginRight': '10px'}),
                "Risk Lab"
            ], href="/risk-lab", className='mobile-menu-item'),
            dcc.Link([
                html.Span("⚙️", style={'marginRight': '10px'}),
                "Settings"
            ], href="/settings", className='mobile-menu-item'),
        ], id='mobile-dropdown-menu', className='mobile-dropdown-menu'),
        
    ], className='mobile-header-container')


def register_callbacks(app):
    """
    Register routing callbacks and theme toggle.
    
    Parameters
    ----------
    app : dash.Dash
        Dash application instance
    """
    
    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname')
    )
    def display_page(pathname):
        """Route to appropriate page based on URL."""
        logger.debug(f"Routing to: {pathname}")
        
        try:
            # Data Quality Gate is the first page (foundation)
            if pathname == '/' or pathname == '/data-quality':
                return data_quality_gate.layout
            
            elif pathname == '/data-studio':
                return data_studio.layout
            
            elif pathname == '/backtest':
                return backtest_arena.layout()
            
            elif pathname == '/risk-lab':
                return risk_lab.layout
            
            elif pathname == '/settings':
                return settings.layout
            
            # Legacy route - redirect to data quality
            elif pathname == '/dashboard':
                return data_quality_gate.layout
            
            else:
                # 404 page
                return create_404_page(pathname)
                
        except Exception as e:
            logger.error(f"Error loading page {pathname}: {e}")
            return create_error_page(str(e))
    
    # Clientside callback for theme toggle (no server roundtrip)
    app.clientside_callback(
        """
        function(n_clicks, n_clicks_mobile, currentTheme) {
            // Determine which button was clicked
            const triggered = dash_clientside.callback_context.triggered;
            if (!triggered || triggered.length === 0 || !triggered[0].value) {
                // Initial load - apply stored theme
                const theme = currentTheme || 'dark';
                document.documentElement.setAttribute('data-theme', theme);
                const wrapper = document.getElementById('main-wrapper');
                if (wrapper) wrapper.setAttribute('data-theme', theme);
                return [theme, theme === 'dark' ? '🌙' : '☀️', theme === 'dark' ? '🌙' : '☀️'];
            }
            
            // Toggle theme
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            const wrapper = document.getElementById('main-wrapper');
            if (wrapper) wrapper.setAttribute('data-theme', newTheme);
            
            const icon = newTheme === 'dark' ? '🌙' : '☀️';
            return [newTheme, icon, icon];
        }
        """,
        [
            Output('theme-store', 'data'),
            Output('theme-toggle-btn', 'children'),
            Output('theme-toggle-mobile', 'children'),
        ],
        [
            Input('theme-toggle-btn', 'n_clicks'),
            Input('theme-toggle-mobile', 'n_clicks'),
        ],
        [
            Input('theme-store', 'data'),
        ],
        prevent_initial_call=False,
    )
    
    # Clientside callback for mobile menu toggle
    app.clientside_callback(
        """
        function(n_clicks) {
            const menu = document.getElementById('mobile-dropdown-menu');
            if (menu) {
                if (menu.classList.contains('show')) {
                    menu.classList.remove('show');
                } else {
                    menu.classList.add('show');
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('mobile-menu-btn', 'n_clicks'),
        Input('mobile-menu-btn', 'n_clicks'),
        prevent_initial_call=True,
    )
    
    # Close mobile menu when navigating
    app.clientside_callback(
        """
        function(pathname) {
            const menu = document.getElementById('mobile-dropdown-menu');
            if (menu) {
                menu.classList.remove('show');
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('url', 'pathname', allow_duplicate=True),
        Input('url', 'pathname'),
        prevent_initial_call=True,
    )


def create_404_page(pathname: str):
    """Create 404 not found page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("404", className='display-1 text-muted'),
                html.H3("Page Not Found"),
                html.P(f"The page '{pathname}' does not exist."),
                dbc.Button(
                    "Go to Dashboard",
                    href="/",
                    color="primary",
                    className="mt-3"
                ),
            ], className='text-center', style={'marginTop': '100px'}),
        ]),
    ])


def create_error_page(error_message: str):
    """Create error page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Error", className='text-danger'),
                html.P(error_message),
                dbc.Button(
                    "Go to Dashboard",
                    href="/",
                    color="primary",
                    className="mt-3"
                ),
            ], className='text-center', style={'marginTop': '100px'}),
        ]),
    ])
