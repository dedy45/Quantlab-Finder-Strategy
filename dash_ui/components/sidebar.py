"""
Sidebar Navigation Component.

Provides consistent navigation across all pages.
Features:
- Responsive design (collapses on mobile)
- Theme toggle button
- 3D shadow effects
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


# Navigation items with icons
# Data Quality Gate is FIRST - must pass before other features
NAV_ITEMS = [
    {'name': 'Data Quality', 'path': '/', 'icon': '🔬', 'description': 'Validate data first'},
    {'name': 'Data Studio', 'path': '/data-studio', 'icon': '💾', 'description': 'Explore ArcticDB'},
    {'name': 'Backtest Arena', 'path': '/backtest', 'icon': '🎯', 'description': 'Run backtests'},
    {'name': 'Risk Lab', 'path': '/risk-lab', 'icon': '⚠️', 'description': 'Risk analysis'},
    {'name': 'Settings', 'path': '/settings', 'icon': '⚙️', 'description': 'Configuration'},
]


def create_sidebar():
    """
    Create sidebar navigation component.
    
    Returns
    -------
    html.Div
        Sidebar component with theme toggle
    """
    return html.Div([
        # Logo/Title section
        html.Div([
            html.H4(
                "⚡ QuantLab",
                className='sidebar-brand-text',
                style={
                    'color': 'var(--primary)',
                    'marginBottom': '5px',
                    'fontWeight': 'bold',
                    'fontSize': '1.3rem',
                }
            ),
            html.Small(
                "v0.7.4",
                style={'color': 'var(--text-muted)', 'fontSize': '0.75rem'}
            ),
        ], style={
            'padding': '15px 10px',
            'borderBottom': '1px solid var(--border)',
            'textAlign': 'center',
        }),
        
        # Navigation links container
        html.Div([
            _create_nav_link(item) for item in NAV_ITEMS
        ], style={
            'padding': '10px 8px',
            'flex': '1',
            'overflowY': 'auto',
        }),
        
        # Footer section (theme toggle + status)
        html.Div([
            html.Hr(style={'borderColor': 'var(--border)', 'margin': '8px 0'}),
            # Theme toggle button
            html.Div([
                html.Button(
                    id='theme-toggle-btn',
                    children='🌙',
                    title='Toggle Dark/Light Theme',
                    style={
                        'width': '36px',
                        'height': '36px',
                        'borderRadius': '50%',
                        'border': '1px solid var(--border)',
                        'background': 'var(--bg-tertiary)',
                        'color': 'var(--text-primary)',
                        'cursor': 'pointer',
                        'fontSize': '1.1rem',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'transition': 'all 0.3s ease',
                        'marginBottom': '8px',
                    }
                ),
            ], style={'textAlign': 'center'}),
            # Connection status
            html.Div([
                html.Span("●", style={
                    'color': 'var(--success)',
                    'marginRight': '6px',
                    'fontSize': '8px',
                }),
                html.Small("ArcticDB Connected", style={
                    'color': 'var(--text-muted)',
                    'fontSize': '0.7rem',
                }),
            ], style={'textAlign': 'center'}),
        ], style={
            'padding': '10px 8px',
            'borderTop': '1px solid var(--border)',
        }),
        
    ], className='sidebar', style={
        'backgroundColor': 'var(--bg-secondary)',
        'height': '100%',
        'display': 'flex',
        'flexDirection': 'column',
    })


def _create_nav_link(item: dict):
    """Create a single navigation link."""
    return dcc.Link(
        html.Div([
            html.Span(
                item['icon'],
                style={
                    'fontSize': '1.1rem',
                    'marginRight': '10px',
                    'width': '24px',
                    'textAlign': 'center',
                }
            ),
            html.Div([
                html.Div(
                    item['name'],
                    style={
                        'color': 'var(--text-primary)',
                        'fontWeight': '500',
                        'fontSize': '0.85rem',
                    }
                ),
                html.Small(
                    item['description'],
                    style={
                        'color': 'var(--text-muted)',
                        'fontSize': '0.65rem',
                    }
                ),
            ], style={'flex': '1'}),
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '10px 12px',
            'borderRadius': '6px',
            'marginBottom': '4px',
            'transition': 'all 0.2s ease',
            'cursor': 'pointer',
        }, className='nav-item-hover'),
        href=item['path'],
        style={'textDecoration': 'none'},
    )
