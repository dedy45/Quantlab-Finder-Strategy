"""
Navigation Sidebar Component.
"""
from dash import html, dcc, page_registry
import dash_bootstrap_components as dbc

from ..theme import COLORS


def create_navbar():
    """Create sidebar navigation."""
    nav_items = [
        {'name': 'Dashboard', 'path': '/', 'icon': '📊'},
        {'name': 'Data Studio', 'path': '/data-studio', 'icon': '💾'},
        {'name': 'Backtest Arena', 'path': '/backtest-arena', 'icon': '🎯'},
        {'name': 'Settings', 'path': '/settings', 'icon': '⚙️'},
    ]
    
    return html.Div([
        # Logo/Title
        html.Div([
            html.H4("QuantLab", style={'color': COLORS['text'], 'marginBottom': '5px'}),
            html.Small("v0.7.2", style={'color': COLORS['text_muted']}),
        ], style={'padding': '20px', 'borderBottom': f"1px solid {COLORS['border']}"}),
        
        # Navigation links
        html.Div([
            dcc.Link(
                html.Div([
                    html.Span(item['icon'], style={'marginRight': '10px'}),
                    html.Span(item['name']),
                ], style={
                    'padding': '12px 20px',
                    'color': COLORS['text'],
                    'display': 'block',
                    'borderRadius': '4px',
                    'marginBottom': '4px',
                }),
                href=item['path'],
                style={'textDecoration': 'none'},
            )
            for item in nav_items
        ], style={'padding': '10px'}),
        
        # Footer
        html.Div([
            html.Hr(style={'borderColor': COLORS['border']}),
            html.Small("Dash Prototype", style={'color': COLORS['text_muted']}),
        ], style={'padding': '20px', 'position': 'absolute', 'bottom': '0', 'width': '100%'}),
    ], style={'height': '100vh', 'position': 'relative'})
