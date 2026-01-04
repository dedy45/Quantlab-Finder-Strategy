"""
QuantLab Dash UI - Theme Configuration.

CYBORG dark theme colors and Plotly layout for trading dashboard.
Based on Bootstrap CYBORG theme for consistency with dash-bootstrap-components.

Features:
- Dark/Light theme support with CSS variables
- 3D shadow effects for modern look
- Responsive mobile support

Usage:
    from dash_ui.theme import COLORS, COLORS_LIGHT, PLOTLY_LAYOUT, CHART_HEIGHT
    
    fig.update_layout(**PLOTLY_LAYOUT)
"""

# =============================================================================
# DARK THEME COLORS (Default - CYBORG)
# =============================================================================
# Based on https://bootswatch.com/cyborg/

COLORS = {
    # Backgrounds
    'bg_primary': '#060606',      # Main background (CYBORG body)
    'bg_secondary': '#222222',    # Card/panel background
    'bg_tertiary': '#282828',     # Elevated elements
    'bg_input': '#191919',        # Input fields
    
    # Text
    'text': '#ADAFAE',            # Primary text (CYBORG body color)
    'text_muted': '#888888',      # Muted/secondary text
    'text_bright': '#FFFFFF',     # Bright/emphasis text
    
    # Bootstrap colors (CYBORG palette)
    'primary': '#2A9FD6',         # CYBORG primary (cyan-blue)
    'secondary': '#555555',       # CYBORG secondary
    'success': '#77B300',         # CYBORG success (lime green)
    'warning': '#FF8800',         # CYBORG warning (orange)
    'danger': '#CC0000',          # CYBORG danger (red)
    'info': '#9933CC',            # CYBORG info (purple)
    
    # Trading specific
    'bullish': '#77B300',         # Green for profit/up
    'bearish': '#CC0000',         # Red for loss/down
    'neutral': '#2A9FD6',         # Blue for neutral
    
    # Borders and grid
    'border': '#282828',          # Border color
    'grid': '#333333',            # Chart grid lines
    
    # Shadow (for 3D effects)
    'shadow': 'rgba(0, 0, 0, 0.5)',
    'shadow_light': 'rgba(0, 0, 0, 0.3)',
    'glow': 'rgba(42, 159, 214, 0.3)',  # Primary color glow
}

# =============================================================================
# LIGHT THEME COLORS
# =============================================================================

COLORS_LIGHT = {
    # Backgrounds
    'bg_primary': '#F5F5F5',      # Main background
    'bg_secondary': '#FFFFFF',    # Card/panel background
    'bg_tertiary': '#E8E8E8',     # Elevated elements
    'bg_input': '#FFFFFF',        # Input fields
    
    # Text
    'text': '#333333',            # Primary text
    'text_muted': '#666666',      # Muted/secondary text
    'text_bright': '#000000',     # Bright/emphasis text
    
    # Bootstrap colors (adjusted for light theme)
    'primary': '#0D6EFD',         # Bootstrap primary blue
    'secondary': '#6C757D',       # Bootstrap secondary
    'success': '#198754',         # Bootstrap success
    'warning': '#FFC107',         # Bootstrap warning
    'danger': '#DC3545',          # Bootstrap danger
    'info': '#0DCAF0',            # Bootstrap info
    
    # Trading specific
    'bullish': '#198754',         # Green for profit/up
    'bearish': '#DC3545',         # Red for loss/down
    'neutral': '#0D6EFD',         # Blue for neutral
    
    # Borders and grid
    'border': '#DEE2E6',          # Border color
    'grid': '#E9ECEF',            # Chart grid lines
    
    # Shadow (for 3D effects)
    'shadow': 'rgba(0, 0, 0, 0.15)',
    'shadow_light': 'rgba(0, 0, 0, 0.08)',
    'glow': 'rgba(13, 110, 253, 0.25)',  # Primary color glow
}

# =============================================================================
# PLOTLY LAYOUT TEMPLATE
# =============================================================================
# Spread into any figure: fig.update_layout(**PLOTLY_LAYOUT)

PLOTLY_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': COLORS['bg_secondary'],
    'plot_bgcolor': COLORS['bg_primary'],
    'font': {
        'color': COLORS['text'],
        'family': 'Roboto, sans-serif',
    },
    'title': {
        'font': {'color': COLORS['text_bright']},
    },
    'xaxis': {
        'gridcolor': COLORS['grid'],
        'linecolor': COLORS['border'],
        'zerolinecolor': COLORS['grid'],
    },
    'yaxis': {
        'gridcolor': COLORS['grid'],
        'linecolor': COLORS['border'],
        'zerolinecolor': COLORS['grid'],
    },
    'legend': {
        'bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': COLORS['text']},
    },
    'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50},
}

# =============================================================================
# CHART DIMENSIONS
# =============================================================================

CHART_HEIGHT = {
    'small': 250,      # KPI sparklines
    'medium': 350,     # Secondary charts
    'large': 450,      # Main charts (candlestick, equity)
    'full': 600,       # Full-page charts
}

# =============================================================================
# COMPONENT STYLES
# =============================================================================

CARD_STYLE = {
    'backgroundColor': COLORS['bg_secondary'],
    'borderRadius': '8px',
    'padding': '16px',
    'marginBottom': '16px',
    'border': f"1px solid {COLORS['border']}",
    # 3D Shadow effect
    'boxShadow': f"0 4px 6px {COLORS['shadow_light']}, 0 1px 3px {COLORS['shadow']}",
    'transition': 'all 0.3s ease',
}

CARD_STYLE_HOVER = {
    **CARD_STYLE,
    'boxShadow': f"0 10px 20px {COLORS['shadow']}, 0 3px 6px {COLORS['shadow_light']}",
    'transform': 'translateY(-2px)',
}

KPI_CARD_STYLE = {
    **CARD_STYLE,
    'textAlign': 'center',
    'minWidth': '150px',
}

# Modern elevated card with stronger shadow
CARD_ELEVATED_STYLE = {
    'backgroundColor': COLORS['bg_secondary'],
    'borderRadius': '12px',
    'padding': '20px',
    'marginBottom': '16px',
    'border': 'none',
    'boxShadow': f"0 8px 16px {COLORS['shadow']}, 0 4px 8px {COLORS['shadow_light']}, inset 0 1px 0 rgba(255,255,255,0.05)",
}

# Glass morphism style (modern frosted glass effect)
CARD_GLASS_STYLE = {
    'backgroundColor': 'rgba(34, 34, 34, 0.8)',
    'backdropFilter': 'blur(10px)',
    'WebkitBackdropFilter': 'blur(10px)',
    'borderRadius': '12px',
    'padding': '20px',
    'marginBottom': '16px',
    'border': '1px solid rgba(255, 255, 255, 0.1)',
    'boxShadow': f"0 8px 32px {COLORS['shadow']}",
}

SIDEBAR_STYLE = {
    'backgroundColor': COLORS['bg_secondary'],
    'padding': '20px',
    'height': '100vh',
    'position': 'fixed',
    'width': '250px',
    'boxShadow': f"4px 0 10px {COLORS['shadow']}",
    'zIndex': '1000',
}

CONTENT_STYLE = {
    'marginLeft': '270px',
    'padding': '20px',
    'backgroundColor': COLORS['bg_primary'],
    'minHeight': '100vh',
}

# Mobile sidebar (collapsed)
SIDEBAR_MOBILE_STYLE = {
    'backgroundColor': COLORS['bg_secondary'],
    'padding': '15px',
    'width': '100%',
    'position': 'relative',
    'boxShadow': f"0 2px 10px {COLORS['shadow']}",
}

CONTENT_MOBILE_STYLE = {
    'marginLeft': '0',
    'padding': '15px',
    'backgroundColor': COLORS['bg_primary'],
    'minHeight': '100vh',
}

# =============================================================================
# CANDLESTICK COLORS
# =============================================================================

CANDLESTICK_COLORS = {
    'increasing_line': COLORS['bullish'],
    'increasing_fill': COLORS['bullish'],
    'decreasing_line': COLORS['bearish'],
    'decreasing_fill': COLORS['bearish'],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_color_for_value(value: float, threshold: float = 0) -> str:
    """Get color based on value (green for positive, red for negative)."""
    if value > threshold:
        return COLORS['bullish']
    elif value < threshold:
        return COLORS['bearish']
    return COLORS['neutral']


def get_bootstrap_color(metric_name: str, value: float) -> str:
    """
    Get Bootstrap color class based on metric and value.
    
    Parameters
    ----------
    metric_name : str
        Name of metric (e.g., 'psr', 'sharpe', 'max_dd')
    value : float
        Metric value
        
    Returns
    -------
    str
        Bootstrap color class ('success', 'warning', 'danger', 'info')
    """
    if metric_name in ('psr', 'dsr'):
        if value >= 0.95:
            return 'success'
        elif value >= 0.80:
            return 'warning'
        return 'danger'
    
    elif metric_name == 'sharpe':
        if value >= 1.5:
            return 'success'
        elif value >= 0.5:
            return 'info'
        return 'warning'
    
    elif metric_name in ('max_dd', 'drawdown'):
        # Drawdown is negative, so we flip the logic
        if abs(value) <= 0.10:
            return 'success'
        elif abs(value) <= 0.20:
            return 'warning'
        return 'danger'
    
    elif metric_name == 'correlation':
        if abs(value) <= 0.30:
            return 'success'
        elif abs(value) <= 0.50:
            return 'warning'
        return 'danger'
    
    return 'info'  # Default
