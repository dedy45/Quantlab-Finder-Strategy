"""
QuantLab Dash UI - Main Application Entry Point.

Multi-page Dash application with CYBORG dark theme for trading dashboard.

Run with: python dash_ui/app.py
Access at: http://localhost:8050

Pages:
- / (Dashboard) - Overview & KPIs
- /data-studio - ArcticDB data explorer
- /backtest - Backtest Arena
- /risk-lab - Risk analysis
- /settings - Configuration
"""
import sys
import logging
from pathlib import Path

# Add QuantLab root to path for imports when running as __main__
QUANTLAB_ROOT = Path(__file__).parent.parent
if str(QUANTLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANTLAB_ROOT))

from dash import Dash
import dash_bootstrap_components as dbc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app with CYBORG theme (dark theme)
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,  # CYBORG dark theme
        dbc.icons.BOOTSTRAP,  # Bootstrap icons
    ],
    suppress_callback_exceptions=True,
    title="QuantLab Dashboard",
    update_title="Loading...",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

# Expose server for WSGI deployment
server = app.server

# Handle imports for both package and direct execution
try:
    # Try relative imports first (when imported as package)
    from .cache import init_cache
    from .index import create_layout, register_callbacks
except ImportError:
    # Fall back to absolute imports (when run as __main__)
    from dash_ui.cache import init_cache
    from dash_ui.index import create_layout, register_callbacks

# Initialize Flask-Caching
try:
    cache = init_cache(app)
    logger.info("Flask-Caching initialized")
except Exception as e:
    logger.warning(f"Flask-Caching not available: {e}")
    cache = None

# Apply layout and register callbacks
app.layout = create_layout()
register_callbacks(app)

logger.info("QuantLab Dash UI initialized")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("  QuantLab Dash UI")
    logger.info("  Theme: CYBORG (Dark)")
    logger.info("  URL: http://localhost:8050")
    logger.info("=" * 60)
    
    app.run(
        debug=True,
        port=8050,
        host='127.0.0.1',
    )
