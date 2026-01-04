"""
Server-Side Cache Manager untuk Dash UI.

Menggunakan Flask-Caching dengan filesystem backend untuk menyimpan:
- OHLCV data dari ArcticDB
- Backtest results
- Risk calculations

PENTING: Jangan gunakan dcc.Store untuk data besar!
Browser localStorage/sessionStorage terbatas ~5-10MB.

Usage:
    from dash_ui.cache import init_cache, save_to_cache, load_from_cache
    
    # Initialize in app.py
    cache = init_cache(app)
    
    # Save data
    cache_id = generate_cache_id('ohlcv')
    save_to_cache(cache_id, df)
    
    # Load data
    df = load_from_cache(cache_id)
"""
import os
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cache directory - relative to QuantLab root
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache' / 'dash_ui'

# Flask-Caching instance (set by init_cache)
_flask_cache = None


def init_cache(app):
    """
    Initialize Flask-Caching for the Dash app.
    
    Parameters
    ----------
    app : dash.Dash
        Dash application instance
        
    Returns
    -------
    flask_caching.Cache
        Initialized cache instance
    """
    global _flask_cache
    
    try:
        from flask_caching import Cache
    except ImportError:
        logger.warning("flask-caching not installed. Using filesystem fallback.")
        return None
    
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure cache
    cache_config = {
        'CACHE_TYPE': 'FileSystemCache',
        'CACHE_DIR': str(CACHE_DIR),
        'CACHE_DEFAULT_TIMEOUT': 3600,  # 1 hour
        'CACHE_THRESHOLD': 500,  # Max items
    }
    
    _flask_cache = Cache(app.server, config=cache_config)
    logger.info(f"Flask-Caching initialized at {CACHE_DIR}")
    
    return _flask_cache


def get_cache():
    """Get the Flask-Caching instance."""
    return _flask_cache


def generate_cache_id(prefix: str = 'data') -> str:
    """
    Generate unique cache ID with timestamp.
    
    Parameters
    ----------
    prefix : str, default='data'
        Prefix for the cache ID
        
    Returns
    -------
    str
        Unique cache ID like 'data_20240101_123456_abc123'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{random_suffix}"


def generate_data_key(
    symbol: str,
    timeframe: str,
    start: str,
    end: str
) -> str:
    """
    Generate deterministic cache key for OHLCV data.
    
    Same inputs always produce same key, enabling cache hits.
    
    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., 'XAUUSD')
    timeframe : str
        Timeframe (e.g., '1H')
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD)
        
    Returns
    -------
    str
        Deterministic cache key
        
    Examples
    --------
    >>> key = generate_data_key('XAUUSD', '1H', '2024-01-01', '2024-12-31')
    >>> print(key)  # 'ohlcv_XAUUSD_1H_20240101_20241231'
    """
    # Normalize inputs
    symbol = symbol.upper().strip()
    timeframe = timeframe.upper().strip()
    start_clean = start.replace('-', '')
    end_clean = end.replace('-', '')
    
    return f"ohlcv_{symbol}_{timeframe}_{start_clean}_{end_clean}"


def save_to_cache(cache_id: str, data: Any) -> bool:
    """
    Save data to server-side cache.
    
    Uses Flask-Caching if available, falls back to pickle files.
    
    Parameters
    ----------
    cache_id : str
        Cache identifier
    data : any
        Data to cache (must be picklable)
        
    Returns
    -------
    bool
        True if saved successfully
    """
    try:
        # Try Flask-Caching first
        if _flask_cache is not None:
            _flask_cache.set(cache_id, data)
            logger.debug(f"Saved to Flask cache: {cache_id}")
            return True
        
        # Fallback to pickle file
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{cache_id}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Saved to pickle cache: {cache_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save cache {cache_id}: {e}")
        return False


def load_from_cache(cache_id: str) -> Optional[Any]:
    """
    Load data from server-side cache.
    
    Parameters
    ----------
    cache_id : str
        Cache identifier
        
    Returns
    -------
    any or None
        Cached data, or None if not found
    """
    try:
        # Try Flask-Caching first
        if _flask_cache is not None:
            data = _flask_cache.get(cache_id)
            if data is not None:
                logger.debug(f"Cache hit (Flask): {cache_id}")
                return data
        
        # Fallback to pickle file
        cache_file = CACHE_DIR / f"{cache_id}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit (pickle): {cache_file}")
            return data
        
        logger.debug(f"Cache miss: {cache_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load cache {cache_id}: {e}")
        return None


def cache_exists(cache_id: str) -> bool:
    """
    Check if cache entry exists.
    
    Parameters
    ----------
    cache_id : str
        Cache identifier
        
    Returns
    -------
    bool
        True if cache exists
    """
    # Try Flask-Caching
    if _flask_cache is not None:
        if _flask_cache.get(cache_id) is not None:
            return True
    
    # Check pickle file
    cache_file = CACHE_DIR / f"{cache_id}.pkl"
    return cache_file.exists()


def delete_from_cache(cache_id: str) -> bool:
    """
    Delete cache entry.
    
    Parameters
    ----------
    cache_id : str
        Cache identifier
        
    Returns
    -------
    bool
        True if deleted successfully
    """
    try:
        # Delete from Flask-Caching
        if _flask_cache is not None:
            _flask_cache.delete(cache_id)
        
        # Delete pickle file
        cache_file = CACHE_DIR / f"{cache_id}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        
        logger.debug(f"Deleted cache: {cache_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete cache {cache_id}: {e}")
        return False


def cleanup_old_cache(max_age_hours: int = 24) -> int:
    """
    Remove cache files older than specified hours.
    
    Parameters
    ----------
    max_age_hours : int, default=24
        Maximum age in hours
        
    Returns
    -------
    int
        Number of files deleted
    """
    if not CACHE_DIR.exists():
        return 0
    
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted = 0
    
    for cache_file in CACHE_DIR.glob('*.pkl'):
        try:
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if mtime < cutoff:
                cache_file.unlink()
                deleted += 1
                logger.debug(f"Deleted old cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file}: {e}")
    
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} old cache files")
    
    return deleted


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns
    -------
    dict
        Cache statistics including count, size, oldest, newest
    """
    if not CACHE_DIR.exists():
        return {'count': 0, 'size_mb': 0, 'oldest': None, 'newest': None}
    
    files = list(CACHE_DIR.glob('*.pkl'))
    
    if not files:
        return {'count': 0, 'size_mb': 0, 'oldest': None, 'newest': None}
    
    total_size = sum(f.stat().st_size for f in files)
    mtimes = [datetime.fromtimestamp(f.stat().st_mtime) for f in files]
    
    return {
        'count': len(files),
        'size_mb': round(total_size / (1024 * 1024), 2),
        'oldest': min(mtimes).isoformat() if mtimes else None,
        'newest': max(mtimes).isoformat() if mtimes else None,
    }
