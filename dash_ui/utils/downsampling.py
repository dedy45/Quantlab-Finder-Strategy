"""
LTTB (Largest-Triangle-Three-Buckets) Downsampling dengan Numba JIT.

Numba meng-compile loop Python menjadi Machine Code (C-speed).
Untuk 1 juta baris: Python native ~3 detik, Numba ~30ms (100x lebih cepat!)

Usage:
    from dash_ui.utils.downsampling import lttb_downsample, lttb_downsample_ohlc
    
    # For line charts (close price)
    df_chart = lttb_downsample(df, target_points=2000)
    
    # For candlestick charts (preserves OHLC integrity)
    df_chart = lttb_downsample_ohlc(df, target_points=2000)
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import numba, fallback to pure Python if not available
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    logger.info("Numba JIT available - LTTB will use optimized implementation")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - LTTB will use pure Python (slower)")
    
    # Create a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _lttb_core(x_data: np.ndarray, y_data: np.ndarray, target_points: int) -> np.ndarray:
    """
    Core LTTB algorithm dengan Numba JIT compilation.
    
    Parameters
    ----------
    x_data : np.ndarray
        X values (biasanya index sebagai float)
    y_data : np.ndarray
        Y values (biasanya close price)
    target_points : int
        Target jumlah points setelah downsampling
        
    Returns
    -------
    np.ndarray
        Array of indices yang dipilih
    """
    n = len(y_data)
    
    if n <= target_points:
        return np.arange(n)
    
    # Bucket size
    bucket_size = (n - 2) / (target_points - 2)
    
    # Output indices
    sampled_indices = np.zeros(target_points, dtype=np.int64)
    sampled_indices[0] = 0  # Always include first point
    sampled_indices[target_points - 1] = n - 1  # Always include last point
    
    a = 0  # Previous selected point index
    
    for i in range(target_points - 2):
        # Current bucket range
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)
        
        # Next bucket range for average calculation
        next_bucket_start = int((i + 2) * bucket_size) + 1
        next_bucket_end = int((i + 3) * bucket_size) + 1
        next_bucket_end = min(next_bucket_end, n)
        
        # Calculate average of next bucket
        avg_x = 0.0
        avg_y = 0.0
        count = 0
        for j in range(next_bucket_start, next_bucket_end):
            avg_x += x_data[j]
            avg_y += y_data[j]
            count += 1
        
        if count > 0:
            avg_x /= count
            avg_y /= count
        
        # Find point in current bucket with largest triangle area
        max_area = -1.0
        max_idx = bucket_start
        
        for j in range(bucket_start, bucket_end):
            # Triangle area calculation (simplified)
            area = abs(
                (x_data[a] - avg_x) * (y_data[j] - y_data[a]) -
                (x_data[a] - x_data[j]) * (avg_y - y_data[a])
            ) * 0.5
            
            if area > max_area:
                max_area = area
                max_idx = j
        
        sampled_indices[i + 1] = max_idx
        a = max_idx
    
    return sampled_indices


def lttb_downsample(
    df: pd.DataFrame,
    target_points: int = 2000,
    y_column: str = 'close'
) -> pd.DataFrame:
    """
    Downsample DataFrame menggunakan LTTB algorithm.
    
    Cocok untuk line charts (single y-value per point).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan DatetimeIndex
    target_points : int, default=2000
        Target jumlah points setelah downsampling
    y_column : str, default='close'
        Column yang digunakan untuk LTTB selection
        
    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame
        
    Examples
    --------
    >>> from core.data_engine import load_data
    >>> df = load_data('XAUUSD', '2020-01-01', '2024-12-31', '1H')
    >>> df_chart = lttb_downsample(df, target_points=2000)
    >>> print(f"Original: {len(df)}, Downsampled: {len(df_chart)}")
    """
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame passed to lttb_downsample")
        return df
    
    n = len(df)
    
    # No downsampling needed
    if n <= target_points:
        logger.debug(f"No downsampling needed: {n} <= {target_points}")
        return df
    
    logger.info(f"LTTB downsampling: {n} -> {target_points} points")
    
    # Prepare data for LTTB
    x_data = np.arange(n, dtype=np.float64)
    y_data = df[y_column].values.astype(np.float64)
    
    # Run LTTB
    indices = _lttb_core(x_data, y_data, target_points)
    
    # Return downsampled DataFrame
    return df.iloc[indices].copy()


def lttb_downsample_ohlc(
    df: pd.DataFrame,
    target_points: int = 2000
) -> pd.DataFrame:
    """
    Downsample OHLCV DataFrame dengan preserving OHLC integrity.
    
    Untuk candlestick charts, kita perlu memastikan:
    - High adalah max dari bucket
    - Low adalah min dari bucket
    - Open adalah first dari bucket
    - Close adalah last dari bucket
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan columns: open, high, low, close, volume (optional)
        dan DatetimeIndex
    target_points : int, default=2000
        Target jumlah candles setelah downsampling
        
    Returns
    -------
    pd.DataFrame
        Downsampled OHLCV DataFrame
        
    Examples
    --------
    >>> from core.data_engine import load_data
    >>> df = load_data('XAUUSD', '2020-01-01', '2024-12-31', '1H')
    >>> df_chart = lttb_downsample_ohlc(df, target_points=2000)
    >>> print(f"Original: {len(df)}, Downsampled: {len(df_chart)}")
    """
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame passed to lttb_downsample_ohlc")
        return df
    
    n = len(df)
    
    # No downsampling needed
    if n <= target_points:
        logger.debug(f"No OHLC downsampling needed: {n} <= {target_points}")
        return df
    
    logger.info(f"LTTB OHLC downsampling: {n} -> {target_points} candles")
    
    # Calculate bucket size
    bucket_size = n / target_points
    
    # Prepare output lists
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    has_volume = 'volume' in df.columns
    
    for i in range(target_points):
        # Bucket range
        start_idx = int(i * bucket_size)
        end_idx = int((i + 1) * bucket_size)
        end_idx = min(end_idx, n)
        
        if start_idx >= end_idx:
            continue
        
        bucket = df.iloc[start_idx:end_idx]
        
        # Aggregate OHLCV
        timestamps.append(bucket.index[0])  # First timestamp
        opens.append(bucket['open'].iloc[0])  # First open
        highs.append(bucket['high'].max())  # Max high
        lows.append(bucket['low'].min())  # Min low
        closes.append(bucket['close'].iloc[-1])  # Last close
        
        if has_volume:
            volumes.append(bucket['volume'].sum())  # Sum volume
    
    # Create result DataFrame
    result = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
    }, index=pd.DatetimeIndex(timestamps))
    
    if has_volume:
        result['volume'] = volumes
    
    return result


def auto_downsample(
    df: pd.DataFrame,
    chart_type: str = 'line',
    target_points: int = 2000,
    y_column: str = 'close'
) -> pd.DataFrame:
    """
    Automatically choose downsampling method based on chart type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    chart_type : str, default='line'
        'line' for line charts, 'candlestick' for OHLC charts
    target_points : int, default=2000
        Target points after downsampling
    y_column : str, default='close'
        Column for line chart LTTB
        
    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame
    """
    if chart_type == 'candlestick':
        return lttb_downsample_ohlc(df, target_points)
    else:
        return lttb_downsample(df, target_points, y_column)
