"""
Input Validation Functions untuk Dash UI.

Semua validator mengembalikan tuple (is_valid: bool, error_message: str).
Jika valid, error_message adalah string kosong.

Usage:
    from dash_ui.utils.validators import validate_symbol, validate_date_range
    
    is_valid, error = validate_symbol('XAUUSD')
    if not is_valid:
        return create_error_alert("Invalid Symbol", error)
"""
import logging
from datetime import datetime, date
from typing import Tuple, Optional, List, Any

logger = logging.getLogger(__name__)


def validate_symbol(
    symbol: Optional[str],
    available_symbols: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate trading symbol.
    
    Parameters
    ----------
    symbol : str or None
        Symbol to validate
    available_symbols : list of str, optional
        List of valid symbols. If None, only checks for non-empty.
        
    Returns
    -------
    tuple of (bool, str)
        (is_valid, error_message)
        
    Examples
    --------
    >>> is_valid, error = validate_symbol('XAUUSD', ['XAUUSD', 'EURUSD'])
    >>> print(is_valid)  # True
    """
    if symbol is None or symbol == '':
        return False, "Symbol is required"
    
    if not isinstance(symbol, str):
        return False, f"Symbol must be a string, got {type(symbol).__name__}"
    
    symbol = symbol.strip().upper()
    
    if len(symbol) < 3:
        return False, "Symbol must be at least 3 characters"
    
    if len(symbol) > 20:
        return False, "Symbol must be at most 20 characters"
    
    if available_symbols is not None:
        available_upper = [s.upper() for s in available_symbols]
        if symbol not in available_upper:
            return False, f"Symbol '{symbol}' not found in available data"
    
    return True, ""


def validate_date_range(
    start_date: Optional[Any],
    end_date: Optional[Any],
    min_days: int = 1,
    max_days: int = 3650  # ~10 years
) -> Tuple[bool, str]:
    """
    Validate date range for data loading.
    
    Parameters
    ----------
    start_date : str, date, or datetime
        Start date
    end_date : str, date, or datetime
        End date
    min_days : int, default=1
        Minimum days in range
    max_days : int, default=3650
        Maximum days in range
        
    Returns
    -------
    tuple of (bool, str)
        (is_valid, error_message)
        
    Examples
    --------
    >>> is_valid, error = validate_date_range('2024-01-01', '2024-12-31')
    >>> print(is_valid)  # True
    """
    if start_date is None:
        return False, "Start date is required"
    
    if end_date is None:
        return False, "End date is required"
    
    # Parse dates
    try:
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start = start_date.date()
        elif isinstance(start_date, date):
            start = start_date
        else:
            return False, f"Invalid start date type: {type(start_date).__name__}"
    except ValueError as e:
        return False, f"Invalid start date format: {e}"
    
    try:
        if isinstance(end_date, str):
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end = end_date.date()
        elif isinstance(end_date, date):
            end = end_date
        else:
            return False, f"Invalid end date type: {type(end_date).__name__}"
    except ValueError as e:
        return False, f"Invalid end date format: {e}"
    
    # Validate range
    if start > end:
        return False, "Start date must be before end date"
    
    days = (end - start).days
    
    if days < min_days:
        return False, f"Date range must be at least {min_days} day(s)"
    
    if days > max_days:
        return False, f"Date range must be at most {max_days} days (~{max_days // 365} years)"
    
    # Check not in future
    today = date.today()
    if start > today:
        return False, "Start date cannot be in the future"
    
    return True, ""


def validate_numeric(
    value: Any,
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_zero: bool = True,
    allow_negative: bool = False
) -> Tuple[bool, str]:
    """
    Validate numeric input.
    
    Parameters
    ----------
    value : any
        Value to validate
    field_name : str
        Name of field for error messages
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value
    allow_zero : bool, default=True
        Whether zero is allowed
    allow_negative : bool, default=False
        Whether negative values are allowed
        
    Returns
    -------
    tuple of (bool, str)
        (is_valid, error_message)
        
    Examples
    --------
    >>> is_valid, error = validate_numeric(100000, 'Initial Capital', min_value=1000)
    >>> print(is_valid)  # True
    """
    if value is None:
        return False, f"{field_name} is required"
    
    # Try to convert to float
    try:
        num = float(value)
    except (ValueError, TypeError):
        return False, f"{field_name} must be a number"
    
    # Check for NaN/Inf
    import math
    if math.isnan(num) or math.isinf(num):
        return False, f"{field_name} must be a finite number"
    
    # Check zero
    if not allow_zero and num == 0:
        return False, f"{field_name} cannot be zero"
    
    # Check negative
    if not allow_negative and num < 0:
        return False, f"{field_name} cannot be negative"
    
    # Check min
    if min_value is not None and num < min_value:
        return False, f"{field_name} must be at least {min_value}"
    
    # Check max
    if max_value is not None and num > max_value:
        return False, f"{field_name} must be at most {max_value}"
    
    return True, ""


def validate_timeframe(
    timeframe: Optional[str],
    available_timeframes: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate timeframe string.
    
    Parameters
    ----------
    timeframe : str or None
        Timeframe to validate (e.g., '1H', '4H', '1D')
    available_timeframes : list of str, optional
        List of valid timeframes. If None, uses default list.
        
    Returns
    -------
    tuple of (bool, str)
        (is_valid, error_message)
    """
    if timeframe is None or timeframe == '':
        return False, "Timeframe is required"
    
    # Default available timeframes from config
    if available_timeframes is None:
        try:
            from config import get_config
            cfg = get_config()
            available_timeframes = cfg.data.available_timeframes
        except Exception:
            available_timeframes = ['15T', '1H', '4H', '1D']
    
    timeframe = timeframe.strip().upper()
    
    if timeframe not in available_timeframes:
        return False, f"Invalid timeframe '{timeframe}'. Available: {', '.join(available_timeframes)}"
    
    return True, ""


def validate_strategy_params(
    strategy_type: str,
    params: dict
) -> Tuple[bool, str]:
    """
    Validate strategy parameters based on strategy type.
    
    Parameters
    ----------
    strategy_type : str
        Type of strategy ('momentum', 'mean_reversion', etc.)
    params : dict
        Strategy parameters
        
    Returns
    -------
    tuple of (bool, str)
        (is_valid, error_message)
    """
    if strategy_type == 'momentum':
        # Validate momentum params
        fast = params.get('fast_period')
        slow = params.get('slow_period')
        
        is_valid, error = validate_numeric(fast, 'Fast Period', min_value=1, max_value=200)
        if not is_valid:
            return False, error
        
        is_valid, error = validate_numeric(slow, 'Slow Period', min_value=1, max_value=500)
        if not is_valid:
            return False, error
        
        if float(fast) >= float(slow):
            return False, "Fast period must be less than slow period"
        
    elif strategy_type == 'mean_reversion':
        # Validate mean reversion params
        lookback = params.get('lookback')
        entry_z = params.get('entry_z')
        exit_z = params.get('exit_z')
        
        is_valid, error = validate_numeric(lookback, 'Lookback', min_value=5, max_value=200)
        if not is_valid:
            return False, error
        
        is_valid, error = validate_numeric(entry_z, 'Entry Z-Score', min_value=0.5, max_value=5.0)
        if not is_valid:
            return False, error
        
        is_valid, error = validate_numeric(exit_z, 'Exit Z-Score', min_value=0, max_value=3.0)
        if not is_valid:
            return False, error
        
        if float(exit_z) >= float(entry_z):
            return False, "Exit Z-Score must be less than Entry Z-Score"
    
    # Validate common params
    stop_loss = params.get('stop_loss')
    take_profit = params.get('take_profit')
    
    if stop_loss is not None:
        is_valid, error = validate_numeric(stop_loss, 'Stop Loss %', min_value=0.1, max_value=50)
        if not is_valid:
            return False, error
    
    if take_profit is not None:
        is_valid, error = validate_numeric(take_profit, 'Take Profit %', min_value=0.1, max_value=100)
        if not is_valid:
            return False, error
    
    return True, ""
