"""
Centralized Error Handling untuk Dash UI.

Menyediakan:
- Custom exception classes
- Error logging ke file
- UI error alerts (dismissable)
- Safe callback decorator

Usage:
    from dash_ui.error_handler import safe_callback, create_error_alert, log_error
    
    @callback(...)
    @safe_callback(default_outputs=[empty_chart(), "Error loading data"])
    def load_data(symbol, timeframe):
        # If exception occurs, returns default_outputs
        df = load_ohlcv(symbol, timeframe)
        return create_chart(df), f"Loaded {len(df)} rows"
"""
import logging
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import dash_bootstrap_components as dbc
from dash import html

# Setup logging
LOG_DIR = Path(__file__).parent.parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / 'dash_ui.log'

# Configure file handler
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
))

# Get logger
logger = logging.getLogger('dash_ui')
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


class CallbackError(Exception):
    """Custom exception for callback errors with context."""
    
    def __init__(self, message: str, callback_name: str = '', context: dict = None):
        self.message = message
        self.callback_name = callback_name
        self.context = context or {}
        super().__init__(self.message)


class DataLoadError(CallbackError):
    """Error loading data from ArcticDB or other sources."""
    pass


class ValidationError(CallbackError):
    """Input validation error."""
    pass


class BacktestError(CallbackError):
    """Backtest execution error."""
    pass


def log_error(
    callback_name: str,
    error: Exception,
    context: dict = None
) -> str:
    """
    Log error to file with full context.
    
    Parameters
    ----------
    callback_name : str
        Name of the callback where error occurred
    error : Exception
        The exception that was raised
    context : dict, optional
        Additional context (inputs, state, etc.)
        
    Returns
    -------
    str
        Error ID for reference
    """
    error_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]
    
    log_message = f"""
================================================================================
ERROR ID: {error_id}
CALLBACK: {callback_name}
TIME: {datetime.now().isoformat()}
ERROR TYPE: {type(error).__name__}
ERROR MESSAGE: {str(error)}
CONTEXT: {context or {}}
TRACEBACK:
{traceback.format_exc()}
================================================================================
"""
    
    logger.error(log_message)
    
    return error_id


def create_error_alert(
    title: str,
    message: str,
    details: str = None,
    dismissable: bool = True,
    color: str = 'danger'
) -> dbc.Alert:
    """
    Create a dismissable error alert for UI.
    
    Parameters
    ----------
    title : str
        Alert title/header
    message : str
        Main error message
    details : str, optional
        Technical details (shown in smaller text)
    dismissable : bool, default=True
        Whether alert can be dismissed
    color : str, default='danger'
        Bootstrap color class
        
    Returns
    -------
    dbc.Alert
        Dash Bootstrap Alert component
    """
    children = [
        html.H5(title, className='alert-heading'),
        html.P(message, className='mb-0'),
    ]
    
    if details:
        children.append(html.Hr())
        children.append(html.Small(details, className='text-muted'))
    
    return dbc.Alert(
        children,
        color=color,
        dismissable=dismissable,
        is_open=True,
        className='mt-3'
    )


def create_connection_error_alert(service: str = 'ArcticDB') -> dbc.Alert:
    """
    Create alert for database connection errors.
    
    Parameters
    ----------
    service : str, default='ArcticDB'
        Name of the service that failed
        
    Returns
    -------
    dbc.Alert
        Connection error alert
    """
    return create_error_alert(
        title=f"Connection Error",
        message=f"Failed to connect to {service}. Please check if the service is running.",
        details="Try restarting the application or check the logs for more details.",
        color='danger'
    )


def create_validation_error_alert(
    field: str,
    message: str
) -> dbc.Alert:
    """
    Create alert for input validation errors.
    
    Parameters
    ----------
    field : str
        Name of the field with invalid input
    message : str
        Validation error message
        
    Returns
    -------
    dbc.Alert
        Validation error alert
    """
    return create_error_alert(
        title="Validation Error",
        message=f"{field}: {message}",
        color='warning',
        dismissable=True
    )


def create_no_data_alert(
    symbol: str = None,
    timeframe: str = None
) -> dbc.Alert:
    """
    Create alert when no data is available.
    
    Parameters
    ----------
    symbol : str, optional
        Symbol that was requested
    timeframe : str, optional
        Timeframe that was requested
        
    Returns
    -------
    dbc.Alert
        No data alert
    """
    if symbol and timeframe:
        message = f"No data available for {symbol} ({timeframe})"
    elif symbol:
        message = f"No data available for {symbol}"
    else:
        message = "No data available"
    
    return create_error_alert(
        title="No Data",
        message=message,
        details="Please check if the data has been imported to ArcticDB.",
        color='info'
    )


def create_success_alert(
    title: str,
    message: str,
    auto_dismiss: bool = True
) -> dbc.Alert:
    """
    Create a success alert.
    
    Parameters
    ----------
    title : str
        Alert title
    message : str
        Success message
    auto_dismiss : bool, default=True
        Whether to auto-dismiss (via CSS animation)
        
    Returns
    -------
    dbc.Alert
        Success alert
    """
    return dbc.Alert(
        [
            html.H5(title, className='alert-heading'),
            html.P(message, className='mb-0'),
        ],
        color='success',
        dismissable=True,
        is_open=True,
        duration=5000 if auto_dismiss else None,
        className='mt-3'
    )


def safe_callback(default_outputs: List[Any] = None):
    """
    Decorator for safe callback execution with error handling.
    
    Wraps callback in try-except, logs errors, and returns default outputs.
    
    Parameters
    ----------
    default_outputs : list, optional
        Default outputs to return on error. If None, returns error alert.
        
    Returns
    -------
    callable
        Decorated callback function
        
    Examples
    --------
    >>> @callback(Output('chart', 'figure'), Output('info', 'children'), Input('btn', 'n_clicks'))
    >>> @safe_callback(default_outputs=[empty_chart(), "Error"])
    >>> def update_chart(n_clicks):
    >>>     # If exception, returns [empty_chart(), "Error"]
    >>>     return create_chart(data), "Success"
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                error_id = log_error(
                    callback_name=func.__name__,
                    error=e,
                    context={'args': str(args)[:500], 'kwargs': str(kwargs)[:500]}
                )
                
                # Log to console as well
                logger.error(f"Callback {func.__name__} failed: {e}")
                
                # Return default outputs or error alert
                if default_outputs is not None:
                    return default_outputs if len(default_outputs) > 1 else default_outputs[0]
                
                # Single output - return error alert
                return create_error_alert(
                    title="Error",
                    message=str(e),
                    details=f"Error ID: {error_id}"
                )
        
        return wrapper
    return decorator


def handle_callback_error(
    error: Exception,
    callback_name: str,
    context: dict = None
) -> Tuple[str, dbc.Alert]:
    """
    Handle callback error and return error ID and alert.
    
    Parameters
    ----------
    error : Exception
        The exception
    callback_name : str
        Name of the callback
    context : dict, optional
        Additional context
        
    Returns
    -------
    tuple of (str, dbc.Alert)
        Error ID and alert component
    """
    error_id = log_error(callback_name, error, context)
    
    alert = create_error_alert(
        title="Error",
        message=str(error),
        details=f"Error ID: {error_id}. Check logs for details."
    )
    
    return error_id, alert
