"""
Toast Notification Components untuk Dash UI.

Auto-dismissing notifications for user feedback.

Usage:
    from dash_ui.components.toast import create_toast, toast_container
    
    toast = create_toast("Data loaded successfully", header="Success", icon="success")
"""
from typing import Optional
from dash import html
import dash_bootstrap_components as dbc

from ..theme import COLORS


# Icon mapping
TOAST_ICONS = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
}

# Color mapping
TOAST_COLORS = {
    'success': COLORS['success'],
    'error': COLORS['danger'],
    'warning': COLORS['warning'],
    'info': COLORS['info'],
}


def create_toast(
    message: str,
    header: str = None,
    icon: str = 'info',
    duration: int = 5000,
    dismissable: bool = True,
    is_open: bool = True,
) -> dbc.Toast:
    """
    Create a toast notification.
    
    Parameters
    ----------
    message : str
        Toast message content
    header : str, optional
        Toast header text
    icon : str, default='info'
        Icon type: 'success', 'error', 'warning', 'info'
    duration : int, default=5000
        Auto-dismiss duration in milliseconds. Set to None for no auto-dismiss.
    dismissable : bool, default=True
        Whether toast can be manually dismissed
    is_open : bool, default=True
        Initial open state
        
    Returns
    -------
    dbc.Toast
        Toast notification component
    """
    icon_char = TOAST_ICONS.get(icon, TOAST_ICONS['info'])
    header_color = TOAST_COLORS.get(icon, COLORS['info'])
    
    header_content = [
        html.Span(icon_char, className="me-2"),
        html.Strong(header or icon.title()),
    ]
    
    return dbc.Toast(
        message,
        header=header_content,
        is_open=is_open,
        dismissable=dismissable,
        duration=duration,
        icon=icon if icon in ('primary', 'secondary', 'success', 'warning', 'danger', 'info') else None,
        style={
            'position': 'fixed',
            'top': 20,
            'right': 20,
            'width': 350,
            'zIndex': 9999,
        },
        header_style={
            'borderLeft': f"4px solid {header_color}",
        },
    )


def success_toast(message: str, header: str = "Success") -> dbc.Toast:
    """Create a success toast."""
    return create_toast(message, header=header, icon='success')


def error_toast(message: str, header: str = "Error") -> dbc.Toast:
    """Create an error toast."""
    return create_toast(message, header=header, icon='error', duration=None)


def warning_toast(message: str, header: str = "Warning") -> dbc.Toast:
    """Create a warning toast."""
    return create_toast(message, header=header, icon='warning', duration=8000)


def info_toast(message: str, header: str = "Info") -> dbc.Toast:
    """Create an info toast."""
    return create_toast(message, header=header, icon='info')


def toast_container(id: str = 'toast-container') -> html.Div:
    """
    Create a container for toast notifications.
    
    Place this in your layout to hold dynamically created toasts.
    
    Parameters
    ----------
    id : str, default='toast-container'
        Container ID for callbacks
        
    Returns
    -------
    html.Div
        Toast container
    """
    return html.Div(
        id=id,
        style={
            'position': 'fixed',
            'top': 20,
            'right': 20,
            'zIndex': 9999,
        }
    )
