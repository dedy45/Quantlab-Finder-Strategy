"""
Data Table Components untuk Dash UI.

Paginated tables dengan dark theme styling.

Usage:
    from dash_ui.components.tables import paginated_table, simple_table
    
    table = paginated_table(df, id='data-table', page_size=20)
"""
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from dash import dash_table, html
import dash_bootstrap_components as dbc

from ..theme import COLORS

logger = logging.getLogger(__name__)

# Default styling for DataTable
TABLE_STYLE_HEADER = {
    'backgroundColor': COLORS['bg_tertiary'],
    'color': COLORS['text_bright'],
    'fontWeight': 'bold',
    'border': f"1px solid {COLORS['border']}",
    'textAlign': 'left',
}

TABLE_STYLE_CELL = {
    'backgroundColor': COLORS['bg_secondary'],
    'color': COLORS['text'],
    'border': f"1px solid {COLORS['border']}",
    'textAlign': 'left',
    'padding': '8px 12px',
    'fontSize': '13px',
}

TABLE_STYLE_DATA_CONDITIONAL = [
    {
        'if': {'row_index': 'odd'},
        'backgroundColor': COLORS['bg_primary'],
    },
    {
        'if': {'state': 'selected'},
        'backgroundColor': COLORS['primary'],
        'border': f"1px solid {COLORS['primary']}",
    },
]


def paginated_table(
    df: pd.DataFrame,
    id: str,
    page_size: int = 20,
    max_rows: int = 10000,
    columns: List[str] = None,
    sortable: bool = True,
    filterable: bool = True,
) -> dash_table.DataTable:
    """
    Create a paginated DataTable with dark theme.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to display
    id : str
        Component ID for callbacks
    page_size : int, default=20
        Rows per page
    max_rows : int, default=10000
        Maximum rows to load (for performance)
    columns : list of str, optional
        Columns to display. If None, shows all columns.
    sortable : bool, default=True
        Enable column sorting
    filterable : bool, default=True
        Enable column filtering
        
    Returns
    -------
    dash_table.DataTable
        Paginated table component
    """
    if df is None or len(df) == 0:
        return html.Div(
            "No data available",
            className="text-muted text-center p-4"
        )
    
    # Limit rows for performance
    if len(df) > max_rows:
        logger.warning(f"Truncating table from {len(df)} to {max_rows} rows")
        df = df.head(max_rows)
    
    # Select columns
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    
    # Reset index if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.columns = ['timestamp'] + list(df.columns[1:])
    
    # Format columns
    table_columns = []
    for col in df.columns:
        col_config = {
            'name': col.replace('_', ' ').title(),
            'id': col,
        }
        
        # Detect numeric columns for formatting
        if df[col].dtype in ['float64', 'float32']:
            col_config['type'] = 'numeric'
            col_config['format'] = {'specifier': ',.4f'}
        elif df[col].dtype in ['int64', 'int32']:
            col_config['type'] = 'numeric'
            col_config['format'] = {'specifier': ',d'}
        
        table_columns.append(col_config)
    
    return dash_table.DataTable(
        id=id,
        columns=table_columns,
        data=df.to_dict('records'),
        page_size=page_size,
        page_action='native',
        sort_action='native' if sortable else 'none',
        filter_action='native' if filterable else 'none',
        style_header=TABLE_STYLE_HEADER,
        style_cell=TABLE_STYLE_CELL,
        style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
        style_table={
            'overflowX': 'auto',
            'minWidth': '100%',
        },
        style_filter={
            'backgroundColor': COLORS['bg_tertiary'],
            'color': COLORS['text'],
        },
    )


def simple_table(
    data: List[Dict[str, Any]],
    columns: List[str] = None,
    striped: bool = True,
    hover: bool = True,
) -> dbc.Table:
    """
    Create a simple Bootstrap table.
    
    Parameters
    ----------
    data : list of dict
        List of row dictionaries
    columns : list of str, optional
        Column names. If None, uses keys from first row.
    striped : bool, default=True
        Enable striped rows
    hover : bool, default=True
        Enable hover effect
        
    Returns
    -------
    dbc.Table
        Bootstrap table component
    """
    if not data:
        return html.Div(
            "No data available",
            className="text-muted text-center p-4"
        )
    
    if columns is None:
        columns = list(data[0].keys())
    
    # Header
    header = html.Thead(html.Tr([
        html.Th(col.replace('_', ' ').title())
        for col in columns
    ]))
    
    # Body
    rows = []
    for row in data:
        cells = [html.Td(row.get(col, '')) for col in columns]
        rows.append(html.Tr(cells))
    
    body = html.Tbody(rows)
    
    return dbc.Table(
        [header, body],
        striped=striped,
        hover=hover,
        dark=True,
        responsive=True,
        className="mb-0",
    )


def stats_table(
    stats: Dict[str, Any],
    title: str = None,
) -> dbc.Card:
    """
    Create a statistics table card.
    
    Parameters
    ----------
    stats : dict
        Dictionary of stat_name: value pairs
    title : str, optional
        Card title
        
    Returns
    -------
    dbc.Card
        Card with stats table
    """
    rows = []
    for key, value in stats.items():
        # Format value
        if isinstance(value, float):
            if abs(value) < 1:
                formatted = f"{value:.4f}"
            else:
                formatted = f"{value:,.2f}"
        else:
            formatted = str(value)
        
        rows.append(html.Tr([
            html.Td(key.replace('_', ' ').title(), className="text-muted"),
            html.Td(formatted, className="text-end"),
        ]))
    
    table = dbc.Table(
        html.Tbody(rows),
        dark=True,
        size="sm",
        className="mb-0",
    )
    
    if title:
        return dbc.Card([
            dbc.CardHeader(title),
            dbc.CardBody(table, className="p-0"),
        ])
    
    return table
