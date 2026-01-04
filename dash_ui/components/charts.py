"""
Chart Components untuk Dash UI.

Plotly chart wrappers dengan CYBORG theme dan auto LTTB downsampling.

Usage:
    from dash_ui.components.charts import candlestick_chart, equity_curve_chart
    
    fig = candlestick_chart(df, title="XAUUSD Price")
    fig = equity_curve_chart(equity_series, benchmark_series, title="Strategy Performance")
"""
import logging
from typing import Optional, List

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import COLORS, PLOTLY_LAYOUT, CHART_HEIGHT, CANDLESTICK_COLORS
from ..utils.downsampling import lttb_downsample, lttb_downsample_ohlc, auto_downsample

logger = logging.getLogger(__name__)

# Default target points for downsampling
DEFAULT_TARGET_POINTS = 2000


def candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    height: int = None,
    show_volume: bool = True,
    target_points: int = DEFAULT_TARGET_POINTS,
) -> go.Figure:
    """
    Create candlestick chart with auto LTTB downsampling.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume (optional)
    title : str, default="Price Chart"
        Chart title
    height : int, optional
        Chart height in pixels. Defaults to CHART_HEIGHT['large']
    show_volume : bool, default=True
        Whether to show volume subplot
    target_points : int, default=2000
        Target points for LTTB downsampling
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if df is None or len(df) == 0:
        return empty_chart("No data available", height=height)
    
    # Auto downsample if needed
    if len(df) > target_points:
        logger.info(f"Downsampling candlestick: {len(df)} -> {target_points}")
        df = lttb_downsample_ohlc(df, target_points)
    
    height = height or CHART_HEIGHT['large']
    has_volume = show_volume and 'volume' in df.columns
    
    if has_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )
    else:
        fig = go.Figure()
    
    # Candlestick trace
    candlestick = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing=dict(
            line=dict(color=COLORS['bullish']),
            fillcolor=COLORS['bullish'],
        ),
        decreasing=dict(
            line=dict(color=COLORS['bearish']),
            fillcolor=COLORS['bearish'],
        ),
    )
    
    if has_volume:
        fig.add_trace(candlestick, row=1, col=1)
        
        # Volume bars
        colors = [
            COLORS['bullish'] if df['close'].iloc[i] >= df['open'].iloc[i]
            else COLORS['bearish']
            for i in range(len(df))
        ]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5,
            ),
            row=2, col=1,
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig.add_trace(candlestick)
        fig.update_yaxes(title_text="Price")
    
    # Apply theme (exclude 'title' from PLOTLY_LAYOUT to avoid duplicate)
    layout_without_title = {k: v for k, v in PLOTLY_LAYOUT.items() if k != 'title'}
    fig.update_layout(
        **layout_without_title,
        title=dict(text=title, font=PLOTLY_LAYOUT.get('title', {}).get('font', {})),
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    
    return fig


def equity_curve_chart(
    equity: pd.Series,
    benchmark: pd.Series = None,
    title: str = "Equity Curve",
    height: int = None,
    target_points: int = DEFAULT_TARGET_POINTS,
) -> go.Figure:
    """
    Create equity curve line chart.
    
    Parameters
    ----------
    equity : pd.Series
        Strategy equity curve (indexed by date)
    benchmark : pd.Series, optional
        Benchmark equity curve for comparison
    title : str, default="Equity Curve"
        Chart title
    height : int, optional
        Chart height in pixels
    target_points : int, default=2000
        Target points for LTTB downsampling
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if equity is None or len(equity) == 0:
        return empty_chart("No equity data available", height=height)
    
    height = height or CHART_HEIGHT['large']
    fig = go.Figure()
    
    # Downsample if needed
    if len(equity) > target_points:
        equity_df = pd.DataFrame({'close': equity})
        equity_df = lttb_downsample(equity_df, target_points, y_column='close')
        equity = equity_df['close']
    
    # Strategy equity
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Strategy',
        line=dict(color=COLORS['primary'], width=2),
    ))
    
    # Benchmark if provided
    if benchmark is not None and len(benchmark) > 0:
        if len(benchmark) > target_points:
            bench_df = pd.DataFrame({'close': benchmark})
            bench_df = lttb_downsample(bench_df, target_points, y_column='close')
            benchmark = bench_df['close']
        
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color=COLORS['text_muted'], width=1, dash='dash'),
        ))
    
    # Apply theme (exclude 'title' and 'legend' from PLOTLY_LAYOUT to avoid duplicate)
    layout_without_conflicts = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('title', 'legend')}
    fig.update_layout(
        **layout_without_conflicts,
        title=dict(text=title, font=PLOTLY_LAYOUT.get('title', {}).get('font', {})),
        height=height,
        yaxis_title='Equity',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
    )
    
    return fig


def drawdown_chart(
    equity: pd.Series,
    title: str = "Drawdown",
    height: int = None,
    target_points: int = DEFAULT_TARGET_POINTS,
) -> go.Figure:
    """
    Create drawdown chart from equity curve.
    
    Parameters
    ----------
    equity : pd.Series
        Equity curve (indexed by date)
    title : str, default="Drawdown"
        Chart title
    height : int, optional
        Chart height in pixels
    target_points : int, default=2000
        Target points for LTTB downsampling
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if equity is None or len(equity) == 0:
        return empty_chart("No equity data available", height=height)
    
    height = height or CHART_HEIGHT['medium']
    
    # Calculate drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100  # As percentage
    
    # Downsample if needed
    if len(drawdown) > target_points:
        dd_df = pd.DataFrame({'close': drawdown})
        dd_df = lttb_downsample(dd_df, target_points, y_column='close')
        drawdown = dd_df['close']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        fillcolor=f"rgba({int(COLORS['danger'][1:3], 16)}, {int(COLORS['danger'][3:5], 16)}, {int(COLORS['danger'][5:7], 16)}, 0.3)",
        line=dict(color=COLORS['danger'], width=1),
        name='Drawdown',
    ))
    
    # Apply theme (exclude 'title' from PLOTLY_LAYOUT to avoid duplicate)
    layout_without_title = {k: v for k, v in PLOTLY_LAYOUT.items() if k != 'title'}
    fig.update_layout(
        **layout_without_title,
        title=dict(text=title, font=PLOTLY_LAYOUT.get('title', {}).get('font', {})),
        height=height,
        yaxis_title='Drawdown (%)',
        showlegend=False,
    )
    
    return fig


def returns_distribution_chart(
    returns: pd.Series,
    title: str = "Returns Distribution",
    height: int = None,
    bins: int = 50,
) -> go.Figure:
    """
    Create returns distribution histogram.
    
    Parameters
    ----------
    returns : pd.Series
        Returns series
    title : str, default="Returns Distribution"
        Chart title
    height : int, optional
        Chart height in pixels
    bins : int, default=50
        Number of histogram bins
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if returns is None or len(returns) == 0:
        return empty_chart("No returns data available", height=height)
    
    height = height or CHART_HEIGHT['medium']
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns.values * 100,  # Convert to percentage
        nbinsx=bins,
        marker_color=COLORS['primary'],
        opacity=0.7,
        name='Returns',
    ))
    
    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash='dash',
        line_color=COLORS['text_muted'],
    )
    
    # Add mean line
    mean_return = returns.mean() * 100
    fig.add_vline(
        x=mean_return,
        line_dash='solid',
        line_color=COLORS['success'] if mean_return > 0 else COLORS['danger'],
        annotation_text=f"Mean: {mean_return:.2f}%",
    )
    
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'title'},
        title=dict(text=title, font=PLOTLY_LAYOUT.get('title', {}).get('font', {})),
        height=height,
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        showlegend=False,
    )
    
    return fig


def rolling_metric_chart(
    metric: pd.Series,
    title: str = "Rolling Metric",
    height: int = None,
    target_points: int = DEFAULT_TARGET_POINTS,
    threshold: float = None,
) -> go.Figure:
    """
    Create rolling metric line chart.
    
    Parameters
    ----------
    metric : pd.Series
        Rolling metric values (indexed by date)
    title : str, default="Rolling Metric"
        Chart title
    height : int, optional
        Chart height in pixels
    target_points : int, default=2000
        Target points for LTTB downsampling
    threshold : float, optional
        Horizontal threshold line
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if metric is None or len(metric) == 0:
        return empty_chart("No metric data available", height=height)
    
    height = height or CHART_HEIGHT['medium']
    
    # Downsample if needed
    if len(metric) > target_points:
        metric_df = pd.DataFrame({'close': metric})
        metric_df = lttb_downsample(metric_df, target_points, y_column='close')
        metric = metric_df['close']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metric.index,
        y=metric.values,
        mode='lines',
        line=dict(color=COLORS['primary'], width=1),
        name=title,
    ))
    
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash='dash',
            line_color=COLORS['warning'],
            annotation_text=f"Threshold: {threshold}",
        )
    
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'title'},
        title=dict(text=title, font=PLOTLY_LAYOUT.get('title', {}).get('font', {})),
        height=height,
        showlegend=False,
    )
    
    return fig


def empty_chart(
    message: str = "No data available",
    height: int = None,
) -> go.Figure:
    """
    Create empty chart with centered message.
    
    Parameters
    ----------
    message : str, default="No data available"
        Message to display
    height : int, optional
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure with centered message
    """
    height = height or CHART_HEIGHT['medium']
    
    fig = go.Figure()
    
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS['text_muted']),
    )
    
    # Apply theme (exclude xaxis/yaxis to avoid duplicate)
    layout_base = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('title', 'xaxis', 'yaxis')}
    fig.update_layout(
        **layout_base,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    
    return fig


def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    height: int = None,
) -> go.Figure:
    """
    Create correlation heatmap.
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    title : str, default="Correlation Matrix"
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if corr_matrix is None or len(corr_matrix) == 0:
        return empty_chart("No correlation data available", height=height)
    
    height = height or CHART_HEIGHT['medium']
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>',
    ))
    
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'title'},
        title=dict(text=title, font=PLOTLY_LAYOUT.get('title', {}).get('font', {})),
        height=height,
    )
    
    return fig
