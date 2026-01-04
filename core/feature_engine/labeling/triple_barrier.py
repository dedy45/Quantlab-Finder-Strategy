"""
Triple-Barrier Labeling Method.

Creates labels based on which barrier is touched first:
1. Upper barrier (take profit) → Label = 1
2. Lower barrier (stop loss) → Label = -1
3. Vertical barrier (time expiry) → Label = sign of return

This method is superior to fixed-horizon labeling because:
- Incorporates path dependency
- Accounts for volatility (dynamic barriers)
- More realistic for trading

Reference:
- Protokol Kausalitas - Fase 1 (Sebab Fundamental)
- Lopez de Prado - Advances in Financial Machine Learning, Chapter 3
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum


class BarrierType(Enum):
    """Type of barrier that was touched."""
    UPPER = "upper"      # Take profit
    LOWER = "lower"      # Stop loss
    VERTICAL = "vertical"  # Time expiry
    NONE = "none"        # No barrier touched


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    # Barrier widths (as multiple of volatility)
    pt_sl_ratio: float = 1.0  # Profit-take / Stop-loss ratio
    upper_barrier: float = 1.0  # Upper barrier (in vol units)
    lower_barrier: float = 1.0  # Lower barrier (in vol units)
    
    # Time parameters
    max_holding_period: int = 10  # Maximum bars to hold
    min_return: float = 0.0  # Minimum return for vertical barrier
    
    # Volatility parameters
    vol_lookback: int = 20  # Lookback for volatility estimation
    vol_type: str = 'std'  # 'std', 'atr', 'parkinson'
    
    # Label options
    binary_labels: bool = False  # True: {0,1}, False: {-1,0,1}
    side_prediction: Optional[pd.Series] = None  # Primary model prediction


@dataclass
class TripleBarrierResult:
    """Result of triple-barrier labeling."""
    labels: pd.Series
    returns: pd.Series
    barrier_touched: pd.Series
    touch_time: pd.Series
    metadata: dict = field(default_factory=dict)
    
    @property
    def n_samples(self) -> int:
        return len(self.labels)
    
    @property
    def label_distribution(self) -> dict:
        return self.labels.value_counts().to_dict()
    
    def __str__(self) -> str:
        dist = self.label_distribution
        return (
            f"TripleBarrierResult:\n"
            f"  Samples: {self.n_samples}\n"
            f"  Distribution: {dist}"
        )


def get_daily_volatility(
    close: pd.Series,
    lookback: int = 20,
    vol_type: str = 'std'
) -> pd.Series:
    """
    Estimate daily volatility.
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    lookback : int
        Lookback window
    vol_type : str
        'std': Standard deviation of returns
        'ewm': Exponentially weighted
        
    Returns
    -------
    pd.Series
        Daily volatility estimate
    """
    returns = close.pct_change()
    
    if vol_type == 'ewm':
        vol = returns.ewm(span=lookback).std()
    else:
        vol = returns.rolling(window=lookback).std()
    
    return vol


def get_vertical_barrier(
    timestamps: pd.DatetimeIndex,
    max_holding_period: int
) -> pd.Series:
    """
    Get vertical barrier timestamps.
    
    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Event timestamps
    max_holding_period : int
        Maximum holding period in bars
        
    Returns
    -------
    pd.Series
        Vertical barrier timestamps
    """
    # For each timestamp, find the timestamp max_holding_period bars later
    vertical_barriers = pd.Series(index=timestamps, dtype='datetime64[ns]')
    
    for i, t in enumerate(timestamps):
        if i + max_holding_period < len(timestamps):
            vertical_barriers.iloc[i] = timestamps[i + max_holding_period]
        else:
            vertical_barriers.iloc[i] = timestamps[-1]
    
    return vertical_barriers


def get_horizontal_barriers(
    close: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    upper_width: float = 1.0,
    lower_width: float = 1.0,
    pt_sl_ratio: float = 1.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Get horizontal barrier levels.
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    events : pd.DatetimeIndex
        Event timestamps
    volatility : pd.Series
        Volatility estimates
    upper_width : float
        Upper barrier width (vol multiplier)
    lower_width : float
        Lower barrier width (vol multiplier)
    pt_sl_ratio : float
        Profit-take / Stop-loss ratio
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (upper_barriers, lower_barriers) as price levels
    """
    # Get volatility at event times
    vol_at_events = volatility.reindex(events).ffill()
    
    # Get prices at event times
    price_at_events = close.reindex(events).ffill()
    
    # Calculate barrier levels
    upper_barriers = price_at_events * (1 + upper_width * vol_at_events * pt_sl_ratio)
    lower_barriers = price_at_events * (1 - lower_width * vol_at_events)
    
    return upper_barriers, lower_barriers


def apply_triple_barrier_single(
    close: pd.Series,
    entry_time: pd.Timestamp,
    upper_barrier: float,
    lower_barrier: float,
    vertical_barrier: pd.Timestamp,
    min_return: float = 0.0
) -> Tuple[int, float, BarrierType, pd.Timestamp]:
    """
    Apply triple-barrier to a single event.
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    entry_time : pd.Timestamp
        Entry timestamp
    upper_barrier : float
        Upper barrier price level
    lower_barrier : float
        Lower barrier price level
    vertical_barrier : pd.Timestamp
        Vertical barrier timestamp
    min_return : float
        Minimum return for vertical barrier label
        
    Returns
    -------
    Tuple[int, float, BarrierType, pd.Timestamp]
        (label, return, barrier_type, touch_time)
    """
    # Get price path from entry to vertical barrier
    path = close.loc[entry_time:vertical_barrier]
    
    if len(path) < 2:
        return 0, 0.0, BarrierType.NONE, entry_time
    
    entry_price = path.iloc[0]
    
    # Check each bar for barrier touch
    for t, price in path.iloc[1:].items():
        # Upper barrier touched (take profit)
        if price >= upper_barrier:
            ret = (price - entry_price) / entry_price
            return 1, ret, BarrierType.UPPER, t
        
        # Lower barrier touched (stop loss)
        if price <= lower_barrier:
            ret = (price - entry_price) / entry_price
            return -1, ret, BarrierType.LOWER, t
    
    # Vertical barrier reached (time expiry)
    exit_price = path.iloc[-1]
    ret = (exit_price - entry_price) / entry_price
    
    # Label based on return at expiry
    if abs(ret) < min_return:
        label = 0
    else:
        label = 1 if ret > 0 else -1
    
    return label, ret, BarrierType.VERTICAL, vertical_barrier


def apply_triple_barrier(
    close: pd.Series,
    events: Optional[pd.DatetimeIndex] = None,
    config: Optional[TripleBarrierConfig] = None
) -> TripleBarrierResult:
    """
    Apply triple-barrier labeling to price series.
    
    Parameters
    ----------
    close : pd.Series
        Close prices with DatetimeIndex
    events : pd.DatetimeIndex, optional
        Event timestamps. If None, uses all timestamps.
    config : TripleBarrierConfig, optional
        Configuration. If None, uses defaults.
        
    Returns
    -------
    TripleBarrierResult
        Labeling results
    """
    config = config or TripleBarrierConfig()
    
    # Use all timestamps if events not specified
    if events is None:
        events = close.index[config.vol_lookback:-config.max_holding_period]
    
    # Calculate volatility
    volatility = get_daily_volatility(
        close, 
        lookback=config.vol_lookback,
        vol_type=config.vol_type
    )
    
    # Get barriers
    vertical_barriers = get_vertical_barrier(events, config.max_holding_period)
    upper_barriers, lower_barriers = get_horizontal_barriers(
        close, events, volatility,
        upper_width=config.upper_barrier,
        lower_width=config.lower_barrier,
        pt_sl_ratio=config.pt_sl_ratio
    )
    
    # Apply to each event
    labels = []
    returns = []
    barrier_types = []
    touch_times = []
    
    for event_time in events:
        if event_time not in upper_barriers.index:
            continue
            
        label, ret, barrier_type, touch_time = apply_triple_barrier_single(
            close,
            entry_time=event_time,
            upper_barrier=upper_barriers.loc[event_time],
            lower_barrier=lower_barriers.loc[event_time],
            vertical_barrier=vertical_barriers.loc[event_time],
            min_return=config.min_return
        )
        
        # Apply side prediction if provided
        if config.side_prediction is not None:
            if event_time in config.side_prediction.index:
                side = config.side_prediction.loc[event_time]
                if side == 0:
                    label = 0
                elif side * label < 0:  # Wrong direction
                    label = 0
        
        labels.append(label)
        returns.append(ret)
        barrier_types.append(barrier_type.value)
        touch_times.append(touch_time)
    
    # Create result series
    result_index = events[:len(labels)]
    
    labels_series = pd.Series(labels, index=result_index, name='label')
    returns_series = pd.Series(returns, index=result_index, name='return')
    barrier_series = pd.Series(barrier_types, index=result_index, name='barrier')
    touch_series = pd.Series(touch_times, index=result_index, name='touch_time')
    
    # Convert to binary if requested
    if config.binary_labels:
        labels_series = (labels_series > 0).astype(int)
    
    return TripleBarrierResult(
        labels=labels_series,
        returns=returns_series,
        barrier_touched=barrier_series,
        touch_time=touch_series,
        metadata={
            'config': {
                'pt_sl_ratio': config.pt_sl_ratio,
                'upper_barrier': config.upper_barrier,
                'lower_barrier': config.lower_barrier,
                'max_holding_period': config.max_holding_period,
                'vol_lookback': config.vol_lookback,
            },
            'n_events': len(labels),
            'label_distribution': labels_series.value_counts().to_dict()
        }
    )


class TripleBarrierLabeler:
    """
    Triple-Barrier Labeling for ML models.
    
    Creates labels based on which barrier is touched first:
    - Upper barrier (profit) → 1
    - Lower barrier (loss) → -1
    - Vertical barrier (time) → sign(return)
    
    Examples
    --------
    >>> labeler = TripleBarrierLabeler(
    ...     pt_sl_ratio=2.0,  # 2:1 reward/risk
    ...     max_holding_period=10
    ... )
    >>> result = labeler.fit_transform(price_df)
    >>> print(result.label_distribution)
    """
    
    def __init__(
        self,
        pt_sl_ratio: float = 1.0,
        upper_barrier: float = 1.0,
        lower_barrier: float = 1.0,
        max_holding_period: int = 10,
        vol_lookback: int = 20,
        vol_type: str = 'std',
        min_return: float = 0.0,
        binary_labels: bool = False,
    ):
        """
        Initialize triple-barrier labeler.
        
        Parameters
        ----------
        pt_sl_ratio : float
            Profit-take / Stop-loss ratio (e.g., 2.0 for 2:1)
        upper_barrier : float
            Upper barrier width in volatility units
        lower_barrier : float
            Lower barrier width in volatility units
        max_holding_period : int
            Maximum holding period in bars
        vol_lookback : int
            Lookback for volatility estimation
        vol_type : str
            Volatility type: 'std' or 'ewm'
        min_return : float
            Minimum return for vertical barrier label
        binary_labels : bool
            If True, output {0, 1} instead of {-1, 0, 1}
        """
        self.config = TripleBarrierConfig(
            pt_sl_ratio=pt_sl_ratio,
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            max_holding_period=max_holding_period,
            vol_lookback=vol_lookback,
            vol_type=vol_type,
            min_return=min_return,
            binary_labels=binary_labels,
        )
        
        self._result: Optional[TripleBarrierResult] = None
    
    def fit(self, data: pd.DataFrame) -> 'TripleBarrierLabeler':
        """
        Fit labeler (no-op, for API consistency).
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data with 'close' column
            
        Returns
        -------
        TripleBarrierLabeler
            Self
        """
        return self
    
    def transform(
        self,
        data: pd.DataFrame,
        events: Optional[pd.DatetimeIndex] = None,
        side_prediction: Optional[pd.Series] = None
    ) -> TripleBarrierResult:
        """
        Generate labels for data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data with 'close' column
        events : pd.DatetimeIndex, optional
            Event timestamps to label
        side_prediction : pd.Series, optional
            Primary model predictions for meta-labeling
            
        Returns
        -------
        TripleBarrierResult
            Labeling results
        """
        # Get close prices
        if 'close' in data.columns:
            close = data['close']
        else:
            close = data.iloc[:, 0]
        
        # Ensure datetime index
        if not isinstance(close.index, pd.DatetimeIndex):
            close.index = pd.to_datetime(close.index)
        
        # Update config with side prediction
        config = TripleBarrierConfig(
            pt_sl_ratio=self.config.pt_sl_ratio,
            upper_barrier=self.config.upper_barrier,
            lower_barrier=self.config.lower_barrier,
            max_holding_period=self.config.max_holding_period,
            vol_lookback=self.config.vol_lookback,
            vol_type=self.config.vol_type,
            min_return=self.config.min_return,
            binary_labels=self.config.binary_labels,
            side_prediction=side_prediction,
        )
        
        self._result = apply_triple_barrier(close, events, config)
        return self._result
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        events: Optional[pd.DatetimeIndex] = None,
        side_prediction: Optional[pd.Series] = None
    ) -> TripleBarrierResult:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data, events, side_prediction)
    
    def get_labels(self) -> pd.Series:
        """Get labels from last transform."""
        if self._result is None:
            raise ValueError("No labels generated. Call transform() first.")
        return self._result.labels
    
    def get_returns(self) -> pd.Series:
        """Get returns from last transform."""
        if self._result is None:
            raise ValueError("No labels generated. Call transform() first.")
        return self._result.returns
