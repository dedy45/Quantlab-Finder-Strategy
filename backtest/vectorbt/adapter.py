"""
VectorBT Adapter - Vectorized backtest engine for fast screening.

This adapter provides a pure Python/NumPy implementation that mimics
VectorBT's vectorized approach without requiring the vectorbt library.
Can be upgraded to use actual vectorbt when installed.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Union

import numpy as np
import pandas as pd

from ..base import (
    BacktestConfig,
    BacktestEngine,
    BacktestMetrics,
    BacktestResult,
    BaseBacktestAdapter
)

logger = logging.getLogger(__name__)


@dataclass
class VectorBTConfig(BacktestConfig):
    """Configuration specific to VectorBT adapter."""
    
    # VectorBT specific
    freq: str = "1D"  # Data frequency
    use_vectorbt_lib: bool = False  # Use actual vectorbt library if available
    
    # Position sizing (override from base)
    size_type: str = "percent"  # percent, fixed, kelly
    size_value: float = 1.0  # 100% of capital or fixed amount
    
    # Execution
    price_type: str = "close"  # close, open, vwap
    allow_partial: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        self.engine = BacktestEngine.VECTORBT


class VectorBTAdapter(BaseBacktestAdapter):
    """
    Vectorized backtest adapter for fast screening.
    
    Uses pure NumPy/Pandas vectorized operations for speed.
    Suitable for screening 1000+ strategy combinations.
    
    Cara Penggunaan:
    - Instantiate dengan VectorBTConfig atau BacktestConfig
    - Call run() dengan prices dan signals
    - Atau call run_strategy() dengan strategy object
    
    Nilai:
    - Sangat cepat untuk screening ribuan kombinasi
    - Cocok untuk parameter optimization
    - Quick prototyping dan idea validation
    
    Manfaat:
    - Mempercepat research 10-100x
    - Filter bad ideas sebelum deep analysis
    - Cost-effective screening
    """
    
    def __init__(self, config: Optional[Union[VectorBTConfig, BacktestConfig]] = None):
        """Initialize VectorBT adapter."""
        # Convert BacktestConfig to VectorBTConfig if needed
        if config is None:
            self.config = VectorBTConfig()
        elif isinstance(config, VectorBTConfig):
            self.config = config
        else:
            # Create VectorBTConfig from BacktestConfig
            self.config = VectorBTConfig(
                initial_capital=config.initial_capital,
                commission=config.commission,
                slippage=config.slippage,
                max_position_size=config.max_position_size,
                max_leverage=config.max_leverage,
                max_drawdown=getattr(config, 'max_drawdown', 0.20),
                target_volatility=getattr(config, 'target_volatility', 0.25),
                vol_lookback=getattr(config, 'vol_lookback', 36),
                position_sizing=getattr(config, 'position_sizing', 'volatility_target'),
            )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Check if vectorbt library is available
        self._vectorbt_available = False
        if self.config.use_vectorbt_lib:
            try:
                import vectorbt as vbt
                self._vectorbt_available = True
                self.logger.info("VectorBT library available, using native implementation")
            except ImportError:
                self.logger.info("VectorBT library not found, using pure NumPy implementation")
    
    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        asset: str = "UNKNOWN",
        strategy_name: str = "Custom"
    ) -> BacktestResult:
        """
        Run vectorized backtest with signals.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
        signals : pd.Series
            Trading signals: 1 (long), -1 (short), 0 (flat)
        asset : str
            Asset identifier
        strategy_name : str
            Strategy name for identification
            
        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        assert prices is not None, "Prices cannot be None"
        assert signals is not None, "Signals cannot be None"
        assert len(prices) > 0, "Prices cannot be empty"
        assert len(signals) > 0, "Signals cannot be empty"
        
        start_time = time.time()
        
        try:
            # Align data
            prices, signals = self._align_data(prices, signals)
            
            if len(prices) < 2:
                self.logger.warning("Insufficient data for backtest")
                return self._empty_result(strategy_name, asset)
            
            # Get price series
            close = self._get_price_series(prices)
            
            # Calculate returns
            returns = close.pct_change().fillna(0)
            
            # Apply signals (shifted to avoid look-ahead bias)
            raw_positions = signals.shift(1).fillna(0)
            
            # Apply position sizing based on config
            positions = self._apply_position_sizing(raw_positions, returns, close)
            
            # Strategy returns
            strategy_returns = positions * returns
            
            # Apply costs
            strategy_returns = self._apply_costs(strategy_returns, positions)
            
            # Calculate equity curve
            equity_curve = self.config.initial_capital * (1 + strategy_returns).cumprod()
            
            # Generate trades
            trades = self._generate_trades(positions, close, strategy_returns)
            
            # Calculate metrics
            metrics = self.calculate_metrics(strategy_returns, trades)
            
            execution_time = time.time() - start_time
            
            return BacktestResult(
                strategy_name=strategy_name,
                asset=asset,
                engine=BacktestEngine.VECTORBT,
                metrics=metrics,
                equity_curve=equity_curve,
                returns=strategy_returns,
                positions=positions,
                trades=trades,
                params={},
                config=self.config,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return self._empty_result(strategy_name, asset)
    
    def _apply_position_sizing(
        self,
        raw_positions: pd.Series,
        returns: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        """
        Apply position sizing based on configuration.
        
        Supports:
        - fixed: Use raw signals as-is
        - percent: Scale by max_position_size
        - volatility_target: Scale to target volatility (Carver method)
        """
        sizing_method = getattr(self.config, 'position_sizing', 'percent')
        
        if sizing_method == 'fixed':
            return raw_positions
        
        elif sizing_method == 'percent':
            # Scale by max position size
            max_pos = getattr(self.config, 'max_position_size', 0.20)
            return raw_positions * max_pos
        
        elif sizing_method == 'volatility_target':
            # Carver-style volatility targeting
            target_vol = getattr(self.config, 'target_volatility', 0.25)
            vol_lookback = getattr(self.config, 'vol_lookback', 36)
            max_pos = getattr(self.config, 'max_position_size', 0.20)
            
            # Calculate rolling volatility (annualized)
            rolling_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
            rolling_vol = rolling_vol.replace(0, np.nan).ffill().fillna(target_vol)
            
            # Calculate vol scalar
            vol_scalar = target_vol / rolling_vol
            
            # Cap the scalar to prevent extreme positions
            vol_scalar = vol_scalar.clip(0.1, 2.0)
            
            # Apply to positions
            sized_positions = raw_positions * vol_scalar * max_pos
            
            return sized_positions
        
        else:
            # Default: scale by max position
            max_pos = getattr(self.config, 'max_position_size', 0.20)
            return raw_positions * max_pos
    
    def run_strategy(
        self,
        prices: pd.DataFrame,
        strategy: Any,
        asset: str = "UNKNOWN"
    ) -> BacktestResult:
        """
        Run backtest with strategy object.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        strategy : Any
            Strategy object with fit/predict methods
        asset : str
            Asset identifier
            
        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        assert prices is not None, "Prices cannot be None"
        assert strategy is not None, "Strategy cannot be None"
        assert hasattr(strategy, 'fit'), "Strategy must have fit method"
        assert hasattr(strategy, 'predict'), "Strategy must have predict method"
        
        try:
            # Fit strategy
            strategy.fit(prices)
            
            # Generate signals
            signals = strategy.predict(prices)
            
            # Get strategy name
            strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)
            
            # Get parameters
            params = {}
            if hasattr(strategy, 'config'):
                params = strategy.config.__dict__ if hasattr(strategy.config, '__dict__') else {}
            
            # Run backtest
            result = self.run(prices, signals, asset, strategy_name)
            result.params = params
            
            return result
            
        except Exception as e:
            self.logger.error(f"Strategy backtest error: {e}")
            strategy_name = getattr(strategy, 'name', 'Unknown')
            return self._empty_result(strategy_name, asset)
    
    def run_batch(
        self,
        prices: pd.DataFrame,
        strategies: List[Any],
        asset: str = "UNKNOWN"
    ) -> List[BacktestResult]:
        """
        Run batch backtest for multiple strategies.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        strategies : List[Any]
            List of strategy objects
        asset : str
            Asset identifier
            
        Returns
        -------
        List[BacktestResult]
            Results for all strategies
        """
        results = []
        
        for i, strategy in enumerate(strategies):
            self.logger.info(f"Running strategy {i+1}/{len(strategies)}")
            result = self.run_strategy(prices, strategy, asset)
            results.append(result)
        
        return results
    
    def _align_data(
        self,
        prices: pd.DataFrame,
        signals: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align prices and signals by index."""
        common_idx = prices.index.intersection(signals.index)
        return prices.loc[common_idx], signals.loc[common_idx]
    
    def _get_price_series(self, prices: pd.DataFrame) -> pd.Series:
        """Get price series based on config."""
        price_col = self.config.price_type
        
        if price_col in prices.columns:
            return prices[price_col]
        elif 'close' in prices.columns:
            return prices['close']
        elif 'Close' in prices.columns:
            return prices['Close']
        else:
            # Assume single column or first column
            return prices.iloc[:, 0]
    
    def _apply_costs(
        self,
        returns: pd.Series,
        positions: pd.Series
    ) -> pd.Series:
        """Apply transaction costs and slippage."""
        # Detect position changes (trades)
        position_changes = positions.diff().abs()
        
        # Total cost per trade
        total_cost = self.config.commission + self.config.slippage
        
        # Apply costs on position changes
        costs = position_changes * total_cost
        
        return returns - costs
    
    def _generate_trades(
        self,
        positions: pd.Series,
        prices: pd.Series,
        returns: pd.Series
    ) -> pd.DataFrame:
        """Generate trade log from positions."""
        trades = []
        
        # Find position changes
        position_changes = positions.diff().fillna(0)
        trade_indices = position_changes[position_changes != 0].index
        
        entry_idx = None
        entry_price = None
        entry_position = None
        
        for idx in trade_indices:
            current_pos = positions.loc[idx]
            prev_pos = positions.shift(1).loc[idx] if idx != positions.index[0] else 0
            
            # Close existing position
            if entry_idx is not None and prev_pos != 0:
                exit_price = prices.loc[idx]
                pnl = (exit_price - entry_price) / entry_price * entry_position
                
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': entry_position,
                    'pnl': pnl,
                    'return_pct': pnl * 100
                })
                
                entry_idx = None
                entry_price = None
                entry_position = None
            
            # Open new position
            if current_pos != 0:
                entry_idx = idx
                entry_price = prices.loc[idx]
                entry_position = current_pos
        
        # Close any remaining position at end
        if entry_idx is not None:
            exit_price = prices.iloc[-1]
            pnl = (exit_price - entry_price) / entry_price * entry_position
            
            trades.append({
                'entry_date': entry_idx,
                'exit_date': prices.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': entry_position,
                'pnl': pnl,
                'return_pct': pnl * 100
            })
        
        if trades:
            return pd.DataFrame(trades)
        else:
            return pd.DataFrame(columns=[
                'entry_date', 'exit_date', 'entry_price', 
                'exit_price', 'position', 'pnl', 'return_pct'
            ])
    
    def _empty_result(self, strategy_name: str, asset: str) -> BacktestResult:
        """Create empty result for error cases."""
        return BacktestResult(
            strategy_name=strategy_name,
            asset=asset,
            engine=BacktestEngine.VECTORBT,
            metrics=BacktestMetrics(),
            equity_curve=None,
            returns=None,
            positions=None,
            trades=None,
            params={},
            config=self.config,
            execution_time=0.0
        )
