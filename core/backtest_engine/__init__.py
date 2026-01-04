"""
Backtest Engine - Quantiacs native backtesting.

Uses qnt.backtester for realistic backtesting with:
- Slippage modeling
- Transaction costs
- Position limits
- Multi-asset support
"""

from .quantiacs_backtest import QuantiacsBacktester, backtest_strategy

__all__ = ['QuantiacsBacktester', 'backtest_strategy']
