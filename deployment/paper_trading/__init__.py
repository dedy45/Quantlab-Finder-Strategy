"""
Paper Trading Module for FASE 5: Production.

Simulates live trading without real money to validate strategy
performance before Alpha Streams submission.

Components:
- PaperTrader: Simulates order execution with realistic slippage
- TradeLogger: Records all trades for analysis
- PerformanceComparer: Compares paper vs backtest performance

Version: 0.6.3
"""

from .paper_trader import PaperTrader, PaperTradingConfig, TradeRecord
from .trade_logger import TradeLogger, TradeLogEntry
from .performance_comparer import PerformanceComparer, ComparisonResult

__all__ = [
    'PaperTrader',
    'PaperTradingConfig',
    'TradeRecord',
    'TradeLogger',
    'TradeLogEntry',
    'PerformanceComparer',
    'ComparisonResult',
]
