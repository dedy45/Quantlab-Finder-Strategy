"""
Deployment Module - Platform adapters and monitoring.

FASE 4: Deployment
- Quantiacs adapter (qnt.backtester)
- QuantConnect adapter (LEAN algorithm)
- Performance monitoring and decay detection

FASE 5: Production
- Paper trading simulation
- Trade logging and audit
- Performance comparison (paper vs backtest)
"""

from .base import (
    BaseAdapter,
    DeploymentConfig,
    DeploymentResult,
    PlatformType,
)

from .paper_trading import (
    PaperTrader,
    PaperTradingConfig,
    TradeRecord,
    TradeLogger,
    TradeLogEntry,
    PerformanceComparer,
    ComparisonResult,
)

__all__ = [
    # Base
    'BaseAdapter',
    'DeploymentConfig',
    'DeploymentResult',
    'PlatformType',
    # Paper Trading
    'PaperTrader',
    'PaperTradingConfig',
    'TradeRecord',
    'TradeLogger',
    'TradeLogEntry',
    'PerformanceComparer',
    'ComparisonResult',
]
