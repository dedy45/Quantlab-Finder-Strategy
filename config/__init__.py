"""
Configuration Module for Quant Lab.

Centralized configuration management using YAML files.
Supports environment-specific configs and runtime overrides.

Usage:
    from config import get_config, BacktestSettings, TradingSettings
    
    # Get full config
    cfg = get_config()
    print(cfg.backtest.initial_capital)
    
    # Get specific settings
    bt_settings = BacktestSettings.load()
"""

from .settings import (
    get_config,
    load_config,
    save_config,
    QuantLabConfig,
    BacktestSettings,
    TradingSettings,
    DataSettings,
    RiskSettings,
)

__all__ = [
    'get_config',
    'load_config',
    'save_config',
    'QuantLabConfig',
    'BacktestSettings',
    'TradingSettings',
    'DataSettings',
    'RiskSettings',
]
