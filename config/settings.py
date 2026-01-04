"""
Settings Module - Centralized configuration management.

Loads configuration from YAML files with support for:
- Default values
- Environment overrides
- Runtime modifications

Version: 0.7.0
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Paths
CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = CONFIG_DIR / "default.yaml"
USER_CONFIG_FILE = CONFIG_DIR / "user.yaml"


@dataclass
class BacktestSettings:
    """Backtest configuration settings."""
    
    # Capital
    initial_capital: float = 100000.0
    currency: str = "USD"
    
    # Costs
    commission_pct: float = 0.001      # 0.1% per trade
    slippage_pct: float = 0.0005       # 0.05% slippage
    spread_pct: float = 0.0001         # 0.01% spread
    
    # Position sizing
    max_position_pct: float = 0.20     # Max 20% per position
    max_leverage: float = 1.0          # No leverage by default
    position_sizing: str = "volatility_target"  # fixed, percent, volatility_target, kelly
    
    # Risk limits
    max_drawdown_pct: float = 0.20     # Stop at 20% drawdown
    daily_loss_limit_pct: float = 0.05 # Stop at 5% daily loss
    
    # Volatility targeting (Carver method)
    target_volatility_pct: float = 0.25  # 25% annual target vol
    vol_lookback: int = 36               # 36 bars for vol estimation
    
    # Execution
    fill_price: str = "close"          # close, open, vwap
    allow_shorting: bool = True
    
    @classmethod
    def load(cls, config_dict: Optional[Dict] = None) -> 'BacktestSettings':
        """Load from dictionary or defaults."""
        if config_dict:
            return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
        return cls()


@dataclass
class TradingSettings:
    """Live/Paper trading settings."""
    
    # Capital
    initial_capital: float = 10000.0
    currency: str = "USD"
    
    # Costs (more conservative for live)
    commission_pct: float = 0.002      # 0.2% per trade
    slippage_pct: float = 0.001        # 0.1% slippage
    
    # Position sizing
    max_position_pct: float = 0.10     # Max 10% per position (conservative)
    max_leverage: float = 1.0
    position_sizing: str = "volatility_target"
    
    # Risk limits (stricter for live)
    max_drawdown_pct: float = 0.15     # Stop at 15% drawdown
    daily_loss_limit_pct: float = 0.03 # Stop at 3% daily loss
    
    # Volatility targeting
    target_volatility_pct: float = 0.15  # 15% annual (more conservative)
    vol_lookback: int = 36
    
    # Execution
    order_type: str = "market"         # market, limit
    timeout_seconds: int = 30
    retry_count: int = 3
    
    @classmethod
    def load(cls, config_dict: Optional[Dict] = None) -> 'TradingSettings':
        """Load from dictionary or defaults."""
        if config_dict:
            return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
        return cls()


@dataclass
class DataSettings:
    """Data management settings."""
    
    # Storage
    primary_storage: str = "arcticdb"  # arcticdb, parquet
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    
    # ArcticDB
    arcticdb_path: str = "data/arcticdb"
    
    # Default data source
    default_source: str = "dukascopy"  # dukascopy, quantiacs, yahoo
    
    # Default symbol and timeframe
    default_symbol: str = "XAUUSD"
    default_timeframe: str = "1H"
    available_timeframes: List[str] = field(default_factory=lambda: ["15T", "1H", "4H", "1D"])
    
    # Data quality
    min_bars_required: int = 1000
    max_missing_pct: float = 0.05      # Max 5% missing data
    
    @classmethod
    def load(cls, config_dict: Optional[Dict] = None) -> 'DataSettings':
        """Load from dictionary or defaults."""
        if config_dict:
            # Handle list fields specially
            result = cls()
            for k, v in config_dict.items():
                if hasattr(result, k):
                    setattr(result, k, v)
            return result
        return cls()


@dataclass
class RiskSettings:
    """Risk management settings."""
    
    # Portfolio level
    max_portfolio_drawdown: float = 0.20
    max_correlation: float = 0.70      # Max correlation between strategies
    min_diversification: int = 3       # Min number of uncorrelated strategies
    
    # Position level
    max_single_position: float = 0.20  # Max 20% in single position
    stop_loss_pct: float = 0.02        # 2% stop loss per trade
    take_profit_pct: float = 0.04      # 4% take profit (2:1 R:R)
    
    # Kelly sizing
    kelly_fraction: float = 0.5        # Half-Kelly for safety
    min_kelly_trades: int = 30         # Min trades for Kelly calculation
    
    # Drawdown control
    dd_reduction_threshold: float = 0.10  # Start reducing at 10% DD
    dd_reduction_factor: float = 0.5      # Reduce position by 50%
    
    @classmethod
    def load(cls, config_dict: Optional[Dict] = None) -> 'RiskSettings':
        """Load from dictionary or defaults."""
        if config_dict:
            return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
        return cls()


@dataclass
class StrategySettings:
    """Strategy-specific settings."""
    
    # Momentum
    momentum_fast_period: int = 20
    momentum_slow_period: int = 50
    
    # Mean Reversion
    mr_lookback: int = 20
    mr_entry_z: float = 2.0
    mr_exit_z: float = 0.5
    
    # EWMAC (Carver)
    ewmac_variations: List[tuple] = field(default_factory=lambda: [
        (8, 32), (16, 64), (32, 128), (64, 256)
    ])
    
    # Regime detection
    hurst_window: int = 100
    vol_regime_window: int = 20
    
    @classmethod
    def load(cls, config_dict: Optional[Dict] = None) -> 'StrategySettings':
        """Load from dictionary or defaults."""
        if config_dict:
            result = cls()
            for k, v in config_dict.items():
                if hasattr(result, k):
                    setattr(result, k, v)
            return result
        return cls()


@dataclass
class ValidationSettings:
    """Validation and testing settings."""
    
    # PSR/DSR targets
    min_psr: float = 0.95              # 95% PSR required
    min_dsr: float = 1.0               # DSR > 1.0 required
    
    # Walk-forward
    wf_n_splits: int = 5
    wf_train_ratio: float = 0.8
    wf_purge_gap: int = 5
    max_sharpe_degradation: float = 0.30  # Max 30% degradation
    
    # Robustness
    robustness_n_simulations: int = 100
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    
    # Screening
    screening_min_sharpe: float = 0.5
    screening_max_dd: float = 0.30
    screening_min_trades: int = 10
    
    @classmethod
    def load(cls, config_dict: Optional[Dict] = None) -> 'ValidationSettings':
        """Load from dictionary or defaults."""
        if config_dict:
            result = cls()
            for k, v in config_dict.items():
                if hasattr(result, k):
                    setattr(result, k, v)
            return result
        return cls()


@dataclass
class QuantLabConfig:
    """Main configuration container."""
    
    # Version
    version: str = "0.7.0"
    
    # Sub-configs
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    data: DataSettings = field(default_factory=DataSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version': self.version,
            'backtest': asdict(self.backtest),
            'trading': asdict(self.trading),
            'data': asdict(self.data),
            'risk': asdict(self.risk),
            'strategy': asdict(self.strategy),
            'validation': asdict(self.validation),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantLabConfig':
        """Create from dictionary."""
        return cls(
            version=data.get('version', '0.7.0'),
            backtest=BacktestSettings.load(data.get('backtest')),
            trading=TradingSettings.load(data.get('trading')),
            data=DataSettings.load(data.get('data')),
            risk=RiskSettings.load(data.get('risk')),
            strategy=StrategySettings.load(data.get('strategy')),
            validation=ValidationSettings.load(data.get('validation')),
        )


# Global config instance
_config: Optional[QuantLabConfig] = None


def load_config(config_file: Optional[Path] = None) -> QuantLabConfig:
    """
    Load configuration from YAML file.
    
    Priority:
    1. Specified config file
    2. User config (config/user.yaml)
    3. Default config (config/default.yaml)
    4. Hardcoded defaults
    """
    global _config
    
    # Try loading from file
    config_data = {}
    
    files_to_try = []
    if config_file:
        files_to_try.append(config_file)
    files_to_try.extend([USER_CONFIG_FILE, DEFAULT_CONFIG_FILE])
    
    for f in files_to_try:
        if f.exists():
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    loaded = yaml.safe_load(fp)
                    if loaded:
                        # Merge with existing (later files override)
                        config_data = _deep_merge(config_data, loaded)
                        logger.info(f"[OK] Loaded config from {f}")
            except Exception as e:
                logger.warning(f"[WARN] Failed to load {f}: {e}")
    
    # Create config
    if config_data:
        _config = QuantLabConfig.from_dict(config_data)
    else:
        _config = QuantLabConfig()
        logger.info("[INFO] Using default configuration")
    
    return _config


def get_config() -> QuantLabConfig:
    """Get current configuration (load if not loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def save_config(config: QuantLabConfig, config_file: Optional[Path] = None) -> None:
    """Save configuration to YAML file."""
    target = config_file or USER_CONFIG_FILE
    
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target, 'w', encoding='utf-8') as fp:
            yaml.dump(config.to_dict(), fp, default_flow_style=False, sort_keys=False)
        
        logger.info(f"[OK] Saved config to {target}")
        
    except Exception as e:
        logger.error(f"[FAIL] Failed to save config: {e}")
        raise


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result
