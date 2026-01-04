"""
Alert System.

Generates alerts for anomalies and risk events.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"[{self.level.value.upper()}] {self.message}"


@dataclass
class AlertRule:
    """Rule for generating alerts."""
    
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'gte', 'lte'
    level: AlertLevel
    message_template: str
    
    def check(self, value: float) -> bool:
        """Check if rule is triggered."""
        if self.comparison == 'gt':
            return value > self.threshold
        elif self.comparison == 'lt':
            return value < self.threshold
        elif self.comparison == 'gte':
            return value >= self.threshold
        elif self.comparison == 'lte':
            return value <= self.threshold
        return False


class AlertSystem:
    """
    Alert system for strategy monitoring.
    
    Generates alerts based on configurable rules.
    
    Parameters
    ----------
    strategy_name : str
        Name of the strategy being monitored
    """
    
    # Default alert rules
    DEFAULT_RULES = [
        AlertRule(
            name='max_drawdown',
            metric='drawdown',
            threshold=0.15,
            comparison='lte',  # drawdown is negative
            level=AlertLevel.WARNING,
            message_template='Drawdown reached {value:.2%} (threshold: {threshold:.2%})'
        ),
        AlertRule(
            name='critical_drawdown',
            metric='drawdown',
            threshold=0.20,
            comparison='lte',
            level=AlertLevel.CRITICAL,
            message_template='CRITICAL: Drawdown {value:.2%} exceeds limit {threshold:.2%}'
        ),
        AlertRule(
            name='sharpe_degradation',
            metric='sharpe_ratio',
            threshold=0.5,
            comparison='lt',
            level=AlertLevel.WARNING,
            message_template='Sharpe ratio {value:.2f} below threshold {threshold:.2f}'
        ),
        AlertRule(
            name='high_correlation',
            metric='correlation',
            threshold=0.5,
            comparison='gt',
            level=AlertLevel.WARNING,
            message_template='High correlation with benchmark: {value:.2f}'
        ),
        AlertRule(
            name='volatility_spike',
            metric='volatility',
            threshold=0.30,
            comparison='gt',
            level=AlertLevel.WARNING,
            message_template='Volatility spike: {value:.2%} (threshold: {threshold:.2%})'
        ),
    ]
    
    def __init__(self, strategy_name: str):
        assert strategy_name, "strategy_name cannot be empty"
        
        self.strategy_name = strategy_name
        self.rules: List[AlertRule] = self.DEFAULT_RULES.copy()
        self.alerts: List[Alert] = []
        self.callbacks: List[Callable[[Alert], None]] = []
        
    def add_rule(self, rule: AlertRule) -> None:
        """Add custom alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback function for alerts."""
        self.callbacks.append(callback)
    
    def check(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check metrics against all rules.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Current metrics to check
            
        Returns
        -------
        List[Alert]
            List of triggered alerts
        """
        assert metrics is not None, "Metrics cannot be None"
        
        triggered = []
        
        for rule in self.rules:
            if rule.metric not in metrics:
                continue
            
            value = metrics[rule.metric]
            
            if rule.check(value):
                alert = Alert(
                    level=rule.level,
                    message=rule.message_template.format(
                        value=value,
                        threshold=rule.threshold
                    ),
                    strategy_name=self.strategy_name,
                    metric_name=rule.metric,
                    metric_value=value,
                    threshold=rule.threshold,
                )
                
                triggered.append(alert)
                self.alerts.append(alert)
                
                # Log alert
                if rule.level == AlertLevel.CRITICAL:
                    logger.critical(str(alert))
                elif rule.level == AlertLevel.WARNING:
                    logger.warning(str(alert))
                else:
                    logger.info(str(alert))
                
                # Execute callbacks
                for callback in self.callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
        
        return triggered
    
    def check_returns(self, returns: pd.Series) -> List[Alert]:
        """
        Check returns series for anomalies.
        
        Parameters
        ----------
        returns : pd.Series
            Return series to check
            
        Returns
        -------
        List[Alert]
            List of triggered alerts
        """
        assert returns is not None, "Returns cannot be None"
        assert len(returns) > 0, "Returns cannot be empty"
        
        # Calculate metrics from returns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = ((cumulative - running_max) / running_max).iloc[-1]
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        metrics = {
            'drawdown': drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
        }
        
        return self.check(metrics)
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        since: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Get historical alerts.
        
        Parameters
        ----------
        level : AlertLevel, optional
            Filter by alert level
        since : datetime, optional
            Filter alerts after this time
            
        Returns
        -------
        List[Alert]
            Filtered alerts
        """
        alerts = self.alerts
        
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def clear_alerts(self) -> None:
        """Clear all historical alerts."""
        self.alerts = []
        logger.info(f"Cleared alerts for {self.strategy_name}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        assert self.strategy_name, "Strategy name must be set"
        
        return {
            'strategy_name': self.strategy_name,
            'total_alerts': len(self.alerts),
            'by_level': {
                'info': len([a for a in self.alerts if a.level == AlertLevel.INFO]),
                'warning': len([a for a in self.alerts if a.level == AlertLevel.WARNING]),
                'critical': len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            },
            'n_rules': len(self.rules),
        }
