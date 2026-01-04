"""
Performance Comparer - Compares paper trading vs backtest performance.

Provides:
- Statistical comparison of returns
- Deviation analysis
- Recommendation engine
- Visualization data

Version: 0.6.3
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DEVIATION_THRESHOLD = 0.20  # 20% deviation
DEFAULT_MIN_OBSERVATIONS = 20


@dataclass
class ComparisonResult:
    """Result of performance comparison."""
    
    # Core metrics
    paper_sharpe: float
    backtest_sharpe: float
    sharpe_deviation: float
    
    paper_return: float
    backtest_return: float
    return_deviation: float
    
    paper_volatility: float
    backtest_volatility: float
    volatility_deviation: float
    
    paper_max_dd: float
    backtest_max_dd: float
    dd_deviation: float
    
    # Statistical tests
    correlation: float
    t_statistic: float
    p_value: float
    
    # Assessment
    is_acceptable: bool
    deviation_score: float  # 0-1, lower is better
    recommendation: str
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'paper_sharpe': self.paper_sharpe,
            'backtest_sharpe': self.backtest_sharpe,
            'sharpe_deviation': self.sharpe_deviation,
            'paper_return': self.paper_return,
            'backtest_return': self.backtest_return,
            'return_deviation': self.return_deviation,
            'paper_volatility': self.paper_volatility,
            'backtest_volatility': self.backtest_volatility,
            'volatility_deviation': self.volatility_deviation,
            'paper_max_dd': self.paper_max_dd,
            'backtest_max_dd': self.backtest_max_dd,
            'dd_deviation': self.dd_deviation,
            'correlation': self.correlation,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'is_acceptable': self.is_acceptable,
            'deviation_score': self.deviation_score,
            'recommendation': self.recommendation,
            'warnings': self.warnings,
        }
    
    def get_summary(self) -> str:
        """Get summary string."""
        status = "[OK]" if self.is_acceptable else "[WARN]"
        
        lines = [
            f"\n{'='*60}",
            f"PERFORMANCE COMPARISON {status}",
            f"{'='*60}",
            f"",
            f"SHARPE RATIO:",
            f"  Paper:    {self.paper_sharpe:.2f}",
            f"  Backtest: {self.backtest_sharpe:.2f}",
            f"  Deviation: {self.sharpe_deviation:+.1%}",
            f"",
            f"RETURNS (Annualized):",
            f"  Paper:    {self.paper_return:.2%}",
            f"  Backtest: {self.backtest_return:.2%}",
            f"  Deviation: {self.return_deviation:+.1%}",
            f"",
            f"VOLATILITY (Annualized):",
            f"  Paper:    {self.paper_volatility:.2%}",
            f"  Backtest: {self.backtest_volatility:.2%}",
            f"  Deviation: {self.volatility_deviation:+.1%}",
            f"",
            f"MAX DRAWDOWN:",
            f"  Paper:    {self.paper_max_dd:.2%}",
            f"  Backtest: {self.backtest_max_dd:.2%}",
            f"  Deviation: {self.dd_deviation:+.1%}",
            f"",
            f"STATISTICAL:",
            f"  Correlation: {self.correlation:.2f}",
            f"  T-statistic: {self.t_statistic:.2f}",
            f"  P-value: {self.p_value:.4f}",
            f"",
            f"ASSESSMENT:",
            f"  Deviation Score: {self.deviation_score:.2f} (lower is better)",
            f"  Acceptable: {'Yes' if self.is_acceptable else 'No'}",
            f"",
            f"RECOMMENDATION:",
            f"  {self.recommendation}",
        ]
        
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)


class PerformanceComparer:
    """
    Compare paper trading performance with backtest.
    
    Analyzes deviation between paper trading and backtest
    to determine if strategy is ready for live deployment.
    
    Parameters
    ----------
    deviation_threshold : float, default=0.20
        Maximum acceptable deviation (20%)
    min_observations : int, default=20
        Minimum observations for comparison
        
    Examples
    --------
    >>> comparer = PerformanceComparer()
    >>> result = comparer.compare(paper_returns, backtest_returns)
    >>> print(result.get_summary())
    """
    
    def __init__(
        self,
        deviation_threshold: float = DEFAULT_DEVIATION_THRESHOLD,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS
    ):
        """Initialize comparer."""
        assert 0 < deviation_threshold < 1, "deviation_threshold must be in (0, 1)"
        assert min_observations > 0, "min_observations must be positive"
        
        self.deviation_threshold = deviation_threshold
        self.min_observations = min_observations
        
        logger.info(
            f"PerformanceComparer initialized: "
            f"threshold={deviation_threshold:.0%}, "
            f"min_obs={min_observations}"
        )
    
    def compare(
        self,
        paper_returns: pd.Series,
        backtest_returns: pd.Series,
        paper_equity: Optional[pd.Series] = None,
        backtest_equity: Optional[pd.Series] = None
    ) -> ComparisonResult:
        """
        Compare paper trading with backtest performance.
        
        Parameters
        ----------
        paper_returns : pd.Series
            Paper trading daily returns
        backtest_returns : pd.Series
            Backtest daily returns
        paper_equity : pd.Series, optional
            Paper trading equity curve
        backtest_equity : pd.Series, optional
            Backtest equity curve
            
        Returns
        -------
        ComparisonResult
            Comparison result with metrics and recommendation
        """
        assert paper_returns is not None, "paper_returns cannot be None"
        assert backtest_returns is not None, "backtest_returns cannot be None"
        assert len(paper_returns) > 0, "paper_returns cannot be empty"
        assert len(backtest_returns) > 0, "backtest_returns cannot be empty"
        
        warnings = []
        
        # Check minimum observations
        if len(paper_returns) < self.min_observations:
            warnings.append(
                f"Paper trading has only {len(paper_returns)} observations "
                f"(minimum: {self.min_observations})"
            )
        
        logger.info(
            f"Comparing performance: "
            f"paper={len(paper_returns)} obs, "
            f"backtest={len(backtest_returns)} obs"
        )
        
        try:
            # Calculate metrics
            paper_metrics = self._calculate_metrics(paper_returns, paper_equity)
            bt_metrics = self._calculate_metrics(backtest_returns, backtest_equity)
            
            # Calculate deviations
            sharpe_dev = self._calc_deviation(
                paper_metrics['sharpe'], bt_metrics['sharpe']
            )
            return_dev = self._calc_deviation(
                paper_metrics['annual_return'], bt_metrics['annual_return']
            )
            vol_dev = self._calc_deviation(
                paper_metrics['volatility'], bt_metrics['volatility']
            )
            dd_dev = self._calc_deviation(
                paper_metrics['max_dd'], bt_metrics['max_dd']
            )
            
            # Statistical comparison
            correlation, t_stat, p_value = self._statistical_comparison(
                paper_returns, backtest_returns
            )
            
            # Calculate deviation score (weighted average)
            deviation_score = self._calculate_deviation_score(
                sharpe_dev, return_dev, vol_dev, dd_dev
            )
            
            # Check acceptability
            is_acceptable = self._check_acceptable(
                sharpe_dev, return_dev, vol_dev, dd_dev, warnings
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                is_acceptable, deviation_score, sharpe_dev, 
                paper_metrics, bt_metrics, warnings
            )
            
            result = ComparisonResult(
                paper_sharpe=paper_metrics['sharpe'],
                backtest_sharpe=bt_metrics['sharpe'],
                sharpe_deviation=sharpe_dev,
                paper_return=paper_metrics['annual_return'],
                backtest_return=bt_metrics['annual_return'],
                return_deviation=return_dev,
                paper_volatility=paper_metrics['volatility'],
                backtest_volatility=bt_metrics['volatility'],
                volatility_deviation=vol_dev,
                paper_max_dd=paper_metrics['max_dd'],
                backtest_max_dd=bt_metrics['max_dd'],
                dd_deviation=dd_dev,
                correlation=correlation,
                t_statistic=t_stat,
                p_value=p_value,
                is_acceptable=is_acceptable,
                deviation_score=deviation_score,
                recommendation=recommendation,
                warnings=warnings
            )
            
            if is_acceptable:
                logger.info(f"Performance comparison: ACCEPTABLE (score={deviation_score:.2f})")
            else:
                logger.warning(f"Performance comparison: NOT ACCEPTABLE (score={deviation_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        n_days = len(returns)
        
        # Sharpe ratio (annualized)
        std = returns.std()
        sharpe = returns.mean() / std * np.sqrt(252) if std > 0 else 0.0
        
        # Annual return
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
        
        # Volatility (annualized)
        volatility = std * np.sqrt(252)
        
        # Max drawdown
        if equity is not None:
            max_dd = self._calculate_max_dd(equity)
        else:
            # Calculate from returns
            cumulative = (1 + returns).cumprod()
            max_dd = self._calculate_max_dd(cumulative)
        
        return {
            'sharpe': sharpe,
            'annual_return': annual_return,
            'volatility': volatility,
            'max_dd': max_dd,
            'n_days': n_days,
        }
    
    def _calculate_max_dd(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min())
    
    def _calc_deviation(self, paper: float, backtest: float) -> float:
        """Calculate percentage deviation."""
        if backtest == 0:
            return 0.0 if paper == 0 else 1.0
        return (paper - backtest) / abs(backtest)
    
    def _statistical_comparison(
        self,
        paper: pd.Series,
        backtest: pd.Series
    ) -> Tuple[float, float, float]:
        """Perform statistical comparison."""
        # Correlation (if same length, use aligned)
        min_len = min(len(paper), len(backtest))
        paper_aligned = paper.iloc[-min_len:]
        bt_aligned = backtest.iloc[-min_len:]
        
        try:
            correlation = paper_aligned.corr(bt_aligned)
            if np.isnan(correlation):
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # T-test for difference in means
        try:
            t_stat, p_value = stats.ttest_ind(paper, backtest, equal_var=False)
        except Exception:
            t_stat, p_value = 0.0, 1.0
        
        return correlation, t_stat, p_value
    
    def _calculate_deviation_score(
        self,
        sharpe_dev: float,
        return_dev: float,
        vol_dev: float,
        dd_dev: float
    ) -> float:
        """
        Calculate overall deviation score.
        
        Weighted average of absolute deviations.
        Lower is better (0 = perfect match).
        """
        weights = {
            'sharpe': 0.40,
            'return': 0.30,
            'volatility': 0.15,
            'drawdown': 0.15,
        }
        
        score = (
            weights['sharpe'] * abs(sharpe_dev) +
            weights['return'] * abs(return_dev) +
            weights['volatility'] * abs(vol_dev) +
            weights['drawdown'] * abs(dd_dev)
        )
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _check_acceptable(
        self,
        sharpe_dev: float,
        return_dev: float,
        vol_dev: float,
        dd_dev: float,
        warnings: List[str]
    ) -> bool:
        """Check if deviations are acceptable."""
        threshold = self.deviation_threshold
        
        # Check each metric
        if abs(sharpe_dev) > threshold:
            warnings.append(f"Sharpe deviation ({sharpe_dev:+.1%}) exceeds threshold")
        
        if abs(return_dev) > threshold * 1.5:  # More lenient for returns
            warnings.append(f"Return deviation ({return_dev:+.1%}) exceeds threshold")
        
        if abs(vol_dev) > threshold:
            warnings.append(f"Volatility deviation ({vol_dev:+.1%}) exceeds threshold")
        
        if dd_dev > threshold:  # Only warn if DD is worse
            warnings.append(f"Drawdown deviation ({dd_dev:+.1%}) exceeds threshold")
        
        # Acceptable if Sharpe deviation is within threshold
        # and no critical warnings
        return abs(sharpe_dev) <= threshold
    
    def _generate_recommendation(
        self,
        is_acceptable: bool,
        deviation_score: float,
        sharpe_dev: float,
        paper_metrics: Dict,
        bt_metrics: Dict,
        warnings: List[str]
    ) -> str:
        """Generate recommendation based on comparison."""
        if is_acceptable and deviation_score < 0.10:
            return (
                "EXCELLENT: Paper trading closely matches backtest. "
                "Strategy is ready for live deployment."
            )
        
        if is_acceptable and deviation_score < 0.20:
            return (
                "GOOD: Paper trading within acceptable range. "
                "Consider extending paper trading period for more confidence."
            )
        
        if is_acceptable:
            return (
                "ACCEPTABLE: Paper trading shows some deviation but within limits. "
                "Monitor closely during initial live trading."
            )
        
        # Not acceptable
        if sharpe_dev < -0.30:
            return (
                "NOT READY: Significant Sharpe degradation detected. "
                "Review strategy parameters and market conditions. "
                "Do not proceed to live trading."
            )
        
        if paper_metrics['max_dd'] > bt_metrics['max_dd'] * 1.5:
            return (
                "NOT READY: Drawdown significantly worse than backtest. "
                "Review risk management and position sizing. "
                "Extend paper trading period."
            )
        
        return (
            "NOT READY: Performance deviates significantly from backtest. "
            "Investigate causes before proceeding. "
            f"Warnings: {len(warnings)}"
        )
    
    def quick_compare(
        self,
        paper_sharpe: float,
        backtest_sharpe: float,
        paper_return: float,
        backtest_return: float
    ) -> Tuple[bool, str]:
        """
        Quick comparison without full returns data.
        
        Parameters
        ----------
        paper_sharpe : float
            Paper trading Sharpe ratio
        backtest_sharpe : float
            Backtest Sharpe ratio
        paper_return : float
            Paper trading return
        backtest_return : float
            Backtest return
            
        Returns
        -------
        Tuple[bool, str]
            (is_acceptable, recommendation)
        """
        sharpe_dev = self._calc_deviation(paper_sharpe, backtest_sharpe)
        return_dev = self._calc_deviation(paper_return, backtest_return)
        
        is_acceptable = abs(sharpe_dev) <= self.deviation_threshold
        
        if is_acceptable:
            recommendation = f"Acceptable: Sharpe deviation {sharpe_dev:+.1%}"
        else:
            recommendation = f"Not acceptable: Sharpe deviation {sharpe_dev:+.1%}"
        
        return is_acceptable, recommendation
