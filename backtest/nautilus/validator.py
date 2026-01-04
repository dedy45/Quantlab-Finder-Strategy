"""
Candidate Validator - Validate Top 50 candidates from VectorBT screening.

Phase 2 of 4-Phase Research Workflow:
- Input: Top 50 candidates from VectorBT
- Filter: Selisih < 10% dari VectorBT
- Output: Top 10 candidates
- Time: ~2-3 hours
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BacktestMetrics, BacktestResult
from ..vectorbt import VectorBTAdapter
from .adapter import NautilusAdapter, NautilusConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for candidate validation."""
    
    # Deviation thresholds
    max_sharpe_deviation: float = 0.10  # 10% max deviation
    max_return_deviation: float = 0.15  # 15% max deviation
    max_drawdown_deviation: float = 0.20  # 20% max deviation
    
    # Filter criteria (same as screening but stricter)
    min_sharpe: float = 0.5
    max_drawdown: float = 0.25  # Stricter than screening
    min_trades: int = 15
    
    # Output
    top_n: int = 10
    
    # Execution
    verbose: bool = True
    
    # Nautilus config
    nautilus_config: Optional[NautilusConfig] = None


@dataclass
class ValidationResult:
    """Result from candidate validation."""
    
    # Identification
    strategy_name: str
    asset: str
    params: Dict[str, Any]
    
    # VectorBT results
    vectorbt_metrics: BacktestMetrics
    
    # Nautilus results
    nautilus_metrics: BacktestMetrics
    
    # Deviation analysis
    sharpe_deviation: float = 0.0
    return_deviation: float = 0.0
    drawdown_deviation: float = 0.0
    
    # Validation status
    passed: bool = False
    rejection_reason: Optional[str] = None
    
    # Score (0-100)
    score: float = 0.0
    
    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationResult({self.strategy_name}, {status}, "
            f"Score={self.score:.1f}, "
            f"SharpeDeviation={self.sharpe_deviation:.1%})"
        )


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    
    # Counts
    total_validated: int = 0
    total_passed: int = 0
    
    # Results
    all_results: List[ValidationResult] = field(default_factory=list)
    passed_results: List[ValidationResult] = field(default_factory=list)
    top_results: List[ValidationResult] = field(default_factory=list)
    
    # Timing
    execution_time: float = 0.0
    
    # Config
    config: Optional[ValidationConfig] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.all_results:
            return pd.DataFrame()
        
        rows = []
        for r in self.all_results:
            rows.append({
                'strategy': r.strategy_name,
                'asset': r.asset,
                'passed': r.passed,
                'score': r.score,
                'vbt_sharpe': r.vectorbt_metrics.sharpe_ratio,
                'naut_sharpe': r.nautilus_metrics.sharpe_ratio,
                'sharpe_dev': r.sharpe_deviation,
                'vbt_return': r.vectorbt_metrics.total_return,
                'naut_return': r.nautilus_metrics.total_return,
                'return_dev': r.return_deviation,
                'vbt_maxdd': r.vectorbt_metrics.max_drawdown,
                'naut_maxdd': r.nautilus_metrics.max_drawdown,
                'dd_dev': r.drawdown_deviation,
                'rejection': r.rejection_reason,
                'params': str(r.params)
            })
        
        return pd.DataFrame(rows)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "VALIDATION RESULTS (Phase 2)",
            "=" * 60,
            f"Total validated: {self.total_validated}",
            f"Total passed: {self.total_passed}",
            f"Top candidates: {len(self.top_results)}",
            f"Execution time: {self.execution_time:.1f}s",
            "-" * 60
        ]
        
        if self.top_results:
            lines.append("TOP 10 VALIDATED CANDIDATES:")
            for i, r in enumerate(self.top_results[:10], 1):
                lines.append(
                    f"  {i}. {r.strategy_name}: "
                    f"Score={r.score:.1f}, "
                    f"VBT_SR={r.vectorbt_metrics.sharpe_ratio:.2f}, "
                    f"Naut_SR={r.nautilus_metrics.sharpe_ratio:.2f}, "
                    f"Dev={r.sharpe_deviation:.1%}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)


class CandidateValidator:
    """
    Validate candidates from VectorBT screening using Nautilus.
    
    Compares VectorBT results with Nautilus results to ensure
    strategies are robust and not artifacts of simplified backtesting.
    
    Cara Penggunaan:
    - Instantiate dengan ValidationConfig
    - Call validate_candidates() dengan prices dan VectorBT results
    - Atau call validate_strategy() untuk single strategy
    
    Nilai:
    - Validasi realistis dengan event-driven backtest
    - Deteksi strategi yang tidak robust
    - Filter berdasarkan deviation dari VectorBT
    
    Manfaat:
    - Confidence sebelum Phase 3 (Deep Analysis)
    - Menghindari false positives dari VectorBT
    - Top 10 candidates yang robust
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize validator."""
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize adapters
        self.vectorbt = VectorBTAdapter()
        
        # Load defaults from config if nautilus_config not provided
        if self.config.nautilus_config is None:
            try:
                from config import get_config
                cfg = get_config()
                nautilus_config = NautilusConfig(
                    slippage=cfg.backtest.slippage_pct,
                    commission=cfg.backtest.commission_pct,
                    slippage_model="volatility",
                    volatility_slippage_factor=0.1
                )
            except Exception:
                # Fallback defaults
                nautilus_config = NautilusConfig(
                    slippage=0.0005,
                    commission=0.001,
                    slippage_model="volatility",
                    volatility_slippage_factor=0.1
                )
        else:
            nautilus_config = self.config.nautilus_config
        
        self.nautilus = NautilusAdapter(nautilus_config)
    
    def validate_candidates(
        self,
        prices: pd.DataFrame,
        vectorbt_results: List[BacktestResult],
        strategies: Optional[List[Any]] = None
    ) -> ValidationSummary:
        """
        Validate list of candidates from VectorBT screening.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data
        vectorbt_results : List[BacktestResult]
            Results from VectorBT screening
        strategies : List[Any], optional
            Strategy objects (if not provided, will use signals from results)
            
        Returns
        -------
        ValidationSummary
            Validation results with top candidates
        """
        assert prices is not None, "Prices cannot be None"
        assert vectorbt_results is not None, "VectorBT results cannot be None"
        
        start_time = time.time()
        all_results = []
        passed_results = []
        
        total = len(vectorbt_results)
        
        for i, vbt_result in enumerate(vectorbt_results):
            if self.config.verbose:
                self.logger.info(f"Validating {i+1}/{total}: {vbt_result.strategy_name}")
            
            try:
                # Get strategy or signals
                strategy = strategies[i] if strategies and i < len(strategies) else None
                
                # Run Nautilus backtest
                if strategy is not None:
                    naut_result = self.nautilus.run_strategy(
                        prices, strategy, vbt_result.asset
                    )
                elif vbt_result.positions is not None:
                    # Use positions as signals
                    naut_result = self.nautilus.run(
                        prices, vbt_result.positions,
                        vbt_result.asset, vbt_result.strategy_name
                    )
                else:
                    self.logger.warning(f"No strategy or signals for {vbt_result.strategy_name}")
                    continue
                
                # Validate
                validation = self._validate_result(vbt_result, naut_result)
                all_results.append(validation)
                
                if validation.passed:
                    passed_results.append(validation)
                    
            except Exception as e:
                self.logger.warning(f"Validation failed for {vbt_result.strategy_name}: {e}")
        
        # Sort by score
        passed_results.sort(key=lambda x: x.score, reverse=True)
        
        # Get top N
        top_results = passed_results[:self.config.top_n]
        
        execution_time = time.time() - start_time
        
        summary = ValidationSummary(
            total_validated=len(all_results),
            total_passed=len(passed_results),
            all_results=all_results,
            passed_results=passed_results,
            top_results=top_results,
            execution_time=execution_time,
            config=self.config
        )
        
        if self.config.verbose:
            self.logger.info(summary.summary())
        
        return summary
    
    def validate_strategy(
        self,
        prices: pd.DataFrame,
        strategy: Any,
        asset: str = "UNKNOWN"
    ) -> ValidationResult:
        """
        Validate single strategy.
        
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
        ValidationResult
            Validation result
        """
        # Run VectorBT
        vbt_result = self.vectorbt.run_strategy(prices, strategy, asset)
        
        # Run Nautilus
        naut_result = self.nautilus.run_strategy(prices, strategy, asset)
        
        # Validate
        return self._validate_result(vbt_result, naut_result)
    
    def _validate_result(
        self,
        vbt_result: BacktestResult,
        naut_result: BacktestResult
    ) -> ValidationResult:
        """Validate Nautilus result against VectorBT result."""
        vbt_metrics = vbt_result.metrics
        naut_metrics = naut_result.metrics
        
        # Calculate deviations
        sharpe_dev = self._calculate_deviation(
            vbt_metrics.sharpe_ratio, naut_metrics.sharpe_ratio
        )
        return_dev = self._calculate_deviation(
            vbt_metrics.total_return, naut_metrics.total_return
        )
        dd_dev = self._calculate_deviation(
            abs(vbt_metrics.max_drawdown), abs(naut_metrics.max_drawdown)
        )
        
        # Check pass criteria
        passed = True
        rejection_reason = None
        
        # Check deviation thresholds
        if sharpe_dev > self.config.max_sharpe_deviation:
            passed = False
            rejection_reason = f"Sharpe deviation too high: {sharpe_dev:.1%}"
        
        elif return_dev > self.config.max_return_deviation:
            passed = False
            rejection_reason = f"Return deviation too high: {return_dev:.1%}"
        
        elif dd_dev > self.config.max_drawdown_deviation:
            passed = False
            rejection_reason = f"Drawdown deviation too high: {dd_dev:.1%}"
        
        # Check absolute thresholds (using Nautilus metrics)
        elif naut_metrics.sharpe_ratio < self.config.min_sharpe:
            passed = False
            rejection_reason = f"Nautilus Sharpe too low: {naut_metrics.sharpe_ratio:.2f}"
        
        elif abs(naut_metrics.max_drawdown) > self.config.max_drawdown:
            passed = False
            rejection_reason = f"Nautilus MaxDD too high: {naut_metrics.max_drawdown:.1%}"
        
        elif naut_metrics.total_trades < self.config.min_trades:
            passed = False
            rejection_reason = f"Too few trades: {naut_metrics.total_trades}"
        
        # Calculate score (0-100)
        score = self._calculate_score(naut_metrics, sharpe_dev, return_dev, dd_dev)
        
        return ValidationResult(
            strategy_name=vbt_result.strategy_name,
            asset=vbt_result.asset,
            params=vbt_result.params,
            vectorbt_metrics=vbt_metrics,
            nautilus_metrics=naut_metrics,
            sharpe_deviation=sharpe_dev,
            return_deviation=return_dev,
            drawdown_deviation=dd_dev,
            passed=passed,
            rejection_reason=rejection_reason,
            score=score
        )
    
    def _calculate_deviation(self, vbt_value: float, naut_value: float) -> float:
        """Calculate relative deviation between VectorBT and Nautilus."""
        if vbt_value == 0:
            return abs(naut_value) if naut_value != 0 else 0.0
        
        return abs(naut_value - vbt_value) / abs(vbt_value)
    
    def _calculate_score(
        self,
        metrics: BacktestMetrics,
        sharpe_dev: float,
        return_dev: float,
        dd_dev: float
    ) -> float:
        """Calculate validation score (0-100)."""
        score = 0.0
        
        # Sharpe contribution (40 points max)
        if metrics.sharpe_ratio >= 2.0:
            score += 40
        elif metrics.sharpe_ratio >= 1.0:
            score += 30
        elif metrics.sharpe_ratio >= 0.5:
            score += 20
        elif metrics.sharpe_ratio > 0:
            score += 10
        
        # Drawdown contribution (20 points max)
        dd = abs(metrics.max_drawdown)
        if dd <= 0.10:
            score += 20
        elif dd <= 0.15:
            score += 15
        elif dd <= 0.20:
            score += 10
        elif dd <= 0.25:
            score += 5
        
        # Deviation penalty (40 points max, lower is better)
        avg_dev = (sharpe_dev + return_dev + dd_dev) / 3
        if avg_dev <= 0.05:
            score += 40
        elif avg_dev <= 0.10:
            score += 30
        elif avg_dev <= 0.15:
            score += 20
        elif avg_dev <= 0.20:
            score += 10
        
        return min(100, max(0, score))
