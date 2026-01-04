"""
Candidate Validator for FASE 5: Production.

Validates strategy candidates with PSR/DSR and robustness testing
to ensure they meet Alpha Streams requirements.

Version: 0.6.1
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Setup path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.validation_engine import calculate_sharpe, calculate_psr, calculate_dsr

logger = logging.getLogger(__name__)

# Constants
MIN_OBSERVATIONS = 60
PSR_THRESHOLD = 0.95
MAX_DRAWDOWN_THRESHOLD = 0.20
MAX_CORRELATION_THRESHOLD = 0.30
SHARPE_STABILITY_THRESHOLD = 0.30
RETURN_STABILITY_THRESHOLD = 0.50
SUBSAMPLE_STABILITY_THRESHOLD = 0.40
MIN_STABILITY_SCORE = 0.70


@dataclass
class ValidationResult:
    """Result of candidate validation."""
    
    strategy_name: str
    asset: str
    
    # Core metrics
    psr: float
    dsr: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Robustness metrics
    parameter_stability: float = 0.0
    noise_stability: float = 0.0
    subsample_stability: float = 0.0
    
    # Correlation metrics
    benchmark_correlation: float = 0.0
    
    # Validation flags
    passes_psr: bool = False
    passes_drawdown: bool = False
    passes_correlation: bool = False
    passes_robustness: bool = False
    
    # Overall
    is_valid: bool = False
    validation_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate validation flags."""
        self.passes_psr = self.psr >= PSR_THRESHOLD
        self.passes_drawdown = self.max_drawdown <= MAX_DRAWDOWN_THRESHOLD
        self.passes_correlation = abs(self.benchmark_correlation) <= MAX_CORRELATION_THRESHOLD
        self.passes_robustness = (
            self.parameter_stability >= MIN_STABILITY_SCORE and
            self.noise_stability >= MIN_STABILITY_SCORE and
            self.subsample_stability >= MIN_STABILITY_SCORE
        )
        
        self.is_valid = (
            self.passes_psr and
            self.passes_drawdown and
            self.passes_correlation
        )
    
    @property
    def score(self) -> float:
        """Calculate overall validation score (0-100)."""
        score = 0.0
        
        # PSR contribution (40 points)
        if self.psr >= 0.99:
            score += 40
        elif self.psr >= 0.95:
            score += 30
        elif self.psr >= 0.90:
            score += 20
        
        # Drawdown contribution (20 points)
        if self.max_drawdown <= 0.10:
            score += 20
        elif self.max_drawdown <= 0.15:
            score += 15
        elif self.max_drawdown <= 0.20:
            score += 10
        
        # Correlation contribution (20 points)
        if abs(self.benchmark_correlation) <= 0.10:
            score += 20
        elif abs(self.benchmark_correlation) <= 0.20:
            score += 15
        elif abs(self.benchmark_correlation) <= 0.30:
            score += 10
        
        # Robustness contribution (20 points)
        avg_stability = (
            self.parameter_stability +
            self.noise_stability +
            self.subsample_stability
        ) / 3
        score += avg_stability * 20
        
        return score


class CandidateValidator:
    """
    Validates strategy candidates for Alpha Streams submission.
    
    Performs comprehensive validation including:
    - PSR/DSR validation
    - Robustness testing (parameter, noise, subsample)
    - Correlation analysis
    - Drawdown analysis
    
    Parameters
    ----------
    min_psr : float
        Minimum PSR threshold (default: 0.95)
    max_drawdown : float
        Maximum drawdown threshold (default: 0.20)
    max_correlation : float
        Maximum benchmark correlation (default: 0.30)
    """
    
    def __init__(
        self,
        min_psr: float = 0.95,
        max_drawdown: float = 0.20,
        max_correlation: float = 0.30
    ):
        assert 0 < min_psr <= 1, "min_psr must be between 0 and 1"
        assert 0 < max_drawdown <= 1, "max_drawdown must be between 0 and 1"
        assert 0 < max_correlation <= 1, "max_correlation must be between 0 and 1"
        
        self.min_psr = min_psr
        self.max_drawdown = max_drawdown
        self.max_correlation = max_correlation
        self.results: List[ValidationResult] = []
    
    def validate(
        self,
        returns: pd.Series,
        strategy_name: str,
        asset: str,
        benchmark_returns: Optional[pd.Series] = None,
        n_robustness_tests: int = 100
    ) -> ValidationResult:
        """
        Validate a strategy candidate.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        strategy_name : str
            Name of strategy
        asset : str
            Asset symbol
        benchmark_returns : pd.Series, optional
            Benchmark returns for correlation
        n_robustness_tests : int
            Number of robustness tests
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        assert returns is not None, "Returns cannot be None"
        assert len(returns) >= MIN_OBSERVATIONS, f"Need at least {MIN_OBSERVATIONS} observations"
        
        # Clean returns
        returns = self._clean_returns(returns)
        
        if len(returns) < MIN_OBSERVATIONS:
            logger.error(f"Insufficient clean returns: {len(returns)}")
            return self._create_failed_result(strategy_name, asset, "Insufficient data")
        
        notes = []
        
        try:
            # Core metrics
            sharpe = calculate_sharpe(returns)
            psr = calculate_psr(returns)
            dsr = calculate_dsr(sharpe, n_trials=10)
            
            # Drawdown (vectorized with safe division)
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = np.divide(
                cumulative - rolling_max,
                rolling_max,
                out=np.zeros_like(cumulative.values, dtype=float),
                where=rolling_max.values != 0
            )
            max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            
            # Correlation with benchmark
            benchmark_corr = 0.0
            if benchmark_returns is not None:
                benchmark_corr = self._calculate_correlation(returns, benchmark_returns)
            
            # Robustness tests
            param_stability = self._test_parameter_stability(returns, n_robustness_tests)
            noise_stability = self._test_noise_stability(returns, n_robustness_tests)
            subsample_stability = self._test_subsample_stability(returns, n_robustness_tests)
            
            # Generate notes
            if psr >= PSR_THRESHOLD:
                notes.append(f"[OK] PSR={psr:.2%} meets threshold")
            else:
                notes.append(f"[FAIL] PSR={psr:.2%} below {PSR_THRESHOLD:.0%}")
            
            if max_dd <= MAX_DRAWDOWN_THRESHOLD:
                notes.append(f"[OK] MaxDD={max_dd:.2%} within limit")
            else:
                notes.append(f"[FAIL] MaxDD={max_dd:.2%} exceeds {MAX_DRAWDOWN_THRESHOLD:.0%}")
            
            if abs(benchmark_corr) <= MAX_CORRELATION_THRESHOLD:
                notes.append(f"[OK] Correlation={benchmark_corr:.2f} is low")
            else:
                notes.append(f"[WARN] Correlation={benchmark_corr:.2f} may be too high")
            
            result = ValidationResult(
                strategy_name=strategy_name,
                asset=asset,
                psr=psr,
                dsr=dsr,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                parameter_stability=param_stability,
                noise_stability=noise_stability,
                subsample_stability=subsample_stability,
                benchmark_correlation=benchmark_corr,
                validation_notes=notes
            )
            
            self.results.append(result)
            
            logger.info(
                f"Validated {strategy_name} on {asset}: "
                f"PSR={psr:.2%}, MaxDD={max_dd:.2%}, Valid={result.is_valid}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for {strategy_name} on {asset}: {e}")
            return self._create_failed_result(strategy_name, asset, str(e))
    
    def _clean_returns(self, returns: pd.Series) -> pd.Series:
        """
        Clean returns by removing inf/NaN.
        
        Parameters
        ----------
        returns : pd.Series
            Raw returns
            
        Returns
        -------
        pd.Series
            Cleaned returns
        """
        # Replace inf with NaN
        cleaned = returns.replace([np.inf, -np.inf], np.nan)
        
        # Drop NaN
        cleaned = cleaned.dropna()
        
        # Clip extreme values (50% daily return is suspicious)
        cleaned = cleaned.clip(-0.5, 0.5)
        
        return cleaned
    
    def _calculate_correlation(
        self,
        returns: pd.Series,
        benchmark: pd.Series
    ) -> float:
        """
        Calculate correlation with benchmark.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        benchmark : pd.Series
            Benchmark returns
            
        Returns
        -------
        float
            Correlation coefficient
        """
        try:
            aligned = pd.concat([returns, benchmark], axis=1).dropna()
            if len(aligned) < 30:
                return 0.0
            return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        except Exception:
            return 0.0
    
    def _create_failed_result(
        self,
        strategy_name: str,
        asset: str,
        reason: str
    ) -> ValidationResult:
        """Create a failed validation result."""
        return ValidationResult(
            strategy_name=strategy_name,
            asset=asset,
            psr=0.0,
            dsr=0.0,
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            validation_notes=[f"[FAIL] {reason}"]
        )
    
    def _test_parameter_stability(
        self,
        returns: pd.Series,
        n_tests: int
    ) -> float:
        """
        Test stability across parameter variations.
        
        Simulates parameter changes by adding small perturbations
        to returns and checking if Sharpe remains stable.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        n_tests : int
            Number of tests
            
        Returns
        -------
        float
            Stability score (0-1)
        """
        try:
            base_sharpe = calculate_sharpe(returns)
            stable_count = 0
            
            # Vectorized noise generation
            noise_matrix = np.random.normal(0, 0.001, (n_tests, len(returns)))
            
            for i in range(n_tests):
                perturbed = returns.values + noise_matrix[i]
                perturbed_sharpe = calculate_sharpe(pd.Series(perturbed))
                
                # Safe division for comparison
                denominator = max(abs(base_sharpe), 0.01)
                if abs(perturbed_sharpe - base_sharpe) / denominator < SHARPE_STABILITY_THRESHOLD:
                    stable_count += 1
            
            return stable_count / n_tests
            
        except Exception as e:
            logger.warning(f"Parameter stability test failed: {e}")
            return 0.5
    
    def _test_noise_stability(
        self,
        returns: pd.Series,
        n_tests: int
    ) -> float:
        """
        Test stability under noise injection.
        
        Adds random noise to returns and checks if
        strategy remains profitable.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        n_tests : int
            Number of tests
            
        Returns
        -------
        float
            Stability score (0-1)
        """
        try:
            base_total = (1 + returns).prod() - 1
            stable_count = 0
            
            # Noise level based on volatility
            noise_level = returns.std() * 0.01
            if noise_level == 0 or np.isnan(noise_level):
                noise_level = 0.001
            
            # Vectorized noise generation
            noise_matrix = np.random.normal(0, noise_level, (n_tests, len(returns)))
            
            for i in range(n_tests):
                noisy_returns = returns.values + noise_matrix[i]
                noisy_total = np.prod(1 + noisy_returns) - 1
                
                # Safe division for comparison
                denominator = max(abs(base_total), 0.01)
                if noisy_total > 0 and abs(noisy_total - base_total) / denominator < RETURN_STABILITY_THRESHOLD:
                    stable_count += 1
            
            return stable_count / n_tests
            
        except Exception as e:
            logger.warning(f"Noise stability test failed: {e}")
            return 0.5
    
    def _test_subsample_stability(
        self,
        returns: pd.Series,
        n_tests: int
    ) -> float:
        """
        Test stability across subsamples.
        
        Randomly samples 80% of data and checks if
        Sharpe remains consistent.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        n_tests : int
            Number of tests
            
        Returns
        -------
        float
            Stability score (0-1)
        """
        try:
            base_sharpe = calculate_sharpe(returns)
            stable_count = 0
            sample_size = int(len(returns) * 0.8)
            
            for _ in range(n_tests):
                # Random subsample (maintaining order for time-series)
                indices = np.random.choice(len(returns), sample_size, replace=False)
                indices.sort()
                subsample = returns.iloc[indices]
                
                subsample_sharpe = calculate_sharpe(subsample)
                
                # Safe division for comparison
                denominator = max(abs(base_sharpe), 0.01)
                if abs(subsample_sharpe - base_sharpe) / denominator < SUBSAMPLE_STABILITY_THRESHOLD:
                    stable_count += 1
            
            return stable_count / n_tests
            
        except Exception as e:
            logger.warning(f"Subsample stability test failed: {e}")
            return 0.5
    
    def validate_all(
        self,
        candidates: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series] = None
    ) -> List[ValidationResult]:
        """
        Validate multiple candidates.
        
        Parameters
        ----------
        candidates : List[Dict]
            List of candidates with keys: returns, strategy_name, asset
        benchmark_returns : pd.Series, optional
            Benchmark returns
            
        Returns
        -------
        List[ValidationResult]
            Validation results
        """
        results = []
        
        for candidate in candidates:
            result = self.validate(
                returns=candidate['returns'],
                strategy_name=candidate['strategy_name'],
                asset=candidate['asset'],
                benchmark_returns=benchmark_returns
            )
            results.append(result)
        
        return results
    
    def get_valid_candidates(self) -> List[ValidationResult]:
        """Get all valid candidates."""
        return [r for r in self.results if r.is_valid]
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            data.append({
                'strategy': r.strategy_name,
                'asset': r.asset,
                'psr': r.psr,
                'dsr': r.dsr,
                'sharpe': r.sharpe_ratio,
                'max_dd': r.max_drawdown,
                'correlation': r.benchmark_correlation,
                'param_stability': r.parameter_stability,
                'noise_stability': r.noise_stability,
                'subsample_stability': r.subsample_stability,
                'score': r.score,
                'is_valid': r.is_valid
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('score', ascending=False)
        
        return df
    
    def print_report(self) -> None:
        """Print validation report."""
        df = self.get_summary()
        
        if df.empty:
            logger.info("No validation results")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 80)
        
        valid = df[df['is_valid']]
        logger.info(f"\nValid Candidates: {len(valid)} / {len(df)}")
        
        if not valid.empty:
            logger.info("\nTop Candidates:")
            for _, row in valid.head(5).iterrows():
                logger.info(
                    f"  [{row['strategy']}] {row['asset']}: "
                    f"Score={row['score']:.0f}, PSR={row['psr']:.2%}, "
                    f"MaxDD={row['max_dd']:.2%}"
                )
        
        # Summary statistics
        logger.info(f"\nAll Results Summary:")
        logger.info(f"  Avg PSR: {df['psr'].mean():.2%}")
        logger.info(f"  Avg Score: {df['score'].mean():.1f}")
        logger.info(f"  Pass Rate: {len(valid)/len(df):.1%}")


def main():
    """Test candidate validator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Generate synthetic returns for testing
    np.random.seed(42)
    n_days = 500
    
    # Good strategy (should pass)
    good_returns = pd.Series(
        np.random.normal(0.001, 0.01, n_days),
        index=pd.date_range('2020-01-01', periods=n_days, freq='B')
    )
    
    # Bad strategy (should fail)
    bad_returns = pd.Series(
        np.random.normal(-0.0005, 0.02, n_days),
        index=pd.date_range('2020-01-01', periods=n_days, freq='B')
    )
    
    # Benchmark
    benchmark = pd.Series(
        np.random.normal(0.0003, 0.012, n_days),
        index=pd.date_range('2020-01-01', periods=n_days, freq='B')
    )
    
    # Validate
    validator = CandidateValidator()
    
    result1 = validator.validate(
        good_returns, "good_strategy", "F_GC", benchmark
    )
    
    result2 = validator.validate(
        bad_returns, "bad_strategy", "F_ES", benchmark
    )
    
    # Print report
    validator.print_report()
    
    return len(validator.get_valid_candidates()) > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
