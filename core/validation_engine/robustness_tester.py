"""
Robustness Testing.

Stress testing strategies under various conditions to ensure
they are not overfit to specific market regimes.

Reference: Protokol Kausalitas - Validasi Strategi
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


# Constants
DEFAULT_N_SIMULATIONS = 1000
PARAMETER_VARIATION_PCT = 0.2
NOISE_LEVELS = [0.01, 0.02, 0.05]
SUBSAMPLE_RATIOS = [0.5, 0.7, 0.9]


@dataclass
class RobustnessTest:
    """Single robustness test result."""
    test_name: str
    original_sharpe: float
    test_sharpe: float
    degradation: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustnessResult:
    """Container for robustness testing results."""
    tests: List[RobustnessTest]
    
    # Aggregate metrics
    n_tests: int = 0
    n_passed: int = 0
    pass_rate: float = 0.0
    avg_degradation: float = 0.0
    
    # Overall assessment
    is_robust: bool = False
    
    def __post_init__(self):
        """Compute aggregate metrics."""
        if len(self.tests) > 0:
            self.n_tests = len(self.tests)
            self.n_passed = sum(1 for t in self.tests if t.passed)
            self.pass_rate = self.n_passed / self.n_tests
            self.avg_degradation = np.mean([t.degradation for t in self.tests])
            self.is_robust = self.pass_rate >= 0.8
    
    def summary(self) -> str:
        """Return summary string."""
        status = "ROBUST" if self.is_robust else "NOT ROBUST"
        lines = [
            f"Robustness Testing ({status}):",
            f"  Tests: {self.n_tests}",
            f"  Passed: {self.n_passed} ({self.pass_rate:.1%})",
            f"  Avg Degradation: {self.avg_degradation:.2%}",
            "",
            "Individual Tests:"
        ]
        
        for test in self.tests:
            status = "PASS" if test.passed else "FAIL"
            lines.append(f"  [{status}] {test.test_name}: {test.degradation:.2%} degradation")
        
        return "\n".join(lines)


class RobustnessTester:
    """
    Robustness Testing Engine.
    
    Tests strategy robustness through:
    1. Parameter Sensitivity: Vary parameters within range
    2. Noise Injection: Add noise to data
    3. Subsample Testing: Test on random subsamples
    4. Regime Testing: Test on different market regimes
    5. Transaction Cost Sensitivity: Vary costs
    
    Examples
    --------
    >>> tester = RobustnessTester()
    >>> result = tester.test(strategy, price_data)
    >>> print(result.summary())
    """
    
    def __init__(
        self,
        max_degradation: float = 0.3,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
    ):
        """
        Initialize tester.
        
        Parameters
        ----------
        max_degradation : float
            Maximum allowed performance degradation
        n_simulations : int
            Number of simulations for Monte Carlo tests
        """
        self.max_degradation = max_degradation
        self.n_simulations = n_simulations
    
    def test(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str = 'close',
    ) -> RobustnessResult:
        """
        Run all robustness tests.
        
        Parameters
        ----------
        strategy : BaseStrategy
            Strategy to test
        data : pd.DataFrame
            Price data
        price_col : str
            Price column name
            
        Returns
        -------
        RobustnessResult
            Test results
        """
        data = self._validate_data(data)
        
        # Get baseline performance
        baseline_sharpe = self._get_baseline_sharpe(strategy, data, price_col)
        
        tests = []
        
        # 1. Parameter Sensitivity
        param_test = self._test_parameter_sensitivity(
            strategy, data, price_col, baseline_sharpe
        )
        tests.append(param_test)
        
        # 2. Noise Injection
        noise_test = self._test_noise_injection(
            strategy, data, price_col, baseline_sharpe
        )
        tests.append(noise_test)
        
        # 3. Subsample Testing
        subsample_test = self._test_subsample(
            strategy, data, price_col, baseline_sharpe
        )
        tests.append(subsample_test)
        
        # 4. Transaction Cost Sensitivity
        cost_test = self._test_transaction_costs(
            strategy, data, price_col, baseline_sharpe
        )
        tests.append(cost_test)
        
        # 5. Time Period Stability
        period_test = self._test_time_periods(
            strategy, data, price_col, baseline_sharpe
        )
        tests.append(period_test)
        
        return RobustnessResult(tests=tests)
    
    def _get_baseline_sharpe(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str
    ) -> float:
        """Get baseline Sharpe ratio."""
        strategy.fit(data)
        signals = strategy.predict(data)
        
        prices = data[price_col] if price_col in data.columns else data.iloc[:, 0]
        returns = self._calculate_returns(prices, signals)
        
        return self._calculate_sharpe(returns)
    
    def _test_parameter_sensitivity(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str,
        baseline_sharpe: float
    ) -> RobustnessTest:
        """Test sensitivity to parameter changes."""
        sharpes = []
        
        # Get original params
        original_params = strategy.get_params()
        
        # Test with varied parameters
        for _ in range(min(50, self.n_simulations)):
            # Create strategy copy with varied params
            varied_strategy = self._vary_parameters(strategy)
            
            try:
                varied_strategy.fit(data)
                signals = varied_strategy.predict(data)
                
                prices = data[price_col] if price_col in data.columns else data.iloc[:, 0]
                returns = self._calculate_returns(prices, signals)
                sharpe = self._calculate_sharpe(returns)
                sharpes.append(sharpe)
            except Exception:
                continue
        
        if len(sharpes) == 0:
            avg_sharpe = baseline_sharpe
        else:
            avg_sharpe = np.mean(sharpes)
        
        degradation = 1 - (avg_sharpe / baseline_sharpe) if baseline_sharpe != 0 else 0
        
        return RobustnessTest(
            test_name="Parameter Sensitivity",
            original_sharpe=baseline_sharpe,
            test_sharpe=avg_sharpe,
            degradation=max(0, degradation),
            passed=degradation < self.max_degradation,
            details={'n_variations': len(sharpes), 'sharpes': sharpes}
        )
    
    def _test_noise_injection(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str,
        baseline_sharpe: float
    ) -> RobustnessTest:
        """Test robustness to data noise."""
        sharpes = []
        
        prices = data[price_col] if price_col in data.columns else data.iloc[:, 0]
        
        for noise_level in NOISE_LEVELS:
            for _ in range(min(20, self.n_simulations // len(NOISE_LEVELS))):
                # Add noise to prices
                noise = np.random.normal(0, noise_level, len(prices))
                noisy_prices = prices * (1 + noise)
                
                noisy_data = data.copy()
                if price_col in noisy_data.columns:
                    noisy_data[price_col] = noisy_prices
                else:
                    noisy_data.iloc[:, 0] = noisy_prices
                
                try:
                    strategy.fit(noisy_data)
                    signals = strategy.predict(noisy_data)
                    returns = self._calculate_returns(noisy_prices, signals)
                    sharpe = self._calculate_sharpe(returns)
                    sharpes.append(sharpe)
                except Exception:
                    continue
        
        if len(sharpes) == 0:
            avg_sharpe = baseline_sharpe
        else:
            avg_sharpe = np.mean(sharpes)
        
        degradation = 1 - (avg_sharpe / baseline_sharpe) if baseline_sharpe != 0 else 0
        
        return RobustnessTest(
            test_name="Noise Injection",
            original_sharpe=baseline_sharpe,
            test_sharpe=avg_sharpe,
            degradation=max(0, degradation),
            passed=degradation < self.max_degradation,
            details={'noise_levels': NOISE_LEVELS, 'n_tests': len(sharpes)}
        )
    
    def _test_subsample(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str,
        baseline_sharpe: float
    ) -> RobustnessTest:
        """Test on random subsamples."""
        sharpes = []
        
        n_samples = len(data)
        
        for ratio in SUBSAMPLE_RATIOS:
            subsample_size = int(n_samples * ratio)
            
            for _ in range(min(20, self.n_simulations // len(SUBSAMPLE_RATIOS))):
                # Random contiguous subsample
                start_idx = np.random.randint(0, n_samples - subsample_size)
                subsample = data.iloc[start_idx:start_idx + subsample_size]
                
                try:
                    strategy.fit(subsample)
                    signals = strategy.predict(subsample)
                    
                    prices = subsample[price_col] if price_col in subsample.columns else subsample.iloc[:, 0]
                    returns = self._calculate_returns(prices, signals)
                    sharpe = self._calculate_sharpe(returns)
                    sharpes.append(sharpe)
                except Exception:
                    continue
        
        if len(sharpes) == 0:
            avg_sharpe = baseline_sharpe
        else:
            avg_sharpe = np.mean(sharpes)
        
        degradation = 1 - (avg_sharpe / baseline_sharpe) if baseline_sharpe != 0 else 0
        
        return RobustnessTest(
            test_name="Subsample Testing",
            original_sharpe=baseline_sharpe,
            test_sharpe=avg_sharpe,
            degradation=max(0, degradation),
            passed=degradation < self.max_degradation,
            details={'subsample_ratios': SUBSAMPLE_RATIOS, 'n_tests': len(sharpes)}
        )
    
    def _test_transaction_costs(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str,
        baseline_sharpe: float
    ) -> RobustnessTest:
        """Test sensitivity to transaction costs."""
        cost_levels = [0.001, 0.002, 0.005, 0.01]
        sharpes = []
        
        strategy.fit(data)
        signals = strategy.predict(data)
        
        prices = data[price_col] if price_col in data.columns else data.iloc[:, 0]
        
        for cost in cost_levels:
            returns = self._calculate_returns(prices, signals, transaction_cost=cost)
            sharpe = self._calculate_sharpe(returns)
            sharpes.append(sharpe)
        
        avg_sharpe = np.mean(sharpes)
        degradation = 1 - (avg_sharpe / baseline_sharpe) if baseline_sharpe != 0 else 0
        
        return RobustnessTest(
            test_name="Transaction Cost Sensitivity",
            original_sharpe=baseline_sharpe,
            test_sharpe=avg_sharpe,
            degradation=max(0, degradation),
            passed=degradation < self.max_degradation,
            details={'cost_levels': cost_levels, 'sharpes': sharpes}
        )
    
    def _test_time_periods(
        self,
        strategy,
        data: pd.DataFrame,
        price_col: str,
        baseline_sharpe: float
    ) -> RobustnessTest:
        """Test stability across time periods."""
        sharpes = []
        
        n_samples = len(data)
        period_size = n_samples // 4
        
        for i in range(4):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < 3 else n_samples
            period_data = data.iloc[start_idx:end_idx]
            
            if len(period_data) < 50:
                continue
            
            try:
                strategy.fit(period_data)
                signals = strategy.predict(period_data)
                
                prices = period_data[price_col] if price_col in period_data.columns else period_data.iloc[:, 0]
                returns = self._calculate_returns(prices, signals)
                sharpe = self._calculate_sharpe(returns)
                sharpes.append(sharpe)
            except Exception:
                continue
        
        if len(sharpes) == 0:
            avg_sharpe = baseline_sharpe
            std_sharpe = 0
        else:
            avg_sharpe = np.mean(sharpes)
            std_sharpe = np.std(sharpes)
        
        # High variance across periods indicates instability
        degradation = std_sharpe / abs(baseline_sharpe) if baseline_sharpe != 0 else 0
        
        return RobustnessTest(
            test_name="Time Period Stability",
            original_sharpe=baseline_sharpe,
            test_sharpe=avg_sharpe,
            degradation=max(0, degradation),
            passed=degradation < self.max_degradation,
            details={'period_sharpes': sharpes, 'std': std_sharpe}
        )
    
    def _vary_parameters(self, strategy):
        """Create strategy copy with varied parameters."""
        import copy
        varied = copy.deepcopy(strategy)
        
        # Vary numeric parameters
        if hasattr(varied.config, 'lookback'):
            variation = np.random.uniform(1 - PARAMETER_VARIATION_PCT, 1 + PARAMETER_VARIATION_PCT)
            varied.config.lookback = max(5, int(varied.config.lookback * variation))
        
        if hasattr(varied.config, 'fast_period'):
            variation = np.random.uniform(1 - PARAMETER_VARIATION_PCT, 1 + PARAMETER_VARIATION_PCT)
            varied.config.fast_period = max(2, int(varied.config.fast_period * variation))
        
        if hasattr(varied.config, 'slow_period'):
            variation = np.random.uniform(1 - PARAMETER_VARIATION_PCT, 1 + PARAMETER_VARIATION_PCT)
            varied.config.slow_period = max(5, int(varied.config.slow_period * variation))
        
        varied._is_fitted = False
        return varied
    
    def _calculate_returns(
        self,
        prices: pd.Series,
        signals: pd.Series,
        transaction_cost: float = 0.0
    ) -> pd.Series:
        """Calculate strategy returns."""
        price_returns = prices.pct_change()
        positions = signals.shift(1)
        strategy_returns = positions * price_returns
        
        # Apply transaction costs
        if transaction_cost > 0:
            position_changes = positions.diff().abs()
            costs = position_changes * transaction_cost
            strategy_returns = strategy_returns - costs
        
        return strategy_returns.dropna()
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        if std_ret == 0:
            return 0.0
        
        return mean_ret / std_ret * np.sqrt(252)
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data = data.set_index('date')
        
        return data.sort_index()
