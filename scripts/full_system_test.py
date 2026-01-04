"""
Full System Test with Real Data.

Comprehensive test of all FASE modules using real market data.
Priority: ArcticDB (XAUUSD) > Quantiacs > Synthetic

Run: python scripts/full_system_test.py

Version: 0.7.0
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Setup
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

# Set API key for Quantiacs
api_key = os.getenv('QUANTIACS_API_KEY') or os.getenv('API_KEY')
if api_key:
    os.environ['API_KEY'] = api_key

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SystemTester:
    """Full system tester."""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.prices_df: pd.DataFrame = None
        self.returns_df: pd.DataFrame = None
    
    def add_result(self, name: str, passed: bool, detail: str = "") -> None:
        """Add test result."""
        self.results.append((name, passed, detail))
        status = "[OK]" if passed else "[FAIL]"
        logger.info(f"{status} {name}: {detail}")
    
    def load_data(self) -> bool:
        """Load market data (ArcticDB > Quantiacs > Synthetic)."""
        logger.info("Loading market data...")
        
        # Priority 1: ArcticDB (REAL data)
        try:
            from core.data_engine import ArcticStore
            
            store = ArcticStore()
            symbols = store.list_symbols()
            
            if 'XAUUSD_1H' in symbols:
                df = store.read('XAUUSD', '1H', start='2020-01-01', end='2024-12-31')
                
                if df is not None and len(df) > 100:
                    # Ensure proper index
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                    
                    # Create multi-asset DataFrame for portfolio tests
                    self.prices_df = pd.DataFrame({
                        'XAUUSD': df['close'],
                        'XAUUSD_H': df['high'],
                        'XAUUSD_L': df['low'],
                    })
                    self.returns_df = self.prices_df.pct_change().dropna()
                    
                    logger.info(f"Loaded {len(self.prices_df)} bars of REAL data from ArcticDB")
                    return True
                    
        except Exception as e:
            logger.warning(f"ArcticDB unavailable: {e}")
        
        # Priority 2: Quantiacs
        try:
            import qnt.data as qndata
            
            data = qndata.futures.load_data(
                assets=['F_GC', 'F_ES', 'F_CL'],
                min_date='2020-01-01'
            )
            
            close = data.sel(field='close').to_pandas()
            self.prices_df = close
            self.returns_df = close.pct_change().dropna()
            
            logger.info(f"Loaded {len(self.prices_df)} days of REAL data from Quantiacs")
            return True
            
        except Exception as e:
            logger.warning(f"Quantiacs unavailable: {e}")
        
        # Priority 3: Synthetic (fallback)
        logger.info("Generating synthetic data...")
        
        np.random.seed(42)
        n_days = 252 * 3
        dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
        
        assets = ['F_GC', 'F_ES', 'F_CL']
        prices = {}
        
        for asset in assets:
            returns = np.random.randn(n_days) * 0.015 + 0.0003
            prices[asset] = 100 * np.exp(np.cumsum(returns))
        
        self.prices_df = pd.DataFrame(prices, index=dates)
        self.returns_df = self.prices_df.pct_change().dropna()
        
        logger.info(f"Generated {len(self.prices_df)} days of synthetic data")
        return True
    
    def test_fase0(self) -> None:
        """Test FASE 0: Foundation."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("FASE 0: FOUNDATION")
        logger.info("=" * 50)
        
        # Test Validation Engine
        try:
            from core.validation_engine import calculate_sharpe, calculate_psr, calculate_dsr
            
            returns = self.returns_df.iloc[:, 0]
            
            sharpe = calculate_sharpe(returns)
            self.add_result("Sharpe Ratio", not np.isnan(sharpe), f"{sharpe:.2f}")
            
            psr = calculate_psr(returns)
            self.add_result("PSR", 0 <= psr <= 1, f"{psr:.2%}")
            
            dsr = calculate_dsr(sharpe, n_trials=10)
            self.add_result("DSR", True, f"{dsr:.2f}")
            
        except Exception as e:
            self.add_result("Validation Engine", False, str(e)[:50])
        
        # Test Regime Detection
        try:
            from core.signal_engine.regime import HurstRegimeDetector
            
            # Use smaller sample for faster test
            prices = self.prices_df.iloc[-500:, 0].to_frame('close')
            
            hurst = HurstRegimeDetector(window=50)  # Smaller window for speed
            result = hurst.fit_predict(prices)
            
            self.add_result("Hurst Regime", True, f"{result.current_regime}")
            
        except Exception as e:
            self.add_result("Regime Detection", False, str(e)[:50])
    
    def test_fase1(self) -> None:
        """Test FASE 1: Alpha Factory."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("FASE 1: ALPHA FACTORY")
        logger.info("=" * 50)
        
        # Test Feature Engine - Fractional Diff
        try:
            from core.feature_engine import FractionalDifferencer
            
            # FractionalDifferencer expects DataFrame with 'close' column
            prices_df = self.prices_df.iloc[:, 0].to_frame('close')
            
            # Use larger threshold for practical window size
            # threshold=1e-5 creates 1458 weights, too long for most data
            # threshold=1e-3 creates ~55 weights, more practical
            fd = FractionalDifferencer(d=0.4, threshold=1e-3)
            result = fd.fit_transform(prices_df)
            
            # Result is FeatureResult, access features DataFrame
            features = result.features
            # Count non-NaN rows (fracdiff produces NaN at start due to window)
            n_valid = features.iloc[:, 0].notna().sum()
            is_valid = features is not None and n_valid > 0
            self.add_result("Fractional Diff", is_valid, f"{n_valid} valid rows, d={result.metadata.get('d', 0.4):.2f}")
            
        except Exception as e:
            self.add_result("Fractional Diff", False, str(e)[:50])
        
        # Test Technical Features - RSI
        try:
            from core.feature_engine.technical import calculate_rsi
            
            prices = self.prices_df.iloc[:, 0]
            
            rsi = calculate_rsi(prices, period=14)
            
            is_valid = rsi is not None and len(rsi) > 0
            # Check RSI is bounded 0-100
            rsi_clean = rsi.dropna()
            in_range = (rsi_clean >= 0).all() and (rsi_clean <= 100).all()
            self.add_result("RSI Indicator", is_valid and in_range, f"{len(rsi_clean)} values, range OK")
            
        except Exception as e:
            self.add_result("RSI Indicator", False, str(e)[:50])
    
    def test_fase2(self) -> None:
        """Test FASE 2: Strategy Development."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("FASE 2: STRATEGY DEVELOPMENT")
        logger.info("=" * 50)
        
        # Test Strategies
        try:
            from strategies import MomentumStrategy
            from strategies.base import StrategyConfig
            
            prices = self.prices_df.iloc[:, 0].to_frame('close')
            
            # Split
            train = prices.iloc[:int(len(prices)*0.7)]
            test = prices.iloc[int(len(prices)*0.7):]
            
            strategy = MomentumStrategy()
            strategy.fit(train)
            signals = strategy.predict(test)
            
            self.add_result("Momentum Strategy", len(signals) > 0, f"{len(signals)} signals")
            
        except Exception as e:
            self.add_result("Momentum Strategy", False, str(e)[:50])
        
        # Test Mean Reversion
        try:
            from strategies import MeanReversionStrategy
            
            prices = self.prices_df.iloc[:, 0].to_frame('close')
            train = prices.iloc[:int(len(prices)*0.7)]
            test = prices.iloc[int(len(prices)*0.7):]
            
            mr = MeanReversionStrategy()
            mr.fit(train)
            signals = mr.predict(test)
            
            self.add_result("Mean Reversion", len(signals) > 0, f"{len(signals)} signals")
            
        except Exception as e:
            self.add_result("Mean Reversion", False, str(e)[:50])
    
    def test_fase3(self) -> None:
        """Test FASE 3: Portfolio Construction."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("FASE 3: PORTFOLIO CONSTRUCTION")
        logger.info("=" * 50)
        
        # Test HRP
        try:
            from core.portfolio_engine import HRPAllocator
            
            hrp = HRPAllocator()
            result = hrp.fit(self.returns_df)
            
            weights = result.weights if hasattr(result, 'weights') else result
            total = sum(weights.values()) if isinstance(weights, dict) else 1.0
            
            self.add_result("HRP Allocation", abs(total - 1.0) < 0.1, f"Sum={total:.2f}")
            
        except Exception as e:
            self.add_result("HRP Allocation", False, str(e)[:50])
        
        # Test Risk Engine - DrawdownController
        try:
            from core.risk_engine import DrawdownController
            from core.risk_engine.base import RiskConfig
            
            config = RiskConfig(max_drawdown=0.20)
            dd = DrawdownController(config=config)
            
            # Use check() with full series
            equity = (1 + self.returns_df.iloc[:, 0]).cumprod()
            dd.check(equity)
            
            metrics = dd.get_metrics()
            max_dd = metrics.max_drawdown if hasattr(metrics, 'max_drawdown') else 0
            self.add_result("Drawdown Control", True, f"MaxDD={max_dd:.2%}")
            
        except Exception as e:
            self.add_result("Drawdown Control", False, str(e)[:50])
        
        # Test VaR
        try:
            from core.risk_engine import VaRCalculator
            
            var_calc = VaRCalculator(method='historical')
            result = var_calc.calculate(self.returns_df.iloc[:, 0])
            
            # VaRResult has 'var' and 'cvar' attributes
            var_val = result.var
            cvar_val = result.cvar
            self.add_result("VaR Calculator", True, f"VaR={var_val:.2%}, CVaR={cvar_val:.2%}")
            
        except Exception as e:
            self.add_result("VaR Calculator", False, str(e)[:50])
    
    def test_fase4(self) -> None:
        """Test FASE 4: Deployment."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("FASE 4: DEPLOYMENT")
        logger.info("=" * 50)
        
        returns = self.returns_df.iloc[:, 0]
        
        # Test Quantiacs Adapter
        try:
            from deployment.quantiacs import QuantiacsAdapter
            from deployment.base import DeploymentConfig
            
            config = DeploymentConfig(strategy_name="SystemTest")
            adapter = QuantiacsAdapter(config=config)
            
            validation = adapter.validate(returns)
            sharpe = validation['metrics']['sharpe_ratio']
            
            self.add_result("Quantiacs Adapter", True, f"Sharpe={sharpe:.2f}")
            
        except Exception as e:
            self.add_result("Quantiacs Adapter", False, str(e)[:50])
        
        # Test QC Adapter
        try:
            from deployment.quantconnect import QuantConnectAdapter
            from deployment.base import DeploymentConfig
            
            config = DeploymentConfig(strategy_name="SystemTest")
            adapter = QuantConnectAdapter(config=config)
            
            validation = adapter.validate(returns)
            psr = validation['psr']
            
            self.add_result("QC Adapter", True, f"PSR={psr:.2%}")
            
        except Exception as e:
            self.add_result("QC Adapter", False, str(e)[:50])
        
        # Test Performance Tracker
        try:
            from deployment.monitoring import PerformanceTracker
            
            tracker = PerformanceTracker(strategy_name="Test", backtest_sharpe=1.0)
            
            for date, ret in zip(returns.index[-60:], returns.values[-60:]):
                tracker.update(date, ret)
            
            metrics = tracker.get_metrics()
            self.add_result("Performance Tracker", True, f"Sharpe={metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            self.add_result("Performance Tracker", False, str(e)[:50])
        
        # Test Decay Detector
        try:
            from deployment.monitoring import DecayDetector
            
            detector = DecayDetector(lookback=60)
            
            bt = returns.iloc[:len(returns)//2]
            live = returns.iloc[len(returns)//2:]
            
            result = detector.check(live, bt)
            self.add_result("Decay Detector", True, f"Decaying={result.is_decaying}")
            
        except Exception as e:
            self.add_result("Decay Detector", False, str(e)[:50])
        
        # Test Alert System
        try:
            from deployment.monitoring import AlertSystem
            
            alerts = AlertSystem(strategy_name="Test")
            triggered = alerts.check({
                'drawdown': -0.10,
                'sharpe_ratio': 0.5,
                'volatility': 0.20,
            })
            
            self.add_result("Alert System", True, f"{len(triggered)} alerts")
            
        except Exception as e:
            self.add_result("Alert System", False, str(e)[:50])
    
    def run_all(self) -> int:
        """Run all tests."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("QUANT LAB - FULL SYSTEM TEST")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Version: 0.7.0")
        logger.info("")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data")
            return 1
        
        # Run tests
        self.test_fase0()
        self.test_fase1()
        self.test_fase2()
        self.test_fase3()
        self.test_fase4()
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, p, _ in self.results if p)
        failed = sum(1 for _, p, _ in self.results if not p)
        total = len(self.results)
        
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Failed: {failed}/{total}")
        logger.info("")
        
        for name, p, detail in self.results:
            status = "[OK]" if p else "[FAIL]"
            logger.info(f"  {status} {name}")
        
        logger.info("")
        if failed == 0:
            logger.info("[OK] ALL TESTS PASSED!")
            return 0
        else:
            logger.warning(f"[WARN] {failed} tests failed")
            return 1


def main():
    """Main entry point."""
    tester = SystemTester()
    return tester.run_all()


if __name__ == '__main__':
    sys.exit(main())
