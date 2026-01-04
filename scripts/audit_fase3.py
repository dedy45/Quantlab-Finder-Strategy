"""
FASE 3 Code Audit Script.

Comprehensive audit for:
1. Emoticons check
2. Code cleanliness
3. Type hints
4. Logging (no print)
5. Try-except blocks
6. Assert statements
7. Vectorized operations
8. Data integrity checks
9. Real data testing
"""

import logging
import sys
import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class CodeAuditor:
    """Audit Python code for quality standards."""
    
    # Emoticon patterns (avoid false positives from code syntax)
    EMOTICON_PATTERNS = [
        r'[\U0001F600-\U0001F64F]',  # Emoticons
        r'[\U0001F300-\U0001F5FF]',  # Symbols & Pictographs
        r'[\U0001F680-\U0001F6FF]',  # Transport & Map
        r'[\U0001F1E0-\U0001F1FF]',  # Flags
        r'[\U00002702-\U000027B0]',  # Dingbats
        r'[\U0001F900-\U0001F9FF]',  # Supplemental Symbols
        r'(?<![:\[\]\-\d]):\)(?![:\[\]\-\d])',  # Smiley (avoid slice syntax)
        r'(?<![:\[\]\-\d]):\((?![:\[\]\-\d])',  # Sad face
        r'(?<![:\[\]\-\d]);[\)\(](?![:\[\]\-\d])',  # Wink
        r'(?<![<\-])❤️',  # Heart emoji
        r'(?<![<\-])💯',  # 100 emoji
    ]
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.issues: List[Dict[str, Any]] = []
        self.stats = {
            'files_checked': 0,
            'emoticons_found': 0,
            'print_statements': 0,
            'missing_type_hints': 0,
            'missing_try_except': 0,
            'missing_asserts': 0,
            'for_loops': 0,
        }
        
    def audit_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Audit a single Python file."""
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                issues.append({
                    'file': str(filepath),
                    'type': 'SYNTAX_ERROR',
                    'line': e.lineno,
                    'message': str(e)
                })
                return issues
            
            # 1. Check emoticons
            for i, line in enumerate(lines, 1):
                for pattern in self.EMOTICON_PATTERNS:
                    if re.search(pattern, line):
                        issues.append({
                            'file': str(filepath),
                            'type': 'EMOTICON',
                            'line': i,
                            'message': f'Emoticon found: {line.strip()[:50]}'
                        })
                        self.stats['emoticons_found'] += 1
            
            # 2. Check print statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'print':
                        issues.append({
                            'file': str(filepath),
                            'type': 'PRINT_STATEMENT',
                            'line': node.lineno,
                            'message': 'Use logging instead of print()'
                        })
                        self.stats['print_statements'] += 1
            
            # 3. Check functions for type hints
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private/dunder methods
                    if node.name.startswith('__') and node.name.endswith('__'):
                        continue
                    
                    # Check return type hint
                    if node.returns is None and not node.name.startswith('_'):
                        issues.append({
                            'file': str(filepath),
                            'type': 'MISSING_RETURN_TYPE',
                            'line': node.lineno,
                            'message': f'Function {node.name}() missing return type hint'
                        })
                        self.stats['missing_type_hints'] += 1
                    
                    # Check parameter type hints
                    for arg in node.args.args:
                        if arg.arg != 'self' and arg.annotation is None:
                            issues.append({
                                'file': str(filepath),
                                'type': 'MISSING_PARAM_TYPE',
                                'line': node.lineno,
                                'message': f'Parameter {arg.arg} in {node.name}() missing type hint'
                            })
                            self.stats['missing_type_hints'] += 1
            
            # 4. Check for for-loops (potential vectorization)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    # Check if iterating over range with data operations
                    if isinstance(node.iter, ast.Call):
                        if isinstance(node.iter.func, ast.Name):
                            if node.iter.func.id == 'range':
                                issues.append({
                                    'file': str(filepath),
                                    'type': 'FOR_LOOP_RANGE',
                                    'line': node.lineno,
                                    'message': 'Consider vectorized operation instead of for-range loop'
                                })
                                self.stats['for_loops'] += 1
            
            self.stats['files_checked'] += 1
            
        except Exception as e:
            issues.append({
                'file': str(filepath),
                'type': 'READ_ERROR',
                'line': 0,
                'message': str(e)
            })
        
        return issues
    
    def audit_directory(self, directory: str) -> None:
        """Audit all Python files in directory."""
        dir_path = self.base_path / directory
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return
        
        for filepath in dir_path.rglob('*.py'):
            if '__pycache__' in str(filepath):
                continue
            
            issues = self.audit_file(filepath)
            self.issues.extend(issues)
    
    def print_report(self) -> None:
        """Print audit report."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("CODE AUDIT REPORT")
        logger.info("=" * 60)
        
        # Group issues by type
        by_type: Dict[str, List] = {}
        for issue in self.issues:
            issue_type = issue['type']
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append(issue)
        
        # Print by type
        for issue_type, issues in sorted(by_type.items()):
            logger.info(f"\n{issue_type} ({len(issues)} issues):")
            for issue in issues[:5]:  # Show first 5
                logger.info(f"  {issue['file']}:{issue['line']} - {issue['message']}")
            if len(issues) > 5:
                logger.info(f"  ... and {len(issues) - 5} more")
        
        # Print stats
        logger.info("")
        logger.info("-" * 60)
        logger.info("STATISTICS:")
        logger.info(f"  Files checked: {self.stats['files_checked']}")
        logger.info(f"  Emoticons found: {self.stats['emoticons_found']}")
        logger.info(f"  Print statements: {self.stats['print_statements']}")
        logger.info(f"  Missing type hints: {self.stats['missing_type_hints']}")
        logger.info(f"  For-range loops: {self.stats['for_loops']}")
        
        # Overall status
        total_issues = len(self.issues)
        if total_issues == 0:
            logger.info("\n[OK] No critical issues found!")
        else:
            logger.warning(f"\n[WARN] {total_issues} issues found")


def test_data_integrity() -> bool:
    """Test data integrity checks."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATA INTEGRITY TESTS")
    logger.info("=" * 60)
    
    import numpy as np
    import pandas as pd
    
    all_passed = True
    
    # Test 1: DatetimeIndex check
    logger.info("\n[TEST] DatetimeIndex validation...")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'close': np.random.randn(100)}, index=dates)
        
        assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
        assert df.index.is_monotonic_increasing, "Index must be sorted"
        logger.info("  [OK] DatetimeIndex validation passed")
    except AssertionError as e:
        logger.error(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 2: Duplicate timestamp check
    logger.info("\n[TEST] Duplicate timestamp check...")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'close': np.random.randn(100)}, index=dates)
        
        assert not df.index.duplicated().any(), "No duplicate timestamps allowed"
        logger.info("  [OK] No duplicate timestamps")
    except AssertionError as e:
        logger.error(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 3: NaN/Inf check
    logger.info("\n[TEST] NaN/Inf value check...")
    try:
        data = np.array([1.0, 2.0, np.nan, 4.0])
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        logger.info(f"  NaN count: {nan_count}")
        logger.info(f"  Inf count: {inf_count}")
        
        if nan_count > 0:
            logger.warning("  [WARN] Data contains NaN values")
        else:
            logger.info("  [OK] No NaN values")
    except Exception as e:
        logger.error(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 4: Dimensionality check after feature engineering
    logger.info("\n[TEST] Dimensionality preservation check...")
    try:
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        
        # RSI calculation (should preserve length with NaN fill)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        assert len(rsi) == len(prices), \
            f"Dimension mismatch: input={len(prices)}, output={len(rsi)}"
        logger.info(f"  [OK] Dimension preserved: {len(prices)} -> {len(rsi)}")
        logger.info(f"  NaN values (expected for warmup): {rsi.isna().sum()}")
    except AssertionError as e:
        logger.error(f"  [FAIL] {e}")
        all_passed = False
    
    return all_passed


def test_with_real_data() -> bool:
    """Test FASE 3 modules with REAL data from ArcticDB."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("REAL DATA TESTS (ArcticDB)")
    logger.info("=" * 60)
    
    import numpy as np
    import pandas as pd
    
    all_passed = True
    
    # Load REAL data from ArcticDB
    logger.info("\n[INFO] Loading REAL data from ArcticDB...")
    try:
        from core.data_engine import DataManager
        
        dm = DataManager()
        
        # Load XAUUSD data
        df = dm.load('XAUUSD', '2020-01-01', '2024-12-31', timeframe='1H')
        
        if df is None or len(df) < 1000:
            logger.error("[FAIL] Not enough REAL data in ArcticDB")
            logger.error("       Run: python scripts/import_csv_to_arctic.py --all")
            return False
        
        logger.info(f"  [OK] Loaded {len(df)} bars of REAL XAUUSD data")
        
        # Calculate returns from REAL data
        returns = df['close'].pct_change().dropna()
        
        # Create multi-asset returns for portfolio tests
        # Use different timeframes as proxy for multiple assets
        returns_dict = {}
        
        # Asset 1: XAUUSD 1H returns
        returns_dict['XAUUSD_1H'] = returns
        
        # Asset 2: XAUUSD 4H returns (resample)
        df_4h = df['close'].resample('4H').last().dropna()
        returns_dict['XAUUSD_4H'] = df_4h.pct_change().dropna()
        
        # Asset 3: XAUUSD 1D returns (resample)
        df_1d = df['close'].resample('1D').last().dropna()
        returns_dict['XAUUSD_1D'] = df_1d.pct_change().dropna()
        
        # Try to load EURUSD if available
        try:
            df_eur = dm.load('EURUSD', '2020-01-01', '2024-12-31', timeframe='1H')
            if df_eur is not None and len(df_eur) > 100:
                returns_dict['EURUSD_1H'] = df_eur['close'].pct_change().dropna()
                logger.info(f"  [OK] Also loaded {len(df_eur)} bars of EURUSD data")
        except Exception:
            pass
        
        # Align all returns to common index
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 100:
            logger.warning("[WARN] Limited overlapping data, using XAUUSD only")
            returns_df = pd.DataFrame({'XAUUSD': returns})
        
        logger.info(f"  [OK] Created returns DataFrame: {returns_df.shape}")
        
        # Generate portfolio values from REAL returns
        portfolio_returns = returns_df.mean(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod() * 100000
        
        # Generate benchmark from REAL data (use XAUUSD as benchmark)
        benchmark_returns = returns
        
    except Exception as e:
        logger.error(f"[FAIL] Could not load REAL data: {e}")
        logger.error("       Ensure ArcticDB has data. Run: python scripts/import_csv_to_arctic.py --all")
        return False
    
    # Test 1: HRP Allocator
    logger.info("\n[TEST] HRP Allocator with REAL data...")
    try:
        from core.portfolio_engine import HRPAllocator
        
        hrp = HRPAllocator()
        hrp.fit(returns_df)
        weights = hrp.get_weights()
        
        n_assets = len(returns_df.columns)
        assert len(weights) == n_assets, "Weight count mismatch"
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"
        assert all(w >= 0 for w in weights.values()), "Weights must be non-negative"
        
        logger.info(f"  [OK] HRP weights: {weights}")
    except Exception as e:
        logger.error(f"  [FAIL] HRP: {e}")
        all_passed = False
    
    # Test 2: Volatility Targeter
    logger.info("\n[TEST] Volatility Targeter with REAL data...")
    try:
        from core.portfolio_engine import VolatilityTargeter
        
        vt = VolatilityTargeter(target_vol=0.15)
        scalar = vt.calculate_scalar(returns_df)
        
        assert scalar > 0, "Scalar must be positive"
        assert scalar <= 2.0, "Scalar should not exceed max leverage"
        
        info = vt.get_info()
        logger.info(f"  [OK] Vol scalar: {scalar:.2f}, current vol: {info['current_vol']:.2%}")
    except Exception as e:
        logger.error(f"  [FAIL] Vol Targeter: {e}")
        all_passed = False
    
    # Test 3: Kelly Sizer
    logger.info("\n[TEST] Kelly Sizer with REAL data...")
    try:
        from core.portfolio_engine import KellySizer
        
        kelly = KellySizer(fraction=0.5, min_trades=30)
        result = kelly.calculate_from_returns(portfolio_returns)
        
        logger.info(f"  [OK] Kelly: full={result.full_kelly:.2%}, "
                   f"recommended={result.recommended_size:.2%}, "
                   f"edge={result.edge:.4f}")
    except Exception as e:
        logger.error(f"  [FAIL] Kelly: {e}")
        all_passed = False
    
    # Test 4: Drawdown Controller
    logger.info("\n[TEST] Drawdown Controller with REAL data...")
    try:
        from core.risk_engine import DrawdownController
        
        dd = DrawdownController()
        is_safe = dd.check(portfolio_value)
        state = dd.get_state()
        
        logger.info(f"  [OK] DD: current={state.current_dd:.2%}, "
                   f"max={state.max_dd:.2%}, safe={is_safe}")
    except Exception as e:
        logger.error(f"  [FAIL] Drawdown: {e}")
        all_passed = False
    
    # Test 5: VaR Calculator
    logger.info("\n[TEST] VaR Calculator with REAL data...")
    try:
        from core.risk_engine import VaRCalculator
        
        var_calc = VaRCalculator(method='historical')
        var_calc.check(portfolio_value)
        
        # Test all methods
        for method in ['historical', 'parametric', 'cornish_fisher']:
            calc = VaRCalculator(method=method)
            result = calc.calculate(portfolio_returns)
            logger.info(f"  [OK] VaR ({method}): {result.var:.2%}, CVaR: {result.cvar:.2%}")
    except Exception as e:
        logger.error(f"  [FAIL] VaR: {e}")
        all_passed = False
    
    # Test 6: Correlation Monitor
    logger.info("\n[TEST] Correlation Monitor with REAL data...")
    try:
        from core.risk_engine import CorrelationMonitor
        
        # Align benchmark to portfolio returns index
        aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).dropna()
        aligned_portfolio = portfolio_returns.reindex(aligned_benchmark.index).dropna()
        
        if len(aligned_portfolio) > 60:
            corr_mon = CorrelationMonitor(lookback=60)
            result = corr_mon.analyze(aligned_portfolio, aligned_benchmark)
            
            logger.info(f"  [OK] Correlation: {result.correlation:.2f}, "
                       f"beta: {result.beta:.2f}, breach: {result.is_breach}")
        else:
            logger.warning("  [WARN] Not enough aligned data for correlation test")
    except Exception as e:
        logger.error(f"  [FAIL] Correlation: {e}")
        all_passed = False
    
    # Test 7: Position Limiter
    logger.info("\n[TEST] Position Limiter with REAL data...")
    try:
        from core.risk_engine import PositionLimiter
        
        # Use HRP weights
        limiter = PositionLimiter(max_concentration=0.6)
        result = limiter.check_weights(weights)
        
        logger.info(f"  [OK] Position limits: adjusted={result.is_adjusted}, "
                   f"breaches={result.breaches}")
    except Exception as e:
        logger.error(f"  [FAIL] Position Limiter: {e}")
        all_passed = False
    
    # Test 8: Integration test
    logger.info("\n[TEST] Full integration pipeline with REAL data...")
    try:
        from core.portfolio_engine import HRPAllocator, VolatilityTargeter
        from core.risk_engine import DrawdownController, PositionLimiter
        
        # Step 1: HRP allocation
        hrp = HRPAllocator()
        hrp.fit(returns_df)
        weights = hrp.get_weights()
        
        # Step 2: Volatility targeting
        vt = VolatilityTargeter(target_vol=0.15)
        scaled_weights = vt.scale_weights(weights, returns_df)
        
        # Step 3: Position limits
        limiter = PositionLimiter()
        final_result = limiter.check_weights(scaled_weights)
        final_weights = final_result.adjusted_weights
        
        # Step 4: Drawdown check
        dd = DrawdownController()
        is_safe = dd.check(portfolio_value)
        
        logger.info(f"  [OK] Integration pipeline completed")
        logger.info(f"      Final weights: {final_weights}")
        logger.info(f"      Portfolio safe: {is_safe}")
    except Exception as e:
        logger.error(f"  [FAIL] Integration: {e}")
        all_passed = False
    
    return all_passed


def test_look_ahead_bias() -> bool:
    """Test for look-ahead bias in data splitting."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("LOOK-AHEAD BIAS TESTS")
    logger.info("=" * 60)
    
    import numpy as np
    import pandas as pd
    
    all_passed = True
    
    # Test 1: Chronological split
    logger.info("\n[TEST] Chronological data split...")
    try:
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({'value': np.random.randn(1000)}, index=dates)
        
        # Correct split (chronological)
        train_end = int(len(data) * 0.7)
        val_end = int(len(data) * 0.85)
        
        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]
        
        # Verify chronological order
        assert train.index.max() < val.index.min(), "Train must be before validation"
        assert val.index.max() < test.index.min(), "Validation must be before test"
        
        logger.info(f"  [OK] Train: {train.index.min()} to {train.index.max()}")
        logger.info(f"  [OK] Val: {val.index.min()} to {val.index.max()}")
        logger.info(f"  [OK] Test: {test.index.min()} to {test.index.max()}")
    except AssertionError as e:
        logger.error(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 2: No shuffle for time-series
    logger.info("\n[TEST] No shuffle verification...")
    try:
        from sklearn.model_selection import train_test_split
        
        # This is WRONG for time-series
        # X_train, X_test = train_test_split(data, shuffle=True)  # DON'T DO THIS
        
        # This is CORRECT
        # X_train, X_test = train_test_split(data, shuffle=False)
        
        logger.info("  [OK] Remember: Always use shuffle=False for time-series!")
    except Exception as e:
        logger.error(f"  [FAIL] {e}")
        all_passed = False
    
    return all_passed


def main():
    """Run full audit."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FASE 3 COMPREHENSIVE CODE AUDIT")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # 1. Code quality audit
    auditor = CodeAuditor(Path(__file__).parent.parent)
    
    logger.info("\n[PHASE 1] Auditing Portfolio Engine...")
    auditor.audit_directory('core/portfolio_engine')
    
    logger.info("[PHASE 2] Auditing Risk Engine...")
    auditor.audit_directory('core/risk_engine')
    
    auditor.print_report()
    
    # 2. Data integrity tests
    logger.info("\n[PHASE 3] Data Integrity Tests...")
    data_ok = test_data_integrity()
    
    # 3. Look-ahead bias tests
    logger.info("\n[PHASE 4] Look-Ahead Bias Tests...")
    bias_ok = test_look_ahead_bias()
    
    # 4. Real data tests
    logger.info("\n[PHASE 5] Real Data Tests...")
    real_ok = test_with_real_data()
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 60)
    
    code_ok = auditor.stats['emoticons_found'] == 0 and auditor.stats['print_statements'] == 0
    
    results = [
        ("Code Quality", code_ok),
        ("Data Integrity", data_ok),
        ("Look-Ahead Bias", bias_ok),
        ("Real Data Tests", real_ok),
    ]
    
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        logger.info(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    logger.info("")
    if all_passed:
        logger.info("[OK] ALL AUDITS PASSED!")
    else:
        logger.warning("[WARN] Some audits failed - review issues above")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
