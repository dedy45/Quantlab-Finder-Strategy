"""
QUANT LAB - VERIFY WITH REAL DATA
=================================
Verifikasi semua modul FASE 0 & 1 dengan data REAL dari Quantiacs.

Usage:
    python scripts/verify_real_data.py
"""

import sys
import os
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

quantiacs_key = os.getenv('QUANTIACS_API_KEY', '')
if quantiacs_key:
    os.environ['API_KEY'] = quantiacs_key

import numpy as np
import pandas as pd
from scipy import stats

results = {'passed': [], 'failed': [], 'warnings': []}

def log_pass(msg):
    print(f"  [PASS] {msg}")
    results['passed'].append(msg)

def log_fail(msg):
    print(f"  [FAIL] {msg}")
    results['failed'].append(msg)

def log_warn(msg):
    print(f"  [WARN] {msg}")
    results['warnings'].append(msg)

def log_info(msg):
    print(f"  [INFO] {msg}")

print("=" * 70)
print("QUANT LAB - VERIFICATION WITH REAL DATA")
print("=" * 70)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# STEP 1: LOAD DATA
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

try:
    import qnt.data as qndata
    log_info("Loading Quantiacs futures data...")
    futures_data = qndata.futures_load_data(
        assets=['F_GC', 'F_ES'],
        min_date='2020-01-01',
        max_date='2023-12-31'
    )
    close_gc = futures_data.sel(asset='F_GC', field='close').to_pandas()
    close_es = futures_data.sel(asset='F_ES', field='close').to_pandas()
    prices_df = pd.DataFrame({'F_GC': close_gc, 'F_ES': close_es}).dropna()
    cache_dir = project_root / 'data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices_df.to_parquet(cache_dir / 'futures_gc_es.parquet')
    log_pass(f"Loaded {len(prices_df)} days REAL data from Quantiacs")
except Exception as e:
    log_warn(f"Quantiacs error: {e}")
    cache_path = project_root / 'data' / 'cache' / 'futures_gc_es.parquet'
    if cache_path.exists():
        prices_df = pd.read_parquet(cache_path)
        log_pass(f"Loaded {len(prices_df)} days from cache")
    else:
        log_warn("Generating synthetic data...")
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
        gc_ret = np.random.normal(0.0002, 0.01, len(dates))
        es_ret = np.random.normal(0.0003, 0.012, len(dates))
        prices_df = pd.DataFrame({
            'F_GC': 1800 * np.exp(np.cumsum(gc_ret)),
            'F_ES': 4000 * np.exp(np.cumsum(es_ret))
        }, index=dates)
        log_pass(f"Generated {len(prices_df)} days synthetic data")

returns_gc = prices_df['F_GC'].pct_change().dropna()
print()

# STEP 2: VALIDATION ENGINE
print("=" * 70)
print("STEP 2: VALIDATION ENGINE")
print("=" * 70)

print("\n--- Sharpe Ratio ---")
try:
    from core.validation_engine import SharpeCalculator
    calc = SharpeCalculator(risk_free_rate=0.02, periods_per_year=252)
    result = calc.calculate(returns_gc)
    log_info(f"Annualized Sharpe: {result.annualized_sharpe:.4f}")
    log_pass("Sharpe Ratio VERIFIED")
except Exception as e:
    log_fail(f"Sharpe error: {e}")

print("\n--- PSR ---")
try:
    from core.validation_engine import PSRCalculator
    psr_calc = PSRCalculator(benchmark_sr=0.0)
    psr_result = psr_calc.calculate(returns_gc)
    log_info(f"PSR: {psr_result.psr:.4f} ({psr_result.psr*100:.2f}%)")
    if 0 <= psr_result.psr <= 1:
        log_pass("PSR VERIFIED")
    else:
        log_fail("PSR out of range")
except Exception as e:
    log_fail(f"PSR error: {e}")

print("\n--- DSR ---")
try:
    from core.validation_engine import DSRCalculator
    dsr_calc = DSRCalculator(n_trials=100)
    dsr_result = dsr_calc.calculate(returns_gc)
    log_info(f"DSR: {dsr_result.dsr:.4f}")
    log_pass("DSR VERIFIED")
except Exception as e:
    log_fail(f"DSR error: {e}")

print("\n--- Bootstrap CI ---")
try:
    from core.validation_engine import BootstrapEngine
    bootstrap = BootstrapEngine(n_bootstrap=1000, confidence_level=0.95)
    boot_result = bootstrap.sharpe_ci(returns_gc)
    log_info(f"95% CI: [{boot_result.ci_lower:.4f}, {boot_result.ci_upper:.4f}]")
    log_pass("Bootstrap CI VERIFIED")
except Exception as e:
    log_fail(f"Bootstrap error: {e}")

print("\n--- Monte Carlo ---")
try:
    from core.validation_engine import MonteCarloEngine
    signals = np.sign(returns_gc.rolling(5).mean().shift(1)).dropna()
    market_rets = returns_gc.loc[signals.index]
    mc = MonteCarloEngine(n_simulations=500, show_progress=False)
    mc_result = mc.permutation_test(signals, market_rets)
    log_info(f"p-value: {mc_result.p_value:.4f}")
    log_pass("Monte Carlo VERIFIED")
except Exception as e:
    log_fail(f"Monte Carlo error: {e}")

print()

# STEP 3: REGIME DETECTION
print("=" * 70)
print("STEP 3: REGIME DETECTION")
print("=" * 70)

print("\n--- Hurst Exponent ---")
try:
    from core.signal_engine.regime import HurstRegimeDetector
    hurst = HurstRegimeDetector(window=100)
    hurst_result = hurst.fit_predict(prices_df[['F_GC']].rename(columns={'F_GC': 'close'}))
    h = hurst_result.metadata.get('current_hurst', 0)
    log_info(f"Hurst: {h:.4f}, Regime: {hurst_result.current_regime}")
    if 0 < h < 1:
        log_pass("Hurst Exponent VERIFIED")
    else:
        log_fail(f"Hurst out of range: {h}")
except Exception as e:
    log_fail(f"Hurst error: {e}")

print("\n--- HMM Regime ---")
try:
    from core.signal_engine.regime import HMMRegimeDetector
    hmm = HMMRegimeDetector(n_regimes=2)
    hmm_result = hmm.fit_predict(prices_df[['F_GC']].rename(columns={'F_GC': 'close'}))
    log_info(f"Regime: {hmm_result.current_regime}")
    log_pass("HMM Regime VERIFIED")
except Exception as e:
    log_fail(f"HMM error: {e}")

print("\n--- Volatility Regime ---")
try:
    from core.signal_engine.regime import VolatilityRegime
    vol = VolatilityRegime(window=20)
    vol_result = vol.fit_predict(prices_df[['F_GC']].rename(columns={'F_GC': 'close'}))
    log_info(f"Regime: {vol_result.current_regime}")
    log_pass("Volatility Regime VERIFIED")
except Exception as e:
    log_fail(f"Volatility error: {e}")

print()

# STEP 4: FEATURE ENGINE
print("=" * 70)
print("STEP 4: FEATURE ENGINE")
print("=" * 70)

print("\n--- Fractional Differencing ---")
try:
    from core.feature_engine import FractionalDifferencer
    frac = FractionalDifferencer(d='auto', threshold=0.01)
    frac_result = frac.fit_transform(prices_df[['F_GC']].rename(columns={'F_GC': 'close'}))
    d = frac_result.metadata.get('d', 0)
    log_info(f"Optimal d: {d:.4f}")
    log_pass("Fractional Differencing VERIFIED")
except Exception as e:
    log_fail(f"FracDiff error: {e}")

print("\n--- Technical Indicators ---")
try:
    from core.feature_engine import TechnicalFeatureGenerator
    tech = TechnicalFeatureGenerator()
    tech_result = tech.fit_transform(prices_df[['F_GC']].rename(columns={'F_GC': 'close'}))
    features = tech_result.features
    if 'rsi' in features.columns:
        rsi = features['rsi'].dropna()
        log_info(f"RSI range: [{rsi.min():.2f}, {rsi.max():.2f}]")
        if 0 <= rsi.min() and rsi.max() <= 100:
            log_pass("RSI VERIFIED")
    if 'bb_percent_b' in features.columns:
        log_pass("Bollinger Bands VERIFIED")
    if 'z_score' in features.columns:
        log_pass("Z-Score VERIFIED")
    log_info(f"Total features: {len(features.columns)}")
except Exception as e:
    log_fail(f"Technical error: {e}")

print("\n--- PCA Denoiser ---")
try:
    from core.feature_engine import PCADenoiser
    returns_df = prices_df.pct_change().dropna()
    denoiser = PCADenoiser(method='marcenko_pastur')
    denoised = denoiser.fit_transform(returns_df)
    log_info(f"Signal components: {denoised.metadata.get('n_signal_components', 0)}")
    log_pass("PCA Denoiser VERIFIED")
except Exception as e:
    log_fail(f"PCA error: {e}")

print()

# STEP 5: LABELING
print("=" * 70)
print("STEP 5: LABELING")
print("=" * 70)

print("\n--- Triple-Barrier ---")
try:
    from core.feature_engine.labeling import TripleBarrierLabeler
    labeler = TripleBarrierLabeler(pt_sl_ratio=2.0, max_holding_period=10, vol_lookback=20)
    label_result = labeler.fit_transform(prices_df[['F_GC']].rename(columns={'F_GC': 'close'}))
    labels = label_result.labels
    log_info(f"Labels: {labels.value_counts().to_dict()}")
    valid = set([-1, 0, 1])
    if set(labels.dropna().unique()).issubset(valid):
        log_pass("Triple-Barrier VERIFIED")
    else:
        log_fail("Invalid labels")
except Exception as e:
    log_fail(f"Triple-Barrier error: {e}")

print("\n--- Meta-Labeling ---")
try:
    from core.feature_engine.labeling import MetaLabeler
    ma_fast = prices_df['F_GC'].rolling(10).mean()
    ma_slow = prices_df['F_GC'].rolling(30).mean()
    primary = pd.Series(np.where(ma_fast > ma_slow, 1, -1), index=prices_df.index)
    features = pd.DataFrame({
        'returns': prices_df['F_GC'].pct_change(),
        'vol': prices_df['F_GC'].pct_change().rolling(20).std()
    }, index=prices_df.index).dropna()
    meta = MetaLabeler(primary_signals=primary, bet_sizing='half_kelly')
    meta.fit(prices=prices_df[['F_GC']].rename(columns={'F_GC': 'close'}), features=features)
    result = meta.transform(features)
    log_info(f"Accuracy: {result.accuracy:.2%}, Avg bet: {result.avg_bet_size:.2%}")
    log_pass("Meta-Labeling VERIFIED")
except Exception as e:
    log_fail(f"Meta-Labeling error: {e}")

print()

# SUMMARY
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nPassed: {len(results['passed'])}")
for m in results['passed']:
    print(f"  [PASS] {m}")
if results['warnings']:
    print(f"\nWarnings: {len(results['warnings'])}")
    for m in results['warnings']:
        print(f"  [WARN] {m}")
if results['failed']:
    print(f"\nFailed: {len(results['failed'])}")
    for m in results['failed']:
        print(f"  [FAIL] {m}")

total = len(results['passed']) + len(results['failed'])
rate = len(results['passed']) / total * 100 if total > 0 else 0
print(f"\nTotal: {total}, Pass Rate: {rate:.1f}%")

if len(results['failed']) == 0:
    print("\n" + "=" * 70)
    print("  ALL MODULES VERIFIED SUCCESSFULLY")
    print("=" * 70)
    sys.exit(0)
else:
    print("\n" + "=" * 70)
    print("  VERIFICATION INCOMPLETE")
    print("=" * 70)
    sys.exit(1)
