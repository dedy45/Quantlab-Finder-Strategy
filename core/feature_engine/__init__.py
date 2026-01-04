"""
Feature Engine - Feature engineering for quantitative strategies.

FASE 1A: Feature Engineering
- base.py - Base classes (FeatureGenerator, FeatureConfig, FeatureResult)
- fractional_diff.py - Fractional differentiation for stationarity
- technical.py - Technical indicators (RSI, Bollinger, Z-Score)
- pca_denoiser.py - PCA-based noise removal (Marcenko-Pastur)

FASE 1B: Labeling
- labeling/triple_barrier.py - Triple-barrier labeling
- labeling/meta_labeling.py - Meta-labeling for bet sizing

Reference: Protokol Kausalitas - Fase 2 (Sebab Statistik)
"""

from .base import (
    FeatureType,
    FeatureConfig,
    FeatureResult,
    FeatureGenerator,
    get_returns,
    normalize_features,
)

from .fractional_diff import (
    FractionalDifferencer,
    FracDiffResult,
    frac_diff,
    frac_diff_expanding,
    find_min_d,
    fractional_difference,
)

from .technical import (
    # RSI
    calculate_rsi,
    rsi_signal,
    # Bollinger
    BollingerBands,
    calculate_bollinger_bands,
    bollinger_signal,
    # Z-Score
    calculate_zscore,
    zscore_signal,
    # Moving Averages
    calculate_sma,
    calculate_ema,
    ma_crossover_signal,
    # ATR
    calculate_atr,
    calculate_atr_ratio,
    # MACD
    MACDResult,
    calculate_macd,
    macd_signal,
    # Combined Generator
    TechnicalFeatureGenerator,
)

from .pca_denoiser import (
    PCADenoiser,
    DenoiseResult,
    marcenko_pastur_pdf,
    get_mp_bounds,
    denoise_covariance,
    denoise_returns,
    cov_to_corr,
    corr_to_cov,
)

from .labeling import (
    # Triple Barrier
    TripleBarrierLabeler,
    TripleBarrierConfig,
    TripleBarrierResult,
    BarrierType,
    apply_triple_barrier,
    get_daily_volatility,
    # Meta Labeling
    MetaLabeler,
    MetaLabelResult,
    generate_meta_labels,
    calculate_bet_size,
)

__all__ = [
    # Base
    'FeatureType',
    'FeatureConfig',
    'FeatureResult',
    'FeatureGenerator',
    'get_returns',
    'normalize_features',
    # Fractional Diff
    'FractionalDifferencer',
    'FracDiffResult',
    'frac_diff',
    'frac_diff_expanding',
    'find_min_d',
    'fractional_difference',
    # Technical Indicators
    'calculate_rsi',
    'rsi_signal',
    'BollingerBands',
    'calculate_bollinger_bands',
    'bollinger_signal',
    'calculate_zscore',
    'zscore_signal',
    'calculate_sma',
    'calculate_ema',
    'ma_crossover_signal',
    'calculate_atr',
    'calculate_atr_ratio',
    'MACDResult',
    'calculate_macd',
    'macd_signal',
    'TechnicalFeatureGenerator',
    # PCA Denoiser
    'PCADenoiser',
    'DenoiseResult',
    'marcenko_pastur_pdf',
    'get_mp_bounds',
    'denoise_covariance',
    'denoise_returns',
    'cov_to_corr',
    'corr_to_cov',
    # Triple Barrier
    'TripleBarrierLabeler',
    'TripleBarrierConfig',
    'TripleBarrierResult',
    'BarrierType',
    'apply_triple_barrier',
    'get_daily_volatility',
    # Meta Labeling
    'MetaLabeler',
    'MetaLabelResult',
    'generate_meta_labels',
    'calculate_bet_size',
]
