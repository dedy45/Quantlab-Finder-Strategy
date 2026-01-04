# Quantlab-Finder-Strategy
QuantLab finder-strategy adalah pipeline+UI trading kuantitatif berstandar institusional.
Membangun infrastruktur data dan validasi statistik yang robust.
---
## Visi

Membangun sistem trading kuantitatif yang:
- Lolos seleksi QuantConnect Alpha Streams
- Menghasilkan strategi dengan PSR > 95% dan korelasi pasar < 0.3
- Berbasis Python murni.

Filosofi: "Membangun SEBAB yang kuat, AKIBAT (profit) akan datang sendiri."
Dengan: 
- 6 FASE development 
- 15 steering files 
- Multi-engine backtest 
 → VectorBT (FAST SCREENING Vectorized) from https://github.com/polakowo/vectorbt
 → Nautilus (FAST VALIDATION EVEN DRIVEN) from https://github.com/nautechsystems/nautilus_trader
 → LEAN (ready to validate real account) from https://github.com/QuantConnect/Lean
- Target: PSR > 95%, Correlation < 0.3, Max DD < 20%

### Libraries
- Data: NumPy, Pandas, PyArrow, ArcticDB
- Scientific: SciPy, Statsmodels, Arch
- ML: Scikit-learn, LightGBM, XGBoost, hmmlearn
- Technical Analysis: TA-Lib (150+ indicators, C-optimized)
- Deep Learning: PyTorch (optional)

### Data Storage
| Storage | Type | Speed | Use Case |
|---------|------|-------|----------|
| ArcticDB | Primary | 3.7x faster | All OHLCV data |
| PyArrow | Backend | - | Required by ArcticDB |

### Backtest Engines
| Engine | Library | Purpose |
|--------|---------|---------|
| VectorBT | vectorbt | Fast screening (1000+ ideas) |
| Nautilus | nautilus_trader | Realistic validation (Top 50) |
| LEAN | QuantConnect | Alpha Streams submission |

## Overview Diagram


                              ┌─────────────┐
                              │ Lyer controler│
                              │  (Dash UI)  │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
              │ Research  │   │ Backtest  │   │ Deployment│
              │ Notebooks │   │  Engines  │   │  (QC/QNT) │
              └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │              CORE LAYER                    │
              ├─────────────────────────────────────────────┤
              │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
              │  │  Data    │ │ Feature  │ │  Signal  │   │
              │  │  Engine  │ │  Engine  │ │  Engine  │   │
              │  └──────────┘ └──────────┘ └──────────┘   │
              │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
              │  │Validation│ │Portfolio │ │   Risk   │   │
              │  │  Engine  │ │  Engine  │ │  Engine  │   │
              │  └──────────┘ └──────────┘ └──────────┘   │
              └─────────────────────┬─────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │              DATA LAYER (ArcticDB)         │
              ├─────────────────────────────────────────────┤
              │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
              │  │  OHLCV   │ │ Features │ │ Signals  │   │
              │  │ Library  │ │ Library  │ │ Library  │   │
              │  └──────────┘ └──────────┘ └──────────┘   │
              └───────────────────────────────────────────┘


## FASE 0: FOUNDATION
### Tujuan
Membangun infrastruktur data dan validasi statistik yang robust.

### Komponen
| Module | Fungsi | File |
|--------|--------|------|
| ArcticStore | Time-series database (3.7x faster) | `data_engine/arctic_store.py` |
| DataManager | Unified data loading | `data_engine/data_manager.py` |
| PSRCalculator | Probabilistic Sharpe Ratio | `validation_engine/psr.py` |
| DSRCalculator | Deflated Sharpe Ratio | `validation_engine/dsr.py` |
| HurstRegime | Trending vs Mean-Reverting | `signal_engine/regime/hurst.py` |

## FASE 1: ALPHA FACTORY
### Tujuan
Membuat fitur dan label yang robust untuk ML.

### Komponen
| Module | Fungsi | File |
|--------|--------|------|
| FractionalDiff | Stationarity dengan memory | `feature_engine/fractional_diff.py` |
| TechnicalFeatures | RSI, Bollinger, Z-Score | `feature_engine/technical.py` |
| PCADenoiser | Marcenko-Pastur denoising | `feature_engine/pca_denoiser.py` |
| TripleBarrier | Path-dependent labeling | `feature_engine/labeling/triple_barrier.py` |
| MetaLabeler | Bet sizing | `feature_engine/labeling/meta_labeling.py` |

## FASE 3: PORTFOLIO CONSTRUCTION
### Tujuan
Alokasi modal dan risk management.

### Komponen
| Module | Fungsi | File |
|--------|--------|------|
| HRPAllocator | Hierarchical Risk Parity | `portfolio_engine/hrp_allocator.py` |
| VolatilityTargeter | Vol targeting (15%) | `portfolio_engine/volatility_target.py` |
| KellySizer | Position sizing | `portfolio_engine/kelly_sizing.py` |
| DrawdownController | DD monitoring | `risk_engine/drawdown_control.py` |
| VaRCalculator | Value at Risk | `risk_engine/var_calculator.py` |

## FASE 4: DEPLOYMENT
### Tujuan
Deploy ke platform institusional.

### Komponen
| Module | Fungsi | File |
|--------|--------|------|
| QuantiacsAdapter | Quantiacs deployment | `deployment/quantiacs/adapter.py` |
| QuantConnectAdapter | QC Alpha Streams | `deployment/quantconnect/adapter.py` |
| PerformanceTracker | Live monitoring | `deployment/monitoring/performance.py` |
| DecayDetector | Strategy decay | `deployment/monitoring/decay_detector.py` |
| AlertSystem | Alerts | `deployment/monitoring/alerts.py` |

## FASE 5: PRODUCTION
### Tujuan
Research pipeline + UI dan multi-engine backtest.

## Dash UI Dashboard

QuantLab Dash UI adalah Command & Control Center untuk platform trading kuantitatif.
### Quick Start
--
Demo APP in https://qlab.bamsbung.id/
--

---

**Built with ❤️ for traders who believe in data-driven decisions.**

*Version 0.7.4 | 04 January 2026*


### 📞 Support
dedy@bamsbung.id
---
