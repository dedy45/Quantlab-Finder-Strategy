# QuantLab Dash UI

> Trading Dashboard dengan Plotly Dash + CYBORG Dark Theme
> Version: 2.1.0 | Status: Production Ready

---

## Quick Start

```bash
conda activate lab-quant
cd QuantLab
python dash_ui/app.py
# Access: http://localhost:8050
```

---

## Pages

| URL | Page | Status | Description |
|-----|------|--------|-------------|
| `/` | **Data Quality Gate** | ✅ COMPLETE | First page - Statistical validation (MUST PASS) |
| `/data-studio` | Data Studio | ✅ COMPLETE | ArcticDB explorer, candlestick chart |
| `/backtest` | Backtest Arena | ✅ COMPLETE | Run backtests with VectorBT |
| `/risk-lab` | Risk Lab | ✅ COMPLETE | VaR, Monte Carlo (POST-TRADE metrics) |
| `/settings` | Settings | ✅ COMPLETE | Config persistence (YAML) |

---

## Data Quality Gate (First Page)

Philosophy: **"Garbage In = Garbage Out"**

Data MUST pass validation before any analysis. Minimum threshold: **70% (Grade C)**.

### Validation Categories (7 Tests)

| Category | Weight | Description |
|----------|--------|-------------|
| Completeness | 20% | Missing values, gaps, bar completeness |
| Distribution | 15% | Jarque-Bera, skewness, kurtosis |
| Stationarity | 20% | ADF test for returns |
| Autocorrelation | 10% | Ljung-Box, ACF analysis |
| Outliers | 15% | Z-score, IQR detection |
| Sample Size | 10% | Minimum observations |
| OHLC Integrity | 10% | Valid OHLC relationships |

### Features
- **All Data Button**: Auto-fill date range from ArcticDB
- **English Advice**: Recommendations for each category
- **Proceed Indicator**: Clear YES/NO for data suitability
- **Diagnostic Charts**: Returns distribution, ACF plot

---

## Architecture

```
dash_ui/
├── app.py              # Entry point
├── index.py            # Router (imports ALL pages at startup)
├── theme.py            # CYBORG colors
├── cache.py            # Server-side caching
├── data_loader.py      # ArcticDB loading + cache
├── error_handler.py    # Error handling
├── pages/
│   ├── data_quality_gate.py  # / (FIRST PAGE)
│   ├── data_studio.py        # /data-studio
│   ├── backtest_arena.py     # /backtest
│   ├── risk_lab.py           # /risk-lab
│   └── settings.py           # /settings
├── components/
│   ├── charts.py       # Plotly wrappers (auto LTTB)
│   ├── cards.py        # KPI cards
│   ├── tables.py       # Paginated tables
│   └── sidebar.py      # Navigation
└── utils/
    ├── downsampling.py # LTTB (Numba JIT)
    ├── validators.py   # Input validation
    └── data_quality.py # Statistical tests
```

---

## Key Rules

1. **REAL DATA ONLY** - All data from ArcticDB, NO synthetic/dummy
2. **NO HARDCODE** - All config from `config/` module
3. **SERVER-SIDE CACHE** - Don't send DataFrame to browser
4. **LTTB DOWNSAMPLING** - Max 2000 points for charts

---

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| LTTB 100k → 2k | < 500ms | ~150ms ✅ |
| OHLCV Load | < 3000ms | ~1000ms ✅ |
| Chart Render | < 1000ms | ~570ms ✅ |
| Memory Reduction | > 90% | > 95% ✅ |

---

## Testing

```bash
python -m pytest tests/dash_ui/ -v
```

| File | Tests | Description |
|------|-------|-------------|
| test_dash_integration.py | 26 | Integration tests |
| test_performance.py | 14 | Performance benchmarks |
| test_property_based.py | 21 | Hypothesis property tests |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ArcticDB error | Check `core.data_engine.ArcticStore().list_symbols()` |
| Slow first load | Numba JIT compiling (one-time) |
| Cache issues | Delete `data/cache/dash_ui/*` |
| Import errors | Run from QuantLab root directory |

---

## Dependencies

- dash >= 2.14.0
- dash-bootstrap-components >= 1.5.0
- plotly >= 5.18.0
- flask-caching >= 2.0.0
- diskcache >= 5.6.0
- numba >= 0.58.0
- scipy, statsmodels (for statistical tests)

---

*QuantLab Project - Internal Use Only*
