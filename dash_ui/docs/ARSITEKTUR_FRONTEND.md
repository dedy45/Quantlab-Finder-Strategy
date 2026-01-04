# ARSITEKTUR FRONTEND: QuantLab Dash UI

> **Version**: 2.1.0
> **Framework**: Dash + Plotly + Bootstrap
> **Theme**: CYBORG (Dark Mode)
> **Status**: Production Ready

---

## 1. OVERVIEW

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                            │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │   Dash Frontend │                          │
│                    │   (React.js)    │                          │
│                    └────────┬────────┘                          │
│                              │ HTTP Callbacks                    │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │   Dash Server   │                          │
│                    │   (Flask)       │                          │
│                    └────────┬────────┘                          │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Data Engine │     │  Backtest   │     │   Config    │       │
│  │  ArcticDB   │     │  VectorBT   │     │   YAML      │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Directory Structure

```
dash_ui/
├── __init__.py          # Public API exports
├── app.py               # Entry point (< 50 lines)
├── index.py             # Router & layout (imports ALL pages at startup!)
├── theme.py             # CYBORG theme colors
├── cache.py             # Server-side caching
├── data_loader.py       # ArcticDB data loading + cache
├── error_handler.py     # Error handling & logging
│
├── assets/
│   └── custom.css       # Style overrides
│
├── pages/               # Page modules (< 300 lines each)
│   ├── __init__.py
│   ├── data_quality_gate.py  # / - FIRST PAGE (validation gate)
│   ├── data_studio.py        # /data-studio
│   ├── backtest_arena.py     # /backtest
│   ├── risk_lab.py           # /risk-lab
│   ├── settings.py           # /settings
│   └── dashboard.py          # Legacy (redirects to /)
│
├── components/          # Reusable UI components
│   ├── __init__.py
│   ├── sidebar.py       # Navigation sidebar
│   ├── charts.py        # Plotly wrappers (auto LTTB)
│   ├── cards.py         # KPI cards
│   ├── tables.py        # Paginated tables
│   └── toast.py         # Notifications
│
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── downsampling.py  # LTTB algorithm (Numba JIT)
│   ├── validators.py    # Input validation
│   └── data_quality.py  # Statistical tests (7 categories)
│
└── docs/                # Documentation
    ├── ARSITEKTUR_FRONTEND.md
    └── PROPOSAL_DATA_QUALITY_GATE.md
```

---

## 2. PAGE FLOW

### 2.1 User Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER JOURNEY                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA QUALITY GATE (/)                                       │
│     ├─ Select Symbol + Timeframe                                │
│     ├─ Click "All Data" → Auto-fill date range                  │
│     ├─ Click "Validate" → Run 7 statistical tests               │
│     ├─ Score >= 70%? → ✅ Proceed enabled                       │
│     └─ Score < 70%? → ❌ Cannot proceed                         │
│                    │                                             │
│                    ▼ (if PASS)                                   │
│  2. DATA STUDIO (/data-studio)                                  │
│     ├─ Explore OHLCV data                                       │
│     ├─ Candlestick chart (auto LTTB)                            │
│     └─ Data table (paginated)                                   │
│                    │                                             │
│                    ▼                                             │
│  3. BACKTEST ARENA (/backtest)                                  │
│     ├─ Select strategy + parameters                             │
│     ├─ Run backtest (background)                                │
│     └─ View results (equity, drawdown, stats)                   │
│                    │                                             │
│                    ▼                                             │
│  4. RISK LAB (/risk-lab)                                        │
│     ├─ POST-TRADE risk analysis                                 │
│     ├─ VaR, Monte Carlo, Sharpe, Sortino                        │
│     └─ Export reports (CSV, PDF)                                │
│                                                                  │
│  5. SETTINGS (/settings)                                        │
│     └─ Config persistence (YAML)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Pages Summary

| URL | Page | Purpose |
|-----|------|---------|
| `/` | Data Quality Gate | **FIRST PAGE** - Validate data before analysis |
| `/data-studio` | Data Studio | Explore OHLCV data from ArcticDB |
| `/backtest` | Backtest Arena | Run backtests with VectorBT |
| `/risk-lab` | Risk Lab | POST-TRADE risk metrics |
| `/settings` | Settings | Config persistence |

---

## 3. DATA QUALITY GATE (Phase 0, Page 1)

### 3.1 Philosophy

**"Garbage In = Garbage Out"**

Data MUST pass statistical validation before any analysis.

### 3.2 Validation Categories

| Category | Weight | Tests |
|----------|--------|-------|
| Completeness | 20% | Missing values, gaps, bar completeness |
| Distribution | 15% | Jarque-Bera, skewness, kurtosis |
| Stationarity | 20% | ADF test for returns |
| Autocorrelation | 10% | Ljung-Box, ACF |
| Outliers | 15% | Z-score, IQR |
| Sample Size | 10% | Minimum observations |
| OHLC Integrity | 10% | Valid OHLC relationships |

### 3.3 Grading System

| Grade | Score | Status | Can Proceed? |
|-------|-------|--------|--------------|
| A | 90-100% | Excellent | ✅ YES |
| B | 80-89% | Good | ✅ YES |
| C | 70-79% | Fair | ✅ YES (with caution) |
| D | 50-69% | Poor | ❌ NO |
| F | 0-49% | Fail | ❌ NO |

### 3.4 Features

- **All Data Button**: Auto-fill Start/End dates from ArcticDB
- **English Advice**: Recommendations for each category
- **Proceed Indicator**: Clear YES/NO status
- **Diagnostic Charts**: Returns distribution, ACF plot
- **Detailed Report**: Expandable test details

---

## 4. CALLBACK REGISTRATION (CRITICAL)

### 4.1 Correct Pattern

```
app.py
  └─ index.py
       ├─ Import ALL pages at module level (BEFORE app.run)
       │   ├─ from .pages import data_quality_gate
       │   ├─ from .pages import data_studio
       │   ├─ from .pages import backtest_arena
       │   ├─ from .pages import risk_lab
       │   └─ from .pages import settings
       │
       └─ Routing callback uses imported layouts
```

**WHY?** Callbacks are registered when `@callback` decorator executes. If pages are imported dynamically inside routing callback, callbacks won't register properly.

---

## 5. DATA FLOW

### 5.1 Loading Flow

```
User Action → Callback → Check Cache → [Hit: Return] / [Miss: Load from ArcticDB]
                                              ↓
                                        Save to Cache
                                              ↓
                                        Apply LTTB (if > 2000 points)
                                              ↓
                                        Create Plotly Figure
                                              ↓
                                        Return JSON to Browser
```

### 5.2 Caching Layers

| Layer | Type | Purpose |
|-------|------|---------|
| 1 | LRU (Memory) | `get_available_symbols()` |
| 2 | Disk (Pickle) | OHLCV data, backtest results |
| 3 | Flask-Caching | Callback memoization |

---

## 6. PERFORMANCE OPTIMIZATIONS

| Optimization | Implementation | Result |
|--------------|----------------|--------|
| LTTB Downsampling | Numba JIT | 100k → 2k in ~150ms |
| Server-side Cache | Disk (Pickle) | Avoid re-loading |
| Background Callbacks | DiskcacheManager | Non-blocking backtest |
| Chart Sampling | Random sample for histograms | Prevent browser hang |

---

## 7. BEST PRACTICES

### 7.1 Code Organization
- **< 300 lines per page** - Split if larger
- **Reusable components** - Use `components/`
- **Config from module** - No hardcoding
- **Real data only** - No synthetic/dummy

### 7.2 Callback Design
- **prevent_initial_call=True** - For user-triggered actions
- **Error handling** - Always wrap with try-except
- **Logging** - Log errors to file

### 7.3 Performance
- **LTTB downsampling** - Max 2000 points for charts
- **Server-side cache** - Don't send DataFrame to browser
- **Background callbacks** - For processes > 10 seconds

---

## 8. CHANGELOG

| Version | Date | Changes |
|---------|------|---------|
| 2.1.0 | 2026-01 | Data Quality Gate as first page, All Data button |
| 2.0.0 | 2025-12 | Callback registration fix, performance optimizations |
| 1.0.0 | 2025-11 | Initial release |

---

*Dokumen ini menjelaskan arsitektur QuantLab Dash UI v2.1.0*
