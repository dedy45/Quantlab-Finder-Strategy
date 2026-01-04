# PROPOSAL: Data Quality Gate (Halaman Pertama)

> **Konsep**: "Garbage In = Garbage Out"
> **Filosofi**: Data WAJIB lolos validasi statistik sebelum fitur lain boleh digunakan
> **Status**: ✅ FULLY IMPLEMENTED (v2.0)

---

## IMPLEMENTATION STATUS

### ✅ Completed Files

| File | Status | Description |
|------|--------|-------------|
| `dash_ui/utils/data_quality.py` | ✅ Done | 7 statistical test categories with advice |
| `dash_ui/pages/data_quality_gate.py` | ✅ Done | New first page with enhanced UI |
| `dash_ui/components/sidebar.py` | ✅ Done | Updated navigation |
| `dash_ui/index.py` | ✅ Done | Fixed callback registration + routing |
| `dash_ui/pages/__init__.py` | ✅ Done | Updated exports |
| `dash_ui/utils/__init__.py` | ✅ Done | Added data_quality exports |

### Key Features Implemented

1. **7 Validation Categories** with weights:
   - Completeness (20%) - Missing values, gaps, bar completeness
   - Distribution (15%) - Jarque-Bera, skewness, kurtosis
   - Stationarity (20%) - ADF test
   - Autocorrelation (10%) - Ljung-Box, ACF
   - Outliers (15%) - Z-score, IQR
   - Sample Size (10%) - Minimum observations
   - OHLC Integrity (10%) - Valid price relationships

2. **Grading System**:
   - Grade A (90-100%): EXCELLENT - Ready for any analysis
   - Grade B (80-89%): GOOD - Minor issues, proceed with caution
   - Grade C (70-79%): FAIR - Review issues before proceeding
   - Grade D (60-69%): POOR - NOT recommended
   - Grade F (0-59%): FAIL - DO NOT proceed

3. **Minimum Threshold**: 70% (Grade C) to proceed

4. **Enhanced UI Features**:
   - Status labels (PASS/WARNING/FAIL) for each category
   - English advice when score < 80%
   - Clear "Can Proceed" indicator
   - English summary in detailed report
   - Expandable test details with individual advice

5. **Gate Mechanism**: Session store tracks validation status

---

## 1. MASALAH DENGAN DESAIN SAAT INI

### 1.1 Dashboard Saat Ini (SALAH)

Dashboard saat ini menampilkan:
- PSR, Sharpe, Max DD, Volatility
- Equity Curve

**Masalah:**
1. Ini adalah metrik **POST-TRADE** (setelah backtest/trading)
2. Seharusnya ada di **Risk Lab** setelah backtest
3. User bisa langsung ke backtest tanpa validasi data
4. Tidak ada "gatekeeper" untuk kualitas data

### 1.2 Alur yang Benar

```
SEHARUSNYA:
───────────
[1] Data Quality Gate (WAJIB PASS)
    ↓
[2] Data Studio (Eksplorasi)
    ↓
[3] Backtest Arena (Uji Strategi)
    ↓
[4] Risk Lab (Analisis Hasil)
    ↓
[5] Settings (Konfigurasi)
```

---

## 2. KONSEP DATA QUALITY GATE

### 2.1 Tujuan

**"Memastikan data STATISTICALLY SOUND (layak uji statistik)"**

Bukan hanya:
- ❌ Bersih dari gap
- ❌ Tidak ada missing values
- ❌ Format benar

Tapi juga:
- ✅ Distribusi returns normal/mendekati normal
- ✅ Tidak ada outlier ekstrem yang merusak statistik
- ✅ Stationarity (untuk time series analysis)
- ✅ Autocorrelation yang wajar
- ✅ Sufficient sample size untuk inferensi statistik


### 2.2 Metrik Validasi Data

| Kategori | Test | Threshold | Alasan |
|----------|------|-----------|--------|
| **Completeness** | Missing Values | < 1% | Data harus lengkap |
| **Completeness** | Gap Detection | < 5% expected bars | Tidak boleh banyak gap |
| **Distribution** | Jarque-Bera Test | p > 0.01 | Normalitas returns |
| **Distribution** | Skewness | \|skew\| < 2 | Tidak terlalu miring |
| **Distribution** | Kurtosis | kurt < 10 | Tidak terlalu fat-tailed |
| **Stationarity** | ADF Test | p < 0.05 | Returns harus stationary |
| **Autocorrelation** | Ljung-Box Test | p > 0.05 | Tidak ada serial correlation |
| **Outliers** | Z-Score | \|z\| < 5 | Outlier ekstrem |
| **Sample Size** | Min Observations | > 252 | Minimal 1 tahun data |
| **Integrity** | OHLC Logic | H >= max(O,C), L <= min(O,C) | Data OHLC valid |

### 2.3 Scoring System

```
QUALITY SCORE = Weighted Average of All Tests

Weights:
- Completeness: 20%
- Distribution: 25%
- Stationarity: 20%
- Autocorrelation: 15%
- Outliers: 10%
- Sample Size: 10%

Grades:
- A (90-100): Excellent - Ready for any analysis
- B (75-89): Good - Minor issues, proceed with caution
- C (60-74): Fair - Significant issues, review recommended
- D (40-59): Poor - Major issues, not recommended
- F (0-39): Fail - Data not suitable for analysis
```

---

## 3. DESAIN UI DATA QUALITY GATE

### 3.1 Layout Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  🔬 DATA QUALITY GATE                              [PASS/FAIL]  │
│  "Garbage In = Garbage Out"                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Symbol    │  │  Timeframe  │  │ Date Range  │  [VALIDATE] │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    QUALITY SCORE                         │   │
│  │                                                          │   │
│  │     ████████████████████░░░░░░░░░░  78% (Grade B)       │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  VALIDATION CHECKLIST                                    │   │
│  │                                                          │   │
│  │  ✅ Completeness      95%   ████████████████████░░░░    │   │
│  │  ✅ Distribution      82%   ████████████████░░░░░░░░    │   │
│  │  ✅ Stationarity      100%  ████████████████████████    │   │
│  │  ⚠️ Autocorrelation   65%   █████████████░░░░░░░░░░░    │   │
│  │  ✅ Outliers          90%   ██████████████████░░░░░░    │   │
│  │  ✅ Sample Size       100%  ████████████████████████    │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  Returns Distribution │  │  Autocorrelation     │            │
│  │  [Histogram + QQ]     │  │  [ACF/PACF Plot]     │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DETAILED REPORT                                         │   │
│  │  - Test results with p-values                            │   │
│  │  - Recommendations                                       │   │
│  │  - Data issues found                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [PROCEED TO DATA STUDIO →]  (disabled if FAIL)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```


### 3.2 Komponen UI

#### A. Header Section
- Title: "Data Quality Gate"
- Subtitle: "Garbage In = Garbage Out"
- Overall Status Badge: PASS (green) / FAIL (red) / WARNING (yellow)

#### B. Control Panel
- Symbol Dropdown (dari ArcticDB)
- Timeframe Dropdown
- Date Range Picker
- "Validate Data" Button (primary action)

#### C. Quality Score Card
- Large circular progress indicator
- Percentage score (0-100%)
- Grade letter (A/B/C/D/F)
- Color-coded (green/yellow/orange/red)

#### D. Validation Checklist
- 6 kategori dengan progress bars
- Status icon (✅/⚠️/❌)
- Percentage per kategori
- Expandable untuk detail

#### E. Diagnostic Charts
- Returns Distribution (Histogram + Normal overlay)
- Q-Q Plot (untuk normalitas)
- ACF/PACF Plot (untuk autocorrelation)
- Time Series dengan outliers highlighted

#### F. Detailed Report
- Collapsible accordion
- Test results dengan p-values
- Specific issues found
- Recommendations untuk perbaikan

#### G. Action Buttons
- "Proceed to Data Studio" - disabled jika FAIL
- "Export Report" - download PDF/CSV
- "Re-validate" - refresh analysis

---

## 4. STATISTICAL TESTS DETAIL

### 4.1 Completeness Tests

```python
def check_completeness(df, timeframe):
    """
    Check data completeness.
    
    Tests:
    1. Missing values percentage
    2. Expected vs actual bars
    3. Gap detection
    """
    results = {}
    
    # Missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    results['missing_values'] = {
        'value': missing_pct,
        'pass': missing_pct < 1,
        'threshold': '< 1%'
    }
    
    # Expected bars
    expected_bars = calculate_expected_bars(df.index.min(), df.index.max(), timeframe)
    actual_bars = len(df)
    completeness = actual_bars / expected_bars * 100
    results['bar_completeness'] = {
        'value': completeness,
        'pass': completeness > 95,
        'threshold': '> 95%'
    }
    
    # Gap detection
    gaps = detect_gaps(df, timeframe)
    gap_pct = len(gaps) / len(df) * 100
    results['gaps'] = {
        'value': gap_pct,
        'pass': gap_pct < 5,
        'threshold': '< 5%',
        'details': gaps
    }
    
    return results
```


### 4.2 Distribution Tests

```python
def check_distribution(returns):
    """
    Check returns distribution properties.
    
    Tests:
    1. Jarque-Bera test for normality
    2. Skewness
    3. Kurtosis
    """
    from scipy import stats
    
    results = {}
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    results['jarque_bera'] = {
        'statistic': jb_stat,
        'p_value': jb_pvalue,
        'pass': jb_pvalue > 0.01,  # Not rejecting normality
        'interpretation': 'Normal' if jb_pvalue > 0.01 else 'Non-normal'
    }
    
    # Skewness
    skew = stats.skew(returns)
    results['skewness'] = {
        'value': skew,
        'pass': abs(skew) < 2,
        'threshold': '|skew| < 2',
        'interpretation': 'Symmetric' if abs(skew) < 0.5 else ('Right-skewed' if skew > 0 else 'Left-skewed')
    }
    
    # Kurtosis (excess)
    kurt = stats.kurtosis(returns)
    results['kurtosis'] = {
        'value': kurt,
        'pass': kurt < 10,
        'threshold': 'kurt < 10',
        'interpretation': 'Normal tails' if kurt < 3 else 'Fat tails'
    }
    
    return results
```

### 4.3 Stationarity Tests

```python
def check_stationarity(returns):
    """
    Check if returns are stationary.
    
    Tests:
    1. Augmented Dickey-Fuller (ADF) test
    2. KPSS test (optional)
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    results = {}
    
    # ADF test (null: non-stationary)
    adf_result = adfuller(returns.dropna())
    results['adf_test'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'pass': adf_result[1] < 0.05,  # Reject null = stationary
        'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
    }
    
    # KPSS test (null: stationary)
    try:
        kpss_result = kpss(returns.dropna(), regression='c')
        results['kpss_test'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'pass': kpss_result[1] > 0.05,  # Not reject null = stationary
            'interpretation': 'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
        }
    except:
        results['kpss_test'] = {'pass': None, 'error': 'Test failed'}
    
    return results
```

### 4.4 Autocorrelation Tests

```python
def check_autocorrelation(returns, lags=20):
    """
    Check for serial correlation in returns.
    
    Tests:
    1. Ljung-Box test
    2. ACF values
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    
    results = {}
    
    # Ljung-Box test
    lb_result = acorr_ljungbox(returns.dropna(), lags=lags, return_df=True)
    significant_lags = (lb_result['lb_pvalue'] < 0.05).sum()
    
    results['ljung_box'] = {
        'significant_lags': significant_lags,
        'total_lags': lags,
        'pass': significant_lags < lags * 0.1,  # Less than 10% significant
        'interpretation': 'No serial correlation' if significant_lags < 2 else 'Serial correlation detected'
    }
    
    # ACF values
    acf_values = acf(returns.dropna(), nlags=lags)
    max_acf = max(abs(acf_values[1:]))  # Exclude lag 0
    
    results['acf'] = {
        'max_absolute_acf': max_acf,
        'pass': max_acf < 0.2,
        'threshold': '< 0.2',
        'values': acf_values.tolist()
    }
    
    return results
```


### 4.5 Outlier Detection

```python
def check_outliers(returns, threshold=5):
    """
    Detect outliers in returns.
    
    Methods:
    1. Z-score method
    2. IQR method
    3. Modified Z-score (MAD)
    """
    import numpy as np
    
    results = {}
    
    # Z-score method
    z_scores = (returns - returns.mean()) / returns.std()
    outliers_z = abs(z_scores) > threshold
    
    results['zscore'] = {
        'outlier_count': outliers_z.sum(),
        'outlier_pct': outliers_z.mean() * 100,
        'pass': outliers_z.mean() < 0.01,  # Less than 1%
        'threshold': f'|z| > {threshold}',
        'outlier_indices': returns.index[outliers_z].tolist()
    }
    
    # IQR method
    Q1 = returns.quantile(0.25)
    Q3 = returns.quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = (returns < Q1 - 1.5 * IQR) | (returns > Q3 + 1.5 * IQR)
    
    results['iqr'] = {
        'outlier_count': outliers_iqr.sum(),
        'outlier_pct': outliers_iqr.mean() * 100,
        'pass': outliers_iqr.mean() < 0.05,  # Less than 5%
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }
    
    # Modified Z-score (MAD)
    median = returns.median()
    mad = np.median(np.abs(returns - median))
    modified_z = 0.6745 * (returns - median) / mad if mad > 0 else 0
    outliers_mad = abs(modified_z) > 3.5
    
    results['mad'] = {
        'outlier_count': outliers_mad.sum(),
        'outlier_pct': outliers_mad.mean() * 100,
        'pass': outliers_mad.mean() < 0.02,  # Less than 2%
        'median': median,
        'mad': mad
    }
    
    return results
```

### 4.6 Sample Size Check

```python
def check_sample_size(df, timeframe):
    """
    Check if sample size is sufficient for statistical inference.
    
    Requirements:
    - Minimum 252 observations (1 year daily)
    - Minimum 30 for basic statistics
    - Recommended 1000+ for robust analysis
    """
    n = len(df)
    
    # Adjust minimum based on timeframe
    min_required = {
        '1D': 252,   # 1 year
        '4H': 252 * 6,  # 1 year
        '1H': 252 * 24,  # 1 year
        '15T': 252 * 24 * 4,  # 1 year
    }.get(timeframe, 252)
    
    results = {
        'observations': n,
        'minimum_required': min_required,
        'pass': n >= min_required,
        'sufficiency': 'Excellent' if n >= min_required * 2 else ('Good' if n >= min_required else 'Insufficient'),
        'recommendation': None
    }
    
    if n < 30:
        results['recommendation'] = 'CRITICAL: Need at least 30 observations for basic statistics'
    elif n < min_required:
        results['recommendation'] = f'Need {min_required - n} more observations for reliable analysis'
    elif n < min_required * 2:
        results['recommendation'] = 'Consider extending date range for more robust results'
    
    return results
```

### 4.7 OHLC Integrity Check

```python
def check_ohlc_integrity(df):
    """
    Check OHLC data integrity.
    
    Rules:
    1. High >= max(Open, Close)
    2. Low <= min(Open, Close)
    3. High >= Low
    4. All prices > 0
    """
    results = {}
    
    # High >= max(Open, Close)
    high_valid = df['high'] >= df[['open', 'close']].max(axis=1)
    results['high_valid'] = {
        'valid_count': high_valid.sum(),
        'invalid_count': (~high_valid).sum(),
        'pass': high_valid.all(),
        'invalid_indices': df.index[~high_valid].tolist()
    }
    
    # Low <= min(Open, Close)
    low_valid = df['low'] <= df[['open', 'close']].min(axis=1)
    results['low_valid'] = {
        'valid_count': low_valid.sum(),
        'invalid_count': (~low_valid).sum(),
        'pass': low_valid.all(),
        'invalid_indices': df.index[~low_valid].tolist()
    }
    
    # High >= Low
    hl_valid = df['high'] >= df['low']
    results['high_low_valid'] = {
        'valid_count': hl_valid.sum(),
        'invalid_count': (~hl_valid).sum(),
        'pass': hl_valid.all(),
        'invalid_indices': df.index[~hl_valid].tolist()
    }
    
    # All prices > 0
    positive = (df[['open', 'high', 'low', 'close']] > 0).all(axis=1)
    results['positive_prices'] = {
        'valid_count': positive.sum(),
        'invalid_count': (~positive).sum(),
        'pass': positive.all(),
        'invalid_indices': df.index[~positive].tolist()
    }
    
    return results
```


---

## 5. IMPLEMENTASI ARSITEKTUR

### 5.1 File Structure Baru

```
dash_ui/
├── pages/
│   ├── data_quality_gate.py    # HALAMAN PERTAMA (baru)
│   ├── data_studio.py          # Halaman kedua
│   ├── backtest_arena.py       # Halaman ketiga
│   ├── risk_lab.py             # Halaman keempat
│   └── settings.py             # Halaman kelima
│
├── components/
│   ├── quality_score.py        # Quality score gauge (baru)
│   ├── validation_checklist.py # Checklist component (baru)
│   ├── diagnostic_charts.py    # ACF, QQ plots (baru)
│   └── ...
│
└── utils/
    ├── data_quality.py         # Statistical tests (baru)
    └── ...
```

### 5.2 Navigation Flow

```python
# sidebar.py - Update navigation
NAV_ITEMS = [
    {'name': 'Data Quality', 'path': '/', 'icon': '🔬', 'description': 'Validate data first'},
    {'name': 'Data Studio', 'path': '/data-studio', 'icon': '💾', 'description': 'Explore data'},
    {'name': 'Backtest Arena', 'path': '/backtest', 'icon': '🎯', 'description': 'Run backtests'},
    {'name': 'Risk Lab', 'path': '/risk-lab', 'icon': '⚠️', 'description': 'Risk analysis'},
    {'name': 'Settings', 'path': '/settings', 'icon': '⚙️', 'description': 'Configuration'},
]
```

### 5.3 Gate Mechanism

```python
# Simpan status validasi di session store
dcc.Store(id='data-quality-status', storage_type='session', data={
    'validated': False,
    'symbol': None,
    'timeframe': None,
    'quality_score': 0,
    'grade': 'F',
    'timestamp': None,
})

# Di halaman lain, cek status sebelum allow akses
@callback(...)
def check_gate_status(pathname, quality_status):
    if pathname in ['/backtest', '/risk-lab']:
        if not quality_status.get('validated') or quality_status.get('grade') == 'F':
            return redirect_to_quality_gate()
```

---

## 6. KEUNTUNGAN DESAIN INI

### 6.1 Untuk User

1. **Confidence** - User tahu data layak untuk analisis
2. **Education** - User belajar tentang kualitas data
3. **Prevention** - Mencegah kesalahan analisis dari data buruk
4. **Transparency** - Jelas apa yang divalidasi dan mengapa

### 6.2 Untuk Sistem

1. **Reliability** - Hasil backtest lebih reliable
2. **Consistency** - Standar kualitas data yang jelas
3. **Debugging** - Mudah trace masalah ke kualitas data
4. **Documentation** - Report kualitas data tersimpan

### 6.3 Untuk Quant Workflow

1. **Best Practice** - Sesuai standar industri
2. **Reproducibility** - Hasil bisa direproduksi
3. **Audit Trail** - Ada bukti validasi data
4. **Risk Management** - Mengurangi risiko dari data buruk

---

## 7. REKOMENDASI IMPLEMENTASI

### 7.1 Prioritas Tinggi (Wajib)

1. **Completeness Check** - Missing values, gaps
2. **OHLC Integrity** - Data OHLC valid
3. **Sample Size** - Cukup data untuk analisis
4. **Basic Distribution** - Skewness, kurtosis

### 7.2 Prioritas Sedang (Recommended)

1. **Stationarity Test** - ADF test
2. **Outlier Detection** - Z-score method
3. **Quality Score** - Aggregate scoring

### 7.3 Prioritas Rendah (Nice to Have)

1. **Autocorrelation** - Ljung-Box test
2. **Advanced Distribution** - Jarque-Bera
3. **Export Report** - PDF generation

---

## 8. KESIMPULAN

### 8.1 Perubahan yang Diperlukan

1. **Rename** dashboard.py → data_quality_gate.py
2. **Rewrite** seluruh layout dan callback
3. **Create** utils/data_quality.py untuk statistical tests
4. **Create** components baru untuk visualisasi
5. **Update** navigation di sidebar.py
6. **Add** gate mechanism untuk halaman lain

### 8.2 Estimasi Effort

| Task | Effort | Priority |
|------|--------|----------|
| data_quality.py (tests) | 4-6 jam | HIGH |
| data_quality_gate.py (page) | 4-6 jam | HIGH |
| Quality score component | 2-3 jam | HIGH |
| Diagnostic charts | 3-4 jam | MEDIUM |
| Gate mechanism | 2-3 jam | MEDIUM |
| Testing | 2-3 jam | HIGH |
| **Total** | **17-25 jam** | - |

### 8.3 Next Steps

1. Review dan approve proposal ini
2. Implementasi utils/data_quality.py
3. Implementasi pages/data_quality_gate.py
4. Update navigation dan routing
5. Testing end-to-end
6. Documentation update

---

*Proposal ini mengusulkan perombakan halaman pertama menjadi Data Quality Gate yang memastikan data statistically sound sebelum analisis lebih lanjut.*
