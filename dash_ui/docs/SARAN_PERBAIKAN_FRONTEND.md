# SARAN & TIPS PERBAIKAN FRONTEND

> **Prioritas**: CRITICAL
> **Estimasi Waktu**: 30 menit - 2 jam
> **Risiko**: Rendah

---

## RINGKASAN MASALAH

Frontend Dash UI tidak berfungsi karena **callback tidak terdaftar** ke app instance. Semua tombol, dropdown, dan form tidak merespons user interaction.

**Bukan masalah:**
- Component ID (sudah benar)
- Callback definition (sudah benar)
- Layout structure (sudah benar)
- Theme/styling (sudah benar)

**Masalah utama:**
- Pages di-import dinamis SETELAH app running
- Callback decorator `@callback` tidak terdaftar ke app

---

## SOLUSI UTAMA (WAJIB)

### Langkah 1: Modifikasi index.py

Buka file `dash_ui/index.py` dan tambahkan import pages di bagian atas:

```python
# TAMBAHKAN setelah import statements yang ada

# Import semua pages untuk register callbacks SEBELUM app running
from .pages import dashboard
from .pages import data_studio
from .pages import backtest_arena
from .pages import risk_lab
from .pages import settings
```

### Langkah 2: Verifikasi

```bash
# Jalankan app
python dash_ui/app.py

# Buka browser http://localhost:8050
# Test setiap halaman dan tombol
```


---

## TIPS TAMBAHAN

### 1. Debugging Callback

Jika callback masih tidak berfungsi setelah fix:

```python
# Tambahkan di app.py sebelum app.run()
print("Registered callbacks:")
for callback_id in app.callback_map:
    print(f"  - {callback_id}")
```

### 2. Browser DevTools

1. Buka DevTools (F12)
2. Tab Network → Filter "dash"
3. Klik tombol di UI
4. Lihat apakah ada POST request ke `/_dash-update-component`
5. Jika tidak ada request → callback tidak terdaftar
6. Jika ada error response → lihat detail error

### 3. Console Logging

Tambahkan logging di callback untuk debug:

```python
@callback(...)
def my_callback(...):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Callback triggered!")
    # ... rest of callback
```

### 4. Suppress Callback Exceptions

Pastikan setting ini ada di app.py:

```python
app = Dash(
    __name__,
    suppress_callback_exceptions=True,  # PENTING!
    ...
)
```

Ini mencegah error saat component belum ada di DOM.


---

## CHECKLIST TESTING

Setelah perbaikan, test setiap fungsi:

### Dashboard (/)
- [ ] KPI cards menampilkan nilai (bukan "-")
- [ ] Equity chart menampilkan data
- [ ] Data status menampilkan info symbol

### Data Studio (/data-studio)
- [ ] Symbol dropdown terisi otomatis
- [ ] Klik "Load Data" memuat chart
- [ ] Candlestick chart tampil
- [ ] Data table tampil

### Backtest Arena (/backtest)
- [ ] Symbol dropdown terisi otomatis
- [ ] Config values terisi dari config
- [ ] Strategy params berubah saat ganti strategy
- [ ] Klik "Run Backtest" menjalankan backtest
- [ ] Results section tampil setelah backtest

### Risk Lab (/risk-lab)
- [ ] Symbol dropdown terisi otomatis
- [ ] Klik "Calculate Risk" menghitung metrics
- [ ] Semua KPI cards terisi
- [ ] Charts (drawdown, volatility, monte carlo) tampil
- [ ] Risk alerts tampil
- [ ] Export CSV berfungsi
- [ ] Export PDF berfungsi

### Settings (/settings)
- [ ] Semua input terisi dari config
- [ ] Klik "Save Settings" menyimpan
- [ ] Klik "Reset to Defaults" mereset
- [ ] Alert success/error tampil

---

## POTENSI MASALAH LAIN

### 1. Import Error

Jika ada error import saat startup:

```
ImportError: cannot import name 'xxx' from 'yyy'
```

**Solusi**: Periksa circular import, pastikan semua dependency tersedia.

### 2. ArcticDB Connection

Jika data tidak muncul:

```python
# Test koneksi ArcticDB
from core.data_engine import ArcticStore
store = ArcticStore()
print(store.list_symbols())
```

### 3. Config Not Found

Jika settings tidak terisi:

```python
# Test config
from config import get_config
cfg = get_config()
print(cfg.backtest.initial_capital)
```


---

## REKOMENDASI JANGKA PANJANG

### 1. Migrasi ke Dash Pages Pattern

Untuk maintainability lebih baik, pertimbangkan migrasi ke Dash Pages:

```python
# app.py
app = Dash(__name__, use_pages=True)

# pages/data_studio.py
import dash
dash.register_page(__name__, path='/data-studio')
```

**Keuntungan:**
- Auto callback registration
- Cleaner routing
- Better code organization

### 2. Unit Testing Callbacks

Tambahkan unit test untuk setiap callback:

```python
# tests/dash_ui/test_callbacks.py
def test_load_data_callback():
    from dash_ui.pages.data_studio import load_data
    result = load_data(1, 'XAUUSD', '1H', '2024-01-01', '2024-12-31')
    assert result[0] is not None  # figure
```

### 3. Error Boundary

Implementasi error boundary untuk graceful degradation:

```python
@callback(...)
def safe_callback(...):
    try:
        return actual_logic(...)
    except Exception as e:
        log_error('callback_name', e)
        return default_output, error_alert
```

### 4. Performance Monitoring

Tambahkan timing untuk callback:

```python
import time

@callback(...)
def monitored_callback(...):
    start = time.time()
    result = actual_logic(...)
    duration = time.time() - start
    logger.info(f"Callback took {duration:.2f}s")
    return result
```

---

## KESIMPULAN

1. **Fix utama**: Import pages di awal index.py
2. **Test**: Verifikasi semua tombol dan fungsi
3. **Monitor**: Gunakan browser DevTools untuk debug
4. **Jangka panjang**: Pertimbangkan Dash Pages pattern

**Setelah fix, frontend akan berfungsi 100%.**

---

*Dokumen ini berisi saran dan tips untuk memperbaiki QuantLab Dash UI Frontend.*
