"""
Import CSV to ArcticDB

Clean workflow untuk import data dari CSV ke ArcticDB:
1. Load CSV (auto-detect format)
2. Clean & validate OHLCV
3. Resample ke multiple timeframes
4. Store ke ArcticDB

Supported CSV formats:
- QuantDataManager (Dukascopy): timestamp,open,high,low,close,volume
- Generic OHLCV: any column order with standard names

Usage:
    python scripts/import_csv_to_arctic.py --file data/raw/XAUUSD.csv --symbol XAUUSD
    python scripts/import_csv_to_arctic.py --list
    python scripts/import_csv_to_arctic.py --status

Examples:
    # Import single file
    python scripts/import_csv_to_arctic.py -f data/raw/m15XAUUSD.csv -s XAUUSD
    
    # Import with custom source timeframe
    python scripts/import_csv_to_arctic.py -f data/raw/EURUSD_tick.csv -s EURUSD -t tick
    
    # List all data in ArcticDB
    python scripts/import_csv_to_arctic.py --list
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================
# CSV LOADING
# ============================================

def detect_csv_format(filepath: Path) -> dict:
    """Auto-detect CSV format."""
    # Read first few lines
    df_sample = pd.read_csv(filepath, nrows=5)
    
    info = {
        'columns': list(df_sample.columns),
        'separator': ',',
        'has_header': True,
        'timestamp_col': None,
        'format': 'unknown',
    }
    
    cols_lower = [c.lower() for c in df_sample.columns]
    
    # Detect timestamp column
    for col in ['timestamp', 'time', 'datetime', 'date']:
        if col in cols_lower:
            idx = cols_lower.index(col)
            info['timestamp_col'] = df_sample.columns[idx]
            break
    
    # Detect format
    if 'gmt time' in cols_lower or 'gmt' in str(df_sample.columns[0]).lower():
        info['format'] = 'quantdatamanager'
    elif info['timestamp_col']:
        info['format'] = 'standard'
    else:
        # Check if first column looks like timestamp
        first_val = str(df_sample.iloc[0, 0])
        if len(first_val) > 10 and any(c in first_val for c in ['-', '/', ':']):
            info['format'] = 'standard'
            info['timestamp_col'] = df_sample.columns[0]
    
    return info


def load_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Load CSV file with auto-detection."""
    logger.info(f"Loading: {filepath.name}")
    
    try:
        # Detect format
        info = detect_csv_format(filepath)
        logger.info(f"  Format: {info['format']}")
        logger.info(f"  Columns: {info['columns']}")
        
        # Load full file
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        df.columns = df.columns.str.replace(' ', '_')
        
        # Handle QuantDataManager format
        if 'gmt_time' in df.columns:
            df = df.rename(columns={'gmt_time': 'timestamp'})
        
        # Find timestamp column
        ts_col = None
        for col in ['timestamp', 'time', 'datetime', 'date']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            # Use first column as timestamp
            ts_col = df.columns[0]
            logger.warning(f"  Using first column as timestamp: {ts_col}")
        
        # Rename to standard
        if ts_col != 'timestamp':
            df = df.rename(columns={ts_col: 'timestamp'})
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)  # Remove timezone
        
        # Ensure OHLCV columns exist
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            logger.error(f"  Missing columns: {missing}")
            return None
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Select and order columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"  Loaded: {len(df):,} rows")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"  Error loading CSV: {e}")
        return None


# ============================================
# DATA CLEANING
# ============================================

def clean_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Clean and validate OHLCV data."""
    logger.info("Cleaning data...")
    
    original_len = len(df)
    
    # 1. Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    dups_removed = original_len - len(df)
    if dups_removed > 0:
        logger.info(f"  Removed {dups_removed:,} duplicates")
    
    # 2. Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 3. Remove NaN in OHLC
    nan_before = df[['open', 'high', 'low', 'close']].isna().sum().sum()
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if nan_before > 0:
        logger.info(f"  Removed {nan_before} NaN values")
    
    # 4. Fix OHLC relationships
    # High should be >= Open, Close, Low
    # Low should be <= Open, Close, High
    invalid_high = df['high'] < df[['open', 'close', 'low']].max(axis=1)
    invalid_low = df['low'] > df[['open', 'close', 'high']].min(axis=1)
    
    if invalid_high.sum() > 0:
        df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close', 'low']].max(axis=1)
        logger.info(f"  Fixed {invalid_high.sum():,} invalid high values")
    
    if invalid_low.sum() > 0:
        df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close', 'high']].min(axis=1)
        logger.info(f"  Fixed {invalid_low.sum():,} invalid low values")
    
    # 5. Remove zero/negative prices
    invalid_price = (df['close'] <= 0) | (df['open'] <= 0)
    if invalid_price.sum() > 0:
        df = df[~invalid_price]
        logger.info(f"  Removed {invalid_price.sum():,} zero/negative prices")
    
    # 6. Add symbol column
    df['symbol'] = symbol.upper()
    
    logger.info(f"  Clean data: {len(df):,} rows")
    
    return df


# ============================================
# RESAMPLING
# ============================================

def detect_source_timeframe(df: pd.DataFrame) -> str:
    """Detect source timeframe from data."""
    if len(df) < 2:
        return '1H'
    
    # Calculate median time difference
    time_diffs = df['timestamp'].diff().dropna()
    median_diff = time_diffs.median().total_seconds()
    
    if median_diff <= 60:
        return '1T'  # 1 minute
    elif median_diff <= 300:
        return '5T'  # 5 minutes
    elif median_diff <= 900:
        return '15T'  # 15 minutes
    elif median_diff <= 1800:
        return '30T'  # 30 minutes
    elif median_diff <= 3600:
        return '1H'
    elif median_diff <= 14400:
        return '4H'
    else:
        return '1D'


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample OHLCV to target timeframe."""
    # Set timestamp as index
    df_indexed = df.set_index('timestamp')
    
    # Map timeframe string
    tf_map = {
        '1H': '1h', '4H': '4h', '1D': '1d',
        '15T': '15min', '30T': '30min', '1T': '1min'
    }
    tf_str = tf_map.get(target_tf, target_tf.lower())
    
    # Resample
    resampled = df_indexed.resample(tf_str).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'symbol': 'first',
    }).dropna()
    
    # Reset index
    resampled = resampled.reset_index()
    
    return resampled


# ============================================
# ARCTICDB STORAGE
# ============================================

def store_to_arctic(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
    """Store DataFrame to ArcticDB."""
    try:
        from core.data_engine.arctic_store import ArcticStore
        
        store = ArcticStore()
        success = store.write(symbol, df, timeframe)
        
        if success:
            logger.info(f"  [OK] Stored {symbol}_{timeframe}: {len(df):,} rows")
        
        return success
        
    except Exception as e:
        logger.error(f"  [FAIL] Store error: {e}")
        return False


def list_arctic_data():
    """List all data in ArcticDB."""
    try:
        from core.data_engine.arctic_store import ArcticStore
        
        store = ArcticStore()
        symbols = store.list_symbols()
        
        logger.info("=" * 70)
        logger.info("ARCTICDB DATA INVENTORY")
        logger.info("=" * 70)
        
        if not symbols:
            logger.info("No data in ArcticDB")
            return
        
        logger.info(f"Total symbols: {len(symbols)}")
        logger.info("")
        
        # Group by base symbol
        by_symbol = {}
        for sym in symbols:
            parts = sym.rsplit('_', 1)
            if len(parts) == 2:
                base, tf = parts
                if base not in by_symbol:
                    by_symbol[base] = []
                by_symbol[base].append(tf)
        
        for base in sorted(by_symbol.keys()):
            logger.info(f"{base}:")
            for tf in sorted(by_symbol[base]):
                info = store.get_info(base, tf)
                if info:
                    logger.info(f"  {tf}: {info['rows']:,} rows, "
                               f"{info['start'].strftime('%Y-%m-%d')} to "
                               f"{info['end'].strftime('%Y-%m-%d')}")
        
        # Storage size
        db_size = store.get_db_size()
        logger.info("")
        logger.info(f"Database size: {db_size['size_mb']:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error listing data: {e}")


# ============================================
# MAIN
# ============================================

def import_csv(filepath: Path, symbol: str, source_tf: str = None) -> bool:
    """Import CSV file to ArcticDB."""
    logger.info("=" * 70)
    logger.info(f"IMPORT: {filepath.name} -> {symbol}")
    logger.info("=" * 70)
    
    # 1. Load CSV
    df = load_csv(filepath)
    if df is None:
        return False
    
    # 2. Clean data
    df = clean_ohlcv(df, symbol)
    if len(df) == 0:
        logger.error("No data after cleaning")
        return False
    
    # 3. Detect source timeframe
    if source_tf is None:
        source_tf = detect_source_timeframe(df)
    logger.info(f"Source timeframe: {source_tf}")
    
    # 4. Store source timeframe
    logger.info(f"\nStoring to ArcticDB...")
    
    # Determine target timeframes based on source
    if source_tf in ['1T', '5T', '15T', '30T']:
        # Resample to 1H, 4H, 1D
        targets = ['1H', '4H', '1D']
        
        # Also store source if it's 15T (useful for detailed analysis)
        if source_tf == '15T':
            store_to_arctic(df, symbol, '15T')
        
    elif source_tf == '1H':
        targets = ['1H', '4H', '1D']
        store_to_arctic(df, symbol, '1H')
        targets = ['4H', '1D']  # Only resample to higher
        
    elif source_tf == '4H':
        store_to_arctic(df, symbol, '4H')
        targets = ['1D']
        
    else:
        store_to_arctic(df, symbol, source_tf)
        targets = []
    
    # 5. Resample and store
    for target_tf in targets:
        logger.info(f"  Resampling to {target_tf}...")
        df_resampled = resample_ohlcv(df, target_tf)
        store_to_arctic(df_resampled, symbol, target_tf)
    
    logger.info("\n[OK] Import complete")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Import CSV to ArcticDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import single file
  python scripts/import_csv_to_arctic.py -f data/raw/XAUUSD.csv -s XAUUSD
  
  # Import with source timeframe hint
  python scripts/import_csv_to_arctic.py -f data/raw/EURUSD_m15.csv -s EURUSD -t 15T
  
  # List all data in ArcticDB
  python scripts/import_csv_to_arctic.py --list
        """
    )
    parser.add_argument('--file', '-f', type=str, help='CSV file to import')
    parser.add_argument('--symbol', '-s', type=str, help='Symbol name (e.g., XAUUSD)')
    parser.add_argument('--timeframe', '-t', type=str, help='Source timeframe (auto-detect if not specified)')
    parser.add_argument('--list', action='store_true', help='List all data in ArcticDB')
    parser.add_argument('--status', action='store_true', help='Show ArcticDB status')
    
    args = parser.parse_args()
    
    if args.list or args.status:
        list_arctic_data()
        return 0
    
    if not args.file:
        parser.print_help()
        return 1
    
    filepath = Path(args.file)
    if not filepath.exists():
        # Try relative to project root
        filepath = project_root / args.file
        if not filepath.exists():
            logger.error(f"File not found: {args.file}")
            return 1
    
    # Determine symbol
    symbol = args.symbol
    if not symbol:
        # Try to extract from filename
        symbol = filepath.stem.upper()
        # Remove common prefixes/suffixes
        for prefix in ['M15', 'M1', 'H1', 'H4', 'D1', 'TICK']:
            symbol = symbol.replace(prefix, '')
        symbol = symbol.strip('_-')
        
        if not symbol:
            logger.error("Cannot determine symbol. Use --symbol option.")
            return 1
        
        logger.info(f"Auto-detected symbol: {symbol}")
    
    success = import_csv(filepath, symbol.upper(), args.timeframe)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
