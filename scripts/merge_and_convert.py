"""
[DEPRECATED] Merge and Convert Dukascopy CSV Files - Use import_csv_to_arctic.py instead.

This script is deprecated. Please use:
    python scripts/import_csv_to_arctic.py --file data/raw/XAUUSD.csv --symbol XAUUSD

ArcticDB is now the primary data storage (3.7x faster than Parquet).
"""

import sys
print("""
[WARN] This script is DEPRECATED.

Please use import_csv_to_arctic.py instead:
    python scripts/import_csv_to_arctic.py --file data/raw/XAUUSD.csv --symbol XAUUSD
    python scripts/import_csv_to_arctic.py --all
    python scripts/import_csv_to_arctic.py --list

ArcticDB is now the primary data storage (3.7x faster than Parquet).
""")
sys.exit(0)

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

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


def find_csv_files(symbol: str, input_dir: Path) -> list:
    """Find all CSV files for a symbol."""
    pattern = f"{symbol.lower()}-*.csv"
    files = list(input_dir.glob(pattern))
    
    # Also check uppercase
    pattern_upper = f"{symbol.upper()}-*.csv"
    files.extend(input_dir.glob(pattern_upper))
    
    # Sort by filename (which includes date)
    files = sorted(set(files))
    
    return files


def load_dukascopy_csv(filepath: Path) -> pd.DataFrame:
    """Load CSV file from dukascopy-node CLI."""
    try:
        # dukascopy-node format: timestamp,open,high,low,close,volume
        df = pd.read_csv(filepath)
        
        # Rename columns to standard format
        df.columns = df.columns.str.lower()
        
        # Parse timestamp (Unix milliseconds)
        if 'timestamp' in df.columns:
            # Check if timestamp is Unix milliseconds (large number)
            if df['timestamp'].iloc[0] > 1e12:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def merge_files(files: list, symbol: str) -> pd.DataFrame:
    """Merge multiple CSV files into single DataFrame."""
    all_dfs = []
    
    for filepath in files:
        logger.info(f"  Loading: {filepath.name}")
        df = load_dukascopy_csv(filepath)
        
        if df is not None and len(df) > 0:
            all_dfs.append(df)
    
    if not all_dfs:
        return None
    
    # Concatenate all
    merged = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates
    merged = merged.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    merged = merged.sort_values('timestamp')
    
    # Add symbol column
    merged['symbol'] = symbol.upper()
    
    return merged


def save_to_parquet(df: pd.DataFrame, symbol: str, timeframe: str, output_dir: Path):
    """Save DataFrame to Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare for saving
    df_save = df.drop(columns=['symbol'], errors='ignore')
    df_save = df_save.set_index('timestamp').sort_index()
    
    # Map timeframe
    tf_map = {'h1': '1H', '1h': '1H', 'h4': '4H', '4h': '4H', 'd1': '1D', '1d': '1D'}
    tf_str = tf_map.get(timeframe.lower(), '1H')
    
    # Save
    filename = f"{symbol.upper()}_{tf_str}.parquet"
    filepath = output_dir / filename
    df_save.to_parquet(filepath, engine='pyarrow', compression='snappy')
    
    file_size = filepath.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {filepath} ({file_size:.2f} MB)")
    
    return filepath


def resample_and_save(df: pd.DataFrame, symbol: str, output_dir: Path):
    """Resample to multiple timeframes and save."""
    # Set timestamp as index
    df_indexed = df.set_index('timestamp').sort_index()
    
    # Detect source timeframe
    if len(df_indexed) > 1:
        time_diff = (df_indexed.index[1] - df_indexed.index[0]).total_seconds()
        if time_diff <= 3600:
            source_tf = '1H'
        elif time_diff <= 14400:
            source_tf = '4H'
        else:
            source_tf = '1D'
    else:
        source_tf = '1H'
    
    logger.info(f"  Source timeframe: {source_tf}")
    
    # Save source timeframe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{symbol.upper()}_{source_tf}.parquet"
    filepath = output_dir / filename
    df_indexed.to_parquet(filepath, engine='pyarrow', compression='snappy')
    logger.info(f"  [OK] {filename}: {len(df_indexed):,} bars")
    
    # Resample to higher timeframes
    resample_map = {
        '1H': ['4H', '1D'],
        '4H': ['1D'],
        '1D': []
    }
    
    for target_tf in resample_map.get(source_tf, []):
        try:
            # Use lowercase timeframe for pandas
            tf_lower = target_tf.replace('H', 'h').replace('D', 'd')
            resampled = df_indexed.resample(tf_lower).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            filename = f"{symbol.upper()}_{target_tf}.parquet"
            filepath = output_dir / filename
            resampled.to_parquet(filepath, engine='pyarrow', compression='snappy')
            logger.info(f"  [OK] {filename}: {len(resampled):,} bars")
            
        except Exception as e:
            logger.warning(f"  [WARN] Failed to resample to {target_tf}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Merge and convert Dukascopy CSV files')
    parser.add_argument('--symbol', '-s', type=str, help='Symbol to process')
    parser.add_argument('--all', action='store_true', help='Process all symbols found')
    parser.add_argument('--input', '-i', type=str, default='data/raw', help='Input directory')
    parser.add_argument('--output', '-o', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--timeframe', '-t', type=str, default='h1', help='Source timeframe')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    logger.info("=" * 60)
    logger.info("MERGE AND CONVERT DUKASCOPY FILES")
    logger.info("=" * 60)
    
    if args.all:
        # Find all unique symbols
        all_files = list(input_dir.glob("*-*.csv"))
        symbols = set()
        for f in all_files:
            # Extract symbol from filename like "eurusd-h1-bid-2024-01-01-2024-12-31.csv"
            parts = f.stem.split('-')
            if parts:
                symbols.add(parts[0].upper())
        
        symbols = sorted(symbols)
        logger.info(f"Found symbols: {symbols}")
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        print("Please specify --symbol or --all")
        return 1
    
    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")
        
        # Find files
        files = find_csv_files(symbol, input_dir)
        
        if not files:
            logger.warning(f"  No files found for {symbol}")
            continue
        
        logger.info(f"  Found {len(files)} files")
        
        # Merge
        df = merge_files(files, symbol)
        
        if df is None or len(df) == 0:
            logger.warning(f"  No data after merge")
            continue
        
        logger.info(f"  Merged: {len(df):,} bars")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Resample and save
        resample_and_save(df, symbol, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext step:")
    logger.info("  python scripts/data_quality_report.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
