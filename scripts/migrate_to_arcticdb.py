"""
Migrate Parquet Files to ArcticDB

This script migrates existing Parquet files to ArcticDB database.

Usage:
    python scripts/migrate_to_arcticdb.py --all
    python scripts/migrate_to_arcticdb.py --symbol XAUUSD
    python scripts/migrate_to_arcticdb.py --list
    python scripts/migrate_to_arcticdb.py --verify

Steps:
1. Scan data/processed/ for Parquet files
2. Load each file
3. Write to ArcticDB
4. Verify data integrity
5. (Optional) Remove old Parquet files
"""

import argparse
import logging
import sys
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


def list_parquet_files(data_dir: Path) -> list:
    """List all Parquet files in data directory."""
    files = list(data_dir.glob("*.parquet"))
    return sorted(files)


def parse_filename(filename: str) -> tuple:
    """Parse filename to extract symbol and timeframe."""
    # Format: SYMBOL_TIMEFRAME.parquet
    stem = Path(filename).stem
    parts = stem.rsplit('_', 1)
    
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, '1H'


def migrate_file(filepath: Path, store) -> dict:
    """Migrate single Parquet file to ArcticDB."""
    result = {
        'file': filepath.name,
        'status': 'pending',
        'rows': 0,
        'error': None,
    }
    
    try:
        # Parse filename
        symbol, timeframe = parse_filename(filepath.name)
        result['symbol'] = symbol
        result['timeframe'] = timeframe
        
        # Load Parquet
        df = pd.read_parquet(filepath)
        result['rows'] = len(df)
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
            df = df.reset_index()
        
        if 'timestamp' not in df.columns:
            # Try to use index as timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.columns = ['timestamp'] + list(df.columns[1:])
            else:
                result['status'] = 'failed'
                result['error'] = 'No timestamp column found'
                return result
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Write to ArcticDB
        success = store.write(symbol, df, timeframe)
        
        if success:
            result['status'] = 'success'
        else:
            result['status'] = 'failed'
            result['error'] = 'Write failed'
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
    
    return result


def verify_migration(filepath: Path, store) -> dict:
    """Verify migrated data matches original."""
    result = {
        'file': filepath.name,
        'status': 'pending',
        'match': False,
    }
    
    try:
        symbol, timeframe = parse_filename(filepath.name)
        
        # Load original
        df_original = pd.read_parquet(filepath)
        
        # Ensure timestamp column
        if 'timestamp' not in df_original.columns:
            if df_original.index.name == 'timestamp':
                df_original = df_original.reset_index()
            elif isinstance(df_original.index, pd.DatetimeIndex):
                df_original = df_original.reset_index()
                df_original.columns = ['timestamp'] + list(df_original.columns[1:])
        
        # Load from ArcticDB
        df_arctic = store.read(symbol, timeframe)
        
        if df_arctic is None:
            result['status'] = 'failed'
            result['error'] = 'Not found in ArcticDB'
            return result
        
        # Compare row counts
        result['original_rows'] = len(df_original)
        result['arctic_rows'] = len(df_arctic)
        
        if len(df_original) == len(df_arctic):
            result['match'] = True
            result['status'] = 'verified'
        else:
            result['status'] = 'mismatch'
            result['error'] = f"Row count mismatch: {len(df_original)} vs {len(df_arctic)}"
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Migrate Parquet files to ArcticDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrate_to_arcticdb.py --list
  python scripts/migrate_to_arcticdb.py --all
  python scripts/migrate_to_arcticdb.py --symbol XAUUSD
  python scripts/migrate_to_arcticdb.py --verify
  python scripts/migrate_to_arcticdb.py --all --cleanup
        """
    )
    parser.add_argument('--list', action='store_true', help='List Parquet files')
    parser.add_argument('--all', action='store_true', help='Migrate all files')
    parser.add_argument('--symbol', '-s', type=str, help='Migrate specific symbol')
    parser.add_argument('--verify', action='store_true', help='Verify migration')
    parser.add_argument('--cleanup', action='store_true', help='Remove Parquet after migration')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Data directory')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = project_root / args.data_dir
    
    logger.info("=" * 70)
    logger.info("MIGRATE PARQUET TO ARCTICDB")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    
    # List files
    files = list_parquet_files(data_dir)
    
    if not files:
        logger.warning("[WARN] No Parquet files found")
        return 0
    
    logger.info(f"Found {len(files)} Parquet files")
    
    # List mode
    if args.list:
        logger.info("\nParquet files:")
        for f in files:
            symbol, tf = parse_filename(f.name)
            try:
                df = pd.read_parquet(f)
                size_kb = f.stat().st_size / 1024
                logger.info(f"  {f.name}: {len(df):,} rows, {size_kb:.1f} KB")
            except Exception as e:
                logger.info(f"  {f.name}: ERROR - {e}")
        return 0
    
    # Import ArcticStore
    try:
        from core.data_engine.arctic_store import ArcticStore
        store = ArcticStore()
    except ImportError as e:
        logger.error(f"[FAIL] Cannot import ArcticStore: {e}")
        logger.error("Run: pip install arcticdb")
        return 1
    
    # Filter by symbol if specified
    if args.symbol:
        files = [f for f in files if f.stem.startswith(args.symbol.upper())]
        if not files:
            logger.warning(f"[WARN] No files found for symbol: {args.symbol}")
            return 0
    
    # Verify mode
    if args.verify:
        logger.info("\nVerifying migration...")
        results = []
        
        for f in files:
            result = verify_migration(f, store)
            results.append(result)
            
            status = "[OK]" if result['match'] else "[FAIL]"
            logger.info(f"  {status} {result['file']}")
            if result.get('error'):
                logger.info(f"       Error: {result['error']}")
        
        # Summary
        verified = sum(1 for r in results if r['match'])
        logger.info(f"\nVerification: {verified}/{len(results)} files match")
        return 0
    
    # Migrate mode
    if args.all or args.symbol:
        logger.info("\nMigrating files...")
        results = []
        
        for f in files:
            result = migrate_file(f, store)
            results.append(result)
            
            status = "[OK]" if result['status'] == 'success' else "[FAIL]"
            logger.info(f"  {status} {result['file']}: {result['rows']:,} rows")
            if result.get('error'):
                logger.info(f"       Error: {result['error']}")
        
        # Summary
        success = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"\nMigration: {success}/{len(results)} files successful")
        
        # Cleanup if requested
        if args.cleanup and success == len(results):
            logger.info("\nCleaning up Parquet files...")
            for f in files:
                f.unlink()
                logger.info(f"  Deleted: {f.name}")
            logger.info("[OK] Cleanup complete")
        elif args.cleanup:
            logger.warning("[WARN] Skipping cleanup due to migration errors")
        
        # Show ArcticDB status
        logger.info("\n" + "=" * 70)
        logger.info("ARCTICDB STATUS")
        logger.info("=" * 70)
        
        symbols = store.list_symbols()
        logger.info(f"Total symbols: {len(symbols)}")
        
        for sym in symbols:
            parts = sym.rsplit('_', 1)
            if len(parts) == 2:
                info = store.get_info(parts[0], parts[1])
                if info:
                    logger.info(f"  {sym}: {info['rows']:,} rows, "
                               f"{info['start']} to {info['end']}")
        
        db_size = store.get_db_size()
        logger.info(f"\nDatabase size: {db_size['size_mb']:.2f} MB")
        
        return 0
    
    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
