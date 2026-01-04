"""
ArcticDB Management Script - Manage and optimize ArcticDB storage.

Features:
- List all libraries and symbols with size info
- Compact/defragment database
- Delete unused libraries
- Delete old versions
- Backup and restore
- Health check

Run: python scripts/manage_arcticdb.py [command]

Commands:
  info      - Show database info and size
  compact   - Compact database (remove old versions)
  cleanup   - Delete empty libraries
  delete    - Delete specific symbol or library
  backup    - Backup to new location
  reset     - Reset database (DANGEROUS!)
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_folder_size(path: str) -> float:
    """Get folder size in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def get_arctic_connection():
    """Get ArcticDB connection."""
    try:
        from arcticdb import Arctic
        db_path = project_root / "data" / "arcticdb"
        return Arctic(f"lmdb://{db_path}")
    except ImportError:
        logger.error("ArcticDB not installed. Run: pip install arcticdb")
        sys.exit(1)


def cmd_info(args):
    """Show database information."""
    print("\n" + "=" * 60)
    print("ARCTICDB DATABASE INFO")
    print("=" * 60)
    
    db_path = project_root / "data" / "arcticdb"
    total_size = get_folder_size(db_path)
    print(f"\nDatabase path: {db_path}")
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    ac = get_arctic_connection()
    libraries = ac.list_libraries()
    
    print(f"\nLibraries: {len(libraries)}")
    print("-" * 60)
    
    total_rows = 0
    for lib_name in libraries:
        lib = ac[lib_name]
        symbols = lib.list_symbols()
        
        # Get library folder size
        lib_path = db_path / lib_name
        lib_size = get_folder_size(lib_path) if lib_path.exists() else 0
        
        print(f"\n📁 {lib_name}")
        print(f"   Size: {lib_size:.2f} MB")
        print(f"   Symbols: {len(symbols)}")
        
        if symbols:
            lib_rows = 0
            for sym in symbols:
                try:
                    info = lib.get_description(sym)
                    rows = info.row_count if hasattr(info, 'row_count') else 0
                    lib_rows += rows
                    
                    # Get versions
                    versions = lib.list_versions(sym)
                    ver_count = len(list(versions)) if versions else 1
                    
                    print(f"   - {sym}: {rows:,} rows, {ver_count} version(s)")
                except Exception as e:
                    print(f"   - {sym}: (error: {e})")
            
            total_rows += lib_rows
            efficiency = (lib_rows * 100) / (lib_size * 1024 * 1024 / 100) if lib_size > 0 else 0
            print(f"   Total rows: {lib_rows:,}")
        else:
            print("   (empty)")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"  Total libraries: {len(libraries)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Efficiency: ~{total_rows / total_size:.0f} rows/MB" if total_size > 0 else "")
    print("=" * 60)


def cmd_compact(args):
    """Compact database by removing old versions."""
    print("\n" + "=" * 60)
    print("COMPACTING ARCTICDB")
    print("=" * 60)
    
    ac = get_arctic_connection()
    
    for lib_name in ac.list_libraries():
        lib = ac[lib_name]
        symbols = lib.list_symbols()
        
        print(f"\n📁 {lib_name}")
        
        for sym in symbols:
            try:
                # Get all versions
                versions = list(lib.list_versions(sym))
                
                if len(versions) > 1:
                    # Keep only latest version
                    print(f"   Compacting {sym}: {len(versions)} versions -> 1")
                    
                    # Delete old versions (keep latest)
                    for ver in versions[1:]:  # Skip first (latest)
                        try:
                            lib.delete_version(sym, ver.version)
                        except:
                            pass
                else:
                    print(f"   {sym}: already compact (1 version)")
                    
            except Exception as e:
                print(f"   {sym}: error - {e}")
    
    print("\n✓ Compaction complete")
    print("Note: LMDB may not immediately release disk space.")
    print("For full space recovery, use 'reset' command to rebuild.")


def cmd_cleanup(args):
    """Delete empty libraries."""
    print("\n" + "=" * 60)
    print("CLEANING UP EMPTY LIBRARIES")
    print("=" * 60)
    
    ac = get_arctic_connection()
    db_path = project_root / "data" / "arcticdb"
    
    empty_libs = []
    for lib_name in ac.list_libraries():
        lib = ac[lib_name]
        if len(lib.list_symbols()) == 0:
            empty_libs.append(lib_name)
    
    if not empty_libs:
        print("\nNo empty libraries found.")
        return
    
    print(f"\nEmpty libraries: {empty_libs}")
    
    if not args.force:
        confirm = input("\nDelete these libraries? (y/N): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
    
    for lib_name in empty_libs:
        try:
            # Delete library
            ac.delete_library(lib_name)
            print(f"✓ Deleted library: {lib_name}")
            
            # Remove folder
            lib_path = db_path / lib_name
            if lib_path.exists():
                shutil.rmtree(lib_path)
                print(f"✓ Removed folder: {lib_path}")
                
        except Exception as e:
            print(f"✗ Error deleting {lib_name}: {e}")
    
    print("\n✓ Cleanup complete")


def cmd_delete(args):
    """Delete specific symbol or library."""
    print("\n" + "=" * 60)
    print("DELETE SYMBOL/LIBRARY")
    print("=" * 60)
    
    ac = get_arctic_connection()
    
    if args.library and args.symbol:
        # Delete specific symbol
        lib = ac[args.library]
        
        if args.symbol not in lib.list_symbols():
            print(f"Symbol {args.symbol} not found in {args.library}")
            return
        
        if not args.force:
            confirm = input(f"Delete {args.library}/{args.symbol}? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
        
        lib.delete(args.symbol)
        print(f"✓ Deleted {args.library}/{args.symbol}")
        
    elif args.library:
        # Delete entire library
        if args.library not in ac.list_libraries():
            print(f"Library {args.library} not found")
            return
        
        lib = ac[args.library]
        symbols = lib.list_symbols()
        
        if not args.force:
            confirm = input(f"Delete library {args.library} ({len(symbols)} symbols)? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
        
        ac.delete_library(args.library)
        
        # Remove folder
        db_path = project_root / "data" / "arcticdb"
        lib_path = db_path / args.library
        if lib_path.exists():
            shutil.rmtree(lib_path)
        
        print(f"✓ Deleted library {args.library}")
    else:
        print("Specify --library and optionally --symbol")


def cmd_backup(args):
    """Backup database to new location."""
    print("\n" + "=" * 60)
    print("BACKUP ARCTICDB")
    print("=" * 60)
    
    db_path = project_root / "data" / "arcticdb"
    
    if args.output:
        backup_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = project_root / "data" / f"arcticdb_backup_{timestamp}"
    
    print(f"Source: {db_path}")
    print(f"Destination: {backup_path}")
    
    if backup_path.exists():
        print(f"Error: Destination already exists")
        return
    
    print("\nCopying...")
    shutil.copytree(db_path, backup_path)
    
    backup_size = get_folder_size(backup_path)
    print(f"\n✓ Backup complete: {backup_size:.2f} MB")


def cmd_reset(args):
    """Reset database - rebuild from scratch."""
    print("\n" + "=" * 60)
    print("⚠️  RESET ARCTICDB (DANGEROUS!)")
    print("=" * 60)
    
    db_path = project_root / "data" / "arcticdb"
    
    if not args.force:
        print("\nThis will:")
        print("1. Export all data to temporary location")
        print("2. Delete the entire database")
        print("3. Recreate with optimized settings")
        print("4. Re-import all data")
        print("\nThis can significantly reduce disk usage.")
        
        confirm = input("\nProceed? (type 'RESET' to confirm): ")
        if confirm != 'RESET':
            print("Cancelled.")
            return
    
    # Step 1: Export data
    print("\n[1/4] Exporting data...")
    ac = get_arctic_connection()
    
    export_data = {}
    for lib_name in ac.list_libraries():
        lib = ac[lib_name]
        export_data[lib_name] = {}
        
        for sym in lib.list_symbols():
            try:
                df = lib.read(sym).data
                export_data[lib_name][sym] = df
                print(f"  Exported {lib_name}/{sym}: {len(df)} rows")
            except Exception as e:
                print(f"  Error exporting {lib_name}/{sym}: {e}")
    
    # Close connection before deleting
    del ac
    
    # Force garbage collection to release file handles
    import gc
    gc.collect()
    
    # Small delay to ensure files are released
    import time
    time.sleep(1)
    
    # Step 2: Delete database
    print("\n[2/4] Deleting old database...")
    try:
        shutil.rmtree(db_path)
        print(f"  Deleted {db_path}")
    except PermissionError as e:
        print(f"  Error: Files still locked. Please close any Python processes using ArcticDB.")
        print(f"  Then run this command again.")
        print(f"\n  Alternatively, manually delete: {db_path}")
        
        # Save export data to temp file for recovery
        import pickle
        temp_file = project_root / "data" / "arcticdb_export_temp.pkl"
        with open(temp_file, 'wb') as f:
            pickle.dump(export_data, f)
        print(f"\n  Data exported to: {temp_file}")
        print(f"  After deleting arcticdb folder, run: python scripts/manage_arcticdb.py restore")
        return
    
    # Step 3: Recreate
    print("\n[3/4] Creating new database...")
    db_path.mkdir(parents=True, exist_ok=True)
    ac = get_arctic_connection()
    
    # Step 4: Re-import
    print("\n[4/4] Re-importing data...")
    for lib_name, symbols in export_data.items():
        if not symbols:
            continue  # Skip empty libraries
            
        lib = ac.get_library(lib_name, create_if_missing=True)
        
        for sym, df in symbols.items():
            lib.write(sym, df)
            print(f"  Imported {lib_name}/{sym}: {len(df)} rows")
    
    # Show new size
    new_size = get_folder_size(db_path)
    print(f"\n✓ Reset complete!")
    print(f"  New size: {new_size:.2f} MB")


def cmd_optimize(args):
    """Optimize database - cleanup + compact + rebuild if needed."""
    print("\n" + "=" * 60)
    print("OPTIMIZE ARCTICDB")
    print("=" * 60)
    
    db_path = project_root / "data" / "arcticdb"
    initial_size = get_folder_size(db_path)
    
    print(f"Initial size: {initial_size:.2f} MB")
    
    # Step 1: Show current state
    ac = get_arctic_connection()
    
    total_rows = 0
    empty_libs = []
    
    for lib_name in ac.list_libraries():
        lib = ac[lib_name]
        symbols = lib.list_symbols()
        
        if not symbols:
            empty_libs.append(lib_name)
        else:
            for sym in symbols:
                try:
                    info = lib.get_description(sym)
                    total_rows += info.row_count if hasattr(info, 'row_count') else 0
                except:
                    pass
    
    print(f"Total rows: {total_rows:,}")
    print(f"Empty libraries: {empty_libs}")
    
    # Calculate expected size (rough estimate: ~100 bytes per row + overhead)
    expected_size = (total_rows * 100) / (1024 * 1024) + 10  # MB
    
    print(f"Expected size: ~{expected_size:.0f} MB")
    print(f"Current size: {initial_size:.0f} MB")
    print(f"Potential savings: ~{initial_size - expected_size:.0f} MB")
    
    if initial_size > expected_size * 5:  # More than 5x expected
        print("\n⚠️  Database is significantly oversized!")
        print("Recommendation: Run 'python scripts/manage_arcticdb.py reset'")
    elif empty_libs:
        print("\n⚠️  Empty libraries found!")
        print("Recommendation: Run 'python scripts/manage_arcticdb.py cleanup'")
    else:
        print("\n✓ Database looks healthy")


def cmd_restore(args):
    """Restore database from exported pickle file."""
    print("\n" + "=" * 60)
    print("RESTORE ARCTICDB FROM EXPORT")
    print("=" * 60)
    
    import pickle
    
    db_path = project_root / "data" / "arcticdb"
    temp_file = project_root / "data" / "arcticdb_export_temp.pkl"
    
    if not temp_file.exists():
        print(f"Error: Export file not found: {temp_file}")
        print("Run 'reset' command first to create export.")
        return
    
    # Check if arcticdb folder exists
    if db_path.exists() and any(db_path.iterdir()):
        print(f"Warning: {db_path} is not empty!")
        if not args.force:
            confirm = input("Overwrite existing data? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
        
        # Delete existing
        shutil.rmtree(db_path)
    
    # Create fresh database
    print("\n[1/2] Creating new database...")
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Load export data
    print("[2/2] Restoring data...")
    with open(temp_file, 'rb') as f:
        export_data = pickle.load(f)
    
    ac = get_arctic_connection()
    
    for lib_name, symbols in export_data.items():
        if not symbols:
            continue
        
        lib = ac.get_library(lib_name, create_if_missing=True)
        
        for sym, df in symbols.items():
            lib.write(sym, df)
            print(f"  Restored {lib_name}/{sym}: {len(df)} rows")
    
    # Show new size
    new_size = get_folder_size(db_path)
    print(f"\n✓ Restore complete!")
    print(f"  New size: {new_size:.2f} MB")
    
    # Optionally delete temp file
    if not args.keep:
        temp_file.unlink()
        print(f"  Deleted temp file: {temp_file}")


def main():
    parser = argparse.ArgumentParser(
        description="ArcticDB Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_arcticdb.py info
  python scripts/manage_arcticdb.py optimize
  python scripts/manage_arcticdb.py cleanup --force
  python scripts/manage_arcticdb.py delete --library signals
  python scripts/manage_arcticdb.py delete --library ohlcv --symbol EURUSD_1D
  python scripts/manage_arcticdb.py backup --output /path/to/backup
  python scripts/manage_arcticdb.py reset --force
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show database info')
    info_parser.set_defaults(func=cmd_info)
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Analyze and suggest optimizations')
    opt_parser.set_defaults(func=cmd_optimize)
    
    # Compact command
    compact_parser = subparsers.add_parser('compact', help='Remove old versions')
    compact_parser.set_defaults(func=cmd_compact)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Delete empty libraries')
    cleanup_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete symbol or library')
    delete_parser.add_argument('--library', '-l', help='Library name')
    delete_parser.add_argument('--symbol', '-s', help='Symbol name')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    delete_parser.set_defaults(func=cmd_delete)
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup database')
    backup_parser.add_argument('--output', '-o', help='Output path')
    backup_parser.set_defaults(func=cmd_backup)
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset and rebuild database')
    reset_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    reset_parser.set_defaults(func=cmd_reset)
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from export file')
    restore_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    restore_parser.add_argument('--keep', '-k', action='store_true', help='Keep temp file after restore')
    restore_parser.set_defaults(func=cmd_restore)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
