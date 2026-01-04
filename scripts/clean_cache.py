"""
Clean Cache Script - Clear cache and temporary files.

Run: python scripts/clean_cache.py [options]

Options:
  --all         Clear everything (cache + processed + logs)
  --cache       Clear cache only (default)
  --processed   Clear processed data
  --logs        Clear log files
  --dry-run     Show what would be deleted without deleting
  --force       Skip confirmation prompt

Examples:
  python scripts/clean_cache.py              # Clear cache only
  python scripts/clean_cache.py --all        # Clear everything
  python scripts/clean_cache.py --dry-run    # Preview deletions
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class CacheCleaner:
    """Cache and temporary file cleaner."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.deleted_files = 0
        self.deleted_dirs = 0
        self.freed_bytes = 0
        self.errors = []
    
    def get_size(self, path: Path) -> int:
        """Get total size of path (file or directory)."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
        return 0
    
    def delete_file(self, path: Path) -> bool:
        """Delete a single file."""
        try:
            size = path.stat().st_size
            if not self.dry_run:
                path.unlink()
            self.deleted_files += 1
            self.freed_bytes += size
            return True
        except Exception as e:
            self.errors.append((str(path), str(e)))
            return False
    
    def delete_dir(self, path: Path) -> bool:
        """Delete a directory and all contents."""
        try:
            size = self.get_size(path)
            file_count = len(list(path.glob('**/*')))
            if not self.dry_run:
                shutil.rmtree(path)
            self.deleted_dirs += 1
            self.deleted_files += file_count
            self.freed_bytes += size
            return True
        except Exception as e:
            self.errors.append((str(path), str(e)))
            return False
    
    def clear_directory(self, path: Path, patterns: List[str] = None) -> int:
        """Clear contents of directory, optionally matching patterns."""
        if not path.exists():
            return 0
        
        count = 0
        if patterns:
            for pattern in patterns:
                for item in path.glob(pattern):
                    if item.is_file():
                        if self.delete_file(item):
                            count += 1
                    elif item.is_dir():
                        if self.delete_dir(item):
                            count += 1
        else:
            for item in path.iterdir():
                if item.is_file():
                    if self.delete_file(item):
                        count += 1
                elif item.is_dir():
                    if self.delete_dir(item):
                        count += 1
        
        return count
    
    def format_size(self, bytes: int) -> str:
        """Format bytes to human readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024
        return f"{bytes:.1f} TB"
    
    def print_summary(self):
        """Print cleanup summary."""
        print("\n" + "-"*60)
        mode = "[DRY RUN] " if self.dry_run else ""
        print(f"{mode}CLEANUP SUMMARY")
        print("-"*60)
        print(f"  Files deleted: {self.deleted_files}")
        print(f"  Directories deleted: {self.deleted_dirs}")
        print(f"  Space freed: {self.format_size(self.freed_bytes)}")
        
        if self.errors:
            print(f"\n  Errors: {len(self.errors)}")
            for path, error in self.errors[:5]:
                print(f"    - {path}: {error}")
        
        if self.dry_run:
            print("\n  [INFO] Dry run - no files were actually deleted")
        else:
            print("\n  Cleanup complete")


def get_cache_info() -> List[Tuple[str, Path, int, int]]:
    """Get info about cacheable directories."""
    info = []
    
    dirs = [
        ('Cache', ROOT / 'data' / 'cache'),
        ('Processed', ROOT / 'data' / 'processed'),
        ('__pycache__', ROOT),  # Will search recursively
        ('Logs', ROOT / 'logs'),
        ('MLflow', ROOT / 'mlruns'),
        ('Backtest Results', ROOT / 'backtest' / 'results'),
    ]
    
    for name, path in dirs:
        if name == '__pycache__':
            # Count all __pycache__ dirs
            pycache_dirs = list(path.glob('**/__pycache__'))
            total_size = sum(
                sum(f.stat().st_size for f in d.glob('**/*') if f.is_file())
                for d in pycache_dirs
            )
            total_files = sum(len(list(d.glob('**/*'))) for d in pycache_dirs)
            info.append((name, path, len(pycache_dirs), total_size))
        elif path.exists():
            files = list(path.glob('**/*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            info.append((name, path, file_count, total_size))
        else:
            info.append((name, path, 0, 0))
    
    return info


def print_cache_status():
    """Print current cache status."""
    print("\n" + "="*60)
    print("CACHE STATUS")
    print("="*60)
    
    info = get_cache_info()
    total_size = 0
    
    for name, path, count, size in info:
        size_str = f"{size / (1024**2):.1f} MB" if size > 0 else "empty"
        status = f"{count} files, {size_str}"
        print(f"  {name:20} {status}")
        total_size += size
    
    print("-"*60)
    print(f"  {'TOTAL':20} {total_size / (1024**2):.1f} MB")


def clean_cache(cleaner: CacheCleaner):
    """Clean cache directory."""
    cache_dir = ROOT / 'data' / 'cache'
    print(f"\n[Cache] Cleaning {cache_dir}...")
    cleaner.clear_directory(cache_dir)


def clean_processed(cleaner: CacheCleaner):
    """Clean processed data directory."""
    processed_dir = ROOT / 'data' / 'processed'
    print(f"\n[Processed] Cleaning {processed_dir}...")
    cleaner.clear_directory(processed_dir)


def clean_pycache(cleaner: CacheCleaner):
    """Clean all __pycache__ directories."""
    print(f"\n[PyCache] Cleaning __pycache__ directories...")
    pycache_dirs = list(ROOT.glob('**/__pycache__'))
    for d in pycache_dirs:
        cleaner.delete_dir(d)


def clean_logs(cleaner: CacheCleaner):
    """Clean log files."""
    logs_dir = ROOT / 'logs'
    print(f"\n[Logs] Cleaning {logs_dir}...")
    if logs_dir.exists():
        cleaner.clear_directory(logs_dir, ['*.log', '*.log.*'])


def clean_temp_files(cleaner: CacheCleaner):
    """Clean temporary files."""
    print(f"\n[Temp] Cleaning temporary files...")
    patterns = ['*.tmp', '*.temp', '*.bak', '*~', '.DS_Store', 'Thumbs.db']
    for pattern in patterns:
        for f in ROOT.glob(f'**/{pattern}'):
            cleaner.delete_file(f)


def confirm_action(message: str) -> bool:
    """Ask for user confirmation."""
    response = input(f"\n{message} [y/N]: ").strip().lower()
    return response in ('y', 'yes')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Clean cache and temporary files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--all', action='store_true', help='Clear everything')
    parser.add_argument('--cache', action='store_true', help='Clear cache only')
    parser.add_argument('--processed', action='store_true', help='Clear processed data')
    parser.add_argument('--logs', action='store_true', help='Clear log files')
    parser.add_argument('--pycache', action='store_true', help='Clear __pycache__ dirs')
    parser.add_argument('--dry-run', action='store_true', help='Preview without deleting')
    parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    # Default to cache only if no specific option
    if not any([args.all, args.cache, args.processed, args.logs, args.pycache]):
        args.cache = True
    
    print("\n" + "="*60)
    print("QUANT LAB - CACHE CLEANER")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Root: {ROOT}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    
    # Show current status
    print_cache_status()
    
    # Determine what to clean
    targets = []
    if args.all:
        targets = ['cache', 'processed', 'logs', 'pycache', 'temp']
    else:
        if args.cache:
            targets.append('cache')
        if args.processed:
            targets.append('processed')
        if args.logs:
            targets.append('logs')
        if args.pycache:
            targets.append('pycache')
    
    print(f"\nTargets: {', '.join(targets)}")
    
    # Confirm if not forced
    if not args.force and not args.dry_run:
        if not confirm_action("Proceed with cleanup?"):
            print("Cancelled.")
            return 0
    
    # Create cleaner and run
    cleaner = CacheCleaner(dry_run=args.dry_run)
    
    if 'cache' in targets:
        clean_cache(cleaner)
    
    if 'processed' in targets:
        clean_processed(cleaner)
    
    if 'logs' in targets:
        clean_logs(cleaner)
    
    if 'pycache' in targets:
        clean_pycache(cleaner)
    
    if 'temp' in targets:
        clean_temp_files(cleaner)
    
    # Print summary
    cleaner.print_summary()
    
    # Show disk space after cleanup
    if not args.dry_run:
        total, used, free = shutil.disk_usage(ROOT)
        print(f"\n  Disk space: {free / (1024**3):.1f} GB free")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
