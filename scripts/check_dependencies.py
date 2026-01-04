"""
Dependency Checker for Quant Lab.

Checks all required and optional dependencies with version info.
Run: python scripts/check_dependencies.py

Version: 0.7.0
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Required dependencies with minimum versions
REQUIRED_DEPS = {
    'numpy': '1.24.0',
    'pandas': '2.0.0',
    'scipy': '1.10.0',
    'scikit-learn': '1.3.0',
    'pyarrow': '14.0.0',
    'arcticdb': '4.0.0',
    'statsmodels': '0.14.0',
    'python-dotenv': '1.0.0',
    'pyyaml': '6.0.0',
    'tqdm': '4.66.0',
}

# Optional dependencies
OPTIONAL_DEPS = {
    'lightgbm': '4.0.0',
    'xgboost': '2.0.0',
    'hmmlearn': '0.3.0',
    'arch': '6.0.0',
    'matplotlib': '3.7.0',
    'plotly': '5.18.0',
    'vectorbt': '0.26.0',
    'nautilus_trader': '1.180.0',
    'qnt': '0.0.0',  # Quantiacs
}

# Development dependencies
DEV_DEPS = {
    'pytest': '7.0.0',
    'jupyter': '1.0.0',
    'jupyterlab': '4.0.0',
    'black': '23.0.0',
    'isort': '5.12.0',
}


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    try:
        parts = version_str.split('.')
        return tuple(int(p.split('+')[0].split('a')[0].split('b')[0].split('rc')[0]) 
                    for p in parts[:3])
    except:
        return (0, 0, 0)


def check_package(name: str, min_version: str = None) -> Dict:
    """Check if package is installed and get version."""
    result = {
        'name': name,
        'installed': False,
        'version': None,
        'min_version': min_version,
        'version_ok': True,
    }
    
    try:
        # Handle package name mapping
        import_name = name.replace('-', '_')
        if name == 'scikit-learn':
            import_name = 'sklearn'
        elif name == 'python-dotenv':
            import_name = 'dotenv'
        elif name == 'pyyaml':
            import_name = 'yaml'
        
        module = __import__(import_name)
        result['installed'] = True
        
        # Get version - try multiple methods
        if hasattr(module, '__version__'):
            result['version'] = module.__version__
        elif hasattr(module, 'VERSION'):
            result['version'] = str(module.VERSION)
        else:
            # Use importlib.metadata (modern API, Python 3.8+)
            try:
                from importlib.metadata import version as get_version
                result['version'] = get_version(name)
            except Exception:
                result['version'] = 'unknown'
        
        # Check version
        if min_version and result['version'] and result['version'] != 'unknown':
            installed_ver = parse_version(result['version'])
            required_ver = parse_version(min_version)
            result['version_ok'] = installed_ver >= required_ver
            
    except ImportError:
        result['installed'] = False
    except Exception as e:
        result['error'] = str(e)
    
    return result


def check_python_version() -> Dict:
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    return {
        'version': version_str,
        'ok': version.major == 3 and version.minor >= 10,
        'recommended': version.major == 3 and version.minor == 11,
    }


def check_conda_env() -> Dict:
    """Check conda environment."""
    import os
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    
    return {
        'active': bool(conda_env),
        'name': conda_env,
        'prefix': conda_prefix,
        'is_lab_quant': conda_env == 'lab-quant',
    }


def print_header(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(result: Dict, show_version: bool = True) -> str:
    """Print check result and return status."""
    name = result['name']
    
    if result['installed']:
        version = result.get('version', 'unknown')
        min_ver = result.get('min_version', '')
        
        if result.get('version_ok', True):
            status = "[OK]"
            if show_version:
                print(f"  {status:8s} {name:20s} v{version}")
            else:
                print(f"  {status:8s} {name}")
            return 'ok'
        else:
            status = "[WARN]"
            print(f"  {status:8s} {name:20s} v{version} (need >= {min_ver})")
            return 'warn'
    else:
        status = "[MISS]"
        print(f"  {status:8s} {name}")
        return 'miss'


def main():
    """Run dependency check."""
    print()
    print("=" * 60)
    print("  QUANT LAB - DEPENDENCY CHECKER")
    print("  Version: 0.7.0")
    print("=" * 60)
    
    stats = {'ok': 0, 'warn': 0, 'miss': 0}
    missing_required = []
    missing_optional = []
    
    # Python Version
    print_header("PYTHON VERSION")
    py = check_python_version()
    if py['recommended']:
        print(f"  [OK]     Python {py['version']} (recommended)")
        stats['ok'] += 1
    elif py['ok']:
        print(f"  [WARN]   Python {py['version']} (3.11 recommended)")
        stats['warn'] += 1
    else:
        print(f"  [FAIL]   Python {py['version']} (need >= 3.10)")
        stats['miss'] += 1
    
    # Conda Environment
    print_header("CONDA ENVIRONMENT")
    conda = check_conda_env()
    if conda['is_lab_quant']:
        print(f"  [OK]     Environment: {conda['name']}")
        stats['ok'] += 1
    elif conda['active']:
        print(f"  [WARN]   Environment: {conda['name']} (expected: lab-quant)")
        stats['warn'] += 1
    else:
        print(f"  [WARN]   No conda environment active")
        stats['warn'] += 1
    
    # Required Dependencies
    print_header("REQUIRED DEPENDENCIES")
    for name, min_ver in REQUIRED_DEPS.items():
        result = check_package(name, min_ver)
        status = print_result(result)
        stats[status] += 1
        if status == 'miss':
            missing_required.append(name)
    
    # Optional Dependencies
    print_header("OPTIONAL DEPENDENCIES")
    for name, min_ver in OPTIONAL_DEPS.items():
        result = check_package(name, min_ver)
        status = print_result(result)
        if status == 'ok':
            stats['ok'] += 1
        elif status == 'miss':
            missing_optional.append(name)
            # Don't count optional as missing in stats
    
    # Development Dependencies
    print_header("DEVELOPMENT DEPENDENCIES")
    for name, min_ver in DEV_DEPS.items():
        result = check_package(name, min_ver)
        status = print_result(result)
        # Don't count dev deps in stats
    
    # Summary
    print_header("SUMMARY")
    print(f"  Passed:   {stats['ok']}")
    print(f"  Warnings: {stats['warn']}")
    print(f"  Missing:  {stats['miss']}")
    
    if missing_required:
        print()
        print("  MISSING REQUIRED:")
        print(f"    pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print()
        print("  MISSING OPTIONAL (install as needed):")
        for pkg in missing_optional:
            print(f"    pip install {pkg}")
    
    # Final Status
    print()
    if stats['miss'] == 0:
        print("  [OK] All required dependencies installed!")
        return 0
    else:
        print("  [FAIL] Some required dependencies missing!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
