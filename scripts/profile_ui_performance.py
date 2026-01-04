#!/usr/bin/env python
"""
UI Performance Profiling Script.

Profiles the initial load time of QuantLab UI components
to ensure they meet the <3 second target.

Requirements: 10.5 - Initial page load within 3 seconds

Usage:
    python scripts/profile_ui_performance.py
    python scripts/profile_ui_performance.py --verbose
    python scripts/profile_ui_performance.py --benchmark
"""
import sys
import os
import time
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.utils.performance import (
    PerformanceProfiler,
    benchmark_data_loading,
    benchmark_downsampling,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PROFILING FUNCTIONS
# =============================================================================

def profile_imports() -> float:
    """
    Profile import times for UI modules.
    
    Returns
    -------
    float
        Total import time in milliseconds
    """
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Profile core imports
    with profiler.measure("import_reflex"):
        import reflex as rx
    
    with profiler.measure("import_plotly"):
        import plotly.graph_objects as go
    
    with profiler.measure("import_numpy"):
        import numpy as np
    
    with profiler.measure("import_pandas"):
        import pandas as pd
    
    # Profile UI module imports
    with profiler.measure("import_ui_state"):
        from ui.state import AppState, DataState
    
    with profiler.measure("import_ui_adapters"):
        from ui.adapters import DataAdapter
    
    with profiler.measure("import_ui_components"):
        from ui.components import sidebar
    
    with profiler.measure("import_ui_utils"):
        from ui.utils.downsampling import lttb_downsample
    
    profiler.stop()
    report = profiler.get_report()
    
    print("\n" + "=" * 60)
    print("IMPORT PROFILING RESULTS")
    print("=" * 60)
    print(report.summary())
    
    return report.total_time_ms


def profile_adapter_initialization() -> float:
    """
    Profile adapter initialization times.
    
    Returns
    -------
    float
        Total initialization time in milliseconds
    """
    profiler = PerformanceProfiler()
    profiler.start()
    
    with profiler.measure("init_data_adapter"):
        from ui.adapters import DataAdapter
        adapter = DataAdapter()
    
    with profiler.measure("check_availability"):
        is_available = adapter.is_available()
    
    if is_available:
        with profiler.measure("list_symbols"):
            symbols = adapter.list_symbols()
        
        with profiler.measure("get_base_symbols"):
            base_symbols = adapter.get_base_symbols()
    
    profiler.stop()
    report = profiler.get_report()
    
    print("\n" + "=" * 60)
    print("ADAPTER INITIALIZATION PROFILING RESULTS")
    print("=" * 60)
    print(report.summary())
    
    return report.total_time_ms


def profile_data_loading(symbol: str = "XAUUSD", timeframe: str = "1H") -> float:
    """
    Profile data loading performance.
    
    Parameters
    ----------
    symbol : str
        Symbol to load
    timeframe : str
        Timeframe to load
        
    Returns
    -------
    float
        Total loading time in milliseconds
    """
    profiler = PerformanceProfiler()
    profiler.start()
    
    from ui.adapters import DataAdapter
    adapter = DataAdapter()
    
    with profiler.measure("load_ohlcv"):
        try:
            data = adapter.load_ohlcv(symbol, timeframe)
            row_count = data.get('row_count', 0)
        except Exception as e:
            logger.warning(f"Data loading failed: {e}")
            row_count = 0
    
    if row_count > 0:
        with profiler.measure("get_data_info"):
            info = adapter.get_data_info(symbol, timeframe)
        
        # Profile downsampling if data is large
        if row_count > 1000:
            with profiler.measure("downsample_ohlcv"):
                from ui.utils.downsampling import downsample_ohlcv
                downsampled = downsample_ohlcv(data, 1000)
    
    profiler.stop()
    report = profiler.get_report()
    
    print("\n" + "=" * 60)
    print(f"DATA LOADING PROFILING RESULTS ({symbol}_{timeframe})")
    print("=" * 60)
    print(f"Rows loaded: {row_count:,}")
    print(report.summary())
    
    return report.total_time_ms


def profile_chart_creation() -> float:
    """
    Profile Plotly chart creation performance.
    
    Returns
    -------
    float
        Total chart creation time in milliseconds
    """
    import numpy as np
    import plotly.graph_objects as go
    
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Generate test data (1000 points - typical after downsampling)
    n_points = 1000
    dates = [f"2024-01-{i//24+1:02d} {i%24:02d}:00:00" for i in range(n_points)]
    opens = [100 + np.random.random() * 10 for _ in range(n_points)]
    highs = [o + np.random.random() * 2 for o in opens]
    lows = [o - np.random.random() * 2 for o in opens]
    closes = [o + np.random.random() * 2 - 1 for o in opens]
    
    with profiler.measure("create_candlestick"):
        fig = go.Figure(data=[
            go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
            )
        ])
    
    with profiler.measure("update_layout"):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#16213e",
            plot_bgcolor="#1a1a2e",
            height=400,
        )
    
    with profiler.measure("to_json"):
        # This is what Reflex does internally
        fig_json = fig.to_json()
    
    profiler.stop()
    report = profiler.get_report()
    
    print("\n" + "=" * 60)
    print("CHART CREATION PROFILING RESULTS")
    print("=" * 60)
    print(f"Data points: {n_points:,}")
    print(report.summary())
    
    return report.total_time_ms


def profile_full_page_load() -> float:
    """
    Profile simulated full page load.
    
    This simulates what happens when a user loads the Data Studio page:
    1. Import modules
    2. Initialize adapters
    3. Load symbol list
    4. Load initial data
    5. Create chart
    
    Returns
    -------
    float
        Total simulated page load time in milliseconds
    """
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Step 1: Import modules (cached after first import)
    with profiler.measure("imports"):
        import reflex as rx
        import plotly.graph_objects as go
        from ui.state import DataState
        from ui.adapters import DataAdapter
        from ui.utils.downsampling import downsample_ohlcv
    
    # Step 2: Initialize adapter
    with profiler.measure("adapter_init"):
        adapter = DataAdapter()
    
    # Step 3: Load symbol list
    with profiler.measure("load_symbols"):
        try:
            symbols = adapter.list_symbols()
            base_symbols = adapter.get_base_symbols()
        except Exception:
            symbols = []
            base_symbols = []
    
    # Step 4: Load initial data (if available)
    data = None
    if base_symbols:
        symbol = base_symbols[0]
        timeframes = adapter.get_timeframes(symbol)
        if timeframes:
            timeframe = timeframes[0]
            with profiler.measure("load_data"):
                try:
                    data = adapter.load_ohlcv(symbol, timeframe)
                except Exception as e:
                    logger.warning(f"Data load failed: {e}")
    
    # Step 5: Create chart (if data available)
    if data and data.get('row_count', 0) > 0:
        with profiler.measure("downsample"):
            if data['row_count'] > 1000:
                data = downsample_ohlcv(data, 1000)
        
        with profiler.measure("create_chart"):
            fig = go.Figure(data=[
                go.Candlestick(
                    x=data['dates'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                )
            ])
            fig.update_layout(
                template="plotly_dark",
                height=400,
            )
    
    profiler.stop()
    report = profiler.get_report()
    
    print("\n" + "=" * 60)
    print("FULL PAGE LOAD SIMULATION RESULTS")
    print("=" * 60)
    print(report.summary())
    
    return report.total_time_ms


def run_benchmarks():
    """Run detailed benchmarks."""
    print("\n" + "=" * 60)
    print("RUNNING DETAILED BENCHMARKS")
    print("=" * 60)
    
    # Data loading benchmark
    print("\n--- Data Loading Benchmark ---")
    result = benchmark_data_loading(iterations=3)
    if 'error' not in result:
        print(f"Symbol: {result['symbol']}_{result['timeframe']}")
        print(f"Average: {result['avg_ms']:.2f}ms")
        print(f"Min: {result['min_ms']:.2f}ms, Max: {result['max_ms']:.2f}ms")
    else:
        print(f"Error: {result['error']}")
    
    # Downsampling benchmark
    print("\n--- Downsampling Benchmark ---")
    result = benchmark_downsampling(data_points=50000, target_points=1000, iterations=3)
    print(f"Input: {result['input_points']:,} points -> {result['output_points']:,} points")
    print(f"Average: {result['avg_ms']:.2f}ms")
    print(f"Min: {result['min_ms']:.2f}ms, Max: {result['max_ms']:.2f}ms")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile QuantLab UI performance"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run detailed benchmarks"
    )
    parser.add_argument(
        "--symbol", "-s",
        default="XAUUSD",
        help="Symbol for data loading tests (default: XAUUSD)"
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="1H",
        help="Timeframe for data loading tests (default: 1H)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("QUANTLAB UI PERFORMANCE PROFILER")
    print("=" * 60)
    print(f"Target: Initial page load < 3000ms")
    print()
    
    total_time = 0.0
    
    # Profile imports
    import_time = profile_imports()
    total_time += import_time
    
    # Profile adapter initialization
    adapter_time = profile_adapter_initialization()
    total_time += adapter_time
    
    # Profile data loading
    data_time = profile_data_loading(args.symbol, args.timeframe)
    total_time += data_time
    
    # Profile chart creation
    chart_time = profile_chart_creation()
    total_time += chart_time
    
    # Profile full page load simulation
    page_time = profile_full_page_load()
    
    # Run benchmarks if requested
    if args.benchmark:
        run_benchmarks()
    
    # Summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Import time:     {import_time:>8.2f}ms")
    print(f"Adapter init:    {adapter_time:>8.2f}ms")
    print(f"Data loading:    {data_time:>8.2f}ms")
    print(f"Chart creation:  {chart_time:>8.2f}ms")
    print("-" * 30)
    print(f"Full page sim:   {page_time:>8.2f}ms")
    print()
    
    target = 3000.0
    if page_time <= target:
        print(f"✓ PASSED: Page load ({page_time:.0f}ms) is within target ({target:.0f}ms)")
        sys.exit(0)
    else:
        print(f"✗ FAILED: Page load ({page_time:.0f}ms) exceeds target ({target:.0f}ms)")
        sys.exit(1)


if __name__ == "__main__":
    main()
