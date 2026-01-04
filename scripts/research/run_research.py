"""
Research Pipeline Runner for FASE 5: Production.

Main script to run the complete research pipeline:
1. Scan strategies across assets
2. Validate candidates with PSR/DSR
3. Generate research report

Version: 0.6.1
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Setup path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.research.strategy_scanner import StrategyScanner, ScanResult
from scripts.research.candidate_validator import CandidateValidator, ValidationResult

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MIN_PSR = 0.95
DEFAULT_MAX_DRAWDOWN = 0.20
FALLBACK_PSR_THRESHOLD = 0.80
MIN_RETURNS_FOR_VALIDATION = 60


class ResearchPipeline:
    """
    Complete research pipeline for finding alpha.
    
    Workflow:
    1. Load historical data
    2. Scan multiple strategies
    3. Filter by PSR threshold
    4. Validate candidates with robustness tests
    5. Generate report
    
    Parameters
    ----------
    assets : List[str]
        List of asset symbols
    min_date : str
        Minimum date for data
    min_psr : float
        Minimum PSR threshold
    max_drawdown : float
        Maximum drawdown threshold
    """
    
    def __init__(
        self,
        assets: List[str],
        min_date: str = '2015-01-01',
        min_psr: float = 0.95,
        max_drawdown: float = 0.20
    ):
        assert assets is not None, "Assets cannot be None"
        assert len(assets) > 0, "Assets cannot be empty"
        
        self.assets = assets
        self.min_date = min_date
        self.min_psr = min_psr
        self.max_drawdown = max_drawdown
        
        self.scanner = StrategyScanner(assets=assets, min_date=min_date)
        self.validator = CandidateValidator(
            min_psr=min_psr,
            max_drawdown=max_drawdown
        )
        
        self.scan_results: List[ScanResult] = []
        self.validation_results: List[ValidationResult] = []
        self.final_candidates: List[ValidationResult] = []
    
    def run(self, use_synthetic: bool = False) -> List[ValidationResult]:
        """
        Run complete research pipeline.
        
        Parameters
        ----------
        use_synthetic : bool
            If True, use synthetic data
            
        Returns
        -------
        List[ValidationResult]
            Final validated candidates
        """
        logger.info("=" * 70)
        logger.info("RESEARCH PIPELINE - Starting")
        logger.info("=" * 70)
        logger.info(f"Assets: {self.assets}")
        logger.info(f"Min Date: {self.min_date}")
        logger.info(f"Min PSR: {self.min_psr:.0%}")
        logger.info(f"Max Drawdown: {self.max_drawdown:.0%}")
        logger.info("")
        
        try:
            # Step 1: Scan strategies
            logger.info("STEP 1: Scanning strategies...")
            self.scan_results = self.scanner.scan_all(use_synthetic=use_synthetic)
            
            if not self.scan_results:
                logger.error("No scan results")
                return []
            
            # Step 2: Filter by PSR
            logger.info("\nSTEP 2: Filtering by PSR threshold...")
            candidates = self.scanner.filter_by_psr(threshold=self.min_psr)
            
            if not candidates:
                logger.warning(f"No candidates with PSR >= {self.min_psr:.0%}")
                # Lower threshold for demonstration
                candidates = self.scanner.filter_by_psr(threshold=FALLBACK_PSR_THRESHOLD)
                logger.info(f"Using lower threshold ({FALLBACK_PSR_THRESHOLD:.0%}): {len(candidates)} candidates")
            
            # Step 3: Validate candidates
            logger.info("\nSTEP 3: Validating candidates...")
            
            # Generate benchmark returns (synthetic)
            np.random.seed(123)
            benchmark_returns = pd.Series(
                np.random.normal(0.0003, 0.012, 500),
                index=pd.date_range('2020-01-01', periods=500, freq='B')
            )
            
            for candidate in candidates:
                try:
                    if candidate.returns is not None and len(candidate.returns) >= MIN_RETURNS_FOR_VALIDATION:
                        result = self.validator.validate(
                            returns=candidate.returns,
                            strategy_name=candidate.strategy_name,
                            asset=candidate.asset,
                            benchmark_returns=benchmark_returns
                        )
                        self.validation_results.append(result)
                except Exception as e:
                    logger.error(f"Validation failed for {candidate.strategy_name}: {e}")
                    continue
            
            # Step 4: Get final candidates
            self.final_candidates = self.validator.get_valid_candidates()
            
            # Step 5: Print report
            self._print_report()
            
            return self.final_candidates
            
        except Exception as e:
            logger.error(f"Research pipeline failed: {e}")
            return []
    
    def _print_report(self) -> None:
        """Print comprehensive research report."""
        logger.info("\n" + "=" * 70)
        logger.info("RESEARCH REPORT")
        logger.info("=" * 70)
        
        # Summary
        logger.info(f"\nScan Summary:")
        logger.info(f"  Total strategies scanned: {len(self.scan_results)}")
        logger.info(f"  Candidates validated: {len(self.validation_results)}")
        logger.info(f"  Final candidates: {len(self.final_candidates)}")
        
        # Scan results
        if self.scan_results:
            scan_df = self.scanner.get_summary()
            logger.info(f"\nTop 5 by PSR:")
            for _, row in scan_df.head(5).iterrows():
                logger.info(
                    f"  [{row['strategy']}] {row['asset']}: "
                    f"PSR={row['psr']:.2%}, Sharpe={row['sharpe']:.2f}"
                )
        
        # Validation results
        if self.validation_results:
            val_df = self.validator.get_summary()
            logger.info(f"\nValidation Results:")
            for _, row in val_df.iterrows():
                status = "[OK]" if row['is_valid'] else "[FAIL]"
                logger.info(
                    f"  {status} [{row['strategy']}] {row['asset']}: "
                    f"Score={row['score']:.0f}, PSR={row['psr']:.2%}"
                )
        
        # Final candidates
        if self.final_candidates:
            logger.info(f"\nFinal Candidates (Ready for Paper Trading):")
            for result in self.final_candidates:
                logger.info(f"  [{result.strategy_name}] {result.asset}")
                logger.info(f"    PSR: {result.psr:.2%}")
                logger.info(f"    Sharpe: {result.sharpe_ratio:.2f}")
                logger.info(f"    Max DD: {result.max_drawdown:.2%}")
                logger.info(f"    Score: {result.score:.0f}")
        else:
            logger.info("\nNo candidates passed all validation criteria.")
            logger.info("Consider:")
            logger.info("  - Adjusting strategy parameters")
            logger.info("  - Testing different assets")
            logger.info("  - Extending data period")
        
        # Next steps
        logger.info(f"\nNext Steps:")
        if self.final_candidates:
            logger.info("  1. Review candidate strategies in detail")
            logger.info("  2. Run walk-forward optimization")
            logger.info("  3. Deploy to paper trading (1-3 months)")
            logger.info("  4. Monitor performance vs backtest")
            logger.info("  5. If deviation < 20%, proceed to Alpha Streams")
        else:
            logger.info("  1. Review scan results for promising strategies")
            logger.info("  2. Adjust parameters or try different assets")
            logger.info("  3. Re-run research pipeline")
    
    def save_report(self, output_dir: str = "output/research") -> None:
        """
        Save research report to files.
        
        Parameters
        ----------
        output_dir : str
            Output directory
        """
        output_path = ROOT / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save scan results
        if self.scan_results:
            scan_df = self.scanner.get_summary()
            scan_file = output_path / f"scan_results_{timestamp}.csv"
            scan_df.to_csv(scan_file, index=False)
            logger.info(f"Saved scan results to {scan_file}")
        
        # Save validation results
        if self.validation_results:
            val_df = self.validator.get_summary()
            val_file = output_path / f"validation_results_{timestamp}.csv"
            val_df.to_csv(val_file, index=False)
            logger.info(f"Saved validation results to {val_file}")


def main():
    """Run research pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Initialize pipeline
    pipeline = ResearchPipeline(
        assets=['F_GC', 'F_ES', 'F_CL'],
        min_date='2015-01-01',
        min_psr=0.95,
        max_drawdown=0.20
    )
    
    # Run pipeline
    candidates = pipeline.run(use_synthetic=False)
    
    # Save report
    pipeline.save_report()
    
    logger.info("\n" + "=" * 70)
    logger.info("RESEARCH PIPELINE - Complete")
    logger.info("=" * 70)
    
    return len(candidates) > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
