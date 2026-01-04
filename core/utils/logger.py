"""
Centralized Logging Configuration for Quant Lab.

Usage:
    from core.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing data...")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    level : int
        Logging level (default INFO)
    log_file : str, optional
        Path to log file (if None, logs to console only)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger for quick access
default_logger = get_logger('quant_lab')
