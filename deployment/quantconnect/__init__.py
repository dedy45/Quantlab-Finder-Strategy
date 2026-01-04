"""
QuantConnect Deployment Module.

Adapter for deploying strategies to QuantConnect LEAN platform.
"""

from .adapter import QuantConnectAdapter
from .algorithm import AlgorithmTemplate

__all__ = [
    'QuantConnectAdapter',
    'AlgorithmTemplate',
]
