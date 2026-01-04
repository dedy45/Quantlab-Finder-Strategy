"""
Hierarchical Risk Parity (HRP) Allocator.

Implements the HRP algorithm from Marcos Lopez de Prado.
More robust than Mean-Variance Optimization as it doesn't require
matrix inversion.

Reference: "Building Diversified Portfolios that Outperform Out-of-Sample"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .base import BaseAllocator, AllocationConfig

logger = logging.getLogger(__name__)


class HRPAllocator(BaseAllocator):
    """
    Hierarchical Risk Parity allocator.
    
    HRP uses hierarchical clustering to build a portfolio that:
    1. Doesn't require covariance matrix inversion
    2. Is more stable than MVO
    3. Provides natural diversification through clustering
    
    Parameters
    ----------
    config : AllocationConfig, optional
        Allocation configuration
    linkage_method : str, default='single'
        Linkage method for hierarchical clustering
        Options: 'single', 'complete', 'average', 'ward'
    """
    
    def __init__(
        self,
        config: Optional[AllocationConfig] = None,
        linkage_method: str = 'single'
    ):
        super().__init__(config)
        self.linkage_method = linkage_method
        self._cov: Optional[pd.DataFrame] = None
        self._corr: Optional[pd.DataFrame] = None
        
    def fit(self, returns: pd.DataFrame) -> 'HRPAllocator':
        """
        Fit HRP allocator to historical returns.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns with assets as columns
            
        Returns
        -------
        HRPAllocator
            Fitted allocator
        """
        self._validate_returns(returns)
        
        logger.info(f"Fitting HRP with {returns.shape[1]} assets, {len(returns)} observations")
        
        try:
            # Step 1: Calculate covariance and correlation
            self._cov = returns.cov()
            self._corr = returns.corr()
            
            # Step 2: Tree clustering
            dist = self._get_distance_matrix(self._corr)
            link = linkage(squareform(dist), method=self.linkage_method)
            sort_ix = self._get_quasi_diag(link)
            sorted_assets = [returns.columns[i] for i in sort_ix]
            
            # Step 3: Recursive bisection
            self._weights = self._get_rec_bipart(
                self._cov, 
                sorted_assets
            )
            
            # Apply constraints
            self._weights = self._apply_constraints(self._weights)
            self._is_fitted = True
            
            logger.info(f"HRP fitted successfully, {len(self._weights)} assets allocated")
            
        except Exception as e:
            logger.error(f"HRP fitting failed: {e}")
            raise
            
        return self
    
    def get_weights(self) -> Dict[str, float]:
        """Get allocation weights."""
        assert self._is_fitted, "Allocator must be fitted first"
        return self._weights.copy()
    
    def _get_distance_matrix(self, corr: pd.DataFrame) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.
        
        Distance = sqrt(0.5 * (1 - correlation))
        """
        dist = ((1 - corr) / 2.0) ** 0.5
        return dist.values
    
    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """
        Sort clustered items by distance.
        
        Returns quasi-diagonal matrix ordering.
        """
        return list(leaves_list(link))
    
    def _get_cluster_var(
        self, 
        cov: pd.DataFrame, 
        cluster_items: List[str]
    ) -> float:
        """
        Calculate variance of a cluster using inverse-variance weights.
        """
        cov_slice = cov.loc[cluster_items, cluster_items]
        
        # Inverse variance weights
        ivp = 1.0 / np.diag(cov_slice)
        ivp = ivp / ivp.sum()
        
        # Cluster variance
        cluster_var = np.dot(ivp, np.dot(cov_slice, ivp))
        
        return cluster_var
    
    def _get_rec_bipart(
        self, 
        cov: pd.DataFrame, 
        sorted_assets: List[str]
    ) -> Dict[str, float]:
        """
        Recursive bisection for weight allocation.
        
        Recursively splits assets into two clusters and allocates
        weights based on inverse variance.
        """
        weights = pd.Series(1.0, index=sorted_assets)
        clusters = [sorted_assets]
        
        while len(clusters) > 0:
            # Bisect each cluster
            new_clusters = []
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Split in half
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]
                    
                    # Calculate cluster variances
                    var_left = self._get_cluster_var(cov, left)
                    var_right = self._get_cluster_var(cov, right)
                    
                    # Allocate based on inverse variance
                    alpha = 1 - var_left / (var_left + var_right)
                    
                    # Update weights
                    weights[left] *= alpha
                    weights[right] *= (1 - alpha)
                    
                    # Add to new clusters for further splitting
                    new_clusters.append(left)
                    new_clusters.append(right)
            
            clusters = new_clusters
        
        return weights.to_dict()
    
    def get_cluster_info(self) -> Dict[str, any]:
        """
        Get clustering information for analysis.
        
        Returns
        -------
        Dict
            Clustering metadata
        """
        assert self._is_fitted, "Allocator must be fitted first"
        
        return {
            'n_assets': len(self._weights),
            'linkage_method': self.linkage_method,
            'correlation_matrix': self._corr,
            'covariance_matrix': self._cov,
        }
