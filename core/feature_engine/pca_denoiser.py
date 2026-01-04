"""
PCA-based Denoising for Covariance Matrix.

Uses Random Matrix Theory (Marcenko-Pastur distribution) to separate
signal eigenvalues from noise eigenvalues in covariance matrices.

Reference:
- Kompendium #16: Marcenko-Pastur PDF
- Lopez de Prado - Advances in Financial Machine Learning, Chapter 2

Key Concept:
- Eigenvalues within Marcenko-Pastur bounds = NOISE
- Eigenvalues outside bounds = SIGNAL
- Denoise by shrinking noise eigenvalues
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

from .base import FeatureGenerator, FeatureConfig, FeatureResult, FeatureType


@dataclass
class DenoiseResult:
    """Result of covariance denoising."""
    cov_denoised: np.ndarray
    corr_denoised: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_signal: int
    n_noise: int
    variance_explained: float
    
    def __str__(self) -> str:
        return (
            f"DenoiseResult:\n"
            f"  Signal eigenvalues: {self.n_signal}\n"
            f"  Noise eigenvalues: {self.n_noise}\n"
            f"  Variance explained: {self.variance_explained:.2%}"
        )


def marcenko_pastur_pdf(
    var: float,
    q: float,
    pts: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Marcenko-Pastur probability density function.
    
    Formula:
        f(λ) = (T/N) × √((λ+ - λ)(λ - λ-)) / (2π × λ × σ²)
        
        λ± = σ² × (1 ± √(N/T))²
    
    Parameters
    ----------
    var : float
        Variance (σ²)
    q : float
        Ratio T/N (observations/variables)
    pts : int
        Number of points for PDF
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (eigenvalues, pdf_values)
    """
    # Eigenvalue bounds
    lambda_min = var * (1 - np.sqrt(1/q)) ** 2
    lambda_max = var * (1 + np.sqrt(1/q)) ** 2
    
    # Generate eigenvalue range
    eigenvalues = np.linspace(lambda_min, lambda_max, pts)
    
    # Marcenko-Pastur PDF
    pdf = q / (2 * np.pi * var * eigenvalues) * \
          np.sqrt((lambda_max - eigenvalues) * (eigenvalues - lambda_min))
    
    return eigenvalues, pdf


def get_mp_bounds(var: float, q: float) -> Tuple[float, float]:
    """
    Get Marcenko-Pastur eigenvalue bounds.
    
    Parameters
    ----------
    var : float
        Variance
    q : float
        Ratio T/N
        
    Returns
    -------
    Tuple[float, float]
        (lambda_min, lambda_max)
    """
    lambda_min = var * (1 - np.sqrt(1/q)) ** 2
    lambda_max = var * (1 + np.sqrt(1/q)) ** 2
    return lambda_min, lambda_max


def fit_mp_distribution(
    eigenvalues: np.ndarray,
    q: float,
    bandwidth: float = 0.01
) -> float:
    """
    Fit Marcenko-Pastur distribution to eigenvalues.
    
    Finds the variance that best fits the noise eigenvalues.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Observed eigenvalues
    q : float
        Ratio T/N
    bandwidth : float
        KDE bandwidth
        
    Returns
    -------
    float
        Fitted variance
    """
    from scipy.stats import gaussian_kde
    
    # Initial guess: variance of smallest eigenvalues
    var_init = np.var(eigenvalues[eigenvalues < np.median(eigenvalues)])
    
    def error_func(var):
        if var <= 0:
            return np.inf
        
        lambda_min, lambda_max = get_mp_bounds(var, q)
        
        # Count eigenvalues within MP bounds
        in_bounds = eigenvalues[(eigenvalues >= lambda_min) & 
                                (eigenvalues <= lambda_max)]
        
        if len(in_bounds) == 0:
            return np.inf
        
        # Error: difference from expected count
        expected_count = len(eigenvalues) * (1 - 1/q) if q > 1 else len(eigenvalues)
        return abs(len(in_bounds) - expected_count)
    
    result = minimize_scalar(
        error_func,
        bounds=(var_init * 0.1, var_init * 10),
        method='bounded'
    )
    
    return result.x if result.success else var_init


def denoise_covariance(
    cov: np.ndarray,
    q: float,
    method: str = 'constant_residual'
) -> Tuple[np.ndarray, int]:
    """
    Denoise covariance matrix using Marcenko-Pastur.
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix
    q : float
        Ratio T/N (observations/variables)
    method : str
        'constant_residual': Replace noise eigenvalues with average
        'shrinkage': Shrink noise eigenvalues toward average
        'targeted': Keep only signal eigenvalues
        
    Returns
    -------
    Tuple[np.ndarray, int]
        (denoised_covariance, n_signal_eigenvalues)
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Fit MP distribution to find noise threshold
    var_noise = fit_mp_distribution(eigenvalues, q)
    lambda_min, lambda_max = get_mp_bounds(var_noise, q)
    
    # Identify signal vs noise
    n_signal = np.sum(eigenvalues > lambda_max)
    
    if method == 'constant_residual':
        # Replace noise eigenvalues with their average
        noise_eigenvalues = eigenvalues[eigenvalues <= lambda_max]
        if len(noise_eigenvalues) > 0:
            avg_noise = np.mean(noise_eigenvalues)
            eigenvalues_denoised = eigenvalues.copy()
            eigenvalues_denoised[eigenvalues <= lambda_max] = avg_noise
        else:
            eigenvalues_denoised = eigenvalues
            
    elif method == 'shrinkage':
        # Shrink noise eigenvalues
        alpha = 0.5  # Shrinkage factor
        eigenvalues_denoised = eigenvalues.copy()
        noise_mask = eigenvalues <= lambda_max
        if noise_mask.any():
            avg_noise = np.mean(eigenvalues[noise_mask])
            eigenvalues_denoised[noise_mask] = (
                alpha * eigenvalues[noise_mask] + (1 - alpha) * avg_noise
            )
            
    elif method == 'targeted':
        # Zero out noise eigenvalues
        eigenvalues_denoised = eigenvalues.copy()
        eigenvalues_denoised[eigenvalues <= lambda_max] = 0
        
    else:
        eigenvalues_denoised = eigenvalues
    
    # Reconstruct covariance
    cov_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
    
    return cov_denoised, n_signal


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    return corr


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to covariance matrix."""
    return corr * np.outer(std, std)



class PCADenoiser(FeatureGenerator):
    """
    PCA-based Denoiser using Marcenko-Pastur distribution.
    
    Denoises covariance/correlation matrices by identifying
    and shrinking noise eigenvalues.
    
    Examples
    --------
    >>> denoiser = PCADenoiser(method='constant_residual')
    >>> result = denoiser.fit_transform(returns_df)
    >>> print(f"Signal components: {result.metadata['n_signal']}")
    """
    
    def __init__(
        self,
        method: str = 'constant_residual',
        min_observations: int = 100,
    ):
        """
        Initialize PCA denoiser.
        
        Parameters
        ----------
        method : str
            Denoising method:
            - 'constant_residual': Replace noise with average
            - 'shrinkage': Shrink noise toward average
            - 'targeted': Zero out noise
        min_observations : int
            Minimum observations required
        """
        config = FeatureConfig(
            name='pca_denoiser',
            feature_type=FeatureType.STATISTICAL,
            params={'method': method}
        )
        super().__init__(config)
        
        self.method = method
        self.min_observations = min_observations
        
        self._cov_denoised: Optional[np.ndarray] = None
        self._corr_denoised: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._n_signal: int = 0
    
    def fit(self, data: pd.DataFrame) -> 'PCADenoiser':
        """
        Fit denoiser to returns data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Returns data (T observations × N assets)
            
        Returns
        -------
        PCADenoiser
            Fitted denoiser
        """
        data = self._validate_data(data)
        
        # Calculate returns if prices provided
        if data.iloc[:, 0].mean() > 1:  # Likely prices
            returns = data.pct_change().dropna()
        else:
            returns = data.dropna()
        
        if len(returns) < self.min_observations:
            raise ValueError(
                f"Insufficient observations: {len(returns)} < {self.min_observations}"
            )
        
        # Calculate covariance
        cov = returns.cov().values
        
        # Calculate q ratio
        T, N = returns.shape
        q = T / N
        
        if q <= 1:
            raise ValueError(
                f"Need more observations than variables: T={T}, N={N}"
            )
        
        # Denoise
        self._cov_denoised, self._n_signal = denoise_covariance(
            cov, q, self.method
        )
        
        # Convert to correlation
        self._corr_denoised = cov_to_corr(self._cov_denoised)
        
        # Store eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self._cov_denoised)
        idx = eigenvalues.argsort()[::-1]
        self._eigenvalues = eigenvalues[idx]
        self._eigenvectors = eigenvectors[:, idx]
        
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> FeatureResult:
        """
        Transform data using denoised covariance.
        
        Returns PCA-transformed features using signal components only.
        
        Parameters
        ----------
        data : pd.DataFrame
            Returns data
            
        Returns
        -------
        FeatureResult
            PCA features (signal components only)
        """
        if not self._is_fitted:
            raise ValueError("Denoiser not fitted. Call fit() first.")
        
        data = self._validate_data(data)
        
        # Calculate returns if prices
        if data.iloc[:, 0].mean() > 1:
            returns = data.pct_change().dropna()
        else:
            returns = data.dropna()
        
        # Project onto signal eigenvectors
        signal_vectors = self._eigenvectors[:, :self._n_signal]
        pca_features = returns.values @ signal_vectors
        
        # Create DataFrame
        feature_names = [f'pca_{i+1}' for i in range(self._n_signal)]
        features_df = pd.DataFrame(
            pca_features,
            index=returns.index,
            columns=feature_names
        )
        
        # Variance explained
        total_var = np.sum(self._eigenvalues)
        signal_var = np.sum(self._eigenvalues[:self._n_signal])
        var_explained = signal_var / total_var if total_var > 0 else 0
        
        return FeatureResult(
            features=features_df,
            feature_names=feature_names,
            config=self.config,
            metadata={
                'n_signal': self._n_signal,
                'n_noise': len(self._eigenvalues) - self._n_signal,
                'variance_explained': var_explained,
                'eigenvalues': self._eigenvalues.tolist(),
                'method': self.method
            }
        )
    
    def get_denoised_covariance(self) -> np.ndarray:
        """Get denoised covariance matrix."""
        if not self._is_fitted:
            raise ValueError("Denoiser not fitted.")
        return self._cov_denoised
    
    def get_denoised_correlation(self) -> np.ndarray:
        """Get denoised correlation matrix."""
        if not self._is_fitted:
            raise ValueError("Denoiser not fitted.")
        return self._corr_denoised
    
    def get_denoise_result(self) -> DenoiseResult:
        """Get full denoise result."""
        if not self._is_fitted:
            raise ValueError("Denoiser not fitted.")
        
        total_var = np.sum(self._eigenvalues)
        signal_var = np.sum(self._eigenvalues[:self._n_signal])
        
        return DenoiseResult(
            cov_denoised=self._cov_denoised,
            corr_denoised=self._corr_denoised,
            eigenvalues=self._eigenvalues,
            eigenvectors=self._eigenvectors,
            n_signal=self._n_signal,
            n_noise=len(self._eigenvalues) - self._n_signal,
            variance_explained=signal_var / total_var if total_var > 0 else 0
        )


def denoise_returns(
    returns: pd.DataFrame,
    method: str = 'constant_residual'
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Quick function to denoise returns covariance.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    method : str
        Denoising method
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (denoised_cov, denoised_corr, n_signal)
    """
    denoiser = PCADenoiser(method=method)
    denoiser.fit(returns)
    
    return (
        denoiser.get_denoised_covariance(),
        denoiser.get_denoised_correlation(),
        denoiser._n_signal
    )
