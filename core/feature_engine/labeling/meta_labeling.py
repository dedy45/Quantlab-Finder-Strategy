"""
Meta-Labeling for Bet Sizing.

Meta-labeling is a two-stage approach:
1. Primary model: Predicts direction (side)
2. Secondary model: Predicts probability of profit given the side

This allows:
- Better calibrated probabilities
- Optimal bet sizing via Kelly criterion
- Filtering low-confidence trades

Reference:
- Protokol Kausalitas - Fase 3 (Adaptabilitas)
- Lopez de Prado - Advances in Financial Machine Learning, Chapter 3

Key Insight:
- Primary model can be simple (MA crossover, momentum)
- Secondary model learns WHEN the primary model is right
- Bet size = f(probability of profit)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .triple_barrier import (
    TripleBarrierLabeler,
    TripleBarrierResult,
    TripleBarrierConfig,
)


@dataclass
class MetaLabelResult:
    """Result of meta-labeling."""
    meta_labels: pd.Series      # Binary: 1 if primary correct, 0 otherwise
    probabilities: pd.Series    # Probability of primary being correct
    bet_sizes: pd.Series        # Recommended bet sizes
    primary_signals: pd.Series  # Original primary model signals
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def accuracy(self) -> float:
        """Accuracy of primary model."""
        return self.meta_labels.mean()
    
    @property
    def avg_bet_size(self) -> float:
        """Average bet size."""
        return self.bet_sizes.mean()
    
    def __str__(self) -> str:
        return (
            f"MetaLabelResult:\n"
            f"  Samples: {len(self.meta_labels)}\n"
            f"  Primary Accuracy: {self.accuracy:.2%}\n"
            f"  Avg Bet Size: {self.avg_bet_size:.2%}"
        )


def generate_meta_labels(
    primary_signals: pd.Series,
    triple_barrier_labels: pd.Series
) -> pd.Series:
    """
    Generate meta-labels from primary signals and actual outcomes.
    
    Meta-label = 1 if primary signal direction matches outcome
    Meta-label = 0 if primary signal direction is wrong
    
    Parameters
    ----------
    primary_signals : pd.Series
        Primary model signals (-1, 0, 1)
    triple_barrier_labels : pd.Series
        Actual outcomes from triple-barrier (-1, 0, 1)
        
    Returns
    -------
    pd.Series
        Meta-labels (0 or 1)
    """
    # Align indices
    common_idx = primary_signals.index.intersection(triple_barrier_labels.index)
    signals = primary_signals.loc[common_idx]
    labels = triple_barrier_labels.loc[common_idx]
    
    # Meta-label: 1 if signal * label > 0 (same direction)
    # Also 1 if signal == 0 (no trade, no loss)
    meta_labels = pd.Series(0, index=common_idx)
    
    # Correct predictions
    meta_labels[(signals * labels) > 0] = 1
    
    # No trade cases (signal = 0)
    meta_labels[signals == 0] = 0  # Don't reward no-trade
    
    return meta_labels


def calculate_bet_size(
    probabilities: pd.Series,
    method: str = 'kelly',
    max_bet: float = 1.0,
    min_prob: float = 0.5
) -> pd.Series:
    """
    Calculate bet sizes from probabilities.
    
    Parameters
    ----------
    probabilities : pd.Series
        Probability of profit
    method : str
        'kelly': Full Kelly criterion
        'half_kelly': Half Kelly (safer)
        'linear': Linear scaling
    max_bet : float
        Maximum bet size (0-1)
    min_prob : float
        Minimum probability to bet
        
    Returns
    -------
    pd.Series
        Bet sizes (0 to max_bet)
    """
    bet_sizes = pd.Series(0.0, index=probabilities.index)
    
    # Only bet when probability > threshold
    mask = probabilities > min_prob
    
    if method == 'kelly':
        # Kelly: f* = p - (1-p)/b = 2p - 1 (for even odds)
        bet_sizes[mask] = 2 * probabilities[mask] - 1
        
    elif method == 'half_kelly':
        # Half Kelly for safety
        bet_sizes[mask] = (2 * probabilities[mask] - 1) / 2
        
    elif method == 'linear':
        # Linear scaling from min_prob to 1
        bet_sizes[mask] = (probabilities[mask] - min_prob) / (1 - min_prob)
    
    # Clip to valid range
    bet_sizes = bet_sizes.clip(0, max_bet)
    
    return bet_sizes


class MetaLabeler:
    """
    Meta-Labeling for bet sizing and trade filtering.
    
    Two-stage approach:
    1. Primary model predicts direction
    2. Secondary model predicts if primary is correct
    
    Examples
    --------
    >>> # Create primary signals (e.g., MA crossover)
    >>> primary_signals = ma_crossover_signal(prices)
    >>> 
    >>> # Create meta-labeler
    >>> meta = MetaLabeler(
    ...     primary_signals=primary_signals,
    ...     bet_sizing='half_kelly'
    ... )
    >>> 
    >>> # Fit and get results
    >>> result = meta.fit_transform(price_df, features_df)
    >>> print(f"Accuracy: {result.accuracy:.2%}")
    """
    
    def __init__(
        self,
        primary_signals: pd.Series,
        pt_sl_ratio: float = 1.0,
        max_holding_period: int = 10,
        vol_lookback: int = 20,
        bet_sizing: str = 'half_kelly',
        max_bet: float = 1.0,
        min_prob: float = 0.55,
        classifier: Optional[Any] = None,
    ):
        """
        Initialize meta-labeler.
        
        Parameters
        ----------
        primary_signals : pd.Series
            Primary model signals (-1, 0, 1)
        pt_sl_ratio : float
            Profit-take / Stop-loss ratio for triple-barrier
        max_holding_period : int
            Maximum holding period
        vol_lookback : int
            Volatility lookback
        bet_sizing : str
            Bet sizing method: 'kelly', 'half_kelly', 'linear'
        max_bet : float
            Maximum bet size
        min_prob : float
            Minimum probability to place bet
        classifier : sklearn classifier, optional
            Custom classifier. Default: RandomForest
        """
        self.primary_signals = primary_signals
        self.pt_sl_ratio = pt_sl_ratio
        self.max_holding_period = max_holding_period
        self.vol_lookback = vol_lookback
        self.bet_sizing = bet_sizing
        self.max_bet = max_bet
        self.min_prob = min_prob
        
        # Default classifier
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1
        )
        
        self._triple_barrier_result: Optional[TripleBarrierResult] = None
        self._meta_labels: Optional[pd.Series] = None
        self._is_fitted: bool = False
    
    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame
    ) -> 'MetaLabeler':
        """
        Fit meta-labeler.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data with 'close' column
        features : pd.DataFrame
            Features for secondary model
            
        Returns
        -------
        MetaLabeler
            Fitted meta-labeler
        """
        # Step 1: Generate triple-barrier labels
        labeler = TripleBarrierLabeler(
            pt_sl_ratio=self.pt_sl_ratio,
            max_holding_period=self.max_holding_period,
            vol_lookback=self.vol_lookback,
            binary_labels=False
        )
        
        # Get events where primary model has signal
        events = self.primary_signals[self.primary_signals != 0].index
        events = events.intersection(prices.index)
        
        self._triple_barrier_result = labeler.fit_transform(
            prices,
            events=events,
            side_prediction=self.primary_signals
        )
        
        # Step 2: Generate meta-labels
        self._meta_labels = generate_meta_labels(
            self.primary_signals,
            self._triple_barrier_result.labels
        )
        
        # Step 3: Fit secondary classifier
        # Align features with meta-labels
        common_idx = self._meta_labels.index.intersection(features.index)
        X = features.loc[common_idx].values
        y = self._meta_labels.loc[common_idx].values
        
        # Remove NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) > 100:
            self.classifier.fit(X, y)
            self._is_fitted = True
        else:
            raise ValueError(f"Insufficient samples for training: {len(X)}")
        
        return self
    
    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict probability of primary model being correct.
        
        Parameters
        ----------
        features : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        pd.Series
            Probabilities
        """
        if not self._is_fitted:
            raise ValueError("Meta-labeler not fitted. Call fit() first.")
        
        X = features.values
        
        # Handle NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        proba = np.zeros(len(X))
        
        if valid_mask.any():
            proba[valid_mask] = self.classifier.predict_proba(X[valid_mask])[:, 1]
        
        return pd.Series(proba, index=features.index, name='probability')
    
    def transform(self, features: pd.DataFrame) -> MetaLabelResult:
        """
        Generate meta-label results.
        
        Parameters
        ----------
        features : pd.DataFrame
            Features for prediction
            
        Returns
        -------
        MetaLabelResult
            Complete meta-labeling results
        """
        if not self._is_fitted:
            raise ValueError("Meta-labeler not fitted. Call fit() first.")
        
        # Get probabilities
        probabilities = self.predict_proba(features)
        
        # Calculate bet sizes
        bet_sizes = calculate_bet_size(
            probabilities,
            method=self.bet_sizing,
            max_bet=self.max_bet,
            min_prob=self.min_prob
        )
        
        # Align with primary signals
        common_idx = probabilities.index.intersection(self.primary_signals.index)
        
        return MetaLabelResult(
            meta_labels=self._meta_labels.reindex(common_idx).fillna(0).astype(int),
            probabilities=probabilities.loc[common_idx],
            bet_sizes=bet_sizes.loc[common_idx],
            primary_signals=self.primary_signals.loc[common_idx],
            metadata={
                'bet_sizing': self.bet_sizing,
                'max_bet': self.max_bet,
                'min_prob': self.min_prob,
                'n_samples': len(common_idx),
            }
        )
    
    def fit_transform(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame
    ) -> MetaLabelResult:
        """Fit and transform in one step."""
        self.fit(prices, features)
        return self.transform(features)
    
    def get_final_signals(
        self,
        features: pd.DataFrame,
        threshold: float = 0.55
    ) -> pd.Series:
        """
        Get final trading signals with bet sizing.
        
        Parameters
        ----------
        features : pd.DataFrame
            Features for prediction
        threshold : float
            Minimum probability to trade
            
        Returns
        -------
        pd.Series
            Final signals: primary_signal * bet_size
        """
        result = self.transform(features)
        
        # Final signal = direction * bet_size
        final_signals = result.primary_signals * result.bet_sizes
        
        # Zero out low probability trades
        final_signals[result.probabilities < threshold] = 0
        
        return final_signals
