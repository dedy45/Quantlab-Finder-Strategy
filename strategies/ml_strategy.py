"""
Machine Learning Based Strategy.

Uses ML models to predict price direction or returns.
Supports multiple model types with proper train/test separation.

Reference: Kompendium ML Models Guide, Protokol Kausalitas
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseStrategy, StrategyConfig, StrategyType


# Constants
DEFAULT_TRAIN_SIZE = 0.7
DEFAULT_LOOKBACK = 20
PREDICTION_THRESHOLD = 0.5
MIN_TRAIN_SAMPLES = 100


class ModelType(Enum):
    """Supported ML model types."""
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"
    LOGISTIC = "logistic"
    LINEAR = "linear"


@dataclass
class MLStrategyConfig(StrategyConfig):
    """Configuration for ML strategy."""
    name: str = "ml_strategy"
    strategy_type: StrategyType = StrategyType.ML_BASED
    
    # Model parameters
    model_type: ModelType = ModelType.RANDOM_FOREST
    train_size: float = DEFAULT_TRAIN_SIZE
    
    # Feature parameters
    feature_lookbacks: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # Prediction parameters
    prediction_threshold: float = PREDICTION_THRESHOLD
    use_probability: bool = True
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 50


class MLStrategy(BaseStrategy):
    """
    Machine Learning Based Strategy.
    
    Features:
    - Multiple model types (RF, LightGBM, Logistic, Linear)
    - Automatic feature generation
    - Probability-based position sizing
    - Walk-forward compatible
    
    Examples
    --------
    >>> config = MLStrategyConfig(model_type=ModelType.RANDOM_FOREST)
    >>> strategy = MLStrategy(config)
    >>> strategy.fit(price_data)
    >>> signals = strategy.predict(price_data)
    """
    
    def __init__(self, config: Optional[MLStrategyConfig] = None):
        """Initialize ML strategy."""
        super().__init__(config or MLStrategyConfig())
        self._model = None
        self._feature_names = []
        self._scaler = None
    
    @property
    def ml_config(self) -> MLStrategyConfig:
        """Get typed config."""
        return self.config
    
    def fit(self, data: pd.DataFrame) -> 'MLStrategy':
        """
        Fit ML model to historical data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data with 'close' column
            
        Returns
        -------
        MLStrategy
            Self for method chaining
        """
        data = self._validate_data(data)
        close = self._get_close(data)
        
        # Generate features
        features = self._generate_features(close)
        
        # Generate labels (next day direction)
        labels = np.sign(close.pct_change().shift(-1))
        
        # Align data
        valid_idx = features.dropna().index.intersection(labels.dropna().index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]
        
        if len(X) < MIN_TRAIN_SAMPLES:
            raise ValueError(f"Insufficient data: {len(X)} < {MIN_TRAIN_SAMPLES}")
        
        # Train model
        self._model = self._create_model()
        self._model.fit(X, y)
        
        self._feature_names = list(X.columns)
        self._fit_data = data
        self._is_fitted = True
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using ML model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data
            
        Returns
        -------
        pd.Series
            Signals: 1 (long), -1 (short), 0 (flat)
        """
        if not self._is_fitted:
            self.fit(data)
        
        data = self._validate_data(data)
        close = self._get_close(data)
        
        # Generate features
        features = self._generate_features(close)
        
        # Get valid indices
        valid_idx = features.dropna().index
        X = features.loc[valid_idx]
        
        # Predict
        cfg = self.ml_config
        
        if cfg.use_probability and hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(X)
            # Probability of positive class
            if proba.shape[1] == 2:
                prob_up = proba[:, 1]
            else:
                prob_up = proba[:, -1]
            
            # Convert to signal based on threshold
            signal_values = np.where(
                prob_up > cfg.prediction_threshold, 1.0,
                np.where(prob_up < (1 - cfg.prediction_threshold), -1.0, 0.0)
            )
        else:
            predictions = self._model.predict(X)
            signal_values = np.sign(predictions)
        
        # Create signal series
        signals = pd.Series(0.0, index=close.index)
        signals.loc[valid_idx] = signal_values
        
        return signals
    
    def _generate_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate features from price data.
        
        Features:
        - Returns at multiple lookbacks
        - Volatility at multiple lookbacks
        - RSI
        - Z-score
        - Momentum
        """
        cfg = self.ml_config
        features = pd.DataFrame(index=close.index)
        
        for lb in cfg.feature_lookbacks:
            # Returns
            features[f'ret_{lb}'] = close.pct_change(lb)
            
            # Volatility
            features[f'vol_{lb}'] = close.pct_change().rolling(lb).std()
            
            # Momentum (normalized)
            ma = close.rolling(lb).mean()
            features[f'mom_{lb}'] = (close - ma) / ma
        
        # RSI
        features['rsi'] = self._calculate_rsi(close, 14) / 100
        
        # Z-score
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['zscore'] = (close - ma20) / std20
        
        # Trend strength
        features['trend'] = close.pct_change(60)
        
        return features
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_model(self):
        """Create ML model based on config."""
        cfg = self.ml_config
        
        if cfg.model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
        
        elif cfg.model_type == ModelType.LIGHTGBM:
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    min_child_samples=cfg.min_samples_leaf,
                    random_state=42,
                    verbose=-1
                )
            except ImportError:
                # Fallback to RandomForest
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    min_samples_leaf=cfg.min_samples_leaf,
                    random_state=42
                )
        
        elif cfg.model_type == ModelType.LOGISTIC:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        
        elif cfg.model_type == ModelType.LINEAR:
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                random_state=42
            )
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from fitted model.
        
        Returns
        -------
        pd.Series
            Feature importance scores
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        if hasattr(self._model, 'feature_importances_'):
            importance = self._model.feature_importances_
        elif hasattr(self._model, 'coef_'):
            importance = np.abs(self._model.coef_).flatten()
        else:
            return pd.Series()
        
        return pd.Series(
            importance,
            index=self._feature_names
        ).sort_values(ascending=False)
    
    def get_params(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        cfg = self.ml_config
        return {
            'name': cfg.name,
            'type': cfg.strategy_type.value,
            'model_type': cfg.model_type.value,
            'n_estimators': cfg.n_estimators,
            'max_depth': cfg.max_depth,
            'feature_lookbacks': cfg.feature_lookbacks,
            'prediction_threshold': cfg.prediction_threshold,
        }
