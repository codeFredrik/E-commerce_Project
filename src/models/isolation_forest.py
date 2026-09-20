import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.ensemble import IsolationForest

class DiscountIsolationForest:
    """
    Unsupervised Isolation Forest model for detecting manufactured pricing discounts.
    """
    DEFAULT_FEATURES: List[str] = [
        'dev_from_5d_mean',
        'dev_from_30d_mean',
        'pre_sale_inflation_pct',
        'volatility_ratio',
        'z_score'
    ]
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42, features: Optional[List[str]] = None):
        self.contamination = contamination
        self.random_state = random_state
        self.features = features or self.DEFAULT_FEATURES
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures required feature columns exist and are filled."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for ML model: {missing}")
        return df[self.features].fillna(0.0)

    def fit(self, df: pd.DataFrame) -> "DiscountIsolationForest":
        """
        Trains the Isolation Forest on product pricing features.
        """
        X = self._prepare_features(df)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scores pricing observations, returning normalized fraud risk scores between 0.0 and 1.0.
        Scores > 0.5 indicate anomaly/suspicion; scores > 0.8 indicate high fraud probability.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        X = self._prepare_features(df)
        
        # decision_function returns negative values for outliers and positive for inliers.
        # We invert it so outliers are positive.
        raw_score = -self.model.decision_function(X)
        
        # Sigmoid transformation: centers boundary (0.0) at 0.5 risk score
        risk_scores = 1.0 / (1.0 + np.exp(-10.0 * raw_score))
        return np.clip(risk_scores, 0.0, 1.0)

    def save_model(self, filepath: str) -> None:
        """Saves trained model weights to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        joblib.dump({'model': self.model, 'features': self.features, 'contamination': self.contamination}, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "DiscountIsolationForest":
        """Loads trained model weights from disk."""
        data = joblib.load(filepath)
        instance = cls(contamination=data['contamination'], features=data['features'])
        instance.model = data['model']
        instance.is_fitted = True
        return instance
