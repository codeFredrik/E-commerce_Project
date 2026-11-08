import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.ensemble import IsolationForest

class DiscountIsolationForest:
    """
    Unsupervised Isolation Forest model calibrated to detect manufactured pricing discounts
    specifically on sale/drop days.
    """
    DEFAULT_FEATURES: List[str] = [
        'deceptive_discount_signal',
        'recent_peak_inflation_ratio',
        'creeping_inflation_slope',
        'discount_exaggeration_gap'
    ]
    
    def __init__(self, contamination: float = 0.02, random_state: int = 42, features: Optional[List[str]] = None):
        self.contamination = contamination
        self.random_state = random_state
        self.features = features or self.DEFAULT_FEATURES
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.is_fitted = False
        self.normal_mode: float = 0.0
        self.spread: float = 0.1

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures required feature columns exist and are filled."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for ML model: {missing}")
        return df[self.features].fillna(0.0)

    def fit(self, df: pd.DataFrame) -> "DiscountIsolationForest":
        """
        Trains the Isolation Forest on product pricing features and computes calibration quantiles.
        """
        X = self._prepare_features(df)
        self.model.fit(X)
        self.is_fitted = True
        
        # Calibration baseline: decision function on training distribution
        raw = -self.model.decision_function(X)
        self.normal_mode = float(np.percentile(raw, 50))
        p95 = float(np.percentile(raw, 95))
        self.spread = max(p95 - self.normal_mode, 1e-4)
        
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scores pricing observations, returning normalized fraud risk scores between 0.0 and 1.0.
        Scores > 0.5 indicate anomaly/suspicion; scores > 0.75 indicate high fraud probability.
        
        Uses an uncoupled soft domain gate: only verifies that today is a discount drop
        and below the recent peak, letting the multivariate tree splits decide whether the
        preceding inflation and discount gap constitute deceptive pricing.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        X = self._prepare_features(df)
        
        # Raw decision function: positive values are outliers
        raw = -self.model.decision_function(X)
        
        # Soft domain gate: only require price drop from yesterday and below recent peak
        if 'prev_price' in df.columns and 'max_price_prev_14' in df.columns:
            drop_pct = np.maximum(0.0, (df['prev_price'] - df['price']) / (df['prev_price'] + 1e-6))
            is_discount_onset = (drop_pct >= 0.08) & (df['price'] < df['max_price_prev_14'])
            raw_gated = np.where(is_discount_onset, raw, self.normal_mode - 0.20)
        else:
            raw_gated = raw
            
        # Dynamic score calibration anchored on normal mode + spread
        # Centers the threshold cleanly at 0.50
        q_thresh = self.normal_mode + 0.95 * self.spread
        risk_scores = 1.0 / (1.0 + np.exp(-30.0 * (raw_gated - q_thresh)))
        return np.clip(risk_scores, 0.0, 1.0)

    def save_model(self, filepath: str) -> None:
        """Saves trained model weights and calibration stats to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        joblib.dump({
            'model': self.model,
            'features': self.features,
            'contamination': self.contamination,
            'normal_mode': self.normal_mode,
            'spread': self.spread
        }, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "DiscountIsolationForest":
        """Loads trained model weights and calibration stats from disk."""
        data = joblib.load(filepath)
        instance = cls(contamination=data['contamination'], features=data['features'])
        instance.model = data['model']
        instance.normal_mode = data.get('normal_mode', 0.0)
        instance.spread = data.get('spread', 0.1)
        instance.is_fitted = True
        return instance
