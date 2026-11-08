import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import shap

from src.models.isolation_forest import DiscountIsolationForest

class DiscountExplainer:
    """
    Explainable AI engine that uses SHAP and domain rules to translate ML anomaly scores
    into plain-English reasons for consumers and compliance teams.
    """
    def __init__(self, if_model: DiscountIsolationForest):
        self.if_model = if_model
        self.explainer: Optional[shap.TreeExplainer] = None
        if self.if_model.is_fitted:
            try:
                self.explainer = shap.TreeExplainer(self.if_model.model)
            except Exception:
                self.explainer = None  # Fallback gracefully if tree explainer fails

    def explain_observation(self, row: pd.Series, risk_score: float) -> Dict[str, Any]:
        """
        Generates plain-English explanation for a single scored price observation.
        
        Args:
            row (pd.Series): Observation row containing feature values and price
            risk_score (float): Normalized 0-1 risk score from DiscountIsolationForest
            
        Returns:
            dict: Verdict, confidence percentage, risk level, plain-English reasons, and feature attributions
        """
        reasons: List[str] = []
        feature_names = self.if_model.features
        attributions: Dict[str, float] = {}
        
        # 1. SHAP attribution computation (if available)
        try:
            if self.explainer is not None:
                x_row = row[feature_names].values.reshape(1, -1).astype(float)
                shap_vals = self.explainer.shap_values(x_row)[0]
                # Invert so positive attribution means contributing towards anomaly
                inv_shap = -shap_vals
                total_abs = np.sum(np.abs(inv_shap)) + 1e-9
                for name, val in zip(feature_names, inv_shap):
                    attributions[name] = round(float(val / total_abs), 4)
        except Exception:
            pass
            
        # 2. Plain-English translation engine based on feature values
        price = row.get('price', 0.0)
        recent_inf = row.get('recent_peak_inflation_ratio', row.get('pre_sale_inflation_pct', 0.0))
        drop_peak = row.get('drop_from_recent_peak', 0.0)
        exagg_gap = row.get('discount_exaggeration_gap', 0.0)
        creep_slope = row.get('creeping_inflation_slope', 0.0)
        fake_idx = row.get('fake_discount_illusion_index', 0.0)
        dev_30 = row.get('dev_from_30d_mean', 0.0)
        z_score = row.get('z_score', 0.0)
        
        if recent_inf > 0.15:
            reasons.append(f"Price was artificially inflated by +{recent_inf*100:.1f}% above historical baseline in the preceding 14 days.")
            
        if fake_idx > 0.05 and drop_peak > 0.15:
            reasons.append(f"Manufactured sale pattern detected: advertised {drop_peak*100:.1f}% discount merely cancels out pre-sale inflation.")
            
        if creep_slope > 0.05:
            reasons.append("Creeping inflation pattern: price was gradually climbed upwards over multiple days prior to this sudden drop.")
            
        if exagg_gap > 0.15:
            reasons.append(f"Advertised discount is exaggerated by {exagg_gap*100:.1f}% compared to real savings against the 30-day baseline.")
            
        if abs(dev_30) < 0.08 and risk_score > 0.5:
            reasons.append(f"Current 'sale price' (${price:.2f}) is essentially equal to the 30-day normal price ({dev_30*100:+.1f}%) — zero genuine consumer savings.")
        elif dev_30 > 0.10:
            reasons.append(f"Current advertised price is actually +{dev_30*100:.1f}% HIGHER than the product's 30-day historical average.")
            
        if abs(z_score) > 2.0:
            reasons.append(f"Price deviation exceeds statistical norms ({abs(z_score):.1f}σ from mean).")
            
        # Fallback / summary reason
        if not reasons:
            if risk_score > 0.5:
                reasons.append("Multi-day pricing pattern strongly matches deceptive discount benchmarks.")
            else:
                reasons.append("Pricing pattern is consistent with genuine discounts within historical baseline variation.")
                
        # 3. Determine final verdict badge and confidence
        if risk_score > 0.75:
            verdict = "🔴 LIKELY FAKE"
            risk_level = "HIGH"
            conf_pct = int(min(risk_score * 105, 99.0))
        elif risk_score > 0.50:
            verdict = "🟠 SUSPICIOUS"
            risk_level = "MODERATE"
            conf_pct = int(risk_score * 100)
        else:
            verdict = "🟢 GENUINE DISCOUNT"
            risk_level = "LOW"
            conf_pct = int((1.0 - risk_score) * 100)
            
        return {
            "verdict": verdict,
            "confidence_pct": conf_pct,
            "risk_level": risk_level,
            "risk_score": round(float(risk_score), 4),
            "reasons": reasons,
            "feature_attributions": attributions
        }

    def explain_dataframe(self, df: pd.DataFrame, risk_scores: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates explanations for an entire scored dataframe.
        """
        explanations = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            score = risk_scores[idx]
            explanations.append(self.explain_observation(row, score))
        return explanations
