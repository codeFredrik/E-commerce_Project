import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from src.pipeline.preprocessing import preprocess_df
from src.pipeline.features import create_features
from src.pipeline.rules import detect_suspicious_events, detect_suspicious_events_advanced
from src.models.isolation_forest import DiscountIsolationForest

class SyntheticFraudBenchmark:
    """
    Generates synthetic e-commerce pricing data with known injected manipulation patterns
    to formally benchmark ML vs. rule-based detection methods.
    """
    def __init__(self, num_products: int = 50, days_per_product: int = 60, random_state: int = 42):
        self.num_products = num_products
        self.days_per_product = max(days_per_product, 25)  # require at least 25 days
        self.random_state = random_state
        np.random.seed(self.random_state)

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generates synthetic time series for products, injecting ground-truth fraud patterns.
        """
        records = []
        start_date = pd.Timestamp("2025-01-01")
        
        for p_id in range(1, self.num_products + 1):
            base_price = np.random.uniform(20.0, 200.0)
            # Normal day-to-day noise: ~1.5% std
            prices = base_price * np.random.normal(1.0, 0.015, self.days_per_product)
            labels = np.zeros(self.days_per_product, dtype=int)
            pattern_types = ['None'] * self.days_per_product
            
            mid = self.days_per_product // 2
            p_type = p_id % 4
            
            if p_type == 0:
                # Genuine Sale: drop 20% for 3 days, no inflation prior
                prices[mid:mid+3] *= 0.80
                pattern_types[mid:mid+3] = ['Genuine Sale'] * min(3, self.days_per_product - mid)
                
            elif p_type == 1:
                # Type 1: Classic Spike + Drop (Fake Discount)
                # Inflate by +40% for 2 days, then drop down to base price advertised as "sale"
                prices[mid-2:mid] *= 1.40
                prices[mid] = base_price * 0.98  # advertised as huge sale
                labels[mid] = 1
                pattern_types[mid-2:mid] = ['Spike+Drop Inflation'] * 2
                pattern_types[mid] = 'Spike+Drop Fraud Sale'
                
            elif p_type == 2:
                # Type 2: Creeping Inflation + Drop
                # Slowly inflate price by +5% each day for 7 days (+35% total), then drop -25%
                for i in range(7):
                    idx = mid - 7 + i
                    if 0 <= idx < self.days_per_product:
                        prices[idx] = base_price * (1.0 + 0.05 * (i + 1))
                        pattern_types[idx] = 'Creeping Inflation'
                prices[mid] = base_price * 1.05  # Still higher than original base price!
                labels[mid] = 1
                pattern_types[mid] = 'Creeping Fraud Sale'
                
            elif p_type == 3:
                # Type 3: Artificial Scarcity (High Volatility oscillations before sale)
                osc_len = min(5, mid)
                prices[mid-osc_len:mid] = base_price * np.random.choice([1.30, 0.85, 1.35, 0.80, 1.25], osc_len)
                prices[mid] = base_price * 0.95
                labels[mid] = 1
                pattern_types[mid-osc_len:mid] = ['Artificial Scarcity'] * osc_len
                pattern_types[mid] = 'Scarcity Fraud Sale'
                
            for d in range(self.days_per_product):
                records.append({
                    'product_id': p_id,
                    'order_date': start_date + pd.Timedelta(days=d),
                    'price': round(float(prices[d]), 2),
                    'is_fake_discount': int(labels[d]),
                    'pattern_type': pattern_types[d]
                })
                
        df = pd.DataFrame(records)
        return df

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Runs the benchmark, evaluating Rule-Based Basic, Rule-Based Advanced, and Isolation Forest.
        """
        df = self.generate_dataset()
        df = preprocess_df(df)
        df = create_features(df)
        
        y_true = df['is_fake_discount'].values
        
        # 1. Rule-Based Basic
        basic_events = detect_suspicious_events(df)
        basic_indices = {e['original_index'] for e in basic_events}
        y_pred_basic = np.array([1 if idx in basic_indices else 0 for idx in df['original_index']])
        
        # 2. Rule-Based Advanced
        adv_events = detect_suspicious_events_advanced(df)
        adv_indices = {e['original_index'] for e in adv_events}
        y_pred_adv = np.array([1 if idx in adv_indices else 0 for idx in df['original_index']])
        
        # 3. Isolation Forest ML
        iso_model = DiscountIsolationForest(contamination=0.05, random_state=self.random_state)
        iso_model.fit(df)
        iso_scores = iso_model.score(df)
        # Threshold: > 0.5 risk score indicates fraud
        y_pred_iso = (iso_scores > 0.50).astype(int)
        
        def compute_metrics(y_t, y_p, name):
            precision = precision_score(y_t, y_p, zero_division=0)
            recall = recall_score(y_t, y_p, zero_division=0)
            f1 = f1_score(y_t, y_p, zero_division=0)
            acc = accuracy_score(y_t, y_p)
            cm = confusion_matrix(y_t, y_p).tolist()
            return {
                'Method': name,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4),
                'Accuracy': round(acc, 4),
                'Confusion_Matrix': cm
            }
            
        results = [
            compute_metrics(y_true, y_pred_basic, "Rule-Based (Basic)"),
            compute_metrics(y_true, y_pred_adv, "Rule-Based (Advanced)"),
            compute_metrics(y_true, y_pred_iso, "Isolation Forest (ML)")
        ]
        
        return {
            'metrics_table': results,
            'dataset_summary': {
                'total_records': len(df),
                'total_products': self.num_products,
                'true_fraud_events': int(sum(y_true))
            },
            'scored_df': df.assign(iso_risk_score=iso_scores, pred_basic=y_pred_basic, pred_adv=y_pred_adv, pred_iso=y_pred_iso)
        }
