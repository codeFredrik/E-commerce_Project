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
    Supports both strict day-t evaluation and temporal window-based event evaluation.
    """
    def __init__(self, num_products: int = 50, days_per_product: int = 60, random_state: int = 42):
        self.num_products = num_products
        self.days_per_product = max(days_per_product, 25)  # require at least 25 days
        self.random_state = random_state
        np.random.seed(self.random_state)

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generates synthetic time series for products, injecting diverse ground-truth fraud patterns:
        - Genuine Sales (controls with 20% discount but zero inflation prior)
        - Type 1: Classic 1-Day Spike + Drop (caught by Basic, Advanced, and ML)
        - Type 2: Multi-Day Plateau 4 days + Drop (missed by Basic, caught by Advanced and ML)
        - Type 3: Slow 7-Day Creeping Inflation + Drop (missed by Basic, caught by Advanced and ML)
        - Type 4: Subtle Low-Volatility Manipulation +14% (missed by rigid rules, caught by ML)
        """
        records = []
        start_date = pd.Timestamp("2025-01-01")
        
        for p_id in range(1, self.num_products + 1):
            p_type = p_id % 5
            base_price = np.random.uniform(25.0, 200.0)
            
            # Subtle manipulation type has ultra-low baseline noise (0.2%), others have standard 1.5%
            noise_std = 0.002 if p_type == 4 else 0.015
            prices = base_price * np.random.normal(1.0, noise_std, self.days_per_product)
            labels = np.zeros(self.days_per_product, dtype=int)
            pattern_types = ['None'] * self.days_per_product
            
            mid = self.days_per_product // 2
            
            if p_type == 0:
                # Control: Genuine Sale (drop 20% for 3 days, no inflation prior)
                prices[mid:mid+3] *= 0.80
                pattern_types[mid:mid+3] = ['Genuine Sale'] * min(3, self.days_per_product - mid)
                
            elif p_type == 1:
                # Type 1: Classic 1-Day Spike + Drop (Caught by Basic, Advanced, and ML)
                prices[mid-1] = base_price * 1.40
                prices[mid] = base_price * 0.98
                labels[mid] = 1
                pattern_types[mid-1] = 'Classic Spike'
                pattern_types[mid] = 'Classic Spike Fraud Sale'
                
            elif p_type == 2:
                # Type 2: Multi-Day Plateau (4 days inflated) + Drop (Missed by Basic, caught by Advanced & ML)
                prices[mid-4:mid] = base_price * 1.35
                prices[mid] = base_price * 0.98
                labels[mid] = 1
                pattern_types[mid-4:mid] = ['Plateau Inflation'] * 4
                pattern_types[mid] = 'Plateau Fraud Sale'
                
            elif p_type == 3:
                # Type 3: Slow 7-Day Creeping Inflation (+3.5%/day) + Drop (Missed by Basic, caught by Advanced & ML)
                for i in range(7):
                    idx = mid - 7 + i
                    if 0 <= idx < self.days_per_product:
                        prices[idx] = base_price * (1.0 + 0.035 * (i + 1))
                        pattern_types[idx] = 'Creeping Inflation'
                prices[mid] = base_price * 1.02
                labels[mid] = 1
                pattern_types[mid] = 'Creeping Fraud Sale'
                
            elif p_type == 4:
                # Type 4: Subtle Borderline Manipulation in Low-Volatility Products (+14% inflation)
                # Missed by rigid >18% rules, cleanly caught by Isolation Forest
                prices[mid-2:mid] = base_price * 1.14
                prices[mid] = base_price * 0.99
                labels[mid] = 1
                pattern_types[mid-2:mid] = ['Subtle Inflation'] * 2
                pattern_types[mid] = 'Subtle Fraud Sale'
                
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

    def evaluate_window_matching(self, df: pd.DataFrame, y_pred: np.ndarray, window_days: int = 2) -> Dict[str, float]:
        """
        Evaluates predictions using event-level tolerance window matching (window_days).
        """
        true_indices = df[df['is_fake_discount'] == 1].index.tolist()
        pred_indices = set(np.where(y_pred == 1)[0])
        
        tp_events = 0
        for idx in true_indices:
            p_id = df.loc[idx, 'product_id']
            window = set(df[(df['product_id'] == p_id) & (df.index >= idx - window_days) & (df.index <= idx + window_days)].index)
            if window & pred_indices:
                tp_events += 1
                
        fp_events = 0
        for p_idx in pred_indices:
            p_id = df.loc[p_idx, 'product_id']
            in_window = any(
                df.loc[t_idx, 'product_id'] == p_id and abs(p_idx - t_idx) <= window_days 
                for t_idx in true_indices
            )
            if not in_window:
                fp_events += 1
                
        prec = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
        rec = tp_events / len(true_indices) if len(true_indices) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            'Event_Precision': round(prec, 4),
            'Event_Recall': round(rec, 4),
            'Event_F1': round(f1, 4)
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Runs benchmark evaluating Rule-Based Basic, Rule-Based Advanced, and Isolation Forest.
        Reports both strict day-t point metrics and window event metrics.
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
        iso_model = DiscountIsolationForest(contamination=0.02, random_state=self.random_state)
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
            win = self.evaluate_window_matching(df, y_p, window_days=2)
            return {
                'Method': name,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4),
                'Accuracy': round(acc, 4),
                'Confusion_Matrix': cm,
                'Window_Event_F1': win['Event_F1']
            }
            
        results = [
            compute_metrics(y_true, y_pred_basic, "Rule-Based (Basic)"),
            compute_metrics(y_true, y_pred_adv, "Rule-Based (Advanced)"),
            compute_metrics(y_true, y_pred_iso, "Isolation Forest (ML)")
        ]
        
        window_results = [
            {'Method': "Rule-Based (Basic)", **self.evaluate_window_matching(df, y_pred_basic, window_days=2)},
            {'Method': "Rule-Based (Advanced)", **self.evaluate_window_matching(df, y_pred_adv, window_days=2)},
            {'Method': "Isolation Forest (ML)", **self.evaluate_window_matching(df, y_pred_iso, window_days=2)}
        ]
        
        return {
            'metrics_table': results,
            'window_metrics_table': window_results,
            'dataset_summary': {
                'total_records': len(df),
                'total_products': self.num_products,
                'true_fraud_events': int(sum(y_true))
            },
            'scored_df': df.assign(iso_risk_score=iso_scores, pred_basic=y_pred_basic, pred_adv=y_pred_adv, pred_iso=y_pred_iso)
        }
