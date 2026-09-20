import os
import pytest
import numpy as np
import pandas as pd
from src.pipeline.preprocessing import preprocess_df
from src.pipeline.features import create_features
from src.models.isolation_forest import DiscountIsolationForest
from src.models.evaluator import SyntheticFraudBenchmark

def test_isolation_forest_workflow(tmp_path):
    # Generate simple test dataset
    data = {
        'product_id': [1]*20 + [2]*20,
        'order_date': list(pd.date_range('2023-01-01', periods=20)) * 2,
        'price': [50.0 + np.random.normal(0, 1) for _ in range(39)] + [150.0]  # One huge outlier
    }
    df = pd.DataFrame(data)
    df = preprocess_df(df)
    df = create_features(df)
    
    model = DiscountIsolationForest(contamination=0.1, random_state=42)
    model.fit(df)
    scores = model.score(df)
    
    assert len(scores) == len(df)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    
    # Test persistence
    model_path = os.path.join(tmp_path, "test_if_model.joblib")
    model.save_model(model_path)
    assert os.path.exists(model_path)
    
    loaded_model = DiscountIsolationForest.load_model(model_path)
    loaded_scores = loaded_model.score(df)
    assert np.allclose(scores, loaded_scores)

def test_synthetic_fraud_benchmark():
    benchmark = SyntheticFraudBenchmark(num_products=12, days_per_product=40, random_state=42)
    results = benchmark.run_benchmark()
    
    assert 'metrics_table' in results
    assert len(results['metrics_table']) == 3
    methods = [row['Method'] for row in results['metrics_table']]
    assert "Rule-Based (Basic)" in methods
    assert "Rule-Based (Advanced)" in methods
    assert "Isolation Forest (ML)" in methods
    
    for row in results['metrics_table']:
        assert 0.0 <= row['Precision'] <= 1.0
        assert 0.0 <= row['Recall'] <= 1.0
        assert 0.0 <= row['F1_Score'] <= 1.0
