import os
import pytest
import numpy as np
import pandas as pd
from src.pipeline.preprocessing import preprocess_df
from src.pipeline.features import create_features
from src.pipeline.ingestion import load_demo_showcases
from src.models.isolation_forest import DiscountIsolationForest
from src.models.explainer import DiscountExplainer
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


def test_showcase_demo_products_verdicts():
    showcases = load_demo_showcases()
    assert len(showcases) == 6

    all_df = pd.concat(list(showcases.values()), ignore_index=True)
    all_df = preprocess_df(all_df)
    all_df = create_features(all_df)

    model = DiscountIsolationForest(contamination=0.03, random_state=42)
    model.fit(all_df)
    scores = model.score(all_df)
    explainer = DiscountExplainer(model)
    exps = explainer.explain_dataframe(all_df, scores)

    all_df['risk_score'] = scores
    all_df['verdict'] = [e['verdict'] for e in exps]

    # 4 Manipulation archetypes: must be flagged as Fake/Suspicious (peak risk_score > 0.50)
    fake_archetype_ids = [9001, 9002, 9003, 9004]
    for p_id in fake_archetype_ids:
        grp = all_df[all_df['product_id'] == p_id]
        peak_score = grp['risk_score'].max()
        assert peak_score > 0.50, f"Product {p_id} should have peak risk > 0.50, got {peak_score}"
        top_verdict = grp.loc[grp['risk_score'].idxmax(), 'verdict']
        assert "FAKE" in top_verdict or "SUSPICIOUS" in top_verdict

    # 2 Genuine discount controls: must be scored as Genuine (peak risk_score < 0.50)
    genuine_control_ids = [9005, 9006]
    for p_id in genuine_control_ids:
        grp = all_df[all_df['product_id'] == p_id]
        peak_score = grp['risk_score'].max()
        assert peak_score < 0.50, f"Product {p_id} should have peak risk < 0.50, got {peak_score}"
        top_verdict = grp.loc[grp['risk_score'].idxmax(), 'verdict']
        assert "GENUINE" in top_verdict
