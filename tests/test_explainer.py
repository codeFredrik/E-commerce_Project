import pytest
import pandas as pd
import numpy as np
from src.pipeline.preprocessing import preprocess_df
from src.pipeline.features import create_features
from src.models.isolation_forest import DiscountIsolationForest
from src.models.explainer import DiscountExplainer

def test_discount_explainer():
    data = {
        'product_id': [1]*15,
        'order_date': pd.date_range('2023-01-01', periods=15),
        'price': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 70.0, 75.0, 70.0, 50.0, 50.0]
    }
    df = pd.DataFrame(data)
    df = preprocess_df(df)
    df = create_features(df)
    
    model = DiscountIsolationForest(contamination=0.1, random_state=42)
    model.fit(df)
    scores = model.score(df)
    
    explainer = DiscountExplainer(model)
    explanations = explainer.explain_dataframe(df, scores)
    
    assert len(explanations) == len(df)
    for exp in explanations:
        assert 'verdict' in exp
        assert 'confidence_pct' in exp
        assert 'reasons' in exp
        assert 'risk_level' in exp
        assert isinstance(exp['reasons'], list)
        assert len(exp['reasons']) > 0
        
    # Check that high risk rows get red or orange badges
    high_risk = [exp for exp in explanations if exp['risk_score'] > 0.6]
    if high_risk:
        for hr in high_risk:
            assert "FAKE" in hr['verdict'] or "SUSPICIOUS" in hr['verdict']
