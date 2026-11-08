import pandas as pd
import numpy as np
import pytest
from src.pipeline.preprocessing import preprocess_df, safe_zscore
from src.pipeline.features import create_features
from src.pipeline.rules import detect_suspicious_events, aggregate_scores
from src.pipeline.pipeline import run_analysis

def test_preprocess_df_sorting():
    data = {
        'product_id': [1, 1, 2, 1],
        'order_date': ['2023-01-03', '2023-01-01', '2023-01-01', '2023-01-02'],
        'price': [100.0, 100.0, 50.0, 100.0]
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_df(df)
    
    assert processed_df.iloc[0]['product_id'] == 1
    assert processed_df.iloc[0]['order_date'] == pd.Timestamp('2023-01-01')
    assert processed_df.iloc[2]['product_id'] == 1
    assert processed_df.iloc[2]['order_date'] == pd.Timestamp('2023-01-03')
    assert processed_df.iloc[3]['product_id'] == 2
    
def test_preprocess_df_invalid():
    df = pd.DataFrame({'product_id': [1], 'price': [100.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_df(df)
        
    df = pd.DataFrame({
        'product_id': [1, 1],
        'order_date': ['2023-01-01', '2023-01-02'],
        'price': [100.0, 100.0]
    })
    with pytest.raises(ValueError, match="Dataset must have more than 3 valid rows"):
        preprocess_df(df)

def test_safe_zscore_zero_std():
    data = {
        'product_id': [1]*5,
        'order_date': pd.date_range('2023-01-01', periods=5),
        'price': [50.0]*5
    }
    df = pd.DataFrame(data)
    features_df = create_features(df)
    
    assert (features_df['z_score'] == 0.0).all()

def test_create_features_advanced():
    # Test pre-sale inflation window and volatility ratios
    data = {
        'product_id': [1]*10,
        'order_date': pd.date_range('2023-01-01', periods=10),
        'price': [50.0, 50.0, 52.0, 70.0, 72.0, 70.0, 50.0, 50.0, 51.0, 50.0]
    }
    df = pd.DataFrame(data)
    df = preprocess_df(df)
    features_df = create_features(df)
    
    assert 'pre_sale_inflation_pct' in features_df.columns
    assert 'volatility_ratio' in features_df.columns
    assert 'dev_from_30d_mean' in features_df.columns
    assert not features_df['pre_sale_inflation_pct'].isna().any()
    assert not features_df['volatility_ratio'].isna().any()

def test_detect_suspicious_events():
    data = {
        'product_id': [1, 1, 1, 1, 1],
        'order_date': pd.date_range('2023-01-01', periods=5),
        'price': [50.0, 50.0, 70.0, 50.0, 50.0]
    }
    df = pd.DataFrame(data)
    df['original_index'] = range(len(df))
    
    anomalies = detect_suspicious_events(df)
    assert len(anomalies) == 1
    anomaly = anomalies[0]
    assert anomaly['product_id'] == 1
    assert anomaly['curr_price'] == 50.0
    assert np.isclose(anomaly['score'], 0.3428, atol=0.01)

def test_aggregate_scores():
    suspicious = [
        {'product_id': 1, 'score': 0.5},
        {'product_id': 1, 'score': 0.7},
        {'product_id': 2, 'score': 0.8}
    ]
    summary = aggregate_scores(suspicious)
    assert len(summary) == 2
    prod1 = next(item for item in summary if item['product'] == 1)
    assert prod1['num_events'] == 2
    assert np.isclose(prod1['final_score'], 0.819, atol=0.01)
