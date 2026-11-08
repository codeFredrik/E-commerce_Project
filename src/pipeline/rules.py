import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.pipeline.features import create_features

def detect_suspicious_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects fake discount events on the sale drop day using a naive short-memory heuristic (Basic Method).
    Looks strictly at immediate 1-to-2 day price jumps (>18-20% above 30-day baseline) followed by 
    an immediate drop back to baseline today.
    
    Naturally catches classic 1-day spike-and-drops, but misses multi-day plateaus, creeping inflation,
    and subtle low-volatility manipulation.
    
    Args:
        df (pd.DataFrame): Dataframe with price observations and features
    
    Returns:
        list: Suspicious events with 'product_id', 'original_index', 'score'
    """
    df_calc = df.copy()
    if 'rolling_mean_30' not in df_calc.columns or 'prev_price' not in df_calc.columns:
        df_calc = create_features(df_calc)
        
    if 'original_index' not in df_calc.columns:
        df_calc['original_index'] = df_calc.index
        
    if 'price_lag_2' not in df_calc.columns:
        df_calc['price_lag_2'] = df_calc.groupby('product_id')['price'].shift(2).fillna(df_calc['price'])
    if 'price_lag_3' not in df_calc.columns:
        df_calc['price_lag_3'] = df_calc.groupby('product_id')['price'].shift(3).fillna(df_calc['price'])
        
    # Naive short memory: immediate 1-2 day spike
    prev_max_2d = np.maximum(df_calc['prev_price'], df_calc['price_lag_2'])
    recent_jump_2d = (prev_max_2d > df_calc['rolling_mean_30'] * 1.18) & (df_calc['price_lag_3'] <= df_calc['rolling_mean_30'] * 1.10)
    
    # Single-day drop today from the spike
    drop_today = (df_calc['prev_price'] - df_calc['price']) / (df_calc['prev_price'] + 1e-6) >= 0.15
    
    # Not a genuine deep sale below baseline
    not_deep_sale = df_calc['price'] >= df_calc['rolling_mean_30'] * 0.88
    
    suspicious_mask = recent_jump_2d & drop_today & not_deep_sale
    anomalies = df_calc[suspicious_mask].copy()
    
    if anomalies.empty:
        return []
        
    # Calculate score based on immediate short jump and drop depth
    inflation_ratio = (anomalies['prev_price'] - anomalies['price_lag_2']) / (anomalies['price_lag_2'] + 1e-6)
    drop_ratio = (anomalies['prev_price'] - anomalies['price']) / (anomalies['prev_price'] + 1e-6)
    
    anomalies['score'] = ((inflation_ratio.clip(0, 1) + drop_ratio.clip(0, 1)) / 2).clip(0.0, 1.0)
    anomalies['curr_price'] = anomalies['price']
    
    cols_to_keep = ['product_id', 'original_index', 'score', 'prev_price', 'curr_price', 'next_price']
    return anomalies[cols_to_keep].to_dict('records')


def detect_suspicious_events_advanced(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects fake discounts combining multi-day spike plateaus (14-day memory), creeping inflation,
    and advertised discount exaggeration patterns on the sale drop day (Advanced Method).
    
    Uses rigid boolean cutoffs (>18% inflation, >18% drop, >15% creeping sum), catching plateaus 
    and creeping inflation while naturally missing subtle borderline variations.
    
    Args:
        df (pd.DataFrame): Dataframe with features
    
    Returns:
        list: Suspicious events with enriched metadata
    """
    df_calc = df.copy()
    if 'recent_peak_inflation_ratio' not in df_calc.columns or 'fake_discount_illusion_index' not in df_calc.columns:
        df_calc = create_features(df_calc)
        
    if 'original_index' not in df_calc.columns:
        df_calc['original_index'] = df_calc.index
        
    # Discount drop today
    drop_today = (df_calc['prev_price'] - df_calc['price']) / (df_calc['prev_price'] + 1e-6) >= 0.08
    not_deep_dip = df_calc['price'] >= df_calc['rolling_median_30'] * 0.88
    
    # Pattern A: Multi-Day Spike / Plateau + Drop (14-day causal window)
    pat_a = (
        (df_calc['recent_peak_inflation_ratio'] >= 0.18) & 
        (df_calc['drop_from_recent_peak'] >= 0.18) & 
        not_deep_dip & 
        drop_today
    )
    
    # Pattern B: Creeping Inflation + Drop
    # Cumulative prior positive gains >= 15% followed by single-day drop of >= 10%
    pos_pct = df_calc.groupby('product_id')['price'].pct_change().clip(lower=0.0).fillna(0.0)
    creeping_prior = (
        pos_pct.groupby(df_calc['product_id']).shift(1).fillna(0.0)
        .groupby(df_calc['product_id']).transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    pat_b = (
        (creeping_prior >= 0.15) & 
        ((df_calc['prev_price'] - df_calc['price']) / (df_calc['prev_price'] + 1e-6) >= 0.10) & 
        (df_calc['price'] >= df_calc['rolling_median_30'] * 0.90)
    )
    
    # Pattern C: Exaggerated Advertised Discount
    pat_c = (
        (df_calc['discount_exaggeration_gap'] >= 0.15) & 
        (df_calc['drop_from_recent_peak'] >= 0.10) & 
        not_deep_dip & 
        drop_today
    )
    
    suspicious_mask = pat_a | pat_b | pat_c
    anomalies = df_calc[suspicious_mask].copy()
    
    if anomalies.empty:
        return []
        
    base_score = (
        0.40 * (anomalies['fake_discount_illusion_index'] / 0.18).clip(upper=1.0) +
        0.35 * (anomalies['drop_from_recent_peak'] / 0.30).clip(upper=1.0) +
        0.25 * (anomalies['discount_exaggeration_gap'] / 0.25).clip(upper=1.0)
    )
    anomaly_bonus = np.where(anomalies['z_score'].abs() > 2, 0.15, 0.0)
    
    anomalies['score'] = (base_score + anomaly_bonus).clip(upper=1.0)
    anomalies['curr_price'] = anomalies['price']
    anomalies['statistical_anomaly'] = anomalies['z_score'].abs() > 2
    
    cols_to_keep = ['product_id', 'original_index', 'score', 'prev_price', 'curr_price', 'next_price', 'z_score', 'statistical_anomaly']
    return anomalies[cols_to_keep].to_dict('records')


def aggregate_scores(suspicious: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregates suspicious events to product-level risk scores with log weighting for repeat offenses.
    
    Args:
        suspicious (list): List of suspicious event dictionaries
    
    Returns:
        list: Product-level summary with aggregated scores
    """
    if not suspicious:
        return []
        
    df_susp = pd.DataFrame(suspicious)
    
    summary_df = df_susp.groupby('product_id').agg(
        num_events=('score', 'count'),
        avg_score=('score', 'mean')
    ).reset_index()
    
    summary_df['final_score'] = summary_df['avg_score'] * (1 + np.log1p(summary_df['num_events']) / 3)
    summary_df['final_score'] = summary_df['final_score'].clip(upper=1.0).round(4)
    summary_df['avg_score'] = summary_df['avg_score'].round(4)
    
    summary_df = summary_df.rename(columns={'product_id': 'product'})
    
    return summary_df.to_dict('records')
