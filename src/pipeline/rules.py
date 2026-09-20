import numpy as np
import pandas as pd
from typing import List, Dict, Any

def detect_suspicious_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects price spikes followed by sharp drops using vectorized operations (Basic Method).
    
    Args:
        df (pd.DataFrame): Dataframe with price, product_id, order_date, prev_price, next_price
    
    Returns:
        list: Suspicious events with 'product_id', 'original_index', 'score'
    """
    df_calc = df.copy()
    if 'prev_price' not in df_calc.columns or 'next_price' not in df_calc.columns:
        grouped = df_calc.groupby('product_id')
        df_calc['prev_price'] = grouped['price'].shift(1)
        df_calc['next_price'] = grouped['price'].shift(-1)
        
    valid_window = df_calc['prev_price'].notna() & df_calc['next_price'].notna()
    
    # Detection conditions: >25% spike followed by <75% drop
    spike_detected = df_calc['price'] > df_calc['prev_price'] * 1.25
    drop_detected = df_calc['next_price'] < df_calc['price'] * 0.75
    
    suspicious_mask = valid_window & spike_detected & drop_detected
    anomalies = df_calc[suspicious_mask].copy()
    
    if anomalies.empty:
        return []
        
    # Calculate scores
    prev_price_safe = anomalies['prev_price'].replace(0, np.nan)
    curr_price_safe = anomalies['price'].replace(0, np.nan)
    
    increase_pct = ((anomalies['price'] - prev_price_safe) / prev_price_safe).abs().fillna(0).clip(upper=1.0)
    drop_pct = ((anomalies['price'] - anomalies['next_price']) / curr_price_safe).abs().fillna(0).clip(upper=1.0)
    
    anomalies['score'] = (increase_pct + drop_pct) / 2
    anomalies['curr_price'] = anomalies['price']
    
    cols_to_keep = ['product_id', 'original_index', 'score', 'prev_price', 'curr_price', 'next_price']
    return anomalies[cols_to_keep].to_dict('records')


def detect_suspicious_events_advanced(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects pricing anomalies using statistical methods + pattern matching (Advanced Method).
    
    Args:
        df (pd.DataFrame): Dataframe with features (rolling_mean, smoothed_price, z_score)
    
    Returns:
        list: Suspicious events with enriched metadata
    """
    df_calc = df.copy()
    if 'prev_price' not in df_calc.columns or 'next_price' not in df_calc.columns:
        grouped = df_calc.groupby('product_id')
        df_calc['prev_price'] = grouped['price'].shift(1)
        df_calc['next_price'] = grouped['price'].shift(-1)
        
    valid_window = df_calc['prev_price'].notna() & df_calc['next_price'].notna()
    
    price_spike = df_calc['smoothed_price'] > df_calc['rolling_mean'] * 1.3
    sharp_drop = df_calc['next_price'] < df_calc['price'] * 0.75
    statistical_anomaly = df_calc['z_score'].abs() > 2
    
    suspicious_mask = valid_window & ((price_spike & sharp_drop) | statistical_anomaly)
    anomalies = df_calc[suspicious_mask].copy()
    
    if anomalies.empty:
        return []
        
    prev_price_safe = anomalies['prev_price'].replace(0, np.nan)
    curr_price_safe = anomalies['price'].replace(0, np.nan)
    
    increase_pct = ((anomalies['price'] - prev_price_safe) / prev_price_safe).abs().fillna(0).clip(upper=1.0)
    drop_pct = ((anomalies['price'] - anomalies['next_price']) / curr_price_safe).abs().fillna(0).clip(upper=1.0)
    
    base_score = (increase_pct + drop_pct) / 2
    anomaly_bonus = np.where(anomalies['z_score'].abs() > 2, 0.2, 0.0)
    
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
