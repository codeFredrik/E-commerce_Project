import pandas as pd
import numpy as np
from src.pipeline.preprocessing import safe_zscore

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates rolling statistics, lag features, baseline deviations, and volatility metrics per product.
    Fully vectorized for maximum performance.
    
    Args:
        df (pd.DataFrame): Preprocessed and chronologically sorted dataframe
    
    Returns:
        pd.DataFrame: Enhanced dataframe with feature columns
    """
    df = df.copy()
    grouped = df.groupby('product_id')['price']
    
    # 1. Standard rolling baselines and smoothing
    df['rolling_mean'] = grouped.transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['rolling_mean_30'] = grouped.transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['smoothed_price'] = grouped.transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['z_score'] = grouped.transform(safe_zscore).fillna(0.0)
    
    # 2. Lag and Lead prices
    df['prev_price'] = grouped.shift(1)
    df['next_price'] = grouped.shift(-1)
    df['price_lag_3'] = grouped.shift(3)
    
    # 3. Pre-Sale Inflation Windows (Max price in preceding 14 observations)
    # Shift by 1 first so we don't include the current price in the preceding window
    prev_prices_grouped = df.groupby('product_id')['prev_price']
    df['max_price_prev_14'] = prev_prices_grouped.transform(lambda x: x.rolling(window=14, min_periods=1).max())
    
    # Fill NA for max_price_prev_14 with current price or rolling mean
    df['max_price_prev_14'] = df['max_price_prev_14'].fillna(df['rolling_mean'])
    
    # Pre-sale inflation percentage: how much higher was the recent max compared to the 30-period baseline?
    rolling_30_safe = df['rolling_mean_30'].replace(0, np.nan)
    df['pre_sale_inflation_pct'] = ((df['max_price_prev_14'] - rolling_30_safe) / rolling_30_safe).fillna(0.0)
    df['pre_sale_inflation_pct'] = df['pre_sale_inflation_pct'].clip(lower=0.0, upper=5.0)  # non-negative inflation
    
    # 4. Baseline Deviations
    rolling_5_safe = df['rolling_mean'].replace(0, np.nan)
    df['dev_from_5d_mean'] = ((df['price'] - rolling_5_safe) / rolling_5_safe).fillna(0.0)
    df['dev_from_30d_mean'] = ((df['price'] - rolling_30_safe) / rolling_30_safe).fillna(0.0)
    
    # 5. Volatility Ratio (short-term 5-period std vs long-term 30-period std)
    df['price_std_5'] = grouped.transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0.0)
    df['price_std_30'] = grouped.transform(lambda x: x.rolling(window=30, min_periods=1).std()).fillna(0.0)
    df['volatility_ratio'] = df['price_std_5'] / (df['price_std_30'] + 1e-5)
    df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0).clip(lower=0.0, upper=10.0)
    
    return df
