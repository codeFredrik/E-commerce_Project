import pandas as pd
import numpy as np
from typing import List

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates, cleans, and sorts the input pricing dataframe.
    
    Args:
        df (pd.DataFrame): Raw dataframe with 'product_id', 'order_date', 'price'
    
    Returns:
        pd.DataFrame: Cleaned dataframe sorted chronologically per product with original_index preserved.
    """
    required_cols: List[str] = ['product_id', 'order_date', 'price']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")
        
    # Remove rows with missing values
    df = df.dropna(subset=required_cols).copy()
    
    # Remove invalid prices (must be strictly positive)
    df = df[df['price'] > 0]
    
    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    if len(df) <= 3:
        raise ValueError("Dataset must have more than 3 valid rows after cleaning")
        
    # Preserve original index for mapping anomalies back to input records
    if 'original_index' not in df.columns:
        df = df.reset_index(names=['original_index'])
    
    # Sort chronologically per product to prevent temporal data leakage
    df = df.sort_values(by=['product_id', 'order_date']).reset_index(drop=True)
    
    return df


def safe_zscore(x: pd.Series) -> pd.Series:
    """
    Calculates z-score safely, returning 0 for constant-price series where std == 0.
    
    Args:
        x (pd.Series): Price series for a product
    
    Returns:
        pd.Series: Z-scores
    """
    std = x.std()
    if std > 0 and not np.isnan(std):
        return (x - x.mean()) / std
    else:
        return pd.Series(0.0, index=x.index)
