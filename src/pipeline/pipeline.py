import pandas as pd
from typing import Tuple, List, Dict, Any
from src.pipeline.preprocessing import preprocess_df
from src.pipeline.features import create_features
from src.pipeline.rules import detect_suspicious_events, detect_suspicious_events_advanced, aggregate_scores

def run_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs complete analysis pipeline: preprocessing → feature engineering → detection.
    
    Args:
        df (pd.DataFrame): Raw dataframe
    
    Returns:
        tuple: (df_processed, suspicious_basic, summary_basic, suspicious_advanced, summary_advanced)
    """
    # Preprocess
    df = preprocess_df(df)
    
    # Feature engineering
    df = create_features(df)
    
    # Run both detection methods
    suspicious_basic = detect_suspicious_events(df)
    suspicious_advanced = detect_suspicious_events_advanced(df)
    
    # Aggregate scores
    summary_basic = aggregate_scores(suspicious_basic)
    summary_advanced = aggregate_scores(suspicious_advanced)
    
    return df, suspicious_basic, summary_basic, suspicious_advanced, summary_advanced
