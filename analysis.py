# Backward compatibility wrapper for legacy imports
from src.pipeline.preprocessing import preprocess_df, safe_zscore
from src.pipeline.features import create_features
from src.pipeline.rules import detect_suspicious_events, detect_suspicious_events_advanced, aggregate_scores
from src.pipeline.pipeline import run_analysis

__all__ = [
    'preprocess_df',
    'safe_zscore',
    'create_features',
    'detect_suspicious_events',
    'detect_suspicious_events_advanced',
    'aggregate_scores',
    'run_analysis'
]