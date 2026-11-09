import pandas as pd
import numpy as np
from src.pipeline.preprocessing import safe_zscore

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates robust baselines, causal rolling extremes, sale-day risk transfer metrics,
    and volatility indicators per product. Fully vectorized.

    Args:
        df (pd.DataFrame): Preprocessed and chronologically sorted dataframe

    Returns:
        pd.DataFrame: Enhanced dataframe with engineered feature columns
    """
    df = df.copy()
    grouped = df.groupby('product_id')['price']

    # 1. Robust Baselines & Rolling Statistics
    df['rolling_mean'] = grouped.transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['rolling_mean_30'] = grouped.transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['rolling_median_30'] = grouped.transform(lambda x: x.rolling(window=30, min_periods=1).median())

    # Fill any fallback baselines with current price
    df['rolling_mean_30'] = df['rolling_mean_30'].fillna(df['price'])
    df['rolling_median_30'] = df['rolling_median_30'].fillna(df['price'])
    df['smoothed_price'] = grouped.transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['z_score'] = grouped.transform(safe_zscore).fillna(0.0)

    # 2. Lag and Lead prices
    df['prev_price'] = grouped.shift(1)
    df['price_lag_2'] = grouped.shift(2)
    df['price_lag_3'] = grouped.shift(3)
    df['next_price'] = grouped.shift(-1)

    # 3. Rolling Extremes Over Preceding 14 Days (Strictly Causal - excludes today's price)
    prev_prices_grouped = df.groupby('product_id')['prev_price']
    df['max_price_prev_14'] = prev_prices_grouped.transform(lambda x: x.rolling(window=14, min_periods=1).max())
    df['min_price_prev_14'] = prev_prices_grouped.transform(lambda x: x.rolling(window=14, min_periods=1).min())
    df['max_price_prev_14'] = df['max_price_prev_14'].fillna(df['price'])
    df['min_price_prev_14'] = df['min_price_prev_14'].fillna(df['price'])

    # 4. Sale-Day Risk Transfer Features (Projected onto drop day t)
    median_30_safe = df['rolling_median_30'].replace(0, np.nan)
    max_prev_safe = df['max_price_prev_14'].replace(0, np.nan)

    # Inflation ratio of recent 14-day peak relative to the 30-day baseline
    df['recent_peak_inflation_ratio'] = ((df['max_price_prev_14'] - median_30_safe) / median_30_safe).fillna(0.0)
    # Backward compatibility with existing pipeline & UI
    df['pre_sale_inflation_pct'] = np.maximum(0.0, df['recent_peak_inflation_ratio']).clip(upper=5.0)

    # Drop depth from recent peak
    df['drop_from_recent_peak'] = ((df['max_price_prev_14'] - df['price']) / max_prev_safe).fillna(0.0)

    # Fake Discount Illusion Index: High ONLY when drop is large immediately following an inflated peak (>3% noise threshold)
    inflation_excess = np.maximum(0.0, df['recent_peak_inflation_ratio'] - 0.03)
    df['fake_discount_illusion_index'] = (
        inflation_excess *
        np.maximum(0.0, df['drop_from_recent_peak'])
    ).fillna(0.0)

    # Drop today and discount signal
    df['price_drop_1d'] = np.maximum(0.0, (df['prev_price'] - df['price']) / (df['prev_price'] + 1e-6))
    df['deceptive_discount_signal'] = df['fake_discount_illusion_index'] * np.clip(df['price_drop_1d'] / 0.06, 0.0, 1.0)

    # Creeping Inflation Slope: cumulative positive daily gains in prior 7 days, scaled by today's drop
    pct_change = grouped.transform(lambda x: x.pct_change()).fillna(0.0)
    pos_pct_change = pct_change.clip(lower=0.0)
    pos_pct_prior = pos_pct_change.groupby(df['product_id']).shift(1).fillna(0.0)
    creeping_prior_sum = (
        pos_pct_prior.groupby(df['product_id'])
        .transform(lambda x: x.rolling(window=7, min_periods=1).sum())
        .fillna(0.0)
    )
    df['creeping_inflation_slope'] = (creeping_prior_sum * np.maximum(0.0, df['drop_from_recent_peak'])).fillna(0.0)

    # 5. Baseline Deviations & Volatility
    rolling_5_safe = df['rolling_mean'].replace(0, np.nan)
    rolling_30_safe = df['rolling_mean_30'].replace(0, np.nan)
    df['dev_from_5d_mean'] = ((df['price'] - rolling_5_safe) / rolling_5_safe).fillna(0.0)
    df['dev_from_30d_mean'] = ((df['price'] - rolling_30_safe) / rolling_30_safe).fillna(0.0)

    df['price_std_5'] = grouped.transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0.0)
    df['price_std_30'] = grouped.transform(lambda x: x.rolling(window=30, min_periods=1).std()).fillna(0.0)
    df['volatility_ratio'] = df['price_std_5'] / (df['price_std_30'] + 1e-5)
    df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0).clip(lower=0.0, upper=10.0)
    df['volatility_deviation'] = (df['max_price_prev_14'] - df['rolling_mean_30']) / (df['price_std_30'] + 1e-4)

    # 6. Explicit Discount Cross-Check
    if 'discount_percent' in df.columns and df['discount_percent'].notna().any():
        effective_price = df['discounted_price'] if 'discounted_price' in df.columns and df['discounted_price'].notna().any() else df['price']
        real_discount = ((median_30_safe - effective_price) / median_30_safe).fillna(0.0)
        df['discount_exaggeration_gap'] = np.maximum(0.0, (df['discount_percent'] / 100.0) - real_discount).fillna(0.0)
    else:
        real_discount = ((median_30_safe - df['price']) / median_30_safe).fillna(0.0)
        df['discount_exaggeration_gap'] = np.maximum(0.0, df['drop_from_recent_peak'] - real_discount).fillna(0.0)

    return df.fillna(0.0)
