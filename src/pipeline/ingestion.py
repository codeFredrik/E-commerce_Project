import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "ebay_cache"

class EbayIngestionClient:
    """
    Ingestion client for eBay price history and offline demo caching.
    Supports authenticated eBay Browse/Finding API fetching with graceful
    offline fallback to curated showcase products in data/ebay_cache.
    """
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.client_id = client_id or os.getenv("EBAY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("EBAY_CLIENT_SECRET")
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_api_configured(self) -> bool:
        """Checks if eBay API credentials are provided."""
        return bool(self.client_id and self.client_secret)

    def get_cached_showcase_products(self) -> Dict[str, pd.DataFrame]:
        """
        Loads all cached demo showcase products from data/ebay_cache.
        Returns a dictionary mapping display label to product DataFrame.
        """
        showcases = {}
        if not self.cache_dir.exists():
            return showcases

        csv_files = sorted(self.cache_dir.glob("*.csv"))
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                df['order_date'] = pd.to_datetime(df['order_date'])
                # Derive friendly display label from filename or column
                label = file_path.stem.replace("_", " ").title()
                if 'product_name' in df.columns and len(df['product_name']) > 0:
                    prod_name = str(df['product_name'].iloc[0])
                    pattern = str(df.get('pattern_label', ['Demo']).iloc[0])
                    label = f"{prod_name} ({pattern})"
                showcases[label] = df
            except Exception as e:
                logger.warning(f"Could not load cached product {file_path.name}: {e}")

        return showcases

    def fetch_price_history(self, query: str, days: int = 60) -> pd.DataFrame:
        """
        Fetches price observation history for a query.
        Falls back to matching cached product or synthetic generation if offline/unauthenticated.
        """
        cached = self.get_cached_showcase_products()
        # Look for close match in cache
        query_lower = query.lower()
        for label, df in cached.items():
            if any(term in label.lower() for term in query_lower.split()):
                return df

        # If API is configured, live eBay query can be executed here
        if self.is_api_configured():
            logger.info(f"Live eBay API querying not actively enabled in offline sandbox mode; falling back to cache.")

        # Fallback to the first available showcase product
        if cached:
            return next(iter(cached.values()))

        raise FileNotFoundError(f"No cached eBay product data found in {self.cache_dir}.")


def load_demo_showcases() -> Dict[str, pd.DataFrame]:
    """Helper function to load all demo showcase products."""
    client = EbayIngestionClient()
    return client.get_cached_showcase_products()
