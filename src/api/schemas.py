from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PriceObservation(BaseModel):
    product_id: int = Field(..., description="Unique product identifier")
    order_date: str = Field(..., description="Order date in YYYY-MM-DD or datetime format")
    price: float = Field(..., gt=0, description="Product price (strictly positive)")

class ClassifyRequest(BaseModel):
    observations: List[PriceObservation] = Field(..., min_length=4, description="List of at least 4 price observations")

class ExplanationResponse(BaseModel):
    verdict: str
    confidence_pct: int
    risk_level: str
    risk_score: float
    reasons: List[str]
    feature_attributions: Dict[str, float]

class ProductClassifyResponse(BaseModel):
    product_id: int
    explanations: List[ExplanationResponse]
    summary: Dict[str, Any]

class BenchmarkResponse(BaseModel):
    metrics_table: List[Dict[str, Any]]
    dataset_summary: Dict[str, Any]
    window_metrics_table: Optional[List[Dict[str, Any]]] = None
