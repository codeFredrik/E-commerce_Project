import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

from src.api.schemas import ClassifyRequest, ProductClassifyResponse, BenchmarkResponse, ExplanationResponse
from src.pipeline.preprocessing import preprocess_df
from src.pipeline.features import create_features
from src.models.isolation_forest import DiscountIsolationForest
from src.models.explainer import DiscountExplainer
from src.models.evaluator import SyntheticFraudBenchmark

app = FastAPI(
    title="Discount Legitimacy Classifier API",
    description="REST API for unsupervised e-commerce price manipulation detection and SHAP plain-English explainability.",
    version="1.0.0"
)

# In-memory cached model for API serving
_global_model: DiscountIsolationForest = DiscountIsolationForest(contamination=0.05, random_state=42)
_global_explainer: DiscountExplainer = None

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "Discount Legitimacy Classifier API"}

@app.post("/api/v1/classify", response_model=List[ProductClassifyResponse], tags=["Classification"])
def classify_discounts(request: ClassifyRequest):
    """
    Classifies a time series of price observations per product, returning fraud verdicts and plain-English reasons.
    """
    try:
        data = [obs.model_dump() for obs in request.observations]
        df_raw = pd.DataFrame(data)
        df = preprocess_df(df_raw)
        df = create_features(df)
        
        # Fit model on incoming data or use existing if fitted
        model = DiscountIsolationForest(contamination=0.05, random_state=42)
        model.fit(df)
        scores = model.score(df)
        
        explainer = DiscountExplainer(model)
        explanations = explainer.explain_dataframe(df, scores)
        
        # Group by product
        df['explanations'] = explanations
        results = []
        for p_id, group in df.groupby('product_id'):
            p_exps = list(group['explanations'])
            p_scores = scores[group.index]
            
            avg_score = round(float(np.mean(p_scores)), 4)
            max_score = round(float(np.max(p_scores)), 4)
            num_events = int(np.sum(p_scores > 0.5))
            
            summary = {
                "avg_risk_score": avg_score,
                "max_risk_score": max_score,
                "suspicious_events_count": num_events,
                "overall_verdict": "🔴 LIKELY FAKE" if max_score > 0.75 else ("🟠 SUSPICIOUS" if max_score > 0.50 else "🟢 GENUINE")
            }
            results.append(ProductClassifyResponse(
                product_id=int(p_id),
                explanations=p_exps,
                summary=summary
            ))
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/benchmark", response_model=BenchmarkResponse, tags=["Evaluation"])
def run_benchmark():
    """
    Runs the synthetic fraud benchmark to evaluate and compare Rule-Based vs. ML detection precision and recall.
    """
    try:
        bench = SyntheticFraudBenchmark(num_products=50, days_per_product=60, random_state=42)
        res = bench.run_benchmark()
        return BenchmarkResponse(
            metrics_table=res['metrics_table'],
            dataset_summary=res['dataset_summary']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
