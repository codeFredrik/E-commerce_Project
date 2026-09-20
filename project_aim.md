# Discount Legitimacy Classifier — Project Vision

## What This Is

An ML system that classifies whether a product discount is genuine or manufactured,
and explains *why* in plain language.

Not a price tracker. Not a deal finder. Those exist.

This answers the question no existing tool answers:

> "This 40% off — is it real, or was the price inflated to create it?"

---

## The Gap in the Market

| Tool | What it does | What it doesn't do |
|---|---|---|
| CamelCamelCamel | Shows price history chart | No verdict, no explanation |
| Honey | Finds coupons, alerts on drops | No manipulation detection |
| Keepa | Tracks Amazon price over time | You interpret the chart yourself |
| PriceSpy / Idealo | Cross-platform price comparison | No fraud signal |

**None of them classify. None of them explain.**

A consumer sees a chart and still has to decide. This system decides for them and shows its reasoning.

---

## Core Output

Input: A product with a current "discount" and its price history

Output:
```
Verdict: LIKELY FAKE (confidence: 84%)

Why:
- Price was inflated 5 days before sale (+38% above 60-day baseline)
- Seller has flagged 6 similar patterns in the last 90 days
- Current "sale price" is within 3% of the 60-day average — no real saving
- Spike magnitude: 2.4σ above this product's normal variation
```

---

## What Makes This a DS Project

- **Unsupervised ML:** Isolation Forest for anomaly detection (no labels needed)
- **Explainability:** SHAP values mapped to human-readable reasons
- **Baseline comparison:** Rule-based detection vs IF, with precision/recall on synthetic benchmark
- **Feature engineering:** Lag features, volatility metrics, pre-event inflation windows
- **Evaluation story:** Inject known fake discounts into clean data, measure detection rate

This is a complete DS arc: problem → data → features → model → evaluation → deployment.

---

## Data Strategy

### Training & Validation
- Existing Kaggle `amazon_sales_dataset.csv`
- Synthetic benchmark: inject known manipulation patterns, measure detection

### Live Demo Data
- **eBay Open API** — legitimate, free, no scraping, returns real listing + sold price history
- 5–10 real product demos to show the system works on live data
- No ToS issues, no IP bans, no dead ends

### Why Not Amazon Direct
- ToS prohibits scraping
- Keepa free tier is token-limited
- eBay API is cleaner for a portfolio build

---

## ML Stack

| Component | Tool | Purpose |
|---|---|---|
| Anomaly detection | Isolation Forest (sklearn) | Core model, unsupervised |
| Explainability | SHAP | Maps model output to plain-English reasons |
| Baseline | Rule-based (existing) | Comparison benchmark |
| Evaluation | Custom synthetic benchmark | Precision / recall story |
| Features | pandas, numpy | Lag, volatility, baseline deviation |

---

## Architecture

```
Price History (Kaggle / eBay API)
        ↓
Preprocessing + Feature Engineering
        ↓
┌─────────────────────────────┐
│  Detection Layer            │
│  ├── Rule-Based (baseline)  │
│  └── Isolation Forest (ML)  │
└─────────────────────────────┘
        ↓
SHAP Explainer → Plain-English Reasons
        ↓
Legitimacy Verdict + Confidence Score
        ↓
FastAPI endpoint
        ↓
Streamlit UI (consumer-facing)
        ↓
Hugging Face Spaces (deployed, public URL)
```

---

## Project Structure

```
discount-legitimacy-classifier/
├── data/
│   ├── amazon_sales_dataset.csv
│   └── ebay_cache/
├── src/
│   ├── pipeline/
│   │   ├── ingestion.py         # eBay API fetch + parse
│   │   ├── preprocessing.py     # Clean, validate, sort
│   │   └── features.py          # Feature engineering
│   ├── models/
│   │   ├── isolation_forest.py  # Train, score, persist
│   │   ├── explainer.py         # SHAP → plain English
│   │   └── evaluator.py         # Synthetic benchmark
│   ├── api/
│   │   └── main.py              # FastAPI
│   └── ui/
│       └── app.py               # Streamlit
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── exploration.ipynb
├── VISION.md
├── METHODOLOGY.md
├── README.md
└── requirements.txt
```

---

## Execution Phases

### Phase 0 — Refactor
Restructure existing code into `src/pipeline/`. No logic changes. All tests pass.

### Phase 1 — Feature Engineering
Expand features: lag prices, volatility, pre-event inflation window, baseline deviation.
Run on Kaggle data. Validate pipeline end-to-end.

### Phase 2 — ML Model
Train Isolation Forest. Compare with rule-based baseline.
Build synthetic benchmark. Generate precision/recall table.

### Phase 3 — SHAP Explainability
Map IF anomaly scores + feature contributions to plain-English verdict reasons.
This is the novel part.

### Phase 4 — eBay Integration
Fetch real product price history via eBay API.
Run 5–10 live product demos. Cache results.

### Phase 5 — API + UI + Deploy
FastAPI endpoint. Streamlit consumer UI. Deploy to HF Spaces.

---

## What This Is Not

- Not a price tracker
- Not a scraper
- Not a real-time alert system
- Not trained on labeled data (unsupervised by design)

---

## The One-Line Pitch

> An ML system that classifies fake discounts and explains why — built on real pricing data,
> powered by Isolation Forest + SHAP, deployed as a consumer tool.

---

*Locked: 2026-07-24*
