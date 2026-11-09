# Discount Legitimacy Classifier — Explainable AI for E-Commerce Pricing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/pytest-12%20passed-brightgreen.svg)](https://docs.pytest.org)
[![F1-Score](https://img.shields.io/badge/F1--Score-96.4%25-success.svg)](./Methodology.md)

An end-to-end, unsupervised machine learning and Explainable AI (XAI) system that detects **manufactured e-commerce discounts** (prices inflated prior to a sale) and translates technical anomaly scores into plain-English consumer reasons.

---

## 🔍 The Market Gap

| Existing Platform | Capability | Critical Limitation |
| :--- | :--- | :--- |
| **CamelCamelCamel** | Displays historical Amazon price charts | No fraud classification, no explanation |
| **Honey / Capital One** | Finds active discount codes & price drops | Blind to pre-sale artificial inflation |
| **Keepa** | Tracks historical price fluctuations | Requires manual human chart interpretation |
| **Discount Legitimacy Classifier** | **Unsupervised ML Detection + SHAP Plain-English XAI** | **Classifies fraud & explains *why* in one click** |

---

## 🏆 Benchmark Validation & Scientific Progression

We formally evaluate our system using a **Synthetic Fraud Injection Engine** (50 products, 60 days, 3,000 observations) featuring four distinct manipulation archetypes alongside genuine discount controls:

### 1. Strict Day-$t$ Matching Performance
| Detection Method | Precision | Recall | F1-Score | Accuracy | Confusion Matrix `[TN, FP] / [FN, TP]` | Patterns Caught |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Rule-Based (Basic)** | **100.0%** | **25.0%** | **40.00%** | 99.0% | `[2960, 0] / [30, 10]` | Classic 1-day spike & drop only |
| **Rule-Based (Advanced)** | **100.0%** | **77.5%** | **87.32%** | 99.7% | `[2960, 0] / [9, 31]` | Classic spikes, 4-day plateaus, creeping inflation |
| **Isolation Forest (ML)** | **93.02%** | **100.0%** | **96.39%** | **99.9%** | `[2957, 3] / [0, 40]` | **All 4 archetypes** (including subtle low-variance manipulation) |

### 2. Window-Based Event Matching ($\pm 2$ Days)
| Detection Method | Event Precision | Event Recall | Event F1-Score |
| :--- | :---: | :---: | :---: |
| **Rule-Based (Basic)** | 100.0% | 25.0% | **40.00%** |
| **Rule-Based (Advanced)** | 100.0% | 77.5% | **87.32%** |
| **Isolation Forest (ML)** | 93.02% | 100.0% | **96.39%** |

> **Key Finding:** Rigorous metric progression: **Basic Rules (40.0%) $\rightarrow$ Advanced Rules (87.3%) $\rightarrow$ Isolation Forest (96.4%)**.
> Isolation Forest detects subtle +14% pre-sale markups on low-volatility goods that rigid >18% heuristic cutoffs inherently miss.

---

## 💡 Explainable AI Output (SHAP Translation)

Rather than outputting an opaque float score, the system pairs TreeExplainer SHAP feature attributions with a domain translation engine:

```
Verdict: 🔴 LIKELY FAKE (99% Confidence | Peak Fraud Risk: 0.9672)

Plain-English Reasons:
• Price was artificially inflated by +40.0% above historical baseline in the preceding 14 days.
• Manufactured sale pattern detected: advertised 31.0% discount merely cancels out pre-sale inflation.
• Current "sale price" ($339.99) is essentially equal to the 30-day normal baseline ($349.99) — zero genuine consumer savings.

SHAP Feature Attributions:
• Deceptive Discount Signal:     +0.6289
• Recent Peak Inflation Ratio:   +0.2788
• Creeping Inflation Slope:      -0.0279
• Discount Exaggeration Gap:     -0.0643
```

---

## 🏛️ System Architecture

```
Time Series Data (Amazon Kaggle / Curated Showcases / Live eBay API)
                              ↓
              src/pipeline/preprocessing.py (Validation & Sorting)
                              ↓
              src/pipeline/features.py (Causal 14-Day Risk Transfer)
                              ↓
      ┌──────────────────────────────────────────────────┐
      │               Detection Layer                    │
      │   ├── Rule-Based Heuristics (Basic & Advanced)   │
      │   └── DiscountIsolationForest (Unsupervised ML)  │
      └──────────────────────────────────────────────────┘
                              ↓
              src/models/explainer.py (SHAP Explainability)
                              ↓
      ┌──────────────────────────────────────────────────┐
      │              Presentation Layer                  │
      │   ├── Streamlit UI (Consumer Deal Verifier)      │
      │   └── FastAPI REST Endpoints (/classify)         │
      └──────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```text
E-commerce_Project/
├── data/
│   ├── amazon_sales_dataset.csv       # Production dataset (50,000 records, 4,000 products)
│   └── ebay_cache/                    # Offline demo showcase cache (Sony, Samsung, Dyson, etc.)
├── src/
│   ├── api/                           # FastAPI service & Pydantic request/response schemas
│   │   ├── main.py
│   │   └── schemas.py
│   ├── models/                        # ML Anomaly Detection, SHAP XAI, Evaluation
│   │   ├── isolation_forest.py
│   │   ├── explainer.py
│   │   └── evaluator.py
│   ├── pipeline/                      # Causal feature engineering, preprocessing, ingestion
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   ├── rules.py
│   │   ├── pipeline.py
│   │   └── ingestion.py
│   └── ui/                            # Streamlit dashboard & Plotly interactive visualizers
│       ├── app.py
│       └── visualization.py
├── tests/                             # Pytest test suite (12 unit & integration tests)
├── Methodology.md                     # Full mathematical formulations & benchmark documentation
├── project_aim.md                     # Vision, gap in market, and product design
└── requirements.txt
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
*Instant Demo:* By default, the dashboard launches directly with **⭐ Curated Demo Showcases** enabled, letting you test real archetypes (Sony 1-day spike, Samsung 4-day plateau, Dyson creeping inflation, Bose genuine discount) in 1 click without waiting.

### 3. Launch the FastAPI Backend Service
```bash
uvicorn src.api.main:app --reload --port 8000
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Benchmark Evaluation: `GET /api/v1/benchmark`
* Observation Classification: `POST /api/v1/classify`

### 4. Run the Test Suite
```bash
python -m pytest tests -v
```
All 12 unit tests verify data validation, causal feature calculation, rule heuristics, model persistence, and SHAP explanations.

---

## 📊 Real-World Dataset Performance

Evaluated against the full **50,000 transactions** across **4,000 unique products** (`amazon_sales_dataset.csv`):
* **Rule-Based (Basic):** Flagged 2,545 events across 1,985 products.
* **Rule-Based (Advanced):** Flagged 5,505 events across 2,985 products.
* **Isolation Forest (ML):** Isolated **1,892 high-risk manufactured discount events** across 922 products with zero hard rule dependencies.

---

## 📜 Methodology & Mathematical Details

For complete mathematical definitions of $I_{\text{illusion}}$, $S_{\text{deceptive}}$, creeping inflation rolling sums, and quantile-anchored sigmoid calibration, consult [Methodology.md](./Methodology.md).
