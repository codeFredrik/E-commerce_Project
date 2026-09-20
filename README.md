# E-commerce Pricing Manipulation Detection

An explainable machine-learning system for identifying suspicious e-commerce
pricing patterns, including artificial price inflation followed by a sharp
discount.

## What it does

- Compares interpretable spike-and-drop rules with an Isolation Forest model.
- Builds rolling-price, volatility, and pre-sale inflation features.
- Produces plain-language explanations using SHAP.
- Provides both a Streamlit dashboard and a FastAPI service.
- Includes a synthetic benchmark for precision, recall, F1, and accuracy.

## Project layout

```text
E-commerce_Project/
├── data/
│   └── amazon_sales_dataset.csv   # Local input data; not committed
├── src/
│   ├── api/                       # FastAPI application and schemas
│   ├── models/                    # Isolation Forest, SHAP, evaluation
│   ├── pipeline/                  # Cleaning, features, and detection rules
│   └── ui/                        # Streamlit dashboard and charts
├── tests/
├── app.py                         # Root compatibility wrapper
├── Methodology.md
├── project_aim.md
└── requirements.txt
```

## Requirements and installation

Python 3.8+ is recommended. Create a virtual environment and install the
dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Data

Place `amazon_sales_dataset.csv` in `data/`. The dataset must contain:

- `product_id`: product identifier
- `order_date`: transaction date
- `price`: positive numeric price

The dataset is intentionally excluded from Git by `.gitignore`.

## Run the applications

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Start the API:

```bash
uvicorn src.api.main:app --reload
```

The API exposes `GET /` for health, `POST /api/v1/classify` for classification,
and `GET /api/v1/benchmark` for the synthetic evaluation.

## Run tests

```bash
python -m pytest -q tests
```

## Interpretation and limitations

Risk scores are indicators for investigation, not proof of seller misconduct.
Legitimate seasonal promotions, supply disruptions, and data-quality issues can
produce false positives. Thresholds should be tuned for the target marketplace
and validated against domain context.

See `Methodology.md` for the detection logic and scoring details.
