# Methodology: Explainable Deceptive Pricing Detection

## 1. Problem Formulation & The Deceptive Discount Phenomenon

In modern e-commerce marketplaces, deceptive pricing schemes frequently exploit consumer cognitive biases through **manufactured reference pricing** (often referred to as "fake discounts" or "was/now manipulation").

```
Price ($)
  |                  [Artificial Peak]
  |                     /‾‾‾‾‾‾\
  |                    /        \   <- Advertised "40% Sale!"
  |                   /          \__________
  |  ────────────────/                      \───────── [True 30-Day Median Baseline]
  |____________________________________________________
                                    Drop Day (t)
```

A merchant artificially elevates a product's price for a brief period (e.g., 1–4 days) or gradually climbs it upwards ("creeping inflation") prior to a promotional event. On sale day $t$, the price is dropped back to or near the normal historical baseline. Although marketed as a deep discount (e.g., "30%–50% OFF!"), the consumer receives **zero genuine savings** relative to the product's historical price.

### Objective
Given a discrete time series of price observations $P = \{(t_i, p_i)\}_{i=1}^N$ per product $k$, determine for each promotional drop day $t$:
1. **Classification:** Whether the discount is **Manufactured / Deceptive** or **Genuine**.
2. **Quantification:** A calibrated fraud risk score $S_t \in [0.0, 1.0]$.
3. **Attribution:** Plain-English, legally defensible explanations grounded in SHAP (SHapley Additive exPlanations).

---

## 2. Causal Feature Engineering Architecture

Standard anomaly detection features fail on the fraudulent sale drop day $t$ because the price returns near normal baseline values ($p_t \approx \text{baseline}$), yielding near-zero z-scores and normal moving averages.

Our system solves this via **Sale-Day Risk Transfer**: projecting pre-sale inflation window statistics directly onto drop day $t$ using strictly causal (past-looking) temporal aggregations:

### 1. Robust Baseline & Rolling Extremes
* **30-Day Median Baseline ($\tilde{M}_{30, t}$):**
  $$\tilde{M}_{30, t} = \text{median}(p_{t-29}, \dots, p_t)$$
  Provides a breakdown-resistant baseline that prevents short pre-sale price spikes from contaminating the reference level.
* **14-Day Preceding Peak Price ($P^{\max}_{14, t}$):**
  $$P^{\max}_{14, t} = \max(p_{t-14}, \dots, p_{t-1})$$
  Excludes today's price $p_t$ to strictly preserve causality.

### 2. Pre-Sale Inflation & Illusion Indices
* **Recent Peak Inflation Ratio ($R_{\text{infl}, t}$):**
  $$R_{\text{infl}, t} = \frac{P^{\max}_{14, t} - \tilde{M}_{30, t}}{\tilde{M}_{30, t} + \epsilon}$$
  Measures the degree to which the recent 14-day peak exceeded the true 30-day baseline.

* **Drop from Recent Peak ($D_{\text{peak}, t}$):**
  $$D_{\text{peak}, t} = \frac{P^{\max}_{14, t} - p_t}{P^{\max}_{14, t} + \epsilon}$$
  Captures the advertised discount depth relative to the manufactured reference peak.

* **Fake Discount Illusion Index ($I_{\text{illusion}, t}$):**
  $$I_{\text{illusion}, t} = \max(0, R_{\text{infl}, t}) \times \max(0, D_{\text{peak}, t})$$
  Evaluates to high values **only** when an advertised price drop directly offsets an artificial pre-sale price surge.

* **Deceptive Discount Signal ($S_{\text{deceptive}, t}$):**
  $$S_{\text{deceptive}, t} = I_{\text{illusion}, t} \times \text{clip}\left(\frac{p_{t-1} - p_t}{0.06 \cdot p_{t-1}}, 0.0, 1.0\right)$$
  Sharpens signal activation specifically onto promotional drop days, suppressing post-sale stationary noise.

### 3. Creeping Inflation Slope ($C_{\text{slope}, t}$)
Detects multi-day stealth price increases (+3% to +5%/day over 7–10 days) preceding a drop:
$$C_{\text{slope}, t} = \left( \sum_{k=1}^7 \max\left(0, \frac{p_{t-k} - p_{t-k-1}}{p_{t-k-1}}\right) \right) \times \max(0, D_{\text{peak}, t})$$

### 4. Advertised vs. Real Discount Gap ($G_{\text{gap}, t}$)
When explicit advertised discounts are provided:
$$G_{\text{gap}, t} = \max\left(0, \frac{\text{Discount\%}}{100} - \frac{\tilde{M}_{30, t} - p_t}{\tilde{M}_{30, t} + \epsilon}\right)$$
Quantifies the divergence between advertised savings and actual savings against historical market baseline.

---

## 3. Unsupervised Model & Calibration (`DiscountIsolationForest`)

### Decoupled Soft Domain Gating
Traditional heuristic-coupled systems impose rigid rule cutoffs on ML predictions. Our `DiscountIsolationForest` avoids this coupling. The tree ensemble splits natively on continuous feature interactions:
* $X_t = [S_{\text{deceptive}, t}, R_{\text{infl}, t}, C_{\text{slope}, t}, G_{\text{gap}, t}]$

A minimal physical domain gate verifies only that day $t$ represents an actual discount drop below the preceding peak:
$$\text{IsDiscountDrop}_t = (p_{t-1} - p_t \ge 0.08 \cdot p_{t-1}) \land (p_t < P^{\max}_{14, t})$$

### Quantile-Anchored Sigmoid Calibration
Raw Isolation Forest decision scores $\phi(x) = - \text{score\_samples}(x)$ are mapped to human-interpretable $[0.0, 1.0]$ probabilities via distribution-anchored sigmoid calibration:
$$\mu_{\text{norm}} = \text{percentile}(\phi, 50), \quad \sigma_{\text{spread}} = \max(\text{percentile}(\phi, 95) - \mu_{\text{norm}}, 10^{-4})$$
$$\theta_{\text{thresh}} = \mu_{\text{norm}} + 0.95 \cdot \sigma_{\text{spread}}$$
$$P(\text{Deceptive} \mid X_t) = \frac{1}{1 + \exp\left(-30 \cdot (\phi_{\text{gated}}(X_t) - \theta_{\text{thresh}})\right)}$$

---

## 4. Benchmark Validation & Detection Progression

We evaluated naive rules, engineered rules, and our decoupled Isolation Forest model on a diverse 3,000-observation benchmark (50 products, 60 days) featuring four injected manipulation archetypes alongside genuine discount controls:

### Archetype Coverage
1. **Type 1: Classic 1-Day Spike & Drop** (+40% for 1 day $\rightarrow$ drop to baseline).
2. **Type 2: Multi-Day Plateau** (+35% for 4 days $\rightarrow$ drop to baseline).
3. **Type 3: Creeping Inflation** (+3.5%/day over 7 days $\rightarrow$ advertised 20% sale).
4. **Type 4: Subtle Low-Volatility Manipulation** (+14% inflation on steady goods $\rightarrow$ drop).
5. **Control: Genuine Deep Discount** (-20% to -25% discount with zero prior inflation).

### Benchmark Results

#### 1. Strict Day-$t$ Matching Table
| Detection Method | Precision | Recall | F1-Score | Accuracy | Confusion Matrix `[TN, FP] / [FN, TP]` | Archetypes Caught |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Rule-Based (Basic)** | **100.0%** | **25.0%** | **40.00%** | 99.0% | `[2960, 0] / [30, 10]` | Classic 1-day spike only |
| **Rule-Based (Advanced)** | **100.0%** | **77.5%** | **87.32%** | 99.7% | `[2960, 0] / [9, 31]` | Classic spikes, 4-day plateaus, creeping inflation |
| **Isolation Forest (ML)** | **93.02%** | **100.0%** | **96.39%** | **99.9%** | `[2957, 3] / [0, 40]` | **All 4 fraud patterns** (including subtle manipulation) |

#### 2. Event-Level Tolerance Window ($\pm 2$ Days)
| Detection Method | Event Precision | Event Recall | Event F1-Score |
| :--- | :---: | :---: | :---: |
| **Rule-Based (Basic)** | 100.0% | 25.0% | **40.00%** |
| **Rule-Based (Advanced)** | 100.0% | 77.5% | **87.32%** |
| **Isolation Forest (ML)** | 93.02% | 100.0% | **96.39%** |

> **Progression Insight:**
> **Basic Rules (40.0% F1)** capture only immediate 1-day jumps. **Advanced Rules (87.3% F1)** capture multi-day plateaus and creeping patterns, but miss subtle +14% manipulation due to rigid >18% cutoffs. **Isolation Forest ML (96.4% F1)** detects all manipulation types by capturing continuous multivariate anomalies relative to local product variance.

---

## 5. Production Dataset Evaluation (`amazon_sales_dataset.csv`)

Running the validated pipeline across **50,000 real-world sales transactions** (4,000 unique products) yielded:
* **Total Transactions:** 50,000
* **Unique Products:** 4,000
* **Basic Rule Alerts:** 2,545 events across 1,985 products
* **Advanced Rule Alerts:** 5,505 events across 2,985 products
* **Isolation Forest High-Risk Alerts ($S_t > 0.50$):** **1,892 events** across 922 products

---

## 6. Explainable AI (SHAP Translation Engine)

`DiscountExplainer` computes exact Shapley attributions using TreeExplainer on the Isolation Forest trees, decomposing the anomaly decision into human-readable evidence:

```
[Observation: Sony WH-1000XM5 on Promotional Drop Day]
Verdict:        🔴 LIKELY FAKE (99% Confidence, Peak Score: 0.9672)
Risk Level:     HIGH

SHAP Attributions:
• Deceptive Discount Signal:     +0.6289
• Recent Peak Inflation Ratio:   +0.2788
• Creeping Inflation Slope:      -0.0279
• Discount Exaggeration Gap:     -0.0643

Plain-English Consumer Reasons:
1. "Price was artificially inflated by +40.0% above historical baseline in the preceding 14 days."
2. "Manufactured sale pattern detected: advertised 31.0% discount merely cancels out pre-sale inflation."
3. "Current 'sale price' ($339.99) is essentially equal to the 30-day normal baseline ($349.99) — zero real savings."
```
