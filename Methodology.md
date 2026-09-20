# Detailed Methodology: Pricing Manipulation Detection

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Detection Approaches](#detection-approaches)
3. [Feature Engineering](#feature-engineering)
4. [Threshold Selection](#threshold-selection)
5. [Scoring System](#scoring-system)
6. [Limitations & Edge Cases](#limitations--edge-cases)

---

## Problem Definition

### What is Pricing Manipulation?

Pricing manipulation in e-commerce typically manifests as:

1. **Fake Discounts (Spike+Drop)**
   - Artificially inflate price
   - Quickly reduce to "original" level
   - Customer perceives large discount that doesn't exist
   - Example: $50 → $65 → $40 (appears as -38% off, but true manipulation)

2. **Artificial Scarcity**
   - Rapid price fluctuations
   - Creates urgency without real supply changes

3. **Seasonal Manipulation**
   - Systematic pattern during peak seasons
   - Detected through aggregation (multiple events)

### Why Detect It?

- **Consumer Protection**: Prevent deceptive pricing
- **Platform Trust**: Maintain marketplace credibility
- **Regulatory Compliance**: Meet FTC/consumer protection laws
- **Competitive Fairness**: Level playing field for honest sellers

---

## Detection Approaches

### Approach 1: Basic Pattern Matching

#### Concept
Detect the classic spike+drop pattern in consecutive prices.

#### Algorithm
```
For each product (chronologically sorted):
  For each middle price point i (where 1 ≤ i < n-1):
    
    prev_price = price[i-1]
    curr_price = price[i]
    next_price = price[i+1]
    
    spike_ratio = curr_price / prev_price
    drop_ratio = next_price / curr_price
    
    IF spike_ratio > 1.25 AND drop_ratio < 0.75:
      flag as suspicious
```

#### Why These Thresholds?

- **1.25 (25% spike)**: 
  - Balances sensitivity vs specificity
  - Distinguishes real spikes from natural variation (<10% typical)
  - Too high (>1.5): Misses moderate manipulation
  - Too low (<1.2): Too many false positives

- **0.75 (25% drop)**:
  - Symmetric with spike threshold
  - Indicates price returned below spike point (manipulation signature)
  - Less than 25% drop: Could be normal correction

#### Strengths ✅
- **Interpretable**: Simple rule, easy to explain
- **Fast**: Single pass through data
- **No Training**: No ML model needed
- **Robust to Outliers**: Only looks at 3-point windows

#### Weaknesses ❌
- **Noise Sensitive**: Natural variation can trigger false alarms
- **Context Blind**: Doesn't know baseline behavior
- **Rigid**: Fixed thresholds don't adapt to product volatility
- **Limited Coverage**: Only catches spike+drop; misses other patterns

---

### Approach 2: Statistical Anomaly Detection

#### Concept
Use product baseline behavior + statistical outlier detection.

#### Features Engineered

**1. Rolling Mean (5-period)**
```
rolling_mean[i] = mean(price[i-4:i+1])
```
- Captures short-term baseline
- Smooths day-to-day noise
- 5-period chosen as balance between responsiveness and smoothing

**2. Smoothed Price (3-period)**
```
smoothed[i] = mean(price[i-2:i+1])
```
- Reduces high-frequency noise
- Simpler moving average
- 3-period: Minimal lag while filtering noise

**3. Z-Score (Statistical Deviation)**
```
z_score[i] = (price[i] - mean(product_prices)) / std(product_prices)
```
- Measures how far price deviates from product average
- Unit: standard deviations (σ)
- |z| > 2: Unusual (95% of normal data falls within ±2σ)
- |z| > 3: Very unusual (99.7% within ±3σ)

#### Detection Logic

```
For each product (chronologically sorted):
  For each price point i:
    
    baseline = rolling_mean[i]
    smoothed = smoothed_price[i]
    deviation = z_score[i]
    
    # Pattern-based detection
    spike = smoothed > baseline * 1.3
    drop = price[i+1] < price[i] * 0.75
    pattern_detected = spike AND drop
    
    # Statistical detection
    statistical_anomaly = abs(deviation) > 2
    
    IF pattern_detected OR statistical_anomaly:
      flag as suspicious
      score = base_score + 0.2 * (1 if statistical_anomaly else 0)
```

#### Why These Thresholds?

- **1.3 (30% above baseline)**:
  - More aggressive than basic (1.25)
  - 30% deviation unusual for same product
  - Indicates manipulation vs normal variation

- **z_score > 2**:
  - Statistical standard (95% confidence)
  - Customizable but 2 is industry default
  - Catches true outliers, not everyday variation

#### Strengths ✅
- **Adaptive**: Baseline adjusts per product
- **Statistically Grounded**: Uses real deviation metrics
- **Detects Isolated Anomalies**: Even without drop pattern
- **Reduces False Positives**: Baseline comparison is contextual

#### Weaknesses ❌
- **Requires History**: Needs multiple prices per product
- **Complex**: Harder to explain than basic method
- **Parameter Sensitivity**: Changes to thresholds affect results significantly

---

## Feature Engineering

### Data Validation Pipeline

```
Raw Data
  ↓
1. Check required columns exist
2. Remove rows with missing values
3. Convert dates to datetime
4. Remove negative/zero prices
5. Group by product_id
  ↓
Validated Data
```

**Safe Z-Score Calculation**
```python
def safe_zscore(x):
    """Handles constant price (std=0) gracefully"""
    std = x.std()
    if std > 0:
        return (x - x.mean()) / std
    else:
        return pd.Series(0, index=x.index)  # No variation = no anomaly
```

Why needed?
- If product has identical prices for all dates: std = 0
- Without safeguard: Division by zero → NaN → errors
- With safeguard: Returns 0 (makes sense: no deviation)

---

## Threshold Selection

### Why These Specific Values?

#### Spike Threshold: 1.25 (Basic) vs 1.3 (Advanced)

| Threshold | Detection Rate | False Positives | Use Case |
|-----------|----------------|-----------------|----------|
| 1.10      | Very High      | Excessive       | Too sensitive |
| 1.15      | High           | High            | Catch subtle |
| **1.25**  | **Balanced**   | **Moderate**    | **Basic method** |
| 1.30      | Balanced       | Low             | Advanced (with baseline) |
| 1.50      | Low            | Very Low        | Conservative |
| 2.00+     | Very Low       | Minimal         | Only extreme spikes |

#### Drop Threshold: 0.75 (25% below spike)

- Symmetric to spike: If up 25%, down 25% completes the manipulation
- 0.80 (20% drop): Too lenient, catches normal corrections
- 0.70 (30% drop): Too strict, misses subtle manipulation
- **0.75**: Goldilocks point

#### Z-Score Threshold: 2.0

- Statistical standard: 2σ = 95% confidence interval
- 1.5: Too permissive (catches 87% of data)
- 2.0: Industry standard (catches 95% as normal)
- 3.0: Too conservative (misses real anomalies)
- **2.0**: Best practice

---

## Scoring System

### Event-Level Score

Individual suspicious event gets a score based on magnitude:

```
increase_pct = min((current - previous) / previous, 1.0)
drop_pct = min((current - next) / current, 1.0)

event_score = (increase_pct + drop_pct) / 2
```

**Example:**
- Price: $50 → $70 → $45
- increase_pct = 20/50 = 0.40
- drop_pct = 25/70 = 0.36
- event_score = (0.40 + 0.36) / 2 = **0.38**

**Interpretation:**
- 0.0-0.3: Mild manipulation
- 0.3-0.6: Moderate manipulation
- 0.6-1.0: Severe manipulation

### Product-Level Aggregation

Multiple events per product → single risk score:

```
events_for_product = [0.35, 0.42, 0.38]

avg_score = mean(events) = 0.38

num_events = 3
event_weight = 1 + log(3) / 3 = 1.366

final_score = avg_score * event_weight = 0.52
final_score = min(0.52, 1.0) = 0.52
```

**Why log weighting?**
- Rewards **consistent** manipulation (multiple events)
- Isolated anomalies stay moderate
- Doesn't explode with many events

### Risk Categorization

```
Score Range | Category    | Action
------------|-------------|----------
0.0 - 0.5   | 🟢 Low      | Monitor
0.5 - 0.8   | 🟠 Moderate | Review
0.8 - 1.0   | 🔴 High     | Investigate
```

---

## Limitations & Edge Cases

### 1. Constant Prices
```
Product with price [50, 50, 50, 50]
z_score = undefined (std = 0)
```
**Solution:** Return 0 (no variation = no anomaly)

### 2. Insufficient Data
```
Product with only 2 price points
```
**Solution:** Skip (need at least 3 for spike+drop pattern)

### 3. Legitimate Seasonal Spikes
```
Example: Winter heating oil prices
- Natural 20-30% spikes seasonally
- Algorithm may flag as manipulation
```
**Mitigation:** Review business context; consider seasonal adjustment

### 4. Data Quality Issues
```
Example: Erroneous price entry ($5000 instead of $50)
```
**Mitigation:** Outliers may be data errors; always validate

### 5. Cross-Product Contamination (FIXED)
```
Product A: [10, 11, 12]
Product B: [13, 14, 15]
```
**Fixed:** Group by product_id first, compare within groups only

---

**Last Updated:** 2026-04-07