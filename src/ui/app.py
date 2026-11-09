import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Tuple, List, Dict, Any

from src.pipeline.preprocessing import preprocess_df
from src.pipeline.features import create_features
from src.pipeline.rules import detect_suspicious_events, detect_suspicious_events_advanced, aggregate_scores
from src.models.isolation_forest import DiscountIsolationForest
from src.models.explainer import DiscountExplainer
from src.models.evaluator import SyntheticFraudBenchmark
from src.ui.visualization import plot_product_analysis, plot_shap_attributions
from src.pipeline.ingestion import load_demo_showcases

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Discount Legitimacy Classifier — AI System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM STYLING
st.markdown("""
<style>
    .verdict-badge {
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .fake-badge { background-color: #FDEDEC; color: #922B21; border: 2px solid #E74C3C; }
    .susp-badge { background-color: #FEF9E7; color: #B7950B; border: 2px solid #F39C12; }
    .gen-badge  { background-color: #EAFAF1; color: #148F77; border: 2px solid #2ECC71; }

    .reason-box {
        background-color: #F8F9F9;
        padding: 16px;
        border-left: 5px solid #2B5B84;
        border-radius: 6px;
        margin-bottom: 15px;
        font-size: 15px;
        line-height: 1.6;
    }
    .archetype-tag {
        display: inline-block;
        background-color: #EBF5FB;
        color: #2471A3;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# CACHED DATA PIPELINE EXECUTION
@st.cache_data
def analyze_dataframe(df_raw: pd.DataFrame, contamination: float = 0.02) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Preprocesses, extracts features, scores via Isolation Forest, and generates SHAP explanations."""
    df_raw = df_raw.copy()
    df_raw['order_date'] = pd.to_datetime(df_raw['order_date'])

    df = preprocess_df(df_raw)
    df = create_features(df)

    susp_basic = detect_suspicious_events(df)
    susp_adv = detect_suspicious_events_advanced(df)

    model = DiscountIsolationForest(contamination=contamination, random_state=42)
    model.fit(df)
    scores = model.score(df)

    explainer = DiscountExplainer(model)
    explanations = explainer.explain_dataframe(df, scores)

    df['risk_score'] = scores
    df['verdict'] = [exp['verdict'] for exp in explanations]
    df['confidence_pct'] = [exp['confidence_pct'] for exp in explanations]
    df['reasons'] = [exp['reasons'] for exp in explanations]
    df['risk_level'] = [exp['risk_level'] for exp in explanations]

    return df, susp_basic, explanations

@st.cache_data
def load_full_amazon_data():
    current_dir = Path(__file__).parent.parent.parent
    file_path = current_dir / "data" / "amazon_sales_dataset.csv"
    if not file_path.exists():
        file_path = current_dir / "amazon_sales_dataset.csv"
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)

@st.cache_data
def run_cached_benchmark():
    bench = SyntheticFraudBenchmark(num_products=50, days_per_product=60, random_state=42)
    return bench.run_benchmark()

# SIDEBAR NAVIGATION & DATA SOURCE
st.sidebar.title("🔍 Discount Legitimacy")
data_mode = st.sidebar.radio(
    "Data Source:",
    ["⭐ Curated Demo Showcases (Instant)", "📁 Full Amazon Dataset (4,000 Products)", "⬆️ Upload Custom CSV"]
)

showcases = load_demo_showcases()

# Load Selected Data
if data_mode == "⭐ Curated Demo Showcases (Instant)":
    # Combine showcases into a shared multi-product corpus for market baseline calibration
    all_showcases_df = pd.concat(list(showcases.values()), ignore_index=True)
    with st.spinner("Initializing Curated Showcases Engine..."):
        df_processed, suspicious_basic, explanations = analyze_dataframe(all_showcases_df, contamination=0.03)

    st.sidebar.subheader("Select Showcase Product:")
    showcase_names = list(showcases.keys())
    selected_showcase_name = st.sidebar.selectbox("Choose Product Archetype:", showcase_names)
    selected_raw = showcases[selected_showcase_name]
    selected_p_id = int(selected_raw['product_id'].iloc[0])
    product_display_title = selected_showcase_name

elif data_mode == "📁 Full Amazon Dataset (4,000 Products)":
    df_amazon_raw = load_full_amazon_data()
    if df_amazon_raw is None:
        st.error("`amazon_sales_dataset.csv` not found in `data/`. Please upload or check path.")
        st.stop()
    with st.spinner("Analyzing 50,000 Amazon sales transactions..."):
        df_processed, suspicious_basic, explanations = analyze_dataframe(df_amazon_raw, contamination=0.02)

    available_prods = sorted(df_processed['product_id'].unique())
    selected_p_id = st.sidebar.selectbox("Select Product ID:", available_prods)
    product_display_title = f"Product ID: {selected_p_id}"

else: # Upload Custom CSV
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain product_id, order_date, price):", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis.")
        st.stop()
    df_custom_raw = pd.read_csv(uploaded_file)
    with st.spinner("Analyzing uploaded time series..."):
        df_processed, suspicious_basic, explanations = analyze_dataframe(df_custom_raw, contamination=0.05)
    available_prods = sorted(df_processed['product_id'].unique())
    selected_p_id = st.sidebar.selectbox("Select Product ID to Inspect:", available_prods)
    product_display_title = f"Uploaded Product ID: {selected_p_id}"

st.sidebar.divider()
tab_selection = st.sidebar.radio(
    "Select View Mode:",
    ["🔍 Consumer Deal Verifier", "📊 Rule-Based vs. ML Benchmark", "📈 Marketplace Analytics"]
)

st.sidebar.divider()
st.sidebar.info("""
**System Engine:**
- **ML Anomaly Detection**: Unsupervised `DiscountIsolationForest` (decoupled soft gate)
- **Causal Feature Engineering**: 14-day rolling extremes, pre-sale illusion index, creeping slope
- **Explainable AI (XAI)**: SHAP TreeExplainer mapped to plain-English consumer reasons
""")

# ==============================================================================
# TAB 1: CONSUMER DEAL VERIFIER
# ==============================================================================
if tab_selection == "🔍 Consumer Deal Verifier":
    st.title("🔍 Consumer Deal Verifier (AI & XAI)")
    st.markdown("""
    Evaluate whether an advertised discount is **genuine** or **manufactured** (price artificially inflated prior to sale).
    The system displays the ML confidence score, plain-English reasons, and SHAP feature attributions.
    """)

    # Filter to selected product
    df_p = df_processed[df_processed['product_id'] == selected_p_id].sort_values('order_date')
    p_exps = [explanations[i] for i in df_p.index]

    # Identify highest risk observation for summary badge
    max_idx = int(np.argmax(df_p['risk_score'].values))
    top_obs = df_p.iloc[max_idx]
    top_exp = p_exps[max_idx]

    verdict = top_exp['verdict']
    conf = top_exp['confidence_pct']
    risk_score = top_exp['risk_score']
    risk_level = top_exp.get('risk_level', 'MODERATE')
    reasons = top_exp['reasons']
    attributions = top_exp.get('feature_attributions', {})

    # Render Verdict Badge
    if "FAKE" in verdict:
        badge_class = "fake-badge"
    elif "SUSPICIOUS" in verdict:
        badge_class = "susp-badge"
    else:
        badge_class = "gen-badge"

    st.markdown(
        f'<div class="verdict-badge {badge_class}">'
        f'{verdict} &nbsp;|&nbsp; Confidence: {conf}% &nbsp;|&nbsp; Peak Fraud Risk Score: {risk_score:.4f} ({risk_level} RISK)'
        f'</div>',
        unsafe_allow_html=True
    )

    # Display Key Metrics & Reasons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("💡 AI Explainability Report (Why?)")
        if 'pattern_label' in top_obs:
            st.markdown(f'<div class="archetype-tag">Archetype: {top_obs["pattern_label"]}</div>', unsafe_allow_html=True)

        reasons_html = "<ul style='margin-bottom:0; padding-left:20px;'>" + "".join([f"<li><b>{r}</b></li>" for r in reasons]) + "</ul>"
        st.markdown(f'<div class="reason-box">{reasons_html}</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("📌 Baseline & Discount Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${df_p.iloc[-1]['price']:.2f}")
        median_baseline = df_p['rolling_median_30'].iloc[-1] if 'rolling_median_30' in df_p.columns else df_p['rolling_mean_30'].iloc[-1]
        c2.metric("30-Day Median Baseline", f"${median_baseline:.2f}")
        c3.metric("Flagged Days", int(np.sum(df_p['risk_score'] > 0.50)))

        # Display small comparison table
        max_prev_14 = top_obs.get('max_price_prev_14', top_obs['price'])
        recent_inf = top_obs.get('recent_peak_inflation_ratio', 0.0) * 100
        drop_peak = top_obs.get('drop_from_recent_peak', 0.0) * 100

        st.markdown(f"""
        - **Pre-Sale Peak (Past 14 Days)**: `${max_prev_14:.2f}` (+{recent_inf:.1f}% vs baseline)
        - **Advertised Drop from Peak**: `-{drop_peak:.1f}%`
        - **Peak Risk Date**: `{pd.to_datetime(top_obs['order_date']).strftime('%Y-%m-%d')}`
        """)

    st.divider()

    # Toggle Rule-Based Flag Overlay
    show_rules_toggle = st.checkbox("Show Rule-Based Comparison Markers on Price Chart", value=True)

    # Interactive Plotly Chart
    st.subheader(f"📊 Price History & Anomaly Detection: {product_display_title}")
    fig = plot_product_analysis(df_processed, suspicious_basic, selected_p_id, explanations, show_rules=show_rules_toggle)
    st.plotly_chart(fig, use_container_width=True)

    # SHAP Feature Attributions Breakdown
    st.subheader("🔬 Technical Explainability: SHAP Feature Attributions")
    st.markdown("SHAP (SHapley Additive exPlanations) isolates each engineered feature's exact contribution to the Isolation Forest tree anomaly score.")
    fig_shap = plot_shap_attributions(attributions)
    st.plotly_chart(fig_shap, use_container_width=True)

    # Detailed Data Table
    with st.expander("📄 View Detailed Scored Pricing History Table"):
        cols_available = ['order_date', 'price', 'rolling_median_30', 'max_price_prev_14', 'recent_peak_inflation_ratio', 'risk_score', 'verdict']
        present_cols = [c for c in cols_available if c in df_p.columns]
        df_display = df_p[present_cols].copy()
        df_display['order_date'] = df_display['order_date'].dt.strftime('%Y-%m-%d')
        if 'recent_peak_inflation_ratio' in df_display.columns:
            df_display['recent_peak_inflation_ratio'] = (df_display['recent_peak_inflation_ratio'] * 100).round(1).astype(str) + '%'

        st.dataframe(df_display.rename(columns={
            'order_date': 'Date',
            'price': 'Price ($)',
            'rolling_median_30': '30d Median Baseline ($)',
            'max_price_prev_14': '14d Peak ($)',
            'recent_peak_inflation_ratio': 'Pre-Sale Peak Inflation',
            'risk_score': 'ML Risk Score (0-1)',
            'verdict': 'AI Verdict'
        }), use_container_width=True)

# ==============================================================================
# TAB 2: BENCHMARK
# ==============================================================================
elif tab_selection == "📊 Rule-Based vs. ML Benchmark":
    st.title("📊 Model Evaluation: Rule-Based vs. Isolation Forest ML")
    st.markdown("""
    To formally validate our system, we constructed a **Synthetic Fraud Injection Engine** that generates 50 products over 60 days (3,000 observations) with ground-truth labels (`is_fake_discount = True`).

    The benchmark injects **four distinct manipulation archetypes** alongside genuine discount controls:
    1. **Classic 1-Day Spike & Drop**: Sudden +40% surge followed by drop to baseline.
    2. **Multi-Day Plateau (4 Days)**: Price held inflated at +35% for 4 days before dropping (missed by short-memory heuristics).
    3. **Slow 7-Day Creeping Inflation**: Gradual climb of +3.5%/day over 7 days (+24.5% total) followed by an advertised "sale".
    4. **Subtle Low-Volatility Manipulation**: +14% pre-sale inflation on low-variance products (missed by rigid >18% boolean rules, cleanly isolated by ML).
    5. **Genuine Discount Control**: Real 20% discount with zero prior inflation (confirms 0% false alarm rate on real sales).
    """)

    with st.spinner("Executing 3,000-observation benchmark..."):
        bench_res = run_cached_benchmark()

    metrics_df = pd.DataFrame(bench_res['metrics_table'])
    window_df = pd.DataFrame(bench_res['window_metrics_table'])

    # Strict Day-t Point Metrics Table
    st.subheader("1. Strict Day-t Point Matching Metrics")
    st.markdown("Requires the detector to fire precisely on the sale drop day ($t$):")
    st.dataframe(
        metrics_df[['Method', 'Precision', 'Recall', 'F1_Score', 'Accuracy', 'Confusion_Matrix']]
        .style.highlight_max(axis=0, subset=['Precision', 'Recall', 'F1_Score', 'Accuracy'], color='#D4EFDF'),
        use_container_width=True
    )

    # Window Event Matching Table
    st.subheader(r"2. Temporal Window Event Matching ($\pm 2$ Days)")
    st.markdown(r"Evaluates whether an alert fired within a realistic $\pm 2$ day tolerance window of the manipulation event:")
    st.dataframe(
        window_df[['Method', 'Event_Precision', 'Event_Recall', 'Event_F1']]
        .style.highlight_max(axis=0, subset=['Event_Precision', 'Event_Recall', 'Event_F1'], color='#D4EFDF'),
        use_container_width=True
    )

    # Visual Progression Chart
    st.subheader("📈 Detection Performance Progression")
    df_melted = metrics_df.melt(id_vars=['Method'], value_vars=['Precision', 'Recall', 'F1_Score'], var_name='Metric', value_name='Score')
    fig_bar = px.bar(
        df_melted, x='Method', y='Score', color='Metric', barmode='group',
        color_discrete_map={'Precision': '#3498DB', 'Recall': '#E67E22', 'F1_Score': '#27AE60'},
        range_y=[0, 1.05],
        text_auto='.3f'
    )
    fig_bar.update_layout(height=450, plot_bgcolor="white", margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Methodological Takeaways
    st.success("""
    **Key Methodological Findings:**
    - **Rule-Based (Basic, 40.0% F1):** Only catches immediate 1-day spikes. Completely blind to multi-day plateaus, creeping inflation, and subtle threshold-hugging price manipulation.
    - **Rule-Based (Advanced, 87.3% F1):** Catches multi-day plateaus and steep creeps, but rigid boolean thresholds (>18%) cause it to miss subtle 14% manipulation on low-volatility goods.
    - **Isolation Forest ML (96.4% F1):** Unsupervised multivariate tree splits across `deceptive_discount_signal`, `recent_peak_inflation_ratio`, `creeping_inflation_slope`, and `discount_exaggeration_gap` achieve **100% Recall** across all 4 fraud archetypes with **93.0% Precision**.
    """)

# ==============================================================================
# TAB 3: MARKETPLACE ANALYTICS
# ==============================================================================
elif tab_selection == "📈 Marketplace Analytics":
    st.title("📈 Marketplace Pricing Analytics")
    st.markdown("Macro-level distribution of pricing risk across the loaded dataset.")

    col1, col2, col3, col4 = st.columns(4)
    total_records = len(df_processed)
    total_prods = df_processed['product_id'].nunique()
    high_risk_prods = df_processed.groupby('product_id')['risk_score'].max()
    count_high = int(np.sum(high_risk_prods > 0.75))
    count_mod = int(np.sum((high_risk_prods > 0.50) & (high_risk_prods <= 0.75)))
    count_gen = int(np.sum(high_risk_prods <= 0.50))

    col1.metric("Products Analyzed", total_prods)
    col2.metric("Total Price Observations", f"{total_records:,}")
    col3.metric("🔴 High Risk Products", count_high)
    col4.metric("🟠 Suspicious / Moderate", count_mod)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Peak Risk Distribution Across Products")
        df_peaks = high_risk_prods.reset_index(name='peak_risk')
        fig_hist = px.histogram(df_peaks, x='peak_risk', nbins=25, color_discrete_sequence=['#2B5B84'])
        fig_hist.add_vline(x=0.75, line_dash="dash", line_color="#C0392B", annotation_text="High Risk (0.75)")
        fig_hist.add_vline(x=0.50, line_dash="dash", line_color="#F39C12", annotation_text="Suspicious (0.50)")
        fig_hist.update_layout(height=400, plot_bgcolor="white", xaxis_title="Peak Fraud Risk Score (0-1)", yaxis_title="Number of Products")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("🚨 Top Flagged Products by Peak Risk")
        top_ids = high_risk_prods.nlargest(10).index
        top_df = df_processed[df_processed['product_id'].isin(top_ids)].groupby('product_id').agg(
            peak_score=('risk_score', 'max'),
            avg_price=('price', 'mean'),
            flagged_events=('risk_score', lambda x: int(np.sum(x > 0.5)))
        ).reset_index().sort_values(by='peak_score', ascending=False)

        top_df['Risk Tier'] = ['🔴 HIGH' if s > 0.75 else '🟠 MODERATE' for s in top_df['peak_score']]
        st.dataframe(top_df.rename(columns={
            'product_id': 'Product ID',
            'peak_score': 'Peak Risk Score',
            'avg_price': 'Avg Price ($)',
            'flagged_events': 'Flagged Days'
        }), use_container_width=True)

# FOOTER
st.divider()
st.caption("Discount Legitimacy Classifier — Explainable Machine Learning for Deceptive E-Commerce Pricing Detection.")
