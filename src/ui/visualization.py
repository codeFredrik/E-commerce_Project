import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional

def plot_product_analysis(df: pd.DataFrame, suspicious: List[Dict[str, Any]], product_id: int, explanations: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
    """
    Visualizes price history, baselines, ML fraud risk trajectory, and SHAP plain-English tooltips using interactive Plotly.
    """
    df_p = df[df['product_id'] == product_id].sort_values(by='order_date').copy()
    
    if len(df_p) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f'No data for Product {product_id}', showarrow=False, font=dict(size=16))
        return fig
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.50, 0.25, 0.25],
        subplot_titles=(
            f'Product {product_id} — Price History, Baselines & Flagged Anomalies',
            'AI Fraud Risk Score (Isolation Forest ML)',
            'Statistical Deviation (Z-Score)'
        )
    )
    
    # ===== ROW 1: MAIN PRICE CHART =====
    fig.add_trace(go.Scatter(
        x=df_p['order_date'], y=df_p['price'],
        mode='lines+markers', name='Actual Price',
        line=dict(color='#2B5B84', width=2.5),
        marker=dict(size=6, color='#2B5B84')
    ), row=1, col=1)
                             
    if 'rolling_mean_30' in df_p.columns:
        fig.add_trace(go.Scatter(
            x=df_p['order_date'], y=df_p['rolling_mean_30'],
            mode='lines', name='30-Period Baseline',
            line=dict(color='#27AE60', width=2, dash='dash')
        ), row=1, col=1)
                             
    if 'rolling_mean' in df_p.columns:
        fig.add_trace(go.Scatter(
            x=df_p['order_date'], y=df_p['rolling_mean'],
            mode='lines', name='5-Period Rolling Mean',
            line=dict(color='#E67E22', width=1.5, dash='dot')
        ), row=1, col=1)
                             
    # Flagged Rule-Based Anomalies
    susp_indices = [item['original_index'] for item in suspicious if item['product_id'] == product_id]
    if susp_indices and 'original_index' in df_p.columns:
        susp_rows = df_p[df_p['original_index'].isin(susp_indices)]
        fig.add_trace(go.Scatter(
            x=susp_rows['order_date'], y=susp_rows['price'],
            mode='markers', name='Rule-Based Flag',
            marker=dict(color='#C0392B', size=11, symbol='x', line=dict(width=2, color='#922B21'))
        ), row=1, col=1)
        
    # Flagged ML Anomalies with SHAP Plain-English Tooltips
    if explanations and len(explanations) == len(df):
        # Extract explanations for this product
        p_exps = [explanations[i] for i in df_p.index if i < len(explanations)]
        high_risk_dates = []
        high_risk_prices = []
        hover_texts = []
        
        for idx_pos, exp in enumerate(p_exps):
            if exp.get('risk_score', 0.0) > 0.50:
                high_risk_dates.append(df_p.iloc[idx_pos]['order_date'])
                high_risk_prices.append(df_p.iloc[idx_pos]['price'])
                reasons_str = "<br>• ".join(exp.get('reasons', ['Anomaly detected']))
                hover_texts.append(f"<b>{exp.get('verdict', 'SUSPICIOUS')}</b> ({exp.get('confidence_pct', 0)}% Conf.)<br>• {reasons_str}")
                
        if high_risk_dates:
            fig.add_trace(go.Scatter(
                x=high_risk_dates, y=high_risk_prices,
                mode='markers', name='AI ML Anomaly',
                text=hover_texts, hoverinfo='text',
                marker=dict(color='#8E44AD', size=13, symbol='diamond', line=dict(width=2, color='#4A235A'))
            ), row=1, col=1)
                                 
    # ===== ROW 2: ML RISK SCORE TRAJECTORY =====
    if explanations and len(explanations) == len(df):
        p_exps = [explanations[i] for i in df_p.index if i < len(explanations)]
        scores = [exp.get('risk_score', 0.0) for exp in p_exps]
        colors_ml = ['#C0392B' if s > 0.75 else ('#E67E22' if s > 0.50 else '#2ECC71') for s in scores]
        
        fig.add_trace(go.Bar(
            x=df_p['order_date'], y=scores,
            marker_color=colors_ml, name='ML Risk Score'
        ), row=2, col=1)
        
        fig.add_hline(y=0.50, line_dash="dash", line_color="#C0392B", line_width=1.5,
                      annotation_text="Fraud Threshold", row=2, col=1)
        fig.update_yaxes(range=[0.0, 1.05], title_text="Risk (0–1)", row=2, col=1)
    else:
        fig.add_annotation(text="Run AI ML Analysis to view risk trajectory", row=2, col=1, showarrow=False)

    # ===== ROW 3: Z-SCORE STATISTICAL DEVIATION =====
    if 'z_score' in df_p.columns:
        colors_z = ['#C0392B' if abs(z) > 2.0 else '#3498DB' for z in df_p['z_score']]
        fig.add_trace(go.Bar(
            x=df_p['order_date'], y=df_p['z_score'],
            marker_color=colors_z, name='Z-Score'
        ), row=3, col=1)
                             
        fig.add_hline(y=2.0, line_dash="dash", line_color="#C0392B", line_width=1.5, annotation_text="+2σ", row=3, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="#C0392B", line_width=1.5, annotation_text="-2σ", row=3, col=1)
        fig.add_hline(y=0, line_width=0.5, line_color="black", row=3, col=1)
        fig.update_yaxes(title_text="Z-Score", row=3, col=1)

    fig.update_layout(
        height=850, hovermode='x unified', showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor="white"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F2F4F4')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F2F4F4')
    fig.update_xaxes(title_text="Order Date", row=3, col=1)
    
    return fig
