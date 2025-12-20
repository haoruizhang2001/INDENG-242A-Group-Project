"""
Interactive Oil Price Prediction Dashboard
Generates a self-contained interactive HTML dashboard for GCP deployment.
All data is embedded within the HTML file.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json


def load_data(file_path: str) -> pd.DataFrame:
    """Load the predictions vs actual data from Excel file."""
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def get_model_metrics():
    """
    Comprehensive model metrics from the performance comparison table.
    """
    metrics = {
        'Pred_M1_Linear_Reg': {
            'display_name': 'Linear Regression',
            'group': 'Basics',
            'color': '#E74C3C',
            'rmse': 25.54, 'mape': 33.20, 'aic': 2450.40, 'r2': -25.08,
            'precision': 0.50, 'recall': 0.91, 'auc_roc': 0.49, 'log_loss': 0.97
        },
        'Pred_M2_Random_Walk': {
            'display_name': 'Random Walk',
            'group': 'Basics',
            'color': '#3498DB',
            'rmse': 1.34, 'mape': 1.48, 'aic': 137.60, 'r2': 0.93,
            'precision': 0.00, 'recall': 0.00, 'auc_roc': 0.50, 'log_loss': 0.69
        },
        'Pred_M3_Smoothed_RW': {
            'display_name': 'Smoothed RW',
            'group': 'Basics',
            'color': '#1ABC9C',
            'rmse': 1.52, 'mape': 1.65, 'aic': 198.93, 'r2': 0.91,
            'precision': 0.52, 'recall': 0.53, 'auc_roc': 0.54, 'log_loss': 0.77
        },
        'Pred_M4_Calibrated_RW': {
            'display_name': 'Calibrated RW',
            'group': 'Basics',
            'color': '#9B59B6',
            'rmse': 1.38, 'mape': 1.54, 'aic': None, 'r2': 0.86,
            'precision': 0.49, 'recall': 0.48, 'auc_roc': 0.45, 'log_loss': 0.93
        },
        'Pred_M5_Random_Forest': {
            'display_name': 'Random Forest',
            'group': 'ML Methods',
            'color': '#F39C12',
            'rmse': 13.13, 'mape': 19.19, 'aic': 1207.10, 'r2': -5.89,
            'precision': 0.50, 'recall': 0.99, 'auc_roc': 0.57, 'log_loss': 1.28
        },
        'Pred_M6_XGBoost': {
            'display_name': 'XGBoost',
            'group': 'ML Methods',
            'color': '#E67E22',
            'rmse': 12.08, 'mape': 17.46, 'aic': 1295.90, 'r2': -4.83,
            'precision': 0.50, 'recall': 0.98, 'auc_roc': 0.57, 'log_loss': 1.18
        },
        'LSTM_Model_1': {
            'display_name': 'LSTM 1',
            'group': 'Neural Networks',
            'color': '#2ECC71',
            'rmse': 1.78, 'mape': 2.14, 'aic': None, 'r2': 0.77,
            'precision': 0.49, 'recall': 0.46, 'auc_roc': 0.47, 'log_loss': 0.86
        },
        'LSTM_Model_2': {
            'display_name': 'LSTM 2',
            'group': 'Neural Networks',
            'color': '#27AE60',
            'rmse': 1.82, 'mape': 2.18, 'aic': None, 'r2': 0.76,
            'precision': 0.48, 'recall': 0.45, 'auc_roc': 0.46, 'log_loss': 0.87
        },
        'LSTM_Model_3': {
            'display_name': 'LSTM 3',
            'group': 'Neural Networks',
            'color': '#16A085',
            'rmse': 1.75, 'mape': 2.10, 'aic': None, 'r2': 0.78,
            'precision': 0.50, 'recall': 0.47, 'auc_roc': 0.48, 'log_loss': 0.85
        },
        'LSTM_Model_4': {
            'display_name': 'LSTM 4',
            'group': 'Neural Networks',
            'color': '#48C9B0',
            'rmse': 1.80, 'mape': 2.16, 'aic': None, 'r2': 0.77,
            'precision': 0.49, 'recall': 0.46, 'auc_roc': 0.47, 'log_loss': 0.86
        },
        'LSTM_Model_5': {
            'display_name': 'LSTM 5',
            'group': 'Neural Networks',
            'color': '#5DADE2',
            'rmse': 1.85, 'mape': 2.22, 'aic': None, 'r2': 0.75,
            'precision': 0.48, 'recall': 0.45, 'auc_roc': 0.46, 'log_loss': 0.88
        },
        'LSTM_Model_6': {
            'display_name': 'LSTM 6',
            'group': 'Neural Networks',
            'color': '#85929E',
            'rmse': 1.79, 'mape': 2.15, 'aic': None, 'r2': 0.77,
            'precision': 0.49, 'recall': 0.46, 'auc_roc': 0.47, 'log_loss': 0.86
        },
        'LSTM_Model_7': {
            'display_name': 'LSTM 7',
            'group': 'Neural Networks',
            'color': '#AF7AC5',
            'rmse': 1.83, 'mape': 2.19, 'aic': None, 'r2': 0.76,
            'precision': 0.48, 'recall': 0.45, 'auc_roc': 0.46, 'log_loss': 0.87
        },
        'LSTM_Model_8': {
            'display_name': 'LSTM 8',
            'group': 'Neural Networks',
            'color': '#F1948A',
            'rmse': 1.88, 'mape': 2.25, 'aic': None, 'r2': 0.74,
            'precision': 0.47, 'recall': 0.44, 'auc_roc': 0.45, 'log_loss': 0.89
        },
        'LSTM_Model_9': {
            'display_name': 'LSTM 9',
            'group': 'Neural Networks',
            'color': '#82E0AA',
            'rmse': 1.76, 'mape': 2.11, 'aic': None, 'r2': 0.78,
            'precision': 0.50, 'recall': 0.47, 'auc_roc': 0.48, 'log_loss': 0.85
        },
        'LSTM_Model_10': {
            'display_name': 'LSTM 10',
            'group': 'Neural Networks',
            'color': '#F7DC6F',
            'rmse': 1.81, 'mape': 2.17, 'aic': None, 'r2': 0.76,
            'precision': 0.48, 'recall': 0.45, 'auc_roc': 0.46, 'log_loss': 0.87
        },
        'Ensemble_Prediction': {
            'display_name': 'Ensemble',
            'group': 'ML Methods',
            'color': '#2C3E50',
            'rmse': 3.65, 'mape': 5.16, 'aic': None, 'r2': 0.01,
            'precision': 0.47, 'recall': 0.42, 'auc_roc': 0.45, 'log_loss': 0.80
        },
    }
    
    return metrics


def create_dashboard(df: pd.DataFrame, output_path: str) -> None:
    """Create a self-contained interactive HTML dashboard."""
    
    model_metrics = get_model_metrics()
    available_models = [col for col in model_metrics.keys() if col in df.columns]
    sorted_models = sorted(available_models, key=lambda x: model_metrics[x]['rmse'])
    
    # Prepare data for JSON
    data_for_json = {
        'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'actual': df['Actual_Price'].tolist(),
        'models': {}
    }
    
    for col in sorted_models:
        m = model_metrics[col]
        data_for_json['models'][col] = {
            'values': df[col].tolist(),
            'name': m['display_name'],
            'color': m['color'],
            'group': m['group'],
            'rmse': m['rmse'],
            'mape': m['mape'],
            'r2': m['r2'],
            'precision': m['precision'],
            'recall': m['recall'],
            'auc_roc': m['auc_roc'],
            'log_loss': m['log_loss']
        }
    
    # Create main prediction figure (single plot, no cumulative error)
    fig_main = go.Figure()
    
    # Add actual price
    fig_main.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Actual_Price'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#000000', width=2.5),
            hovertemplate='<b>Actual Price</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
        )
    )
    
    trace_map = {}
    for idx, col in enumerate(sorted_models):
        m = model_metrics[col]
        trace_map[col] = idx + 1
        fig_main.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df[col],
                mode='lines',
                name=m['display_name'],
                line=dict(color=m['color'], width=1.5),
                visible='legendonly',
                hovertemplate=f"<b>{m['display_name']}</b><br>Date: %{{x|%Y-%m-%d}}<br>Prediction: $%{{y:.2f}}<extra></extra>"
            )
        )
    
    fig_main.update_layout(
        title=dict(text='<b>Price Predictions vs Actual</b>', x=0.5, font=dict(size=18)),
        height=600,
        template='plotly_white',
        legend=dict(
            title=dict(text='<b>Models (Click to Toggle)</b>', font=dict(size=11)),
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E0E0E0',
            borderwidth=1,
            font=dict(size=9),
        ),
        hovermode='x unified',
        margin=dict(l=70, r=20, t=80, b=60),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            showgrid=True,
            gridcolor='#EAECEE'
        ),
        yaxis=dict(title='Price ($)', showgrid=True, gridcolor='#EAECEE')
    )
    
    main_plot_html = fig_main.to_html(include_plotlyjs='cdn', full_html=False, div_id='main-plot')
    
    # Create metric bar charts
    model_names = [model_metrics[col]['display_name'] for col in sorted_models]
    model_colors = [model_metrics[col]['color'] for col in sorted_models]
    
    def create_bar_chart(metric_key, title, y_label, higher_better=False):
        values = [model_metrics[col][metric_key] for col in sorted_models]
        data = list(zip(model_names, values, model_colors))
        data_sorted = sorted(data, key=lambda x: x[1], reverse=higher_better)
        names, vals, colors = zip(*data_sorted)
        
        fig = go.Figure(data=[
            go.Bar(x=list(names), y=list(vals), marker_color=list(colors),
                   hovertemplate='<b>%{x}</b><br>' + y_label + ': %{y:.2f}<extra></extra>')
        ])
        fig.update_layout(
            title=dict(text=f'<b>{title}</b>', x=0.5, font=dict(size=14)),
            height=320,
            template='plotly_white',
            margin=dict(l=50, r=30, t=50, b=80),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(title=dict(text=y_label, font=dict(size=11))),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)
    
    rmse_html = create_bar_chart('rmse', 'RMSE Comparison', 'RMSE')
    mape_html = create_bar_chart('mape', 'MAPE (%) Comparison', 'MAPE (%)')
    r2_html = create_bar_chart('r2', 'R² Comparison', 'R²', higher_better=True)
    precision_html = create_bar_chart('precision', 'Precision Comparison', 'Precision', higher_better=True)
    recall_html = create_bar_chart('recall', 'Recall Comparison', 'Recall', higher_better=True)
    auc_html = create_bar_chart('auc_roc', 'AUC-ROC Comparison', 'AUC-ROC', higher_better=True)
    logloss_html = create_bar_chart('log_loss', 'Log Loss Comparison', 'Log Loss')
    
    total_traces = len(sorted_models) + 1
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oilman Sachs Prediction Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .dashboard-container {{
            max-width: 1600px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
            color: white;
            padding: 25px 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header p {{ opacity: 0.8; font-size: 14px; }}
        .main-content {{
            width: 100%;
        }}
        .chart-container {{
            padding: 30px;
        }}
        .metrics-section {{
            padding: 30px;
            background: #F8F9FA;
            border-top: 2px solid #E0E0E0;
        }}
        .metrics-header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .metrics-header h2 {{ font-size: 24px; color: #2C3E50; margin-bottom: 8px; }}
        .metrics-header p {{ font-size: 14px; color: #7F8C8D; }}
        .section-title {{
            font-size: 18px;
            color: #2C3E50;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498DB;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .metric-chart {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            padding: 10px;
        }}
        @media (max-width: 1200px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>Oilman Sachs Prediction Dashboard</h1>
            <p>Data: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')} ({len(df)} observations)</p>
        </div>
        
        <div class="main-content">
            <div class="chart-container">
                {main_plot_html}
            </div>
        </div>
        
        <div class="metrics-section">
            <div class="metrics-header">
                <h2>Performance Comparison of Baseline and Machine Learning Models</h2>
            </div>
            
            <h3 class="section-title">Price Prediction (Regression) Metrics</h3>
            <div class="chart-grid">
                <div class="metric-chart">{rmse_html}</div>
                <div class="metric-chart">{mape_html}</div>
                <div class="metric-chart">{r2_html}</div>
            </div>
            
            <h3 class="section-title">Directional Forecasting (Classification) Metrics</h3>
            <div class="chart-grid">
                <div class="metric-chart">{precision_html}</div>
                <div class="metric-chart">{recall_html}</div>
                <div class="metric-chart">{auc_html}</div>
                <div class="metric-chart">{logloss_html}</div>
            </div>
        </div>
    </div>
    
    <script>
        // Dashboard is fully interactive through Plotly's built-in legend
        // Click legend items to show/hide models
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Dashboard saved to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "results" / "model_predictions_vs_actual.xlsx"
    output_path = project_root / "dashboard" / "prediction_dashboard.html"
    
    print("📊 Loading prediction data...")
    df = load_data(str(data_path))
    print(f"   Loaded {len(df)} records")
    
    print("🎨 Creating dashboard...")
    create_dashboard(df, str(output_path))


if __name__ == "__main__":
    main()
