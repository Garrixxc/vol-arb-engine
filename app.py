import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))

from core.iv_surface import compute_iv_surface
from core.svi import fit_surface
from backtest.engine import generate_synthetic_backtest_data, BacktestEngine, BacktestConfig
from data.fetchers.yfinance_fetcher import fetch_options_chain

# Configuration
LIVE_DATA_MODE = True  # Set to True to fetch real SPY data
TICKER = "SPY"

# Initialize Dash app
app = dash.Dash(__name__, title="Aether Vol-Arb Engine")

# Global state to store backtest results
BACKTEST_RESULTS = None

def get_backtest_results():
    global BACKTEST_RESULTS
    if BACKTEST_RESULTS is None:
        print("Initializing synthetic backtest for dashboard...")
        snapshots = generate_synthetic_backtest_data(n_days=90)
        engine = BacktestEngine(BacktestConfig(capital=1000000))
        BACKTEST_RESULTS = engine.run(snapshots, verbose=False)
    return BACKTEST_RESULTS

def get_latest_data():
    """Generates the latest surface and signal data."""
    if LIVE_DATA_MODE:
        try:
            print(f"Fetching live data for {TICKER}...")
            chain, spot = fetch_options_chain(TICKER)
            snap = {"date": datetime.now().strftime("%Y-%m-%d"), "spot": spot, "chain": chain}
            
            # Try to compute surface, if too sparse, fallback to synthetic
            iv_surface = compute_iv_surface(snap["chain"], r=0.053, q=0.013)
        except Exception as e:
            print(f"Live data failed or too sparse: {e}. Falling back to synthetic.")
            snapshots = generate_synthetic_backtest_data(n_days=1)
            snap = snapshots[0]
            iv_surface = compute_iv_surface(snap["chain"], r=0.053, q=0.013)
    else:
        snapshots = generate_synthetic_backtest_data(n_days=1)
        snap = snapshots[0]
        iv_surface = compute_iv_surface(snap["chain"], r=0.053, q=0.013)
    params, fitted_df, _ = fit_surface(iv_surface, verbose=False)
    
    market_points = fitted_df[fitted_df["option_type"] != "model"]
    model_surface = fitted_df[fitted_df["option_type"] == "model"]
    
    signals = market_points.dropna(subset=["snr"])
    signals = signals.sort_values("snr", ascending=False).head(20)
    
    return snap, market_points, model_surface, signals

# Layout components
def create_header(capital, cum_pnl, spot_price=None):
    pnl_color = "#3fb950" if cum_pnl >= 0 else "#f85149"
    mode_label = "LIVE TRADING" if LIVE_DATA_MODE else "SIMULATION"
    mode_color = "#3fb950" if LIVE_DATA_MODE else "#8b949e"
    
    return html.Header([
        html.Div([
            html.Div([
                html.H1("AETHER VOL-ARB", style={'margin': 0, 'fontSize': '1.5rem', 'fontWeight': '900', 'color': '#ffffff', 'letterSpacing': '-0.02em'}),
                html.Div([
                    html.Span(mode_label, style={'fontSize': '0.65rem', 'color': mode_color, 'fontWeight': 'bold', 'border': f'1px solid {mode_color}', 'padding': '2px 6px', 'borderRadius': '4px', 'marginRight': '8px'}),
                    html.Span("M4 PRO v2.0", style={'fontSize': '0.65rem', 'color': '#8b949e', 'textTransform': 'uppercase', 'letterSpacing': '0.1em'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '4px'})
            ])
        ], className="header-left"),
        
        html.Div([
            html.Div([
                html.P("INDEX SPOT", style={'fontSize': '0.65rem', 'color': '#8b949e', 'margin': 0, 'fontWeight': 'bold'}),
                html.P(f"${spot_price:,.2f}" if spot_price else "---", style={'fontSize': '1.25rem', 'fontFamily': 'monospace', 'fontWeight': '900', 'margin': 0, 'color': '#ffffff'})
            ], style={'textAlign': 'right'}),
            html.Div([
                html.P("PORTFOLIO CAPITAL", style={'fontSize': '0.65rem', 'color': '#8b949e', 'margin': 0, 'fontWeight': 'bold'}),
                html.P(f"${capital:,.0f}", style={'fontSize': '1.25rem', 'fontFamily': 'monospace', 'fontWeight': '900', 'margin': 0, 'color': '#ffffff'})
            ], style={'textAlign': 'right'}),
            html.Div([
                html.P("UNREALIZED P&L", style={'fontSize': '0.65rem', 'color': '#8b949e', 'margin': 0, 'fontWeight': 'bold'}),
                html.P(f"${cum_pnl:,.0f}", style={'fontSize': '1.25rem', 'fontFamily': 'monospace', 'fontWeight': '900', 'margin': 0, 'color': pnl_color})
            ], style={'textAlign': 'right'})
        ], style={'display': 'flex', 'gap': '2.5rem'})
    ], className="glass-panel main-header", style={'padding': '1.25rem 2rem', 'borderRadius': '1rem', 'marginBottom': '1.5rem'})

app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0),
    html.Div(id='dashboard-content')
], style={'backgroundColor': '#0d1117', 'color': '#e6edf3', 'minHeight': '100vh', 'padding': '1.5rem', 'fontFamily': 'Inter, system-ui, sans-serif'})

@app.callback(Output('dashboard-content', 'children'),
              Input('interval-component', 'n_intervals'))
def update_dashboard(n):
    snap, market, model, signals = get_latest_data()
    bt_results = get_backtest_results()
    event_log = bt_results["event_log"]
    cum_pnl = event_log["cum_pnl"].iloc[-1]
    
    # 3D Surface Plot
    expiries = sorted(model['dte'].unique())
    logM = sorted(model['log_moneyness'].unique())
    z = []
    for dte in expiries:
        row = []
        for k in logM:
            point = model[(model['dte'] == dte) & (np.isclose(model['log_moneyness'], k, atol=1e-5))]
            row.append(point['svi_iv'].iloc[0] if not point.empty else None)
        z.append(row)

    fig_3d = go.Figure()
    fig_3d.add_trace(go.Surface(
        x=logM, y=expiries, z=z,
        colorscale='Viridis', opacity=0.8, showscale=False, name='SVI Surface'
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=market['log_moneyness'], y=market['dte'], z=market['market_iv'],
        mode='markers',
        marker=dict(size=4, color=market['mispricing_bps'], colorscale='RdBu', reversescale=True, opacity=1),
        name='Market Data'
    ))
    fig_3d.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title='Log-Moneyness', gridcolor='#30363d'),
            yaxis=dict(title='DTE', gridcolor='#30363d'),
            zaxis=dict(title='Implied Vol', gridcolor='#30363d'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )
    )

    # P&L Chart
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=event_log['date'], y=event_log['cum_pnl'],
        fill='tozeroy', mode='lines', line=dict(color='#3fb950', width=3),
        fillcolor='rgba(63, 185, 80, 0.1)', name='Cumulative P&L'
    ))
    fig_pnl.update_layout(
        template='plotly_dark',
        margin=dict(l=40, r=20, b=40, t=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title=None, gridcolor='#30363d', showgrid=False),
        yaxis=dict(title='P&L ($)', gridcolor='#30363d', zerolinecolor='#8b949e'),
        font=dict(family="JetBrains Mono, monospace", size=10)
    )

    return [
        create_header(1000000, cum_pnl, snap.get("spot")),
        html.Div([
            # Left Column
            html.Div([
                html.Div([
                    html.H2("3D Implied Volatility Surface", style={'fontSize': '1rem', 'marginBottom': '1rem'}),
                    dcc.Graph(figure=fig_3d, style={'height': '500px'}, config={'displayModeBar': False})
                ], className="glass-panel", style={'padding': '1rem', 'borderRadius': '0.75rem', 'marginBottom': '1.5rem'}),
                
                html.Div([
                    dcc.Graph(figure=fig_pnl, config={'displayModeBar': False})
                ], className="glass-panel", style={'padding': '1rem', 'borderRadius': '0.75rem'})
            ], style={'flex': '2'}),
            
            # Right Column
            html.Div([
                html.Div([
                    html.H2("Ranked Arb Signals", style={'fontSize': '1rem', 'marginBottom': '1rem'}),
                    dash_table.DataTable(
                        data=signals.to_dict('records'),
                        columns=[
                            {"name": "Expiry", "id": "expiry"},
                            {"name": "Strike", "id": "strike"},
                            {"name": "Type", "id": "option_type"},
                            {"name": "SNR", "id": "snr"},
                            {"name": "Mispricing", "id": "mispricing_bps"}
                        ],
                        style_header={'backgroundColor': '#0d1117', 'color': '#8b949e', 'border': 'none', 'fontWeight': 'bold', 'textAlign': 'left'},
                        style_cell={'backgroundColor': '#161b22', 'color': '#e6edf3', 'border': 'none', 'padding': '8px', 'fontSize': '12px', 'textAlign': 'left'},
                        style_data_conditional=[
                            {'if': {'column_id': 'snr', 'filter_query': '{snr} > 1'}, 'color': '#3fb950'},
                            {'if': {'column_id': 'mispricing_bps', 'filter_query': '{mispricing_bps} > 0'}, 'color': '#3fb950'},
                            {'if': {'column_id': 'mispricing_bps', 'filter_query': '{mispricing_bps} < 0'}, 'color': '#f85149'}
                        ],
                        style_table={'height': '600px', 'overflowY': 'auto'}
                    )
                ], className="glass-panel", style={'padding': '1rem', 'borderRadius': '0.75rem', 'height': '100%'})
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1.5rem'})
    ]

# Inject CSS for glass panel effect and premium fonts
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body { 
                margin: 0; 
                background-color: #05070a; 
                background-image: 
                    radial-gradient(circle at 50% 0%, rgba(33, 38, 45, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 0% 100%, rgba(31, 111, 235, 0.05) 0%, transparent 30%);
                color: #e6edf3;
                font-family: 'Inter', sans-serif;
            }
            .glass-panel {
                background: rgba(13, 17, 23, 0.7);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(48, 54, 61, 0.6);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
            }
            .main-header {
                background: linear-gradient(135deg, rgba(22, 27, 34, 0.8), rgba(13, 17, 23, 0.8));
                border-bottom: 1px solid rgba(56, 139, 253, 0.3);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            h1, h2, h3 { color: #f0f6fc; }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet {
                border: none !important;
            }
            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #0d1117; }
            ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #484f58; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=False, port=8050, host='0.0.0.0')
