import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random, threading, time
from collections import deque
import base64, io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

# ==========================================================
live_stream = deque(maxlen=400)
uploaded_dataset_name = "No dataset uploaded"
ATTACK_TYPES = ["brute_force", "MITM", "zero_day", "ransomware", "phishing", "data_exfiltration"]
crypto_mode_global = "AUTO"

# ==========================================================
MITIGATION_DB = {
    "brute_force": {"risk": "High", "mitigation": "Rate limiting, behavioral CAPTCHA, AI anomaly detection"},
    "MITM": {"risk": "Critical", "mitigation": "TLS 1.3 enforcement, Certificate Pinning, AI traffic inspection"},
    "zero_day": {"risk": "Critical", "mitigation": "Behavioral analysis, rapid patching, threat hunting"},
    "ransomware": {"risk": "Critical", "mitigation": "Immutable backups, AI endpoint detection, Zero Trust"},
    "phishing": {"risk": "High", "mitigation": "AI email filtering, MFA, security awareness training"},
    "data_exfiltration": {"risk": "High", "mitigation": "DLP, outbound anomaly detection, AI data flow monitoring"}
}

# ==========================================================
def crypto_engine(mode, risk):
    if mode == "PQC": return "PQC", 0.1
    elif mode == "RSA": return "RSA", 0.3
    elif mode == "HYBRID": return ("PQC+AES", 0.08) if risk > 0.6 else ("RSA+AES", 0.2)
    else:
        if risk > 0.75: return "PQC-AUTO", 0.1
        elif risk > 0.45: return "RSA-AUTO", 0.3
        return "AES-256", 0.05

def pipeline():
    global crypto_mode_global
    while True:
        risk = random.random()
        latency = random.uniform(50, 400)
        crypto, penalty = crypto_engine(crypto_mode_global, risk)
        live_stream.append({
            "attack_type": random.choice(ATTACK_TYPES),
            "risk_score": min(1, risk + penalty),
            "latency": latency,
            "crypto": crypto
        })
        time.sleep(1)

threading.Thread(target=pipeline, daemon=True).start()

# ==========================================================
STYLE = {"backgroundColor": "#050914", "color": "#e5e7eb", "fontFamily": "Arial", "minHeight": "100vh", "padding": "16px"}
CARD = {"background": "rgba(255,255,255,0.05)", "border": "1px solid rgba(255,255,255,0.08)", "borderRadius": "14px", "padding": "14px", "marginBottom": "12px"}

# ==========================================================
def parse_upload(contents):
    if not contents:
        return pd.DataFrame(), ""
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="ignore")))
        
        original_rows = len(df)
        if original_rows > 3000:
            df = df.sample(3000, random_state=42).reset_index(drop=True)
            warning = f" (Limited: {original_rows:,} → 3,000 rows)"
        else:
            warning = ""
        
        # Ensure required columns
        if "attack_type" not in df.columns:
            df["attack_type"] = np.random.choice(ATTACK_TYPES, len(df))
        if "risk_score" not in df.columns:
            df["risk_score"] = np.random.uniform(0.1, 0.95, len(df))
        if "latency" not in df.columns:
            df["latency"] = np.random.uniform(50, 400, len(df))
            
        return df, warning
    except:
        return pd.DataFrame(), " (Error loading file)"

# ==========================================================
def generate_mitigation_table(df):
    if df.empty: return []
    attack_stats = df.groupby("attack_type")["risk_score"].agg(["mean", "count"]).reset_index()
    attack_stats.columns = ["attack_type", "avg_risk", "occurrences"]
    
    data = []
    for _, row in attack_stats.iterrows():
        attack = row["attack_type"]
        info = MITIGATION_DB.get(attack, {})
        mitigation = info.get("mitigation", f"Apply AI monitoring, anomaly detection and response for {attack.replace('_', ' ')}")
        
        data.append({
            "Threat": attack.replace("_", " ").title(),
            "Avg Risk Score": round(row["avg_risk"], 3),
            "Risk Level": info.get("risk", "High"),
            "Occurrences": int(row["occurrences"]),
            "Potential Impact": f"Significant impact from {attack.replace('_', ' ')}",
            "Latency Impact": "Medium",
            "Cryptographic Concern": "Potential encryption exposure",
            "AI/ML Relevance": "High",
            "Recommended Mitigation": mitigation
        })
    return pd.DataFrame(data).sort_values("Avg Risk Score", ascending=False).to_dict('records')

# ==========================================================
app.layout = html.Div(style=STYLE, children=[
    html.H1("🛡 AO-LACCP SOC Intelligence Platform", style={"textAlign": "center", "color": "#38bdf8"}),
    
    html.Div(style=CARD, children=[
        dcc.Upload(id="upload_data", 
                   children=html.Div("📁 Upload Dataset (Large files will be limited to 3000 rows)"), 
                   style={"border": "2px dashed #38bdf8", "padding": "20px", "textAlign": "center"}),
        html.Div(id="dataset_name", style={"marginTop": "10px", "fontWeight": "bold"}),
        
        html.Div(style={"display": "flex", "gap": "15px", "marginTop": "15px"}, children=[
            dcc.Dropdown(id="model_selector", options=[
                {"label": "Random Forest", "value": "RF"},
                {"label": "XGBoost", "value": "XGB", "disabled": not XGB_AVAILABLE}
            ], value="RF", style={"flex": 1}),
            dcc.Dropdown(id="crypto_selector", options=[
                {"label": "Auto", "value": "AUTO"},
                {"label": "PQC", "value": "PQC"},
                {"label": "RSA", "value": "RSA"},
                {"label": "Hybrid", "value": "HYBRID"}
            ], value="AUTO", style={"flex": 1}),
        ]),
        html.Div(id="crypto_advice")
    ]),

    html.Div(style=CARD, children=[html.H3("🔥 Live Threat Stream"), html.Div(id="live_feed")]),

    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, 
             children=[dcc.Graph(id="risk_graph"), dcc.Graph(id="latency_graph")]),
    
    html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px"}, 
             children=[dcc.Graph(id="soc_attack_stream"), dcc.Graph(id="soc_risk_gauge")]),

    html.Div(id="soc_summary", style=CARD),

    html.Div(style=CARD, children=[
        html.H3("🛠 SOC Mitigation & Action Plan"),
        html.Button("📥 Export as CSV", id="export-btn", style={"marginBottom": "10px"}),
        html.Div(id="mitigation_table_container", 
                 style={"maxHeight": "620px", "overflow": "auto", "border": "1px solid rgba(255,255,255,0.15)", "borderRadius": "8px"})
    ]),

    dcc.Download(id="download-mitigation"),
    dcc.Interval(id="interval", interval=2500),
])

# ==========================================================
@app.callback(
    [Output("live_feed", "children"), Output("risk_graph", "figure"), Output("latency_graph", "figure"),
     Output("soc_attack_stream", "figure"), Output("soc_risk_gauge", "figure"), Output("dataset_name", "children"),
     Output("crypto_advice", "children"), Output("mitigation_table_container", "children")],
    [Input("interval", "n_intervals"), Input("model_selector", "value"), Input("crypto_selector", "value")],
    [State("upload_data", "contents"), State("upload_data", "filename")]
)
def update_dashboard(n, model_type, crypto_mode, upload_content, filename):
    global uploaded_dataset_name, crypto_mode_global
    crypto_mode_global = crypto_mode

    if upload_content:
        df_up, warning = parse_upload(upload_content)
        if not df_up.empty:
            uploaded_dataset_name = (filename or "Uploaded Dataset") + warning
            for _, r in df_up.iterrows():
                live_stream.append({
                    "attack_type": str(r.get("attack_type")),
                    "risk_score": float(r.get("risk_score", 0.5)),
                    "latency": float(r.get("latency", 150)),
                    "crypto": "UPLOAD"
                })

    df = pd.DataFrame(list(live_stream))
    if df.empty:
        empty = go.Figure()
        return "Initializing...", empty, empty, empty, empty, "No dataset", "Booting...", html.Div("Waiting for data...")

    latest = df.iloc[-1]
    avg_risk = df["risk_score"].mean()

    # ML Prediction
    ml_text = "ML Model: Waiting..."
    if len(df) > 12:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X = encoder.fit_transform(df[["attack_type"]])
            y = (df["risk_score"] > 0.65).astype(int)

            if model_type == "XGB" and XGB_AVAILABLE:
                model = XGBClassifier(n_estimators=80, random_state=42, verbosity=0)
                model_name = "XGBoost"
            else:
                model = RandomForestClassifier(n_estimators=80, random_state=42)
                model_name = "Random Forest"

            model.fit(X[:-1], y[:-1])
            proba = model.predict_proba(X[[-1]])[0][1]
            pred = "HIGH RISK" if proba > 0.5 else "LOW RISK"
            ml_text = f"✅ {model_name}: {pred} ({proba:.1%})"
        except:
            ml_text = f"{model_type}: Processing..."

    # Figures
    risk_fig = px.line(df, y="risk_score").update_layout(template="plotly_dark", height=340)
    latency_fig = px.line(df, y="latency").update_layout(template="plotly_dark", height=340)
    attack_fig = px.histogram(df, x="attack_type", color="attack_type").update_layout(template="plotly_dark", height=380)

    gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(avg_risk),
        gauge={"axis": {"range": [0, 1]}, "steps": [
            {"range": [0, 0.35], "color": "#16a34a"},
            {"range": [0.35, 0.7], "color": "#facc15"},
            {"range": [0.7, 1], "color": "#ef4444"}
        ]}))
    gauge.update_layout(template="plotly_dark", height=380)

    advice = "⚠ PQC" if avg_risk > 0.7 else "⚡ Hybrid" if avg_risk > 0.5 else "✅ Stable"

    table_data = generate_mitigation_table(df)

    return (
        f"{latest['attack_type'].upper()} | {latest['crypto']} | {ml_text}",
        risk_fig, latency_fig, attack_fig, gauge,
        f"Dataset: {uploaded_dataset_name}",
        advice,
        dash_table.DataTable(data=table_data, page_size=12,
                             style_table={'overflowX': 'auto'},
                             style_header={'backgroundColor': '#1e2937', 'color': 'white', 'fontWeight': 'bold', 'position': 'sticky', 'top': 0},
                             style_data={'backgroundColor': '#0f172a', 'color': '#e2e8f0', 'whiteSpace': 'normal'})
    )

@app.callback(Output("download-mitigation", "data"), Input("export-btn", "n_clicks"), prevent_initial_call=True)
def export_mitigation(n_clicks):
    if n_clicks is None: return None
    df = pd.DataFrame(list(live_stream))
    if df.empty: return None
    table_data = generate_mitigation_table(df)
    return dcc.send_data_frame(pd.DataFrame(table_data).to_csv, "AO-LACCP_Mitigation_Plan.csv", index=False)

# ==========================================================
if __name__ == "__main__":
    print("🚀 AO-LACCP SOC Dashboard started (Stable Mode)")
    print("→ Open: http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)