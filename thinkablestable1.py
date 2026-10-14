
import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import threading
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

# ==========================================================
# LIVE STREAM AND GLOBAL SETTINGS
# ==========================================================
live_stream = deque(maxlen=400)

ATTACK_TYPES = [
    "brute_force",
    "MITM",
    "zero_day",
    "ransomware",
    "phishing",
    "data_exfiltration"
]

crypto_mode_global = "AUTO"

# ==========================================================
# MITIGATION DATABASE
# ==========================================================
MITIGATION_DB = {
    "brute_force": {
        "risk": "High",
        "mitigation": "Rate limiting, behavioral CAPTCHA, AI anomaly detection"
    },
    "MITM": {
        "risk": "Critical",
        "mitigation": "TLS 1.3 enforcement, certificate pinning, AI traffic inspection"
    },
    "zero_day": {
        "risk": "Critical",
        "mitigation": "Behavioral analysis, rapid patching, threat hunting"
    },
    "ransomware": {
        "risk": "Critical",
        "mitigation": "Immutable backups, AI endpoint detection, Zero Trust"
    },
    "phishing": {
        "risk": "High",
        "mitigation": "AI email filtering, MFA, security awareness training"
    },
    "data_exfiltration": {
        "risk": "High",
        "mitigation": "DLP, outbound anomaly detection, AI data flow monitoring"
    }
}

# ==========================================================
# CRYPTOGRAPHIC RECOMMENDATION ENGINE
# ==========================================================
def crypto_engine(mode, risk):
    if mode == "PQC":
        return "PQC", 0.1

    elif mode == "RSA":
        return "RSA", 0.3

    elif mode == "HYBRID":
        return ("PQC+AES", 0.08) if risk > 0.6 else ("RSA+AES", 0.2)

    else:
        if risk > 0.75:
            return "PQC-AUTO", 0.1
        elif risk > 0.45:
            return "RSA-AUTO", 0.3

        return "AES-256", 0.05


# ==========================================================
# SYNTHETIC LIVE THREAT GENERATOR
# ==========================================================
def pipeline():
    global crypto_mode_global

    while True:
        risk = random.random()
        latency = random.uniform(50, 400)

        crypto, penalty = crypto_engine(
            crypto_mode_global,
            risk
        )

        live_stream.append({
            "attack_type": random.choice(ATTACK_TYPES),
            "risk_score": min(1.0, risk + penalty),
            "latency": latency,
            "crypto": crypto
        })

        time.sleep(1)


threading.Thread(
    target=pipeline,
    daemon=True
).start()

# ==========================================================
# DASHBOARD STYLING
# ==========================================================
STYLE = {
    "backgroundColor": "#050914",
    "color": "#e5e7eb",
    "fontFamily": "Arial",
    "minHeight": "100vh",
    "padding": "16px"
}

CARD = {
    "background": "rgba(255,255,255,0.05)",
    "border": "1px solid rgba(255,255,255,0.08)",
    "borderRadius": "14px",
    "padding": "14px",
    "marginBottom": "12px"
}

# ==========================================================
# MITIGATION TABLE GENERATOR
# ==========================================================
def generate_mitigation_table(df):
    if df.empty:
        return []

    attack_stats = (
        df.groupby("attack_type")["risk_score"]
        .agg(["mean", "count"])
        .reset_index()
    )

    attack_stats.columns = [
        "attack_type",
        "avg_risk",
        "occurrences"
    ]

    data = []

    for _, row in attack_stats.iterrows():
        attack = row["attack_type"]
        info = MITIGATION_DB.get(attack, {})

        mitigation = info.get(
            "mitigation",
            f"Apply AI monitoring and anomaly detection for {attack}"
        )

        data.append({
            "Threat": attack.replace("_", " ").title(),
            "Avg Risk Score": round(row["avg_risk"], 3),
            "Risk Level": info.get("risk", "High"),
            "Occurrences": int(row["occurrences"]),
            "Potential Impact": (
                f"Potential impact from {attack.replace('_', ' ')}"
            ),
            "Latency Impact": "Medium",
            "Cryptographic Concern": "Potential encryption exposure",
            "AI/ML Relevance": "High",
            "Recommended Mitigation": mitigation
        })

    return sorted(
        data,
        key=lambda item: item["Avg Risk Score"],
        reverse=True
    )


# ==========================================================
# DASHBOARD LAYOUT — NO DATASET UPLOAD
# ==========================================================
app.layout = html.Div(
    style=STYLE,
    children=[
        html.H1(
            "🛡 AO-LACCP SOC Intelligence Platform",
            style={
                "textAlign": "center",
                "color": "#38bdf8"
            }
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("⚙️ Model and Cryptographic Configuration"),

                html.Div(
                    style={
                        "display": "flex",
                        "gap": "15px",
                        "marginTop": "15px"
                    },
                    children=[
                        dcc.Dropdown(
                            id="model_selector",
                            options=[
                                {
                                    "label": "Random Forest",
                                    "value": "RF"
                                },
                                {
                                    "label": "XGBoost",
                                    "value": "XGB",
                                    "disabled": not XGB_AVAILABLE
                                }
                            ],
                            value="RF",
                            clearable=False,
                            style={"flex": 1}
                        ),

                        dcc.Dropdown(
                            id="crypto_selector",
                            options=[
                                {"label": "Auto", "value": "AUTO"},
                                {"label": "PQC", "value": "PQC"},
                                {"label": "RSA", "value": "RSA"},
                                {"label": "Hybrid", "value": "HYBRID"}
                            ],
                            value="AUTO",
                            clearable=False,
                            style={"flex": 1}
                        )
                    ]
                ),

                html.Div(
                    id="crypto_advice",
                    style={"marginTop": "12px"}
                )
            ]
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("🔥 Live Threat Stream"),
                html.Div(id="live_feed")
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px"
            },
            children=[
                dcc.Graph(id="risk_graph"),
                dcc.Graph(id="latency_graph")
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 1fr",
                "gap": "16px"
            },
            children=[
                dcc.Graph(id="soc_attack_stream"),
                dcc.Graph(id="soc_risk_gauge")
            ]
        ),

        html.Div(id="soc_summary", style=CARD),

        html.Div(
            style=CARD,
            children=[
                html.H3("🛠 SOC Mitigation & Action Plan"),

                html.Button(
                    "📥 Export Mitigation Plan CSV",
                    id="export-btn",
                    style={"marginBottom": "10px"}
                ),

                html.Div(
                    id="mitigation_table_container",
                    style={
                        "maxHeight": "620px",
                        "overflow": "auto",
                        "border": "1px solid rgba(255,255,255,0.15)",
                        "borderRadius": "8px"
                    }
                )
            ]
        ),

        dcc.Download(id="download-mitigation"),

        dcc.Interval(
            id="interval",
            interval=2500,
            n_intervals=0
        )
    ]
)

# ==========================================================
# DASHBOARD UPDATE CALLBACK — NO UPLOAD INPUTS
# ==========================================================
@app.callback(
    [
        Output("live_feed", "children"),
        Output("risk_graph", "figure"),
        Output("latency_graph", "figure"),
        Output("soc_attack_stream", "figure"),
        Output("soc_risk_gauge", "figure"),
        Output("crypto_advice", "children"),
        Output("mitigation_table_container", "children"),
        Output("soc_summary", "children")
    ],
    [
        Input("interval", "n_intervals"),
        Input("model_selector", "value"),
        Input("crypto_selector", "value")
    ]
)
def update_dashboard(n, model_type, crypto_mode):
    global crypto_mode_global

    crypto_mode_global = crypto_mode

    df = pd.DataFrame(list(live_stream))

    if df.empty:
        empty = go.Figure()

        return (
            "Initializing threat stream...",
            empty,
            empty,
            empty,
            empty,
            "Initializing cryptographic recommendations...",
            html.Div("Waiting for generated events..."),
            "Waiting for threat data..."
        )

    latest = df.iloc[-1]
    avg_risk = float(df["risk_score"].mean())

    # ------------------------------------------------------
    # MACHINE-LEARNING DEMONSTRATION
    # ------------------------------------------------------
    ml_text = "ML Model: Collecting events..."

    if len(df) > 12:
        try:
            try:
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                )
            except TypeError:
                encoder = OneHotEncoder(
                    sparse=False,
                    handle_unknown="ignore"
                )

            X = encoder.fit_transform(df[["attack_type"]])

            y = (df["risk_score"] > 0.65).astype(int)

            if model_type == "XGB" and XGB_AVAILABLE:
                model = XGBClassifier(
                    n_estimators=80,
                    random_state=42,
                    verbosity=0
                )
                model_name = "XGBoost"
            else:
                model = RandomForestClassifier(
                    n_estimators=80,
                    random_state=42
                )
                model_name = "Random Forest"

            # Train on earlier events, predict the latest event.
            X_train = X[:-1]
            y_train = y.iloc[:-1]
            X_latest = X[[-1]]

            if y_train.nunique() < 2:
                ml_text = (
                    f"{model_name}: Insufficient label diversity"
                )
            else:
                model.fit(X_train, y_train)

                proba = model.predict_proba(X_latest)[0]
                class_indices = list(model.classes_)

                if 1 in class_indices:
                    high_risk_probability = float(
                        proba[class_indices.index(1)]
                    )
                else:
                    high_risk_probability = 0.0

                prediction = (
                    "HIGH RISK"
                    if high_risk_probability > 0.5
                    else "LOW RISK"
                )

                ml_text = (
                    f"{model_name}: {prediction} "
                    f"({high_risk_probability:.1%})"
                )

        except Exception as exc:
            ml_text = f"ML demonstration unavailable: {exc}"

    # ------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------
    risk_fig = px.line(
        df,
        y="risk_score",
        title="Live Risk Score"
    )
    risk_fig.update_layout(
        template="plotly_dark",
        height=340,
        yaxis={"range": [0, 1]}
    )

    latency_fig = px.line(
        df,
        y="latency",
        title="Live Response Latency"
    )
    latency_fig.update_layout(
        template="plotly_dark",
        height=340
    )

    attack_fig = px.histogram(
        df,
        x="attack_type",
        color="attack_type",
        title="Generated Threat Distribution"
    )
    attack_fig.update_layout(
        template="plotly_dark",
        height=380,
        showlegend=False
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_risk,
            title={"text": "Average Risk"},
            gauge={
                "axis": {"range": [0, 1]},
                "steps": [
                    {
                        "range": [0, 0.35],
                        "color": "#16a34a"
                    },
                    {
                        "range": [0.35, 0.7],
                        "color": "#facc15"
                    },
                    {
                        "range": [0.7, 1],
                        "color": "#ef4444"
                    }
                ]
            }
        )
    )

    gauge.update_layout(
        template="plotly_dark",
        height=380
    )

    # ------------------------------------------------------
    # CRYPTOGRAPHIC RECOMMENDATION
    # ------------------------------------------------------
    advice = (
        "⚠️ High risk: PQC recommended for demonstration"
        if avg_risk > 0.7
        else "⚡ Elevated risk: Review hybrid posture"
        if avg_risk > 0.5
        else "✅ Lower simulated risk: Review policy requirements"
    )

    # ------------------------------------------------------
    # MITIGATION TABLE
    # ------------------------------------------------------
    table_data = generate_mitigation_table(df)

    table = dash_table.DataTable(
        data=table_data,
        page_size=12,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1e2937",
            "color": "white",
            "fontWeight": "bold"
        },
        style_data={
            "backgroundColor": "#0f172a",
            "color": "#e2e8f0",
            "whiteSpace": "normal",
            "height": "auto"
        }
    )

    # ------------------------------------------------------
    # LIVE SUMMARY
    # ------------------------------------------------------
    summary = html.Div([
        html.H3("📊 SOC Performance Summary"),
        html.P(f"Generated events retained: {len(df)}"),
        html.P(f"Average simulated risk: {avg_risk:.3f}"),
        html.P(
            f"Average simulated latency: "
            f"{df['latency'].mean():.2f} ms"
        ),
        html.P(f"Latest threat: {latest['attack_type']}"),
        html.P(f"Latest cryptographic label: {latest['crypto']}"),
        html.P(ml_text),
        html.P(
            "Research note: events and labels are synthetic. "
            "Model outputs and cryptographic recommendations "
            "are demonstrative, not validated production results."
        )
    ])

    return (
        f"{latest['attack_type'].upper()} | {latest['crypto']} | {ml_text}",
        risk_fig,
        latency_fig,
        attack_fig,
        gauge,
        advice,
        table,
        summary
    )


# ==========================================================
# EXPORT MITIGATION PLAN
# ==========================================================
@app.callback(
    Output("download-mitigation", "data"),
    Input("export-btn", "n_clicks"),
    prevent_initial_call=True
)
def export_mitigation(n_clicks):
    if not n_clicks:
        return None

    df = pd.DataFrame(list(live_stream))

    if df.empty:
        return None

    table_data = generate_mitigation_table(df)

    return dcc.send_data_frame(
        pd.DataFrame(table_data).to_csv,
        "AO-LACCP_Mitigation_Plan.csv",
        index=False
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================
if __name__ == "__main__":
    print("AO-LACCP SOC Dashboard started")
    print("Open: http://127.0.0.1:8050")

    app.run(
        debug=False,
        host="0.0.0.0",
        port=8050
    )
