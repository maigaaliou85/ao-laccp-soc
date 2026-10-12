import base64
import io
import random
import threading
import time
from collections import deque

import dash
from dash import dcc, html, Input, Output, State, dash_table
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ==========================================================
# App configuration
# ==========================================================
app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

MAX_STREAM_ROWS = 400
MAX_UPLOAD_ROWS = 3000
ATTACK_TYPES = [
    "brute_force",
    "MITM",
    "zero_day",
    "ransomware",
    "phishing",
    "data_exfiltration",
]

live_stream = deque(maxlen=MAX_STREAM_ROWS)
uploaded_dataset_name = "No dataset uploaded"
last_uploaded_signature = None
crypto_mode_global = "AUTO"
stream_lock = threading.Lock()
pipeline_started = False

MITIGATION_DB = {
    "brute_force": {
        "risk": "High",
        "mitigation": "Rate limiting, behavioral CAPTCHA, and anomaly detection",
    },
    "MITM": {
        "risk": "Critical",
        "mitigation": "Enforce TLS 1.3, certificate validation/pinning where appropriate, and traffic inspection",
    },
    "zero_day": {
        "risk": "Critical",
        "mitigation": "Behavioral detection, threat hunting, and risk-based containment",
    },
    "ransomware": {
        "risk": "Critical",
        "mitigation": "Immutable backups, endpoint detection, and Zero Trust controls",
    },
    "phishing": {
        "risk": "High",
        "mitigation": "Email filtering, MFA, and security-awareness training",
    },
    "data_exfiltration": {
        "risk": "High",
        "mitigation": "DLP, outbound anomaly detection, and data-flow monitoring",
    },
}


# ==========================================================
# Cryptographic decision logic
# Note: overhead is a simulation parameter, not a benchmark.
# ==========================================================
def crypto_engine(mode, risk):
    risk = float(np.clip(risk, 0.0, 1.0))

    if mode == "PQC":
        return "PQC", 0.10
    if mode == "RSA":
        return "RSA", 0.30
    if mode == "HYBRID":
        return ("PQC+AES", 0.08) if risk > 0.60 else ("RSA+AES", 0.20)

    # AUTO mode: illustrative policy thresholds.
    if risk > 0.75:
        return "PQC-AUTO", 0.10
    if risk > 0.45:
        return "RSA-AUTO", 0.30
    return "AES-256", 0.05


def make_stream_event():
    """Create one illustrative synthetic event; not real SIEM telemetry."""
    base_risk = random.random()
    latency = random.uniform(50, 400)
    attack_type = random.choice(ATTACK_TYPES)
    crypto, crypto_overhead = crypto_engine(crypto_mode_global, base_risk)

    # Keep the observed risk score separate from crypto overhead.
    # The overhead is retained as its own field for display/analysis.
    return {
        "timestamp": pd.Timestamp.now(),
        "attack_type": attack_type,
        "risk_score": round(float(np.clip(base_risk, 0.0, 1.0)), 4),
        "latency": round(float(latency), 2),
        "crypto": crypto,
        "crypto_overhead": crypto_overhead,
        "source": "Synthetic stream",
    }


def pipeline():
    while True:
        with stream_lock:
            live_stream.append(make_stream_event())
        time.sleep(1)


def start_pipeline_once():
    global pipeline_started
    if not pipeline_started:
        thread = threading.Thread(target=pipeline, daemon=True, name="ao-laccp-synthetic-stream")
        thread.start()
        pipeline_started = True


# ==========================================================
# Dataset upload and normalization
# ==========================================================
def parse_upload(contents):
    if not contents:
        return pd.DataFrame(), "", "No file selected."

    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8-sig", errors="replace")))

        if df.empty:
            return pd.DataFrame(), "", "The uploaded CSV contains no data rows."

        original_rows = len(df)
        warning = ""
        if original_rows > MAX_UPLOAD_ROWS:
            df = df.sample(MAX_UPLOAD_ROWS, random_state=42).reset_index(drop=True)
            warning = f" (sampled {MAX_UPLOAD_ROWS:,} of {original_rows:,} rows)"

        # Normalize column names so common variants can be recognized.
        df.columns = [str(c).strip() for c in df.columns]
        lower_to_original = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

        # Recognize a few common column aliases.
        aliases = {
            "attack_type": ["attack", "attacktype", "threat_type", "label", "class"],
            "risk_score": ["risk", "risk_score", "risk_score_value", "anomaly_score"],
            "latency": ["latency_ms", "response_time", "response_time_ms", "delay"],
        }
        rename = {}
        for canonical, possible_names in aliases.items():
            if canonical in df.columns:
                continue
            for alias in possible_names:
                if alias in lower_to_original:
                    rename[lower_to_original[alias]] = canonical
                    break
        if rename:
            df = df.rename(columns=rename)

        if "attack_type" not in df.columns:
            df["attack_type"] = np.random.choice(ATTACK_TYPES, size=len(df))

        df["attack_type"] = df["attack_type"].fillna("unknown").astype(str).str.strip()
        df.loc[df["attack_type"] == "", "attack_type"] = "unknown"

        if "risk_score" not in df.columns:
            df["risk_score"] = np.random.uniform(0.10, 0.95, size=len(df))
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
        df["risk_score"] = df["risk_score"].fillna(0.50)

        # If the source uses percentages on a 0–100 scale, normalize them.
        if not df["risk_score"].empty and df["risk_score"].max() > 1.0:
            if df["risk_score"].between(0, 100).all():
                df["risk_score"] = df["risk_score"] / 100.0
        df["risk_score"] = df["risk_score"].clip(0.0, 1.0)

        if "latency" not in df.columns:
            df["latency"] = np.random.uniform(50, 400, size=len(df))
        df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
        df["latency"] = df["latency"].fillna(150.0).clip(lower=0.0)

        df["crypto"] = "UPLOAD"
        df["crypto_overhead"] = np.nan
        df["source"] = "Uploaded dataset"
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NaT

        return df[[
            "timestamp", "attack_type", "risk_score", "latency",
            "crypto", "crypto_overhead", "source"
        ]], warning, ""

    except Exception as exc:
        return pd.DataFrame(), "", f"Could not read CSV: {exc}"


# ==========================================================
# Mitigation recommendations
# ==========================================================
def generate_mitigation_table(df):
    if df is None or df.empty:
        return []

    required = {"attack_type", "risk_score"}
    if not required.issubset(df.columns):
        return []

    working = df.copy()
    working["risk_score"] = pd.to_numeric(working["risk_score"], errors="coerce")
    working = working.dropna(subset=["risk_score"])
    if working.empty:
        return []

    attack_stats = (
        working.groupby("attack_type", dropna=False)["risk_score"]
        .agg(["mean", "count"])
        .reset_index()
    )
    attack_stats.columns = ["attack_type", "avg_risk", "occurrences"]

    rows = []
    for _, row in attack_stats.iterrows():
        attack = str(row["attack_type"])
        info = MITIGATION_DB.get(attack, {})
        avg_risk = float(row["avg_risk"])

        if avg_risk >= 0.75:
            risk_level = "Critical"
        elif avg_risk >= 0.50:
            risk_level = "High"
        elif avg_risk >= 0.30:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        rows.append({
            "Threat": attack.replace("_", " ").title(),
            "Avg Risk Score": round(avg_risk, 3),
            "Risk Level": info.get("risk", risk_level),
            "Occurrences": int(row["occurrences"]),
            "Potential Impact": f"Potential impact associated with {attack.replace('_', ' ')}",
            "Latency Impact": "Review observed event latency",
            "Cryptographic Concern": "Assess protection requirements and crypto-agility needs",
            "AI/ML Relevance": "Use validated detection models and monitor model drift",
            "Recommended Mitigation": info.get(
                "mitigation",
                f"Investigate {attack.replace('_', ' ')}; apply monitoring, containment, and recovery controls",
            ),
        })

    return sorted(rows, key=lambda x: x["Avg Risk Score"], reverse=True)


# ==========================================================
# Visual style
# ==========================================================
STYLE = {
    "backgroundColor": "#050914",
    "color": "#e5e7eb",
    "fontFamily": "Arial, sans-serif",
    "minHeight": "100vh",
    "padding": "16px",
}
CARD = {
    "background": "rgba(255,255,255,0.05)",
    "border": "1px solid rgba(255,255,255,0.08)",
    "borderRadius": "14px",
    "padding": "14px",
    "marginBottom": "12px",
}
DROPDOWN_STYLE = {"color": "#111827", "backgroundColor": "#ffffff"}


# ==========================================================
# Layout
# ==========================================================
app.layout = html.Div(style=STYLE, children=[
    html.H1(
        "🛡 AO-LACCP SOC Intelligence Platform",
        style={"textAlign": "center", "color": "#38bdf8"},
    ),
    html.P(
        "Demonstration mode: the built-in live stream uses synthetic events, not live security telemetry.",
        style={"textAlign": "center", "color": "#fbbf24"},
    ),

    html.Div(style=CARD, children=[
        dcc.Upload(
            id="upload_data",
            children=html.Div("📁 Upload CSV Dataset (up to 3,000 rows sampled)"),
            style={
                "border": "2px dashed #38bdf8",
                "padding": "20px",
                "textAlign": "center",
                "cursor": "pointer",
                "borderRadius": "8px",
            },
            multiple=False,
        ),
        html.Div(id="dataset_name", style={"marginTop": "10px", "fontWeight": "bold"}),
        html.Div(id="upload_status", style={"marginTop": "6px", "color": "#fbbf24"}),
        html.Div(
            style={"display": "flex", "gap": "15px", "marginTop": "15px", "flexWrap": "wrap"},
            children=[
                html.Div(style={"flex": "1", "minWidth": "220px"}, children=[
                    html.Label("ML model"),
                    dcc.Dropdown(
                        id="model_selector",
                        options=[
                            {"label": "Random Forest", "value": "RF"},
                            {"label": "XGBoost", "value": "XGB", "disabled": not XGB_AVAILABLE},
                        ],
                        value="RF",
                        clearable=False,
                        style=DROPDOWN_STYLE,
                    ),
                ]),
                html.Div(style={"flex": "1", "minWidth": "220px"}, children=[
                    html.Label("Cryptographic policy"),
                    dcc.Dropdown(
                        id="crypto_selector",
                        options=[
                            {"label": "Auto", "value": "AUTO"},
                            {"label": "PQC", "value": "PQC"},
                            {"label": "RSA", "value": "RSA"},
                            {"label": "Hybrid", "value": "HYBRID"},
                        ],
                        value="AUTO",
                        clearable=False,
                        style=DROPDOWN_STYLE,
                    ),
                ]),
            ],
        ),
        html.Div(id="crypto_advice", style={"marginTop": "12px"}),
    ]),

    html.Div(style=CARD, children=[
        html.H3("🔥 Live Threat Stream"),
        html.Div(id="live_feed"),
    ]),

    html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))", "gap": "16px"},
        children=[dcc.Graph(id="risk_graph"), dcc.Graph(id="latency_graph")],
    ),

    html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))", "gap": "16px"},
        children=[dcc.Graph(id="soc_attack_stream"), dcc.Graph(id="soc_risk_gauge")],
    ),

    html.Div(id="soc_summary", style=CARD),

    html.Div(style=CARD, children=[
        html.H3("🛠 SOC Mitigation & Action Plan"),
        html.Button("📥 Export Mitigation Plan as CSV", id="export-btn", n_clicks=0,
                    style={"marginBottom": "10px", "padding": "9px 14px", "cursor": "pointer"}),
        html.Div(
            id="mitigation_table_container",
            style={
                "maxHeight": "620px",
                "overflow": "auto",
                "border": "1px solid rgba(255,255,255,0.15)",
                "borderRadius": "8px",
            },
        ),
    ]),

    dcc.Download(id="download-mitigation"),
    dcc.Interval(id="interval", interval=2500, n_intervals=0),
])


# ==========================================================
# Dashboard update
# ==========================================================
@app.callback(
    [
        Output("live_feed", "children"),
        Output("risk_graph", "figure"),
        Output("latency_graph", "figure"),
        Output("soc_attack_stream", "figure"),
        Output("soc_risk_gauge", "figure"),
        Output("dataset_name", "children"),
        Output("upload_status", "children"),
        Output("crypto_advice", "children"),
        Output("soc_summary", "children"),
        Output("mitigation_table_container", "children"),
    ],
    [
        Input("interval", "n_intervals"),
        Input("model_selector", "value"),
        Input("crypto_selector", "value"),
        Input("upload_data", "contents"),
    ],
    [State("upload_data", "filename")],
)
def update_dashboard(n, model_type, crypto_mode, upload_content, filename):
    global uploaded_dataset_name, crypto_mode_global, last_uploaded_signature

    crypto_mode_global = crypto_mode or "AUTO"
    upload_status = ""

    # Only import a selected upload once, not on every interval tick.
    if upload_content:
        signature = (filename, hash(upload_content))
        if signature != last_uploaded_signature:
            df_up, warning, error = parse_upload(upload_content)
            if error:
                upload_status = error
            elif not df_up.empty:
                with stream_lock:
                    # Keep the most recent MAX_STREAM_ROWS records from the upload.
                    for record in df_up.tail(MAX_STREAM_ROWS).to_dict("records"):
                        live_stream.append(record)
                uploaded_dataset_name = (filename or "Uploaded dataset") + warning
                last_uploaded_signature = signature
                upload_status = (
                    f"Loaded {min(len(df_up), MAX_STREAM_ROWS):,} row(s) into the recent-event stream. "
                    "Missing fields, if any, may have been generated for demonstration."
                )

    with stream_lock:
        df = pd.DataFrame(list(live_stream))

    if df.empty:
        empty = go.Figure()
        empty.update_layout(template="plotly_dark")
        return (
            "Initializing synthetic stream...", empty, empty, empty, empty,
            uploaded_dataset_name, upload_status, "Initializing policy engine...",
            html.Div("Waiting for events..."), html.Div("Waiting for data..."),
        )

    df = df.copy()
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.5).clip(0, 1)
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce").fillna(150).clip(lower=0)
    df["attack_type"] = df["attack_type"].fillna("unknown").astype(str)

    latest = df.iloc[-1]
    avg_risk = float(df["risk_score"].mean())
    max_risk = float(df["risk_score"].max())
    avg_latency = float(df["latency"].mean())

    # Illustrative ML demonstration. This does not establish real-world model performance.
    ml_text = "ML model: waiting for sufficient events"
    if len(df) >= 15:
        try:
            features = df[["attack_type", "latency"]].copy()
            features["attack_type"] = features["attack_type"].astype(str)
            X = pd.get_dummies(features, columns=["attack_type"], dtype=float)
            y = (df["risk_score"] >= 0.65).astype(int)

            X_train, X_latest = X.iloc[:-1], X.iloc[[-1]]
            y_train = y.iloc[:-1]

            if y_train.nunique() < 2:
                ml_text = "ML model: insufficient class diversity to train"
            else:
                if model_type == "XGB" and XGB_AVAILABLE:
                    model = XGBClassifier(
                        n_estimators=80,
                        max_depth=3,
                        learning_rate=0.08,
                        random_state=42,
                        eval_metric="logloss",
                        verbosity=0,
                    )
                    model_name = "XGBoost"
                else:
                    model = RandomForestClassifier(
                        n_estimators=80,
                        random_state=42,
                        class_weight="balanced",
                    )
                    model_name = "Random Forest"

                model.fit(X_train, y_train)
                # Align feature columns between training data and latest row.
                X_latest = X_latest.reindex(columns=X_train.columns, fill_value=0)
                class_index = list(model.classes_).index(1)
                probability = float(model.predict_proba(X_latest)[0][class_index])
                prediction = "HIGH RISK" if probability >= 0.5 else "LOW RISK"
                ml_text = (
                    f"{model_name} demonstration estimate: {prediction} "
                    f"({probability:.1%}); synthetic/mixed data, not validated"
                )
        except Exception as exc:
            ml_text = f"ML model unavailable: {type(exc).__name__}"

    # Risk trend
    risk_fig = px.line(df, y="risk_score", title="Observed Risk Score")
    risk_fig.update_layout(template="plotly_dark", height=340, yaxis={"range": [0, 1]})

    # Latency trend
    latency_fig = px.line(df, y="latency", title="Observed Latency (ms)")
    latency_fig.update_layout(template="plotly_dark", height=340)

    # Attack distribution
    attack_counts = (
        df["attack_type"].value_counts().rename_axis("attack_type").reset_index(name="count")
    )
    attack_fig = px.bar(
        attack_counts,
        x="attack_type",
        y="count",
        color="attack_type",
        title="Threat Events by Attack Type",
    )
    attack_fig.update_layout(template="plotly_dark", height=380, showlegend=False)

    # Average-risk gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_risk,
        number={"valueformat": ".3f"},
        title={"text": "Mean Risk Score"},
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.35], "color": "#16a34a"},
                {"range": [0.35, 0.70], "color": "#facc15"},
                {"range": [0.70, 1.0], "color": "#ef4444"},
            ],
            "bar": {"color": "#38bdf8"},
        },
    ))
    gauge.update_layout(template="plotly_dark", height=380)

    if avg_risk >= 0.70:
        advice = "⚠ Elevated mean risk: review high-risk events and evaluate stronger controls."
    elif avg_risk >= 0.50:
        advice = "⚡ Moderate mean risk: review hybrid-policy suitability and operational constraints."
    else:
        advice = "✅ Mean risk is currently lower under the displayed event stream."

    mitigation_data = generate_mitigation_table(df)
    mitigation_table = dash_table.DataTable(
        data=mitigation_data,
        columns=[{"name": col, "id": col} for col in (
            mitigation_data[0].keys() if mitigation_data else [
                "Threat", "Avg Risk Score", "Risk Level", "Occurrences",
                "Potential Impact", "Latency Impact", "Cryptographic Concern",
                "AI/ML Relevance", "Recommended Mitigation",
            ]
        )],
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "minWidth": "100%"},
        style_header={
            "backgroundColor": "#1e293b",
            "color": "white",
            "fontWeight": "bold",
            "whiteSpace": "normal",
            "position": "sticky",
            "top": 0,
        },
        style_cell={
            "backgroundColor": "#0f172a",
            "color": "#e2e8f0",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "110px",
            "maxWidth": "280px",
            "padding": "8px",
            "textAlign": "left",
        },
    )

    latest_timestamp = latest.get("timestamp")
    if pd.notna(latest_timestamp):
        latest_time_text = str(latest_timestamp)
    else:
        latest_time_text = "time unavailable"

    feed = html.Div([
        html.Div(
            f"{str(latest['attack_type']).upper()} | Risk: {float(latest['risk_score']):.3f} | "
            f"Latency: {float(latest['latency']):.1f} ms | Crypto: {latest.get('crypto', 'N/A')}",
            style={"fontWeight": "bold", "color": "#38bdf8"},
        ),
        html.Div(f"Event time: {latest_time_text} | {ml_text}", style={"marginTop": "6px"}),
        html.Div(
            f"Event source: {latest.get('source', 'Unknown')}",
            style={"fontSize": "12px", "color": "#9ca3af", "marginTop": "4px"},
        ),
    ])

    summary = html.Div([
        html.H3("SOC Snapshot"),
        html.Div(
            f"Events displayed: {len(df)} | Mean risk: {avg_risk:.3f} | "
            f"Maximum risk: {max_risk:.3f} | Mean latency: {avg_latency:.1f} ms",
        ),
        html.Div(
            "Interpretation note: uploaded records and synthetic events may coexist in the stream. "
            "Treat the figures as a prototype demonstration unless the data source and model are validated.",
            style={"color": "#fbbf24", "marginTop": "8px"},
        ),
    ])

    return (
        feed,
        risk_fig,
        latency_fig,
        attack_fig,
        gauge,
        f"Dataset: {uploaded_dataset_name}",
        upload_status,
        advice,
        summary,
        mitigation_table,
    )


# ==========================================================
# Export the currently displayed stream's mitigation plan
# ==========================================================
@app.callback(
    Output("download-mitigation", "data"),
    Input("export-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_mitigation(n_clicks):
    if not n_clicks:
        return None

    with stream_lock:
        df = pd.DataFrame(list(live_stream))

    if df.empty:
        return None

    table_data = generate_mitigation_table(df)
    export_df = pd.DataFrame(table_data)
    if export_df.empty:
        return None

    return dcc.send_data_frame(
        export_df.to_csv,
        "AO-LACCP_Mitigation_Plan.csv",
        index=False,
    )


# Start the demonstration stream on direct launch and WSGI import.
# For multi-worker deployments, replace this in-process thread with a shared event source.
start_pipeline_once()

# ==========================================================
if __name__ == "__main__":
    print("AO-LACCP SOC Dashboard started.")
    print("Synthetic demonstration stream enabled; no live SIEM connection is configured.")
    print("Open: http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
