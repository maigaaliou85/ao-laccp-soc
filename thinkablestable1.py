
import base64
import io
import threading
import time
import random
import traceback
from collections import deque

import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ==========================================================
# CONFIGURATION
# ==========================================================

app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DATASET_ROWS = 3000
MAX_CHART_ROWS = 500

live_stream = deque(maxlen=200)
stream_lock = threading.Lock()

ATTACK_TYPES = [
    "brute_force", "MITM", "zero_day",
    "ransomware", "phishing", "data_exfiltration"
]

MITIGATION_DB = {
    "brute_force": "Rate limiting, behavioral CAPTCHA, AI anomaly detection",
    "MITM": "TLS 1.3 enforcement, certificate pinning, AI traffic inspection",
    "zero_day": "Behavioral analysis, rapid patching, threat hunting",
    "ransomware": "Immutable backups, endpoint detection, Zero Trust",
    "phishing": "AI email filtering, MFA, security awareness training",
    "data_exfiltration": "DLP and outbound anomaly detection",
}

STYLE = {
    "backgroundColor": "#050914",
    "color": "#e5e7eb",
    "fontFamily": "Arial",
    "minHeight": "100vh",
    "padding": "16px",
}

CARD = {
    "background": "rgba(255,255,255,0.05)",
    "border": "1px solid rgba(255,255,255,0.12)",
    "borderRadius": "12px",
    "padding": "16px",
    "marginBottom": "14px",
}

GRAPH_CONFIG = {"displaylogo": False, "responsive": True}


# ==========================================================
# CRYPTOGRAPHIC ENGINE
# ==========================================================

def crypto_engine(mode, risk):
    if mode == "PQC":
        return "PQC", 0.10
    if mode == "RSA":
        return "RSA", 0.30
    if mode == "HYBRID":
        return ("PQC+AES", 0.08) if risk > 0.60 else ("RSA+AES", 0.20)

    if risk > 0.75:
        return "PQC-AUTO", 0.10
    if risk > 0.45:
        return "RSA-AUTO", 0.30
    return "AES-256", 0.05


# ==========================================================
# SYNTHETIC LIVE STREAM
# ==========================================================

def pipeline():
    while True:
        risk = random.random()
        latency = random.uniform(50, 400)
        crypto, penalty = crypto_engine("AUTO", risk)

        event = {
            "attack_type": random.choice(ATTACK_TYPES),
            "risk_score": min(1.0, risk + penalty),
            "latency": latency,
            "crypto": crypto,
        }

        with stream_lock:
            live_stream.append(event)

        time.sleep(1)


# ==========================================================
# DATASET NORMALIZATION
# ==========================================================

def normalize_dataset(df):
    df = df.copy()

    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    aliases = {
        "attack_type": [
            "attack", "attacktype", "label", "class",
            "type", "threat", "category", "event_type"
        ],
        "risk_score": [
            "risk", "risk_probability", "risk_prob",
            "anomaly_score", "threat_score"
        ],
        "latency": [
            "response_time", "latency_ms", "delay",
            "processing_time", "response_latency"
        ],
    }

    for target, names in aliases.items():
        if target not in df.columns:
            for name in names:
                if name in df.columns:
                    df.rename(columns={name: target}, inplace=True)
                    break

    notes = []

    if "attack_type" not in df.columns:
        df["attack_type"] = "uploaded_event"
        notes.append(
            "No attack label found. Events are labeled uploaded_event."
        )
    else:
        df["attack_type"] = (
            df["attack_type"].fillna("unknown").astype(str).str.strip()
        )
        df.loc[df["attack_type"] == "", "attack_type"] = "unknown"

    if "risk_score" not in df.columns:
        df["risk_score"] = 0.5
        notes.append(
            "Risk score missing: placeholder 0.5 used, not a measured score."
        )
    else:
        risk = pd.to_numeric(df["risk_score"], errors="coerce")
        risk = risk.replace([np.inf, -np.inf], np.nan)

        valid = risk.dropna()
        if not valid.empty and valid.min() >= 0 and valid.max() > 1:
            if valid.max() <= 100:
                risk = risk / 100.0
                notes.append("Risk scores converted from percentages.")
            else:
                risk[:] = np.nan

        risk = risk.where(risk.between(0, 1))
        if risk.notna().any():
            risk = risk.fillna(risk.median())
        else:
            risk = pd.Series(0.5, index=df.index)
            notes.append("Invalid risk scores replaced with placeholder 0.5.")

        df["risk_score"] = risk.astype(float)

    if "latency" not in df.columns:
        df["latency"] = 150.0
        notes.append(
            "Latency missing: placeholder 150 ms used, not a measurement."
        )
    else:
        latency = pd.to_numeric(df["latency"], errors="coerce")
        latency = latency.replace([np.inf, -np.inf], np.nan)
        latency = latency.where(latency >= 0)

        if latency.notna().any():
            latency = latency.fillna(latency.median())
        else:
            latency = pd.Series(150.0, index=df.index)
            notes.append("Invalid latency replaced with placeholder 150 ms.")

        df["latency"] = latency.astype(float)

    return df.reset_index(drop=True), notes


# ==========================================================
# FILE UPLOAD PARSER
# ==========================================================

def parse_upload(contents, filename):
    if not contents or not filename:
        raise ValueError("Select a dataset first.")

    extension = filename.lower().rsplit(".", 1)[-1]
    if extension not in ("csv", "xlsx", "xls"):
        raise ValueError("Supported formats: CSV, XLSX, XLS.")

    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("The uploaded file could not be decoded.") from exc

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 10 MB upload limit.")

    if not raw:
        raise ValueError("The file is empty.")

    try:
        if extension == "csv":
            df = pd.read_csv(
                io.BytesIO(raw),
                nrows=MAX_DATASET_ROWS + 1
            )
        elif extension == "xlsx":
            df = pd.read_excel(
                io.BytesIO(raw),
                nrows=MAX_DATASET_ROWS + 1,
                engine="openpyxl"
            )
        else:
            df = pd.read_excel(
                io.BytesIO(raw),
                nrows=MAX_DATASET_ROWS + 1
            )
    except Exception as exc:
        raise ValueError(
            "Could not read this file. Check its format and install "
            "the required Excel reader if needed."
        ) from exc

    if df.empty:
        raise ValueError("The dataset contains no data rows.")

    notes = []
    if len(df) > MAX_DATASET_ROWS:
        df = df.iloc[:MAX_DATASET_ROWS].copy()
        notes.append(f"Limited to {MAX_DATASET_ROWS:,} rows.")

    df, normalization_notes = normalize_dataset(df)
    notes.extend(normalization_notes)

    return df, notes


# ==========================================================
# MITIGATION TABLE
# ==========================================================

def generate_mitigation_table(df):
    if df.empty:
        return []

    grouped = (
        df.groupby("attack_type")["risk_score"]
        .agg(["mean", "count"])
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():
        attack = str(row["attack_type"])
        risk = float(row["mean"])

        if risk >= 0.70:
            level = "Critical"
        elif risk >= 0.50:
            level = "High"
        elif risk >= 0.30:
            level = "Moderate"
        else:
            level = "Low"

        results.append({
            "Threat": attack.replace("_", " ").title(),
            "Average Risk": round(risk, 3),
            "Risk Level": level,
            "Occurrences": int(row["count"]),
            "Recommended Mitigation": MITIGATION_DB.get(
                attack, "Investigate the event and validate appropriate controls."
            ),
        })

    return sorted(
        results,
        key=lambda x: x["Average Risk"],
        reverse=True
    )


# ==========================================================
# LAYOUT
# ==========================================================

app.layout = html.Div(style=STYLE, children=[
    html.H1(
        "AO-LACCP SOC Intelligence Platform",
        style={"textAlign": "center", "color": "#38bdf8"}
    ),

    # Critical: stores uploaded records for the graph callback.
    dcc.Store(id="dataset_store", storage_type="memory"),

    html.Div(style=CARD, children=[
        html.H3("Upload Dataset"),

        dcc.Upload(
            id="upload_data",
            children=html.Div([
                "📁 Select or drag a CSV / Excel dataset here",
                html.Br(),
                html.Small("Maximum size: 10 MB; maximum rows: 3,000")
            ]),
            accept=".csv,.xlsx,.xls",
            multiple=False,
            style={
                "border": "2px dashed #38bdf8",
                "padding": "24px",
                "textAlign": "center",
                "cursor": "pointer",
                "borderRadius": "10px",
            },
        ),

        html.Div(
            id="dataset_name",
            style={"marginTop": "12px", "whiteSpace": "pre-wrap"}
        ),

        html.Div(
            style={
                "display": "flex",
                "gap": "14px",
                "flexWrap": "wrap",
                "marginTop": "16px",
            },
            children=[
                html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                    html.Label("ML Model"),
                    dcc.Dropdown(
                        id="model_selector",
                        options=[
                            {"label": "Random Forest", "value": "RF"},
                            {
                                "label": "XGBoost",
                                "value": "XGB",
                                "disabled": not XGB_AVAILABLE
                            }
                        ],
                        value="RF",
                        clearable=False
                    )
                ]),
                html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                    html.Label("Cryptographic Mode"),
                    dcc.Dropdown(
                        id="crypto_selector",
                        options=[
                            {"label": "Auto", "value": "AUTO"},
                            {"label": "PQC", "value": "PQC"},
                            {"label": "RSA", "value": "RSA"},
                            {"label": "Hybrid", "value": "HYBRID"}
                        ],
                        value="AUTO",
                        clearable=False
                    )
                ])
            ]
        ),
        html.Div(id="crypto_advice", style={"marginTop": "12px"})
    ]),

    html.Div(style=CARD, children=[
        html.H3("Threat Stream"),
        html.Div(id="live_feed")
    ]),

    html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
            "gap": "14px"
        },
        children=[
            dcc.Graph(id="risk_graph", config=GRAPH_CONFIG),
            dcc.Graph(id="latency_graph", config=GRAPH_CONFIG),
        ]
    ),

    html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
            "gap": "14px"
        },
        children=[
            dcc.Graph(id="soc_attack_stream", config=GRAPH_CONFIG),
            dcc.Graph(id="soc_risk_gauge", config=GRAPH_CONFIG),
        ]
    ),

    html.Div(id="soc_summary", style=CARD),

    html.Div(style=CARD, children=[
        html.H3("SOC Mitigation & Action Plan"),
        html.Button("Export as CSV", id="export-btn", n_clicks=0),
        html.Div(
            id="mitigation_table_container",
            style={"overflowX": "auto", "marginTop": "12px"}
        )
    ]),

    dcc.Download(id="download-mitigation"),
    dcc.Interval(id="interval", interval=2500, n_intervals=0),
])


# ==========================================================
# UPLOAD CALLBACK
# This stores the uploaded dataset and triggers the graphs.
# ==========================================================

@app.callback(
    Output("dataset_store", "data"),
    Output("dataset_name", "children"),
    Input("upload_data", "contents"),
    State("upload_data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents:
        raise PreventUpdate

    try:
        df, notes = parse_upload(contents, filename)

        # JSON-compatible records are stored in the browser session.
        records = df.to_dict("records")

        message = (
            f"UPLOAD SUCCESSFUL: {filename}\n"
            f"Rows loaded: {len(df):,}\n"
            f"Columns: {len(df.columns)}\n"
            f"Detected columns: {', '.join(df.columns.astype(str))}"
        )

        if notes:
            message += "\n\nNotes:\n- " + "\n- ".join(notes)

        return records, message

    except Exception as exc:
        traceback.print_exc()
        return no_update, f"UPLOAD FAILED: {exc}"


# ==========================================================
# ML PREDICTION
# ==========================================================

def predict_risk(df, model_type):
    if len(df) < 20:
        return "ML prediction unavailable: at least 20 rows recommended."

    y = (df["risk_score"] > 0.65).astype(int)

    if y.nunique() < 2:
        return "ML prediction unavailable: only one risk class is present."

    try:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        X = encoder.fit_transform(df[["attack_type"]].astype(str))

        if model_type == "XGB" and XGB_AVAILABLE:
            model = XGBClassifier(
                n_estimators=40,
                max_depth=3,
                random_state=42,
                verbosity=0,
                n_jobs=1,
                eval_metric="logloss"
            )
            model_name = "XGBoost"
        else:
            model = RandomForestClassifier(
                n_estimators=40,
                max_depth=8,
                random_state=42,
                n_jobs=1
            )
            model_name = "Random Forest"

        model.fit(X[:-1], y.iloc[:-1])
        classes = list(model.classes_)

        probability = (
            float(model.predict_proba(X[-1:])[0][classes.index(1)])
            if 1 in classes else 0.0
        )

        label = "HIGH RISK" if probability > 0.5 else "LOW RISK"

        return f"{model_name}: {label} ({probability:.1%})"

    except Exception as exc:
        print("ML prediction error:", exc)
        return "ML prediction unavailable; see Python terminal."


# ==========================================================
# GRAPH CALLBACK
# Reads the dataset from dataset_store.
# Uploading a dataset directly triggers this callback.
# ==========================================================

@app.callback(
    Output("live_feed", "children"),
    Output("risk_graph", "figure"),
    Output("latency_graph", "figure"),
    Output("soc_attack_stream", "figure"),
    Output("soc_risk_gauge", "figure"),
    Output("crypto_advice", "children"),
    Output("soc_summary", "children"),
    Output("mitigation_table_container", "children"),

    Input("interval", "n_intervals"),
    Input("dataset_store", "data"),
    Input("model_selector", "value"),
    Input("crypto_selector", "value"),
)
def update_dashboard(n, stored_data, model_type, crypto_mode):

    # Uploaded dataset takes priority over generated events.
    if stored_data:
        df = pd.DataFrame(stored_data)
        source_name = "Uploaded dataset"
        source_type = "User-provided data"
    else:
        with stream_lock:
            df = pd.DataFrame(list(live_stream))

        source_name = "Synthetic demonstration stream"
        source_type = "Generated demonstration data"

    empty = go.Figure()
    empty.update_layout(template="plotly_dark")

    if df.empty:
        return (
            "Waiting for data...",
            empty, empty, empty, empty,
            "No data available.",
            "Upload a dataset to display the analysis.",
            html.Div("No data available.")
        )

    # Ensure numerical types after reading records from dcc.Store.
    df["risk_score"] = pd.to_numeric(
        df["risk_score"], errors="coerce"
    ).fillna(0.5).clip(0, 1)

    df["latency"] = pd.to_numeric(
        df["latency"], errors="coerce"
    ).fillna(150).clip(lower=0)

    df["attack_type"] = (
        df["attack_type"].fillna("unknown").astype(str)
    )

    chart_df = df.tail(MAX_CHART_ROWS).copy()

    avg_risk = float(df["risk_score"].mean())
    avg_latency = float(df["latency"].mean())

    # 1. Risk score graph
    risk_fig = px.line(
        chart_df,
        x=chart_df.index,
        y="risk_score",
        title="Risk Score Across Dataset Records",
        labels={"x": "Record", "risk_score": "Risk Score"},
    )
    risk_fig.update_layout(
        template="plotly_dark",
        height=350,
        yaxis_range=[0, 1],
    )

    # 2. Latency graph
    latency_fig = px.line(
        chart_df,
        x=chart_df.index,
        y="latency",
        title="Latency Across Dataset Records",
        labels={"x": "Record", "latency": "Latency (ms)"},
    )
    latency_fig.update_layout(
        template="plotly_dark",
        height=350,
    )

    # 3. Attack-count graph
    attack_counts = (
        chart_df["attack_type"]
        .value_counts()
        .rename_axis("attack_type")
        .reset_index(name="count")
    )

    attack_fig = px.bar(
        attack_counts,
        x="attack_type",
        y="count",
        color="attack_type",
        title="Event Counts by Attack Label",
        labels={
            "attack_type": "Attack Label",
            "count": "Number of Records",
        },
    )
    attack_fig.update_layout(
        template="plotly_dark",
        height=380,
        showlegend=False,
    )

    # 4. Risk gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_risk,
        title={"text": "Average Dataset Risk"},
        number={"valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.35], "color": "#16a34a"},
                {"range": [0.35, 0.70], "color": "#facc15"},
                {"range": [0.70, 1], "color": "#ef4444"},
            ],
        },
    ))
    gauge.update_layout(
        template="plotly_dark",
        height=380,
    )

    # Advice
    if avg_risk > 0.70:
        advice = (
            "Elevated average risk: review mitigations and "
            "cryptographic readiness."
        )
    elif avg_risk > 0.50:
        advice = (
            "Moderate-to-high average risk: investigate events "
            "and evaluate applicable controls."
        )
    else:
        advice = (
            "Lower average risk observed. Continue monitoring "
            "and validate risk-score provenance."
        )

    # ML model
    ml_text = predict_risk(df, model_type)

    feed = html.Div([
        html.P(f"Data source: {source_name}"),
        html.P(f"Records analyzed: {len(df):,}"),
        html.P(ml_text),
    ])

    # Summary
    summary = html.Div([
        html.H3("AO-LACCP SOC Summary"),
        html.P(f"Source: {source_type}"),
        html.P(f"Rows analyzed: {len(df):,}"),
        html.P(f"Average risk score: {avg_risk:.3f}"),
        html.P(f"Average latency: {avg_latency:.2f} ms"),
        html.P(
            "Placeholder values and synthetic events are not "
            "empirical measurements. Validate dataset labels and "
            "risk-score methodology before interpreting results."
        ),
    ])

    # Mitigation table
    table_data = generate_mitigation_table(df)

    columns = [
        {"name": col, "id": col}
        for col in (list(table_data[0].keys()) if table_data else [])
    ]

    table = dash_table.DataTable(
        data=table_data,
        columns=columns,
        page_size=10,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#0f172a",
            "color": "#e2e8f0",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "100px",
            "maxWidth": "300px",
            "textAlign": "left",
            "padding": "8px",
        },
        style_header={
            "backgroundColor": "#1e2937",
            "color": "white",
            "fontWeight": "bold",
        },
    )

    return (
        feed,
        risk_fig,
        latency_fig,
        attack_fig,
        gauge,
        advice,
        summary,
        table,
    )


# ==========================================================
# EXPORT MITIGATION PLAN
# ==========================================================

@app.callback(
    Output("download-mitigation", "data"),
    Input("export-btn", "n_clicks"),
    State("dataset_store", "data"),
    prevent_initial_call=True,
)
def export_mitigation(n_clicks, stored_data):
    if not n_clicks:
        raise PreventUpdate

    if stored_data:
        df = pd.DataFrame(stored_data)
    else:
        with stream_lock:
            df = pd.DataFrame(list(live_stream))

    if df.empty:
        raise PreventUpdate

    data = generate_mitigation_table(df)

    return dcc.send_data_frame(
        pd.DataFrame(data).to_csv,
        "AO-LACCP_Mitigation_Plan.csv",
        index=False,
    )


# ==========================================================
# START APPLICATION
# ==========================================================

if __name__ == "__main__":
    threading.Thread(
        target=pipeline,
        daemon=True,
        name="AO-LACCP-Live-Stream",
    ).start()

    print("AO-LACCP SOC started.")
    print("CSV/XLSX/XLS uploads enabled.")
    print("Maximum upload size: 10 MB")
    print("Maximum dataset rows: 3,000")
    print("XGBoost available:", XGB_AVAILABLE)
    print("Open http://127.0.0.1:8050")

    app.run(
        debug=False,
        host="127.0.0.1",
        port=8050,
    )
