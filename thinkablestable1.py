
import base64
import io
import threading
import time
import random
import traceback
from collections import deque

import dash
from dash import dcc, html, Input, Output, State, dash_table
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
# AO-LACCP SOC CONFIGURATION
# ==========================================================

app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_DATASET_ROWS = 3000
MAX_CHART_ROWS = 500
MAX_LIVE_ROWS = 200

live_stream = deque(maxlen=MAX_LIVE_ROWS)
stream_lock = threading.Lock()
state_lock = threading.Lock()

uploaded_df = pd.DataFrame()
uploaded_dataset_name = "No dataset uploaded"
upload_status = "Upload a CSV or Excel dataset."

crypto_mode_global = "AUTO"

ATTACK_TYPES = [
    "brute_force",
    "MITM",
    "zero_day",
    "ransomware",
    "phishing",
    "data_exfiltration",
]

MITIGATION_DB = {
    "brute_force": {
        "risk": "High",
        "mitigation": "Rate limiting, behavioral CAPTCHA, AI anomaly detection",
    },
    "MITM": {
        "risk": "Critical",
        "mitigation": "TLS 1.3 enforcement, certificate pinning, AI traffic inspection",
    },
    "zero_day": {
        "risk": "Critical",
        "mitigation": "Behavioral analysis, rapid patching, threat hunting",
    },
    "ransomware": {
        "risk": "Critical",
        "mitigation": "Immutable backups, AI endpoint detection, Zero Trust",
    },
    "phishing": {
        "risk": "High",
        "mitigation": "AI email filtering, MFA, security awareness training",
    },
    "data_exfiltration": {
        "risk": "High",
        "mitigation": "DLP, outbound anomaly detection, AI data flow monitoring",
    },
}


# ==========================================================
# CRYPTOGRAPHIC DECISION ENGINE
# ==========================================================

def crypto_engine(mode, risk):
    if mode == "PQC":
        return "PQC", 0.10

    if mode == "RSA":
        return "RSA", 0.30

    if mode == "HYBRID":
        if risk > 0.60:
            return "PQC+AES", 0.08
        return "RSA+AES", 0.20

    if risk > 0.75:
        return "PQC-AUTO", 0.10

    if risk > 0.45:
        return "RSA-AUTO", 0.30

    return "AES-256", 0.05


# ==========================================================
# SYNTHETIC LIVE THREAT STREAM
# ==========================================================

def pipeline():
    global crypto_mode_global

    while True:
        risk = random.random()
        latency = random.uniform(50, 400)

        with state_lock:
            mode = crypto_mode_global

        crypto, penalty = crypto_engine(mode, risk)

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
        str(column).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    aliases = {
        "attack_type": [
            "attack", "attacktype", "label", "class",
            "type", "threat", "category", "event_type",
        ],
        "risk_score": [
            "risk", "risk_probability", "risk_prob",
            "anomaly_score", "threat_score",
        ],
        "latency": [
            "response_time", "latency_ms", "delay",
            "processing_time", "response_latency",
        ],
    }

    for target, candidates in aliases.items():
        if target not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df.rename(
                        columns={candidate: target},
                        inplace=True,
                    )
                    break

    notes = []

    # Attack labels
    if "attack_type" not in df.columns:
        df["attack_type"] = "uploaded_event"
        notes.append(
            "No attack-label column found. Events are labeled "
            "'uploaded_event'; attack types were not inferred."
        )
    else:
        df["attack_type"] = (
            df["attack_type"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )

    # Risk scores
    if "risk_score" not in df.columns:
        df["risk_score"] = 0.5
        notes.append(
            "No risk-score column found. Placeholder risk 0.5 used; "
            "this is not a measured risk score."
        )
    else:
        risk = pd.to_numeric(
            df["risk_score"],
            errors="coerce",
        )

        risk = risk.replace([np.inf, -np.inf], np.nan)
        valid = risk.dropna()

        if not valid.empty and valid.min() >= 0 and valid.max() > 1:
            if valid.max() <= 100:
                risk = risk / 100.0
                notes.append(
                    "Risk values converted from percentages to 0-1."
                )
            else:
                risk[:] = np.nan

        risk = risk.where(risk.between(0, 1))

        if risk.notna().any():
            risk = risk.fillna(risk.median())
        else:
            risk = pd.Series(0.5, index=df.index)
            notes.append(
                "Risk values were missing or invalid. Placeholder "
                "risk 0.5 used."
            )

        df["risk_score"] = risk.astype(float)

    # Latency
    if "latency" not in df.columns:
        df["latency"] = 150.0
        notes.append(
            "No latency column found. Placeholder latency of "
            "150 ms used; this is not a measured latency."
        )
    else:
        latency = pd.to_numeric(
            df["latency"],
            errors="coerce",
        )

        latency = latency.replace([np.inf, -np.inf], np.nan)
        latency = latency.where(latency >= 0)

        if latency.notna().any():
            latency = latency.fillna(latency.median())
        else:
            latency = pd.Series(150.0, index=df.index)
            notes.append(
                "Latency values were invalid. Placeholder 150 ms used."
            )

        df["latency"] = latency.astype(float)

    return df.reset_index(drop=True), notes


# ==========================================================
# CSV AND EXCEL UPLOAD
# ==========================================================

def parse_upload(contents, filename):
    if not contents:
        raise ValueError("No file was received.")

    if not filename:
        raise ValueError("The filename is missing.")

    extension = filename.lower().rsplit(".", 1)[-1]

    if extension not in ("csv", "xlsx", "xls"):
        raise ValueError(
            "Unsupported file. Please upload CSV, XLSX, or XLS."
        )

    # Approximate encoded size check before decoding.
    estimated_size = len(contents) * 3 // 4

    if estimated_size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit. "
            "Reduce its size and try again."
        )

    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(
            "The file could not be decoded. Try saving it again."
        ) from exc

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
        )

    if not raw:
        raise ValueError("The uploaded file is empty.")

    try:
        if extension == "csv":
            df = pd.read_csv(
                io.BytesIO(raw),
                nrows=MAX_DATASET_ROWS + 1,
                low_memory=True,
            )
        else:
            df = pd.read_excel(
                io.BytesIO(raw),
                nrows=MAX_DATASET_ROWS + 1,
                engine="openpyxl" if extension == "xlsx" else None,
            )

    except ImportError as exc:
        raise ValueError(
            "An Excel reader is missing. Install openpyxl for XLSX "
            "or xlrd for older XLS files."
        ) from exc

    except Exception as exc:
        raise ValueError(
            "Could not read the file. Check that it is a valid CSV "
            "or Excel file and is not corrupted."
        ) from exc

    if df.empty:
        raise ValueError(
            "No data rows were found in the uploaded file."
        )

    notes = []

    if len(df) > MAX_DATASET_ROWS:
        df = df.iloc[:MAX_DATASET_ROWS].copy()
        notes.append(
            f"Dataset limited to the first {MAX_DATASET_ROWS:,} rows."
        )

    df, normalization_notes = normalize_dataset(df)
    notes.extend(normalization_notes)

    return df, notes


# ==========================================================
# MITIGATION RECOMMENDATIONS
# ==========================================================

def generate_mitigation_table(df):
    if df.empty:
        return []

    grouped = (
        df.groupby("attack_type", dropna=False)["risk_score"]
        .agg(["mean", "count"])
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():
        attack = str(row["attack_type"])
        risk = float(row["mean"])

        info = MITIGATION_DB.get(attack, {})

        if risk >= 0.70:
            risk_level = "Critical"
        elif risk >= 0.50:
            risk_level = "High"
        elif risk >= 0.30:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        results.append({
            "Threat": attack.replace("_", " ").title(),
            "Avg Risk Score": round(risk, 3),
            "Risk Level": info.get("risk", risk_level),
            "Occurrences": int(row["count"]),
            "Potential Impact": (
                f"Potential impact from {attack.replace('_', ' ')}"
            ),
            "Latency Impact": "Requires measurement",
            "Cryptographic Concern": (
                "Review applicable encryption exposure"
            ),
            "AI/ML Relevance": "Requires validation",
            "Recommended Mitigation": info.get(
                "mitigation",
                f"Investigate and validate {attack.replace('_', ' ')} events",
            ),
        })

    return sorted(
        results,
        key=lambda item: item["Avg Risk Score"],
        reverse=True,
    )


# ==========================================================
# STYLING
# ==========================================================

STYLE = {
    "backgroundColor": "#050914",
    "color": "#e5e7eb",
    "fontFamily": "Arial",
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

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}


# ==========================================================
# DASHBOARD LAYOUT
# ==========================================================

app.layout = html.Div(style=STYLE, children=[
    html.H1(
        "AO-LACCP SOC Intelligence Platform",
        style={
            "textAlign": "center",
            "color": "#38bdf8",
        },
    ),

    html.Div(style=CARD, children=[
        html.H3("Dataset Upload"),

        dcc.Upload(
            id="upload_data",
            children=html.Div([
                "Select or drag a CSV / Excel dataset here",
                html.Br(),
                html.Small("Maximum upload size: 10 MB"),
            ]),
            accept=".csv,.xlsx,.xls",
            multiple=False,
            style={
                "border": "2px dashed #38bdf8",
                "padding": "22px",
                "textAlign": "center",
                "cursor": "pointer",
                "borderRadius": "10px",
            },
        ),

        html.Div(
            id="dataset_name",
            children=upload_status,
            style={
                "marginTop": "12px",
                "whiteSpace": "pre-wrap",
            },
        ),

        html.Div(
            style={
                "display": "flex",
                "gap": "15px",
                "marginTop": "15px",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": "200px"},
                    children=[
                        html.Label("Machine Learning Model"),
                        dcc.Dropdown(
                            id="model_selector",
                            options=[
                                {
                                    "label": "Random Forest",
                                    "value": "RF",
                                },
                                {
                                    "label": "XGBoost",
                                    "value": "XGB",
                                    "disabled": not XGB_AVAILABLE,
                                },
                            ],
                            value="RF",
                            clearable=False,
                        ),
                    ],
                ),

                html.Div(
                    style={"flex": "1", "minWidth": "200px"},
                    children=[
                        html.Label("Cryptographic Mode"),
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
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            id="crypto_advice",
            style={"marginTop": "12px"},
        ),
    ]),

    html.Div(style=CARD, children=[
        html.H3("Live Threat Stream"),
        html.Div(id="live_feed"),
    ]),

    html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(auto-fit, minmax(300px, 1fr))"
            ),
            "gap": "16px",
        },
        children=[
            dcc.Graph(
                id="risk_graph",
                config=GRAPH_CONFIG,
            ),
            dcc.Graph(
                id="latency_graph",
                config=GRAPH_CONFIG,
            ),
        ],
    ),

    html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(auto-fit, minmax(300px, 1fr))"
            ),
            "gap": "16px",
        },
        children=[
            dcc.Graph(
                id="soc_attack_stream",
                config=GRAPH_CONFIG,
            ),
            dcc.Graph(
                id="soc_risk_gauge",
                config=GRAPH_CONFIG,
            ),
        ],
    ),

    html.Div(id="soc_summary", style=CARD),

    html.Div(style=CARD, children=[
        html.H3("SOC Mitigation & Action Plan"),

        html.Button(
            "Export as CSV",
            id="export-btn",
            n_clicks=0,
            style={"marginBottom": "10px"},
        ),

        html.Div(
            id="mitigation_table_container",
            style={"overflowX": "auto"},
        ),
    ]),

    dcc.Download(id="download-mitigation"),

    dcc.Interval(
        id="interval",
        interval=2500,
        n_intervals=0,
    ),
])


# ==========================================================
# DATASET UPLOAD CALLBACK
# The upload is processed only when a file is selected.
# ==========================================================

@app.callback(
    Output("dataset_name", "children"),
    Input("upload_data", "contents"),
    State("upload_data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    global uploaded_df
    global uploaded_dataset_name
    global upload_status

    if not contents:
        raise PreventUpdate

    try:
        df, notes = parse_upload(contents, filename)

        with state_lock:
            uploaded_df = df
            uploaded_dataset_name = filename or "Uploaded dataset"

        message = (
            f"Upload successful: {uploaded_dataset_name}\n"
            f"Rows loaded: {len(df):,}\n"
            f"Columns: {len(df.columns)}"
        )

        if notes:
            message += "\n\nNotes:\n- " + "\n- ".join(notes)

        upload_status = message
        print(message)

        return message

    except Exception as exc:
        with state_lock:
            uploaded_df = pd.DataFrame()
            uploaded_dataset_name = "Upload failed"

        upload_status = f"Upload failed: {exc}"
        print(upload_status)
        traceback.print_exc()

        return upload_status


# ==========================================================
# MACHINE LEARNING
# ==========================================================

def predict_risk(df, model_type):
    if len(df) < 20:
        return (
            "ML prediction unavailable: at least 20 rows are recommended."
        )

    # This demonstration target is derived from risk_score.
    # It is not an independently verified attack label.
    y = (df["risk_score"] > 0.65).astype(int)

    if y.nunique() < 2:
        return (
            "ML prediction unavailable: only one risk class is present."
        )

    try:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            # Compatibility with older scikit-learn releases.
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

        X = encoder.fit_transform(
            df[["attack_type"]].astype(str)
        )

        if model_type == "XGB" and XGB_AVAILABLE:
            model = XGBClassifier(
                n_estimators=40,
                max_depth=3,
                random_state=42,
                verbosity=0,
                n_jobs=1,
                eval_metric="logloss",
            )
            model_name = "XGBoost"
        else:
            model = RandomForestClassifier(
                n_estimators=40,
                max_depth=8,
                random_state=42,
                n_jobs=1,
            )
            model_name = "Random Forest"

        model.fit(X[:-1], y.iloc[:-1])

        classes = list(model.classes_)

        if 1 not in classes:
            probability = 0.0
        else:
            probability = float(
                model.predict_proba(X[-1:])[0][classes.index(1)]
            )

        prediction = (
            "HIGH RISK" if probability > 0.5 else "LOW RISK"
        )

        return (
            f"{model_name}: {prediction} "
            f"(estimated probability {probability:.1%})"
        )

    except Exception as exc:
        print(f"ML prediction error: {exc}")
        traceback.print_exc()
        return "ML prediction unavailable. See Python terminal."


# ==========================================================
# DASHBOARD REFRESH CALLBACK
# ==========================================================

@app.callback(
    [
        Output("live_feed", "children"),
        Output("risk_graph", "figure"),
        Output("latency_graph", "figure"),
        Output("soc_attack_stream", "figure"),
        Output("soc_risk_gauge", "figure"),
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
)
def update_dashboard(n, model_type, crypto_mode, upload_contents):
    global crypto_mode_global

    with state_lock:
        crypto_mode_global = crypto_mode
        dataset = uploaded_df.copy()
        dataset_name = uploaded_dataset_name

    with stream_lock:
        live_df = pd.DataFrame(list(live_stream))

    if not dataset.empty:
        analysis_df = dataset
        source_name = dataset_name
        source_type = "Uploaded dataset"
    else:
        analysis_df = live_df
        source_name = "Generated demonstration stream"
        source_type = "Synthetic demonstration data"

    if analysis_df.empty:
        empty = go.Figure()
        empty.update_layout(template="plotly_dark")

        return (
            "Initializing live stream...",
            empty,
            empty,
            empty,
            empty,
            "Waiting for data...",
            "Waiting for dataset or live events...",
            html.Div("No data available."),
        )

    # Limit the number of points sent to the browser.
    chart_df = analysis_df.tail(MAX_CHART_ROWS).copy()

    avg_risk = float(analysis_df["risk_score"].mean())
    avg_latency = float(analysis_df["latency"].mean())

    ml_text = predict_risk(analysis_df, model_type)

    # Risk chart
    risk_fig = px.line(
        chart_df,
        y="risk_score",
        title="Risk Score",
        render_mode="webgl",
    )
    risk_fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=35, r=20, t=55, b=35),
    )

    # Latency chart
    latency_fig = px.line(
        chart_df,
        y="latency",
        title="Latency (ms)",
        render_mode="webgl",
    )
    latency_fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=35, r=20, t=55, b=35),
    )

    # Attack count chart
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
        title="Event Counts by Label",
    )
    attack_fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=35, r=20, t=55, b=80),
    )

    # Risk gauge
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_risk,
            title={"text": "Average Risk Score"},
            number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1]},
                "steps": [
                    {"range": [0, 0.35], "color": "#16a34a"},
                    {"range": [0.35, 0.70], "color": "#facc15"},
                    {"range": [0.70, 1.0], "color": "#ef4444"},
                ],
            },
        )
    )
    gauge.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=25, r=25, t=65, b=25),
    )

    # Advisory text
    if avg_risk > 0.70:
        advice = (
            "Elevated average risk: review PQC readiness and "
            "validate mitigation actions."
        )
    elif avg_risk > 0.50:
        advice = (
            "Moderate-to-high average risk: evaluate hybrid "
            "cryptographic controls."
        )
    else:
        advice = (
            "Lower average risk observed. Continue monitoring "
            "and validate against actual threat labels."
        )

    # Latest synthetic event
    if live_df.empty:
        feed_text = "No live events generated yet."
    else:
        latest = live_df.iloc[-1]
        feed_text = (
            f"Latest simulated event: {latest['attack_type'].upper()} | "
            f"{latest['crypto']} | "
            f"Risk: {latest['risk_score']:.3f} | "
            f"Latency: {latest['latency']:.1f} ms | {ml_text}"
        )

    # Mitigation table
    table_data = generate_mitigation_table(analysis_df)

    columns = [
        {"name": column, "id": column}
        for column in (
            list(table_data[0].keys()) if table_data else []
        )
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
            "maxWidth": "280px",
            "textAlign": "left",
            "padding": "8px",
        },
        style_header={
            "backgroundColor": "#1e2937",
            "color": "white",
            "fontWeight": "bold",
        },
    )

    summary = html.Div([
        html.H3("AO-LACCP SOC Summary"),
        html.P(f"Data source: {source_name}"),
        html.P(f"Source type: {source_type}"),
        html.P(f"Rows analyzed: {len(analysis_df):,}"),
        html.P(f"Average risk score: {avg_risk:.3f}"),
        html.P(f"Average latency: {avg_latency:.2f} ms"),
        html.P(
            "Research note: synthetic live events and placeholder "
            "values are not measured security outcomes. Validate "
            "the uploaded dataset's labels and scoring methodology."
        ),
    ])

    return (
        feed_text,
        risk_fig,
        latency_fig,
        attack_fig,
        gauge,
        advice,
        summary,
        table,
    )


# ==========================================================
# EXPORT MITIGATION TABLE
# ==========================================================

@app.callback(
    Output("download-mitigation", "data"),
    Input("export-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_mitigation(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    with state_lock:
        df = uploaded_df.copy()

    if df.empty:
        with stream_lock:
            df = pd.DataFrame(list(live_stream))

    if df.empty:
        raise PreventUpdate

    table_data = generate_mitigation_table(df)

    if not table_data:
        raise PreventUpdate

    return dcc.send_data_frame(
        pd.DataFrame(table_data).to_csv,
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

    print("=" * 55)
    print("AO-LACCP SOC Intelligence Platform")
    print("CSV / Excel uploads enabled")
    print(f"Maximum upload: {MAX_UPLOAD_MB} MB")
    print(f"Maximum dataset rows: {MAX_DATASET_ROWS}")
    print(f"XGBoost available: {XGB_AVAILABLE}")
    print("Open: http://127.0.0.1:8050")
    print("=" * 55)

    app.run(
        debug=False,
        host="127.0.0.1",
        port=8050,
    )
