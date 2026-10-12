import base64
import io
import logging
import random
from datetime import datetime, timezone

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# AO-LACCP SOC research prototype. Events generated here are synthetic;
# cryptographic recommendations are advisory and do not deploy controls.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AO_LACCP_SOC")
app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

MAX_EVENTS = 3000
MAX_UPLOAD_ROWS = 3000
RISK_THRESHOLD = 0.65
ATTACK_TYPES = [
    "brute_force", "MITM", "zero_day", "ransomware",
    "phishing", "data_exfiltration", "unknown"
]

MITIGATION_DB = {
    "brute_force": {
        "mitigation": "Apply rate limiting, MFA, account lockout controls, and authentication anomaly monitoring.",
        "impact": "Unauthorized account access",
    },
    "MITM": {
        "mitigation": "Validate certificates, enforce approved TLS configurations, and investigate suspicious network traffic.",
        "impact": "Interception or alteration of communications",
    },
    "zero_day": {
        "mitigation": "Use behavioral detection, investigate indicators, isolate affected assets when warranted, and prioritize remediation.",
        "impact": "Potential exploitation of an unknown vulnerability",
    },
    "ransomware": {
        "mitigation": "Isolate affected endpoints when warranted, protect immutable backups, and verify incident-response procedures.",
        "impact": "Service disruption and potential data loss",
    },
    "phishing": {
        "mitigation": "Investigate suspicious messages, strengthen email filtering, use MFA, and review reported links or credentials.",
        "impact": "Credential theft or malware delivery",
    },
    "data_exfiltration": {
        "mitigation": "Review outbound transfers, investigate DLP alerts, and validate data access against approved policies.",
        "impact": "Unauthorized disclosure of sensitive data",
    },
    "unknown": {
        "mitigation": "Validate the event classification, enrich the evidence, and investigate before selecting a response.",
        "impact": "Undetermined until investigated",
    },
}

STYLE = {
    "backgroundColor": "#050914", "color": "#e5e7eb",
    "fontFamily": "Arial, sans-serif", "minHeight": "100vh", "padding": "16px"
}
CARD = {
    "background": "rgba(255,255,255,0.05)",
    "border": "1px solid rgba(255,255,255,0.10)",
    "borderRadius": "14px", "padding": "16px", "marginBottom": "14px"
}
DROPDOWN_STYLE = {"color": "#111827", "backgroundColor": "#ffffff"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def classify_risk(risk):
    if risk >= 0.70:
        return "Critical"
    if risk >= 0.50:
        return "High"
    if risk >= 0.30:
        return "Moderate"
    return "Low"


def crypto_engine(mode, risk):
    """Return advisory labels only; no cryptographic operation is performed."""
    if mode == "PQC":
        return "PQC selected for assessment", "Validate standardized algorithms, implementation, interoperability, and migration requirements."
    if mode == "RSA":
        return "RSA selected for assessment", "Validate approved key sizes, protocol usage, and organizational cryptographic policy."
    if mode == "HYBRID":
        return "Hybrid configuration selected for assessment", "Validate the actual algorithms, protocol design, interoperability, and performance."
    if risk >= 0.70:
        return "High-risk cryptographic review", "Review approved hybrid options and PQC readiness; do not automatically switch production cryptography."
    if risk >= 0.45:
        return "Cryptographic agility review", "Assess whether an approved hybrid configuration is appropriate for the system and latency constraints."
    return "Maintain approved cryptographic configuration", "Continue monitoring and follow the organization's cryptographic policy."


def generate_synthetic_event(crypto_mode):
    """Create one illustrative event, not real network telemetry."""
    attack = random.choice(ATTACK_TYPES[:-1])
    risk = float(np.random.uniform(0.05, 0.98))
    latency = float(np.random.uniform(50, 400))
    crypto_label, _ = crypto_engine(crypto_mode, risk)
    return {
        "timestamp": utc_now(), "attack_type": attack,
        "risk_score": round(risk, 4), "latency": round(latency, 2),
        "crypto": crypto_label, "source": "Synthetic simulation",
    }


def normalize_column_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def parse_upload(contents):
    """Parse and validate a CSV upload; missing fields use explicit placeholders."""
    if not contents:
        return [], "No file was supplied."
    try:
        _, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded)
        raw_df = pd.read_csv(io.BytesIO(decoded), nrows=MAX_UPLOAD_ROWS + 1, encoding="utf-8-sig")
        if raw_df.empty:
            return [], "The uploaded CSV contains no data rows."
        raw_df = raw_df.head(MAX_UPLOAD_ROWS).copy()
        df = raw_df.copy()
        df.columns = [normalize_column_name(c) for c in df.columns]
        aliases = {
            "attack": "attack_type", "threat": "attack_type",
            "threat_type": "attack_type", "attack_category": "attack_type",
            "transaction_type": "attack_type", "event_type": "attack_type",
            "label": "attack_type", "class": "attack_type",
            "risk": "risk_score", "risk_level_score": "risk_score",
            "risk_probability": "risk_score", "fraud_probability": "risk_score",
            "probability": "risk_score", "score": "risk_score",
            "latency_ms": "latency", "response_time": "latency",
            "response_time_ms": "latency", "processing_time": "latency",
            "processing_time_ms": "latency", "event_time": "timestamp", "time": "timestamp",
        }
        df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
        notes = []
        # A binary fraud indicator can be used as an explicitly disclosed risk proxy.
        fraud_col = next((c for c in ("is_fraud", "fraud", "fraudulent", "is_fraudulent") if c in df.columns), None)
        if "risk_score" not in df.columns and fraud_col:
            raw = df[fraud_col].astype(str).str.strip().str.lower()
            positive = {"1", "true", "yes", "fraud", "fraudulent", "positive"}
            negative = {"0", "false", "no", "legitimate", "not_fraud", "negative"}
            mapped = raw.map(lambda value: 1.0 if value in positive else (0.0 if value in negative else np.nan))
            if mapped.notna().any():
                df["risk_score"] = mapped.fillna(0.5)
                notes.append(f"Risk chart uses '{fraud_col}' as a binary fraud-indicator proxy, not a calibrated risk probability.")
        if len(df) > MAX_UPLOAD_ROWS:
            df = df.head(MAX_UPLOAD_ROWS).copy()
            notes.append(f"Upload limited to {MAX_UPLOAD_ROWS:,} rows.")
        if "attack_type" not in df.columns:
            # Prefer a useful categorical field for the event-count chart.
            categorical = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 30]
            if categorical:
                chosen = categorical[0]
                df["attack_type"] = df[chosen].fillna("unknown").astype(str)
                notes.append(f"Event-count chart uses '{chosen}' as a category; categories are not automatically verified attack labels.")
            else:
                df["attack_type"] = "unknown"
                notes.append("No suitable categorical field found: event categories set to 'unknown'.")
        if "risk_score" not in df.columns:
            df["risk_score"] = 0.5
            notes.append("No risk_score, risk probability, or recognized fraud indicator found: risk chart uses a placeholder score of 0.5. Add a meaningful risk column for data-driven risk values.")
        if "latency" not in df.columns:
            df["latency"] = 150.0
            notes.append("No latency/response-time column found: latency chart uses a placeholder of 150 ms. Add a measured latency column for data-driven latency values.")
        df["attack_type"] = df["attack_type"].fillna("unknown").astype(str).str.strip().str.lower().replace("", "unknown")
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.5)
        df["latency"] = pd.to_numeric(df["latency"], errors="coerce").fillna(150.0)
        if ((df["risk_score"] < 0) | (df["risk_score"] > 1)).any():
            notes.append("Risk values outside 0–1 were clipped.")
        df["risk_score"] = df["risk_score"].clip(0, 1)
        if (df["latency"] < 0).any():
            notes.append("Negative latency values replaced with 150 ms.")
            df.loc[df["latency"] < 0, "latency"] = 150.0
        if "timestamp" not in df.columns:
            df["timestamp"] = utc_now()
        else:
            df["timestamp"] = df["timestamp"].fillna(utc_now()).astype(str)
        # Vectorized conversion is much faster than iterating over every row.
        df["risk_score"] = df["risk_score"].round(4)
        df["latency"] = df["latency"].round(2)
        df["crypto"] = df["risk_score"].map(
            lambda value: crypto_engine("AUTO", float(value))[0]
        )
        df["source"] = "Uploaded CSV"
        events = df[["timestamp", "attack_type", "risk_score", "latency", "crypto", "source"]].to_dict(orient="records")
        message = " ".join(notes) if notes else f"Successfully loaded {len(events):,} rows."
        message += f" Dataset contains {len(df.columns):,} columns."
        return events, message
    except Exception as exc:
        logger.exception("CSV upload failed")
        return [], f"Could not read the CSV: {type(exc).__name__}: {exc}"


def generate_mitigation_table(df):
    if df.empty:
        return []
    grouped = df.groupby("attack_type", dropna=False).agg(
        avg_risk=("risk_score", "mean"),
        occurrences=("attack_type", "size"),
        avg_latency=("latency", "mean"),
    ).reset_index().sort_values("avg_risk", ascending=False)
    result = []
    for _, row in grouped.iterrows():
        attack = str(row["attack_type"])
        info = MITIGATION_DB.get(attack, MITIGATION_DB["unknown"])
        result.append({
            "Threat": attack.replace("_", " ").title(),
            "Average Risk": round(float(row["avg_risk"]), 3),
            "Risk Level": classify_risk(float(row["avg_risk"])),
            "Occurrences": int(row["occurrences"]),
            "Average Latency (ms)": round(float(row["avg_latency"]), 2),
            "Potential Impact": info["impact"],
            "Recommended Mitigation": info["mitigation"],
        })
    return result


def evaluate_and_predict(df, model_type):
    """Fit on prior rows, predict the latest row, and report held-out metrics when possible."""
    waiting = {"text": "Insufficient labeled history for ML analysis.", "metrics": None}
    if len(df) < 30:
        return waiting
    history = df.iloc[:-1].copy()
    latest = df.iloc[[-1]].copy()
    history["target"] = (history["risk_score"] > RISK_THRESHOLD).astype(int)
    if history["target"].nunique() < 2:
        return {"text": "ML unavailable: training history has one class only.", "metrics": None}
    X = history[["attack_type", "latency"]]
    y = history["target"]
    X_latest = latest[["attack_type", "latency"]]
    preprocessor = ColumnTransformer([
        ("attack", OneHotEncoder(handle_unknown="ignore"), ["attack_type"]),
        ("latency", "passthrough", ["latency"]),
    ])
    def new_classifier():
        if model_type == "XGB" and XGB_AVAILABLE:
            return XGBClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                random_state=42, eval_metric="logloss", verbosity=0,
            )
        return RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        )
    model_name = "XGBoost" if model_type == "XGB" and XGB_AVAILABLE else "Random Forest"
    metrics = None
    counts = y.value_counts()
    if len(counts) == 2 and counts.min() >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            evaluation_model = Pipeline([
                ("preprocessor", ColumnTransformer([
                    ("attack", OneHotEncoder(handle_unknown="ignore"), ["attack_type"]),
                    ("latency", "passthrough", ["latency"]),
                ])),
                ("classifier", new_classifier()),
            ])
            evaluation_model.fit(X_train, y_train)
            y_pred = evaluation_model.predict(X_test)
            metrics = {
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "F1": float(f1_score(y_test, y_pred, zero_division=0)),
                "Test observations": int(len(y_test)),
            }
        except Exception:
            logger.exception("ML evaluation split failed")
    try:
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", new_classifier()),
        ])
        model.fit(X, y)
        classifier = model.named_steps["classifier"]
        probabilities = model.predict_proba(X_latest)[0]
        classes = list(classifier.classes_)
        p_high = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
        prediction = "HIGH-RISK CLASS" if p_high >= 0.5 else "LOWER-RISK CLASS"
        text = f"{model_name}: {prediction} (estimated class probability {p_high:.1%})"
        if metrics:
            text += f" | Held-out accuracy: {metrics['Accuracy']:.1%} | F1: {metrics['F1']:.3f}"
        else:
            text += " | Held-out metrics unavailable"
        return {"text": text, "metrics": metrics}
    except Exception:
        logger.exception("ML prediction failed")
        return {"text": f"{model_name}: prediction unavailable for this data.", "metrics": metrics}


def empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", height=350,
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False}],
    )
    return fig


def make_risk_figure(df):
    if df.empty:
        return empty_figure("Waiting for events")
    fig = px.line(df, x="event_number", y="risk_score", title="Risk Score by Event",
                  labels={"event_number": "Event", "risk_score": "Risk score (0–1)"},
                  hover_data=["attack_type", "source"])
    fig.add_hline(y=RISK_THRESHOLD, line_dash="dash", annotation_text="ML label threshold")
    fig.update_layout(template="plotly_dark", height=350)
    return fig


def make_latency_figure(df):
    if df.empty:
        return empty_figure("Waiting for events")
    fig = px.line(df, x="event_number", y="latency", title="Event Latency",
                  labels={"event_number": "Event", "latency": "Latency (ms)"},
                  hover_data=["attack_type", "source"])
    fig.update_layout(template="plotly_dark", height=350)
    return fig


def make_attack_figure(df):
    if df.empty:
        return empty_figure("Waiting for events")
    counts = df["attack_type"].value_counts().rename_axis("Attack type").reset_index(name="Events")
    fig = px.bar(counts, x="Attack type", y="Events", title="Event Counts by Attack Classification", color="Attack type")
    fig.update_layout(template="plotly_dark", height=380)
    return fig


def make_risk_gauge(avg_risk):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(avg_risk), number={"valueformat": ".3f"},
        title={"text": "Mean Risk Score"},
        gauge={"axis": {"range": [0, 1]},
               "steps": [{"range": [0, .30], "color": "#16a34a"},
                         {"range": [.30, .50], "color": "#84cc16"},
                         {"range": [.50, .70], "color": "#facc15"},
                         {"range": [.70, 1], "color": "#ef4444"}],
               "threshold": {"line": {"color": "white", "width": 3},
                             "thickness": .75, "value": .70}},
    ))
    fig.update_layout(template="plotly_dark", height=380)
    return fig


app.layout = html.Div(style=STYLE, children=[
    html.H1("AO-LACCP SOC Intelligence Platform", style={"textAlign": "center", "color": "#38bdf8"}),
    html.P("Research prototype | Synthetic events and uploaded CSV data | Advisory decision support only",
           style={"textAlign": "center", "color": "#cbd5e1"}),
    # Session storage survives browser refreshes in the same tab. It is not a server-side database.
    dcc.Store(id="event-store", storage_type="local", data={
        "events": [], "dataset_name": "Synthetic event generator", "upload_message": ""
    }),
    dcc.Store(id="ml-cache", storage_type="memory", data=None),
    html.Div(style=CARD, children=[
        html.H3("Data Source"),
        dcc.Upload(id="upload_data", children=html.Div([
            "📁 Upload CSV Dataset", html.Br(), html.Small("Up to 3,000 rows per upload")
        ]), style={"border": "2px dashed #38bdf8", "padding": "20px", "textAlign": "center",
                    "borderRadius": "8px", "cursor": "pointer"}, multiple=False),
        # This filename label updates independently, before CSV parsing finishes.
        html.Div(id="selected_filename", style={"marginTop": "10px", "fontWeight": "bold", "color": "#7dd3fc"}),
        html.Div(id="dataset_name", style={"marginTop": "6px"}),
        html.Div(id="upload_status", style={"display": "none"}),
    ]),
    html.Div(style=CARD, children=[
        html.H3("Model and Cryptographic Configuration"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "15px"}, children=[
            html.Div([html.Label("ML Model"), dcc.Dropdown(id="model_selector", options=[
                {"label": "Random Forest", "value": "RF"},
                {"label": "XGBoost", "value": "XGB", "disabled": not XGB_AVAILABLE},
            ], value="RF", clearable=False, style=DROPDOWN_STYLE)]),
            html.Div([html.Label("Cryptographic Decision Mode"), dcc.Dropdown(id="crypto_selector", options=[
                {"label": "Auto", "value": "AUTO"}, {"label": "PQC", "value": "PQC"},
                {"label": "RSA", "value": "RSA"}, {"label": "Hybrid", "value": "HYBRID"},
            ], value="AUTO", clearable=False, style=DROPDOWN_STYLE)]),
        ]),
        html.Div(id="crypto_advice", style={"marginTop": "14px"}),
    ]),
    html.Div(style=CARD, children=[html.H3("Live Threat Stream"), html.Div(id="live_feed", children="Initializing..."),
                                     dcc.Interval(id="interval", interval=5000, n_intervals=0)]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "16px"},
             children=[dcc.Graph(id="risk_graph"), dcc.Graph(id="latency_graph")]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "16px"},
             children=[dcc.Graph(id="soc_attack_stream"), dcc.Graph(id="soc_risk_gauge")]),
    html.Div(id="soc_summary", style=CARD),
    html.Div(style=CARD, children=[html.H3("SOC Mitigation and Action Plan"),
        html.Button("Export Mitigation Plan as CSV", id="export-btn", n_clicks=0,
                    style={"padding": "10px 16px", "marginBottom": "12px", "cursor": "pointer"}),
        html.Div(id="mitigation_table_container", style={"maxHeight": "620px", "overflow": "auto",
                  "border": "1px solid rgba(255,255,255,0.15)", "borderRadius": "8px"})]),
    dcc.Download(id="download-mitigation"),
    html.Div(style=CARD, children=[html.H3("Research Interpretation"), html.P(
        "Synthetic events are generated for demonstration. Uploaded data is not independently verified. "
        "ML metrics depend on the supplied features and labels. Cryptographic and compliance recommendations "
        "are advisory and do not deploy controls or establish certification.")]),
])


@app.callback(
    Output("selected_filename", "children"),
    Input("upload_data", "filename"),
)
def show_selected_filename(filename):
    """Show the selected filename immediately, before parsing completes."""
    if not filename:
        return "No new file selected. The last saved dataset remains loaded if available."
    return f"Selected file: {filename}"


@app.callback(
    Output("event-store", "data"),
    Output("upload_status", "children"),
    Output("interval", "disabled"),
    Input("upload_data", "contents"),
    Input("interval", "n_intervals"),
    State("event-store", "data"),
    State("upload_data", "filename"),
    prevent_initial_call=False,
)
def load_dataset_or_generate_events(upload_contents, n_intervals, stored_state, upload_filename):
    """Parse each new upload once; persist records in browser local storage.

    Once a dataset is loaded, stop synthetic timer events from replacing it.
    The parsed records and filename survive a browser refresh in the same browser.
    """
    stored_state = stored_state if isinstance(stored_state, dict) else {
        "events": [], "dataset_name": "Synthetic event generator", "upload_message": ""
    }
    trigger = ctx.triggered_id
    current_name = stored_state.get("dataset_name", "Synthetic event generator")

    if trigger == "upload_data" and upload_contents:
        events, message = parse_upload(upload_contents)
        if not events:
            # Keep the currently loaded data if a new file fails to parse.
            return dash.no_update, message, current_name != "Synthetic event generator"
        name = upload_filename or "Uploaded CSV"
        message = f"Loaded {len(events):,} rows from {name}. {message}"
        new_state = {
            "events": events[-MAX_EVENTS:],
            "dataset_name": name,
            "upload_message": message,
            "data_source": "uploaded_csv",
            "upload_version": utc_now(),
        }
        return new_state, message, True

    if current_name != "Synthetic event generator":
        # A persisted uploaded dataset is already loaded: do not overwrite it.
        # Restore a visible status after a page refresh as well.
        restored_status = stored_state.get("upload_message") or f"Restored {current_name} from browser storage."
        return dash.no_update, restored_status, True

    if trigger == "interval" or trigger is None:
        events = list(stored_state.get("events", []))
        events.append(generate_synthetic_event("AUTO"))
        events = events[-MAX_EVENTS:]
        new_state = {
            "events": events,
            "dataset_name": "Synthetic event generator",
            "upload_message": "",
            "data_source": "synthetic",
        }
        return new_state, dash.no_update, False

    return dash.no_update, dash.no_update, False


@app.callback(
    [Output("live_feed", "children"),
     Output("risk_graph", "figure"), Output("latency_graph", "figure"),
     Output("soc_attack_stream", "figure"), Output("soc_risk_gauge", "figure"),
     Output("dataset_name", "children"), Output("crypto_advice", "children"),
     Output("mitigation_table_container", "children"), Output("soc_summary", "children"),
     Output("ml-cache", "data")],
    [Input("event-store", "data"), Input("model_selector", "value"),
     Input("crypto_selector", "value")],
    [State("ml-cache", "data")],
)
def render_dashboard(stored_state, model_type, crypto_mode, cached_ml_result):
    """Render all charts and summaries automatically when stored data changes."""
    stored_state = stored_state if isinstance(stored_state, dict) else {}
    events = stored_state.get("events", [])
    dataset_name = stored_state.get("dataset_name", "Synthetic event generator")
    df = pd.DataFrame(events)

    if df.empty:
        empty = empty_figure("Waiting for events")
        return ("Waiting for the first event...", empty, empty, empty, make_risk_gauge(0),
                f"Dataset: {dataset_name}", "No recommendation available yet.",
                html.Div("Waiting for events..."), html.Div("No event data is available yet."),
                None)

    defaults = {"attack_type": "unknown", "risk_score": 0.5, "latency": 150.0,
                "crypto": "Unknown", "timestamp": "", "source": "Unknown"}
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(.5).clip(0, 1)
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce").fillna(150).clip(lower=0)
    df["attack_type"] = df["attack_type"].fillna("unknown").astype(str)
    df["event_number"] = np.arange(1, len(df) + 1)

    latest = df.iloc[-1]
    avg_risk = float(df["risk_score"].mean())
    avg_latency = float(df["latency"].mean())
    high_risk_count = int((df["risk_score"] >= .70).sum())

    # Refit only when the dataset or model choice changes; crypto-mode changes
    # update the advisory label without retraining the ML model.
    data_signature = (
        dataset_name,
        stored_state.get("upload_version", ""),
        len(df),
        str(df["risk_score"].sum()),
        str(df["latency"].sum()),
    )
    cached_signature = cached_ml_result.get("signature") if isinstance(cached_ml_result, dict) else None
    if cached_signature != data_signature or not isinstance(cached_ml_result, dict) or cached_ml_result.get("model_type") != model_type:
        ml_result = evaluate_and_predict(df, model_type)
        ml_cache_out = {"signature": data_signature, "model_type": model_type, "result": ml_result}
    else:
        ml_result = cached_ml_result.get("result", {"text": "ML analysis unavailable.", "metrics": None})
        ml_cache_out = cached_ml_result

    crypto_label, crypto_reason = crypto_engine(crypto_mode, float(latest["risk_score"]))
    advice = html.Div([
        html.H4("Cryptographic Decision Support"),
        html.P([html.Strong("Selected mode: "), crypto_mode]),
        html.P([html.Strong("Recommendation: "), crypto_label]),
        html.P(crypto_reason),
        html.Small("Advisory only: no encryption, key exchange, or production cryptographic switch is performed."),
    ])
    live_feed = html.Div([
        html.P([html.Strong("Latest event: "), str(latest["attack_type"]).upper()]),
        html.P([html.Strong("Source: "), str(latest["source"])]),
        html.P([html.Strong("Risk score: "), f"{float(latest['risk_score']):.3f} ({classify_risk(float(latest['risk_score']))})"]),
        html.P([html.Strong("Latency: "), f"{float(latest['latency']):.2f} ms"]),
        html.P([html.Strong("Crypto decision label: "), str(latest["crypto"])]),
        html.P([html.Strong("ML result: "), ml_result["text"]]),
    ])

    mitigation_data = generate_mitigation_table(df)
    columns = [{"name": col, "id": col} for col in mitigation_data[0].keys()] if mitigation_data else []
    table = dash_table.DataTable(
        data=mitigation_data, columns=columns, page_size=12, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1e293b", "color": "white", "fontWeight": "bold", "whiteSpace": "normal", "height": "auto"},
        style_cell={"backgroundColor": "#0f172a", "color": "#e2e8f0", "whiteSpace": "normal", "height": "auto",
                    "minWidth": "110px", "maxWidth": "300px", "textAlign": "left", "padding": "8px"},
        style_data_conditional=[
            {"if": {"filter_query": '{Risk Level} = "Critical"', "column_id": "Risk Level"}, "backgroundColor": "#7f1d1d", "color": "white"},
            {"if": {"filter_query": '{Risk Level} = "High"', "column_id": "Risk Level"}, "backgroundColor": "#9a3412", "color": "white"},
        ],
    )
    summary = [html.H3("SOC Operational Summary"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"}, children=[
            html.Div([html.H4("Dataset rows"), html.H2(str(len(df)))]),
            html.Div([html.H4("Mean risk"), html.H2(f"{avg_risk:.3f}")]),
            html.Div([html.H4("High-risk events"), html.H2(str(high_risk_count))]),
            html.Div([html.H4("Mean latency"), html.H2(f"{avg_latency:.1f} ms")]),
        ]), html.Hr(), html.P([html.Strong("ML assessment: "), ml_result["text"]])]
    if ml_result.get("metrics"):
        m = ml_result["metrics"]
        summary.append(html.P(f"Held-out evaluation: accuracy {m['Accuracy']:.1%}; F1 {m['F1']:.3f}; test rows {m['Test observations']}."))
    summary.append(html.Small("Synthetic-data metrics are not evidence of production detection performance."))

    return (live_feed, make_risk_figure(df), make_latency_figure(df), make_attack_figure(df),
            make_risk_gauge(avg_risk), f"Dataset: {dataset_name}", advice, table, summary, ml_cache_out)


@app.callback(
    Output("download-mitigation", "data"),
    Input("export-btn", "n_clicks"),
    State("event-store", "data"),
    prevent_initial_call=True,
)
def export_mitigation(n_clicks, stored_state):
    if not n_clicks or not isinstance(stored_state, dict):
        return None
    df = pd.DataFrame(stored_state.get("events", []))
    if df.empty:
        return None
    return dcc.send_data_frame(
        pd.DataFrame(generate_mitigation_table(df)).to_csv,
        "AO-LACCP_Mitigation_Plan.csv", index=False,
    )


if __name__ == "__main__":
    print("AO-LACCP SOC Dashboard starting...")
    print("Open http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
