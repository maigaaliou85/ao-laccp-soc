"""
AO-LACCP SOC Research Prototype
-------------------------------
A reproducible, synthetic-data SOC demonstrator designed for transparent evaluation.

Run:
    pip install dash pandas numpy plotly scikit-learn
    python ao_laccp_soc_research.py

Then open the local URL printed by Dash (usually http://127.0.0.1:8050).

Research integrity notes:
- All events are SYNTHETIC. This is not evidence of production SOC performance.
- The model is evaluated on a held-out test split. Training/test split occurs before model fitting.
- Baseline rules and AI-assisted decisions operate on the SAME held-out events.
- Risk scores, labels, and policy actions are explicitly defined in this file.
- Results depend on the documented synthetic generator and assumptions; validate on authorized,
  representative operational data before making real-world claims.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

APP_TITLE = "AO-LACCP SOC | Reproducible Research Prototype"
RANDOM_SEED = 42
ATTACK_TYPES = [
    "benign",
    "brute_force",
    "MITM",
    "zero_day",
    "ransomware",
    "phishing",
    "data_exfiltration",
]
FEATURES = [
    "failed_logins",
    "packet_rate",
    "latency_ms",
    "payload_entropy",
    "bytes_out_mb",
    "geo_anomaly",
    "endpoint_alerts",
    "auth_fail_ratio",
]
ACTION_ORDER = ["ALLOW_MONITOR", "STEP_UP_AUTH", "ISOLATE_HOST", "BLOCK_AND_ROTATE"]

COLORS = {
    "background": "#0b1220",
    "panel": "#121c2e",
    "panel2": "#17243a",
    "text": "#e7edf7",
    "muted": "#9aabc4",
    "accent": "#62d6c7",
    "orange": "#f2b66d",
    "red": "#f07178",
}


def generate_synthetic_events(n_events: int = 2400, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate transparent, labeled synthetic SOC events with overlapping distributions."""
    rng = np.random.default_rng(int(seed))
    labels = rng.choice(
        ATTACK_TYPES,
        size=n_events,
        p=[0.48, 0.12, 0.08, 0.06, 0.09, 0.10, 0.07],
    )
    rows = []
    for label in labels:
        # Start with noisy benign-like background, then apply attack-specific signals.
        failed = rng.poisson(1.4)
        packet = rng.lognormal(mean=4.15, sigma=0.48)
        latency = np.clip(rng.normal(105, 28), 20, 450)
        entropy = np.clip(rng.normal(0.48, 0.14), 0, 1)
        bytes_out = rng.lognormal(mean=1.8, sigma=0.65)
        geo = rng.binomial(1, 0.035)
        endpoint = rng.poisson(0.25)
        auth_ratio = np.clip(rng.beta(2, 12), 0, 1)

        if label == "brute_force":
            failed += rng.poisson(15) + 5
            auth_ratio = np.clip(rng.beta(8, 2), 0, 1)
            geo = rng.binomial(1, 0.18)
        elif label == "MITM":
            latency += rng.normal(55, 22)
            entropy = np.clip(entropy + rng.normal(0.16, 0.12), 0, 1)
            endpoint += rng.poisson(1.0)
            geo = rng.binomial(1, 0.16)
        elif label == "zero_day":
            endpoint += rng.poisson(3.0) + 1
            entropy = np.clip(entropy + rng.normal(0.12, 0.18), 0, 1)
            packet *= rng.uniform(1.15, 2.0)
            geo = rng.binomial(1, 0.14)
        elif label == "ransomware":
            endpoint += rng.poisson(4.0) + 2
            entropy = np.clip(entropy + rng.normal(0.25, 0.12), 0, 1)
            bytes_out *= rng.uniform(1.2, 3.5)
            packet *= rng.uniform(1.0, 2.0)
        elif label == "phishing":
            failed += rng.poisson(3)
            auth_ratio = np.clip(auth_ratio + rng.uniform(0.05, 0.4), 0, 1)
            geo = rng.binomial(1, 0.12)
            endpoint += rng.poisson(0.8)
        elif label == "data_exfiltration":
            bytes_out *= rng.uniform(4.0, 10.0)
            packet *= rng.uniform(1.3, 3.0)
            entropy = np.clip(entropy + rng.normal(0.12, 0.15), 0, 1)
            geo = rng.binomial(1, 0.22)

        # Add common noise so categories overlap and the task is not artificially perfect.
        failed = max(0, int(failed + rng.integers(-1, 2)))
        packet = max(1.0, packet * rng.lognormal(0, 0.12))
        latency = float(np.clip(latency + rng.normal(0, 10), 10, 600))
        bytes_out = max(0.01, bytes_out * rng.lognormal(0, 0.15))
        endpoint = max(0, int(endpoint))
        auth_ratio = float(np.clip(auth_ratio + rng.normal(0, 0.035), 0, 1))

        rows.append(
            {
                "failed_logins": int(failed),
                "packet_rate": round(float(packet), 3),
                "latency_ms": round(latency, 2),
                "payload_entropy": round(float(entropy), 3),
                "bytes_out_mb": round(float(bytes_out), 3),
                "geo_anomaly": int(geo),
                "endpoint_alerts": int(endpoint),
                "auth_fail_ratio": round(auth_ratio, 3),
                "true_label": label,
            }
        )
    df = pd.DataFrame(rows)
    # Event identifiers are metadata only; they are not used as model features.
    df.insert(0, "event_id", [f"SYN-{int(seed):04d}-{i+1:05d}" for i in range(len(df))])
    return df


def rule_based_baseline(row: pd.Series) -> str:
    """Explicit, deterministic baseline classifier using only observable event features."""
    if (
        row["failed_logins"] >= 12
        or row["auth_fail_ratio"] >= 0.78
    ):
        return "brute_force"
    if row["bytes_out_mb"] >= 35 and row["packet_rate"] >= 75:
        return "data_exfiltration"
    if row["endpoint_alerts"] >= 5 and row["payload_entropy"] >= 0.72:
        return "ransomware"
    if row["endpoint_alerts"] >= 3 or row["payload_entropy"] >= 0.82:
        return "zero_day"
    if row["latency_ms"] >= 185 and row["geo_anomaly"] == 1:
        return "MITM"
    if row["geo_anomaly"] == 1 and row["failed_logins"] >= 3:
        return "phishing"
    return "benign"


def risk_score(row: pd.Series, predicted_label: str, confidence: float) -> float:
    """Transparent bounded policy score; this is a policy heuristic, not a calibrated probability."""
    score = 0.10
    score += min(float(row["failed_logins"]) / 25.0, 0.22)
    score += min(float(row["endpoint_alerts"]) / 10.0, 0.22)
    score += min(float(row["bytes_out_mb"]) / 100.0, 0.18)
    score += min(float(row["payload_entropy"]), 1.0) * 0.10
    score += 0.08 if int(row["geo_anomaly"]) else 0.0
    score += 0.10 if predicted_label != "benign" else 0.0
    score += 0.08 * float(np.clip(confidence, 0, 1))
    return float(np.clip(score, 0, 1))


def choose_crypto_mode(row: pd.Series, predicted_label: str, score: float) -> str:
    """Illustrative crypto-agility policy response; no actual cryptography is performed."""
    if score >= 0.72 or predicted_label in {"ransomware", "data_exfiltration", "zero_day"}:
        return "PQC-READY / REKEY REVIEW"
    if predicted_label in {"MITM", "brute_force", "phishing"} or row["geo_anomaly"] == 1:
        return "HYBRID / STEP-UP"
    return "CURRENT POLICY / MONITOR"


def choose_action(predicted_label: str, score: float) -> str:
    if score >= 0.78 or predicted_label in {"ransomware", "data_exfiltration"} and score >= 0.62:
        return "BLOCK_AND_ROTATE"
    if score >= 0.58 or predicted_label in {"zero_day", "MITM"}:
        return "ISOLATE_HOST"
    if score >= 0.34 or predicted_label in {"brute_force", "phishing"}:
        return "STEP_UP_AUTH"
    return "ALLOW_MONITOR"


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=int(seed),
                    n_jobs=-1,
                ),
            ),
        ]
    )


@dataclass
class Evaluation:
    events: pd.DataFrame
    metrics: Dict[str, Dict[str, float]]
    report: Dict[str, Dict[str, Dict[str, float]]]
    confusion: Dict[str, np.ndarray]
    feature_importance: pd.DataFrame
    run_info: Dict[str, object]


def evaluate(seed: int = RANDOM_SEED, n_events: int = 2400, test_size: float = 0.25) -> Evaluation:
    data = generate_synthetic_events(n_events=n_events, seed=seed)
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=int(seed),
        stratify=data["true_label"],
    )
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    model = build_model(seed)
    model.fit(train[FEATURES], train["true_label"])

    t0 = time.perf_counter()
    ai_pred = model.predict(test[FEATURES])
    ai_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    probabilities = model.predict_proba(test[FEATURES])
    classes = list(model.named_steps["classifier"].classes_)
    ai_conf = probabilities.max(axis=1)

    # Baseline and AI model are evaluated against the same held-out records.
    base_pred = test.apply(rule_based_baseline, axis=1).to_numpy()
    out = test.copy().reset_index(drop=True)
    out["baseline_prediction"] = base_pred
    out["ai_prediction"] = ai_pred
    out["ai_confidence"] = ai_conf
    out["baseline_correct"] = out["baseline_prediction"] == out["true_label"]
    out["ai_correct"] = out["ai_prediction"] == out["true_label"]

    # Decision/action logic is applied after prediction and does not use the true label.
    base_scores, ai_scores, base_actions, ai_actions, crypto_modes = [], [], [], [], []
    for i, row in out.iterrows():
        base_score = risk_score(row, row["baseline_prediction"], 0.50)
        ai_score = risk_score(row, row["ai_prediction"], float(row["ai_confidence"]))
        base_scores.append(base_score)
        ai_scores.append(ai_score)
        base_actions.append(choose_action(row["baseline_prediction"], base_score))
        ai_actions.append(choose_action(row["ai_prediction"], ai_score))
        crypto_modes.append(choose_crypto_mode(row, row["ai_prediction"], ai_score))
    out["baseline_risk_score"] = base_scores
    out["ai_risk_score"] = ai_scores
    out["baseline_action"] = base_actions
    out["ao_laccp_action"] = ai_actions
    out["crypto_policy_recommendation"] = crypto_modes

    metric_dict = {}
    report_dict = {}
    confusion_dict = {}
    for name, pred in [("Rule baseline", base_pred), ("Random Forest", ai_pred)]:
        metric_dict[name] = {
            "accuracy": float(accuracy_score(test["true_label"], pred)),
            "macro_precision": float(precision_score(test["true_label"], pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(test["true_label"], pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(test["true_label"], pred, average="macro", zero_division=0)),
        }
        report_dict[name] = classification_report(
            test["true_label"], pred, labels=ATTACK_TYPES, output_dict=True, zero_division=0
        )
        confusion_dict[name] = confusion_matrix(test["true_label"], pred, labels=ATTACK_TYPES)

    importances = model.named_steps["classifier"].feature_importances_
    fi = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    info = {
        "seed": int(seed),
        "n_events": int(n_events),
        "train_events": int(len(train)),
        "test_events": int(len(test)),
        "test_fraction": float(test_size),
        "model": "RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=2, class_weight='balanced_subsample')",
        "features": FEATURES,
        "labels": ATTACK_TYPES,
        "prediction_elapsed_ms_for_test_batch": round(float(ai_elapsed_ms), 3),
        "warning": "Synthetic benchmark only; this is not production evidence.",
    }
    return Evaluation(out, metric_dict, report_dict, confusion_dict, fi, info)


def card(title: str, value: str, detail: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"}),
            html.Div(value, style={"fontSize": "25px", "fontWeight": "700", "color": COLORS["text"]}),
            html.Div(detail, style={"fontSize": "11px", "color": COLORS["muted"], "marginTop": "6px"}),
        ],
        style={
            "background": COLORS["panel2"],
            "border": "1px solid #263651",
            "borderRadius": "12px",
            "padding": "16px",
            "minWidth": "150px",
            "flex": "1",
        },
    )


def figure_style(fig, title: str):
    fig.update_layout(
        title=title,
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font={"color": COLORS["text"]},
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
        legend={"bgcolor": COLORS["panel"]},
    )
    fig.update_xaxes(gridcolor="#293953")
    fig.update_yaxes(gridcolor="#293953")
    return fig


app = Dash(__name__)
app.title = APP_TITLE

app.layout = html.Div(
    style={
        "background": COLORS["background"],
        "color": COLORS["text"],
        "minHeight": "100vh",
        "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        "padding": "22px",
    },
    children=[
        html.Div(
            [
                html.Div("AO-LACCP", style={"fontSize": "13px", "letterSpacing": "3px", "color": COLORS["accent"], "fontWeight": "700"}),
                html.H1("SOC Intelligence & Evaluation Workbench", style={"margin": "6px 0", "fontSize": "30px"}),
                html.Div(
                    "Reproducible synthetic-event benchmark · Rule baseline vs. AI-assisted decision pipeline",
                    style={"color": COLORS["muted"], "fontSize": "14px"},
                ),
            ],
            style={"marginBottom": "18px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Synthetic events", style={"color": COLORS["muted"], "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="n-events",
                            options=[{"label": str(x), "value": x} for x in [600, 1200, 2400, 5000]],
                            value=2400,
                            clearable=False,
                            style={"color": "#152238"},
                        ),
                    ],
                    style={"minWidth": "180px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Random seed", style={"color": COLORS["muted"], "display": "block", "marginBottom": "6px"}),
                        dcc.Input(id="seed", type="number", value=42, min=0, step=1, style={"width": "100%", "padding": "9px", "borderRadius": "6px"}),
                    ],
                    style={"minWidth": "140px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Test split", style={"color": COLORS["muted"], "display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="test-split",
                            options=[
                                {"label": "20% test", "value": 0.20},
                                {"label": "25% test", "value": 0.25},
                                {"label": "30% test", "value": 0.30},
                            ],
                            value=0.25,
                            clearable=False,
                            style={"color": "#152238"},
                        ),
                    ],
                    style={"minWidth": "160px", "flex": "1"},
                ),
                html.Button(
                    "Run reproducible evaluation",
                    id="run-evaluation",
                    n_clicks=0,
                    style={
                        "background": COLORS["accent"],
                        "color": "#09202a",
                        "border": "none",
                        "borderRadius": "8px",
                        "fontWeight": "700",
                        "padding": "12px 18px",
                        "cursor": "pointer",
                        "alignSelf": "end",
                    },
                ),
            ],
            style={"display": "flex", "gap": "14px", "alignItems": "end", "flexWrap": "wrap", "background": COLORS["panel"], "padding": "16px", "borderRadius": "12px", "marginBottom": "16px"},
        ),
        html.Div(
            "RESEARCH BOUNDARY: Generated data only. The model's scores are benchmark results under these assumptions—not proof of real-world SOC effectiveness. No live network traffic is collected, no hosts are isolated, and no cryptographic keys are changed.",
            style={"background": "#2c2430", "border": "1px solid #68404d", "padding": "12px 14px", "borderRadius": "8px", "color": "#ffd9df", "fontSize": "12px", "marginBottom": "16px"},
        ),
        dcc.Store(id="evaluation-store"),
        html.Div(id="status-line", style={"color": COLORS["muted"], "marginBottom": "12px"}),
        html.Div(id="metric-cards", style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}),
        html.Div(
            [
                html.Div(dcc.Graph(id="metric-chart"), style={"flex": "1", "minWidth": "300px", "background": COLORS["panel"], "borderRadius": "12px", "padding": "8px"}),
                html.Div(dcc.Graph(id="action-chart"), style={"flex": "1", "minWidth": "300px", "background": COLORS["panel"], "borderRadius": "12px", "padding": "8px"}),
            ],
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "14px"},
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="confusion-chart"), style={"flex": "1", "minWidth": "320px", "background": COLORS["panel"], "borderRadius": "12px", "padding": "8px"}),
                html.Div(dcc.Graph(id="feature-chart"), style={"flex": "1", "minWidth": "320px", "background": COLORS["panel"], "borderRadius": "12px", "padding": "8px"}),
            ],
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "14px"},
        ),
        html.Div(
            [
                html.H3("Held-out event decision trace", style={"marginTop": "4px"}),
                html.Div("Each row shows the same test event passed through both the explicit rule baseline and the AI-assisted pipeline. True labels are shown for offline evaluation only; the decision functions do not receive them.", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "10px"}),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="label-filter",
                            options=[{"label": "All true labels", "value": "ALL"}] + [{"label": x, "value": x} for x in ATTACK_TYPES],
                            value="ALL",
                            clearable=False,
                            style={"color": "#152238", "minWidth": "210px"},
                        ),
                        dcc.Input(id="table-search", type="text", value="", placeholder="Filter event ID / prediction...", style={"padding": "9px", "borderRadius": "6px", "minWidth": "240px"}),
                        html.Button("Export test results CSV", id="export-button", n_clicks=0, style={"padding": "10px 14px", "borderRadius": "7px", "border": "1px solid #405573", "background": COLORS["panel2"], "color": COLORS["text"], "cursor": "pointer"}),
                        dcc.Download(id="download-csv"),
                        html.Button("Download run manifest JSON", id="manifest-button", n_clicks=0, style={"padding": "10px 14px", "borderRadius": "7px", "border": "1px solid #405573", "background": COLORS["panel2"], "color": COLORS["text"], "cursor": "pointer"}),
                        dcc.Download(id="download-manifest"),
                    ],
                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "alignItems": "center", "marginBottom": "10px"},
                ),
                html.Div(id="event-table"),
            ],
            style={"background": COLORS["panel"], "padding": "16px", "borderRadius": "12px", "marginBottom": "14px"},
        ),
        html.Details(
            [
                html.Summary("Model, data-generation, and decision-policy documentation", style={"cursor": "pointer", "fontWeight": "700", "padding": "10px 0"}),
                dcc.Markdown(
                    """
**Synthetic event generation.** Each record has eight observable features: failed logins, packet rate, latency, payload entropy, outbound volume, geographic anomaly, endpoint alerts, and authentication-failure ratio. A synthetic `true_label` is sampled from the documented class mix and then feature distributions are perturbed by class-specific signals plus noise. This creates a controlled benchmark, not a validated representation of any bank or production network.

**AI model.** A scikit-learn pipeline imputes missing numeric values and trains a Random Forest classifier. The split is stratified and performed before model fitting. The `true_label` and `event_id` are excluded from the model inputs. Macro precision, macro recall, macro F1, accuracy, per-class report, and confusion matrix are calculated on held-out rows.

**Baseline.** A deterministic rule-based classifier uses only the same eight observable features. It is not intentionally given a different event population. The thresholds are illustrative and should be pre-registered or tuned only on training/validation data for a formal study.

**Decision pipeline.** Predicted labels and observable features feed an explicit bounded policy-risk heuristic. Action thresholds map the score and predicted class to `ALLOW_MONITOR`, `STEP_UP_AUTH`, `ISOLATE_HOST`, or `BLOCK_AND_ROTATE`. A separate illustrative policy recommends a crypto posture. These are recommendations in the simulator; no actual control-plane enforcement or cryptographic operation occurs.

**For a defensible research claim.** Freeze the generator and policies before evaluation; run multiple seeds; report uncertainty intervals and effect sizes; compare on identical test events; test against independently curated or authorized operational data; and report false-positive/false-negative costs. Do not describe synthetic results as proof of operational effectiveness.
                    """,
                    style={"color": COLORS["muted"], "lineHeight": "1.6"},
                ),
            ],
            style={"background": COLORS["panel"], "padding": "14px 18px", "borderRadius": "12px"},
        ),
    ],
)


def _get_eval(store):
    if not store:
        return None
    return store


@app.callback(
    Output("evaluation-store", "data"),
    Output("status-line", "children"),
    Input("run-evaluation", "n_clicks"),
    State("n-events", "value"),
    State("seed", "value"),
    State("test-split", "value"),
    prevent_initial_call=False,
)
def run_eval(n_clicks, n_events, seed, test_split):
    seed = RANDOM_SEED if seed is None else int(seed)
    n_events = 2400 if n_events is None else int(n_events)
    test_split = 0.25 if test_split is None else float(test_split)
    try:
        ev = evaluate(seed=seed, n_events=n_events, test_size=test_split)
        store = {
            "events": ev.events.to_dict("records"),
            "metrics": ev.metrics,
            "report": ev.report,
            "confusion": {k: v.tolist() for k, v in ev.confusion.items()},
            "feature_importance": ev.feature_importance.to_dict("records"),
            "run_info": ev.run_info,
        }
        status = (
            f"Evaluation complete · seed={seed} · train={ev.run_info['train_events']:,} · "
            f"held-out test={ev.run_info['test_events']:,} · model prediction time for test batch="
            f"{ev.run_info['prediction_elapsed_ms_for_test_batch']:.2f} ms. "
            "This timing is local inference timing and is not end-to-end network latency."
        )
        return store, status
    except Exception as exc:
        return no_update, f"Evaluation failed: {type(exc).__name__}: {exc}"


@app.callback(
    Output("metric-cards", "children"),
    Output("metric-chart", "figure"),
    Output("action-chart", "figure"),
    Output("confusion-chart", "figure"),
    Output("feature-chart", "figure"),
    Input("evaluation-store", "data"),
)
def render_evaluation(store):
    if not store:
        empty = figure_style(go.Figure(), "Run evaluation to display results")
        return [], empty, empty, empty, empty

    metrics = store["metrics"]
    events = pd.DataFrame(store["events"])
    cards = [
        card("AI macro F1", f"{metrics['Random Forest']['macro_f1']:.3f}", "Held-out test set"),
        card("Baseline macro F1", f"{metrics['Rule baseline']['macro_f1']:.3f}", "Same held-out events"),
        card("AI macro recall", f"{metrics['Random Forest']['macro_recall']:.3f}", "Across all event classes"),
        card("AI accuracy", f"{metrics['Random Forest']['accuracy']:.3f}", "Overall held-out accuracy"),
        card("Test events", f"{len(events):,}", "No training rows included"),
    ]

    metric_rows = []
    for model_name, vals in metrics.items():
        for metric_name in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            metric_rows.append({"Model": model_name, "Metric": metric_name.replace("_", " ").title(), "Score": vals[metric_name]})
    metric_df = pd.DataFrame(metric_rows)
    fig_metrics = px.bar(
        metric_df, x="Metric", y="Score", color="Model", barmode="group",
        category_orders={"Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]},
        range_y=[0, 1], text_auto=".3f",
    )
    fig_metrics = figure_style(fig_metrics, "Fair held-out classifier comparison")

    action_counts = pd.DataFrame({
        "Action": ACTION_ORDER,
        "Rule baseline": [int((events["baseline_action"] == a).sum()) for a in ACTION_ORDER],
        "AO-LACCP AI-assisted": [int((events["ao_laccp_action"] == a).sum()) for a in ACTION_ORDER],
    }).melt(id_vars="Action", var_name="Pipeline", value_name="Event count")
    fig_actions = px.bar(action_counts, x="Action", y="Event count", color="Pipeline", barmode="group")
    fig_actions = figure_style(fig_actions, "Policy actions on identical test events")

    cm = np.array(store["confusion"]["Random Forest"])
    fig_cm = px.imshow(
        cm,
        x=ATTACK_TYPES,
        y=ATTACK_TYPES,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Teal",
        labels={"x": "Predicted label", "y": "True label", "color": "Count"},
    )
    fig_cm = figure_style(fig_cm, "AI model confusion matrix (held-out test)")
    fig_cm.update_xaxes(tickangle=-35)

    fi = pd.DataFrame(store["feature_importance"]).sort_values("importance", ascending=True)
    fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", text_auto=".3f")
    fig_fi = figure_style(fig_fi, "Random Forest feature importance")
    return cards, fig_metrics, fig_actions, fig_cm, fig_fi


@app.callback(
    Output("event-table", "children"),
    Input("evaluation-store", "data"),
    Input("label-filter", "value"),
    Input("table-search", "value"),
)
def render_table(store, label_filter, search):
    if not store:
        return html.Div("Run an evaluation to inspect held-out events.", style={"color": COLORS["muted"]})
    df = pd.DataFrame(store["events"])
    if label_filter and label_filter != "ALL":
        df = df[df["true_label"] == label_filter]
    if search:
        q = str(search).strip().lower()
        if q:
            mask = (
                df["event_id"].str.lower().str.contains(q, na=False)
                | df["true_label"].str.lower().str.contains(q, na=False)
                | df["baseline_prediction"].str.lower().str.contains(q, na=False)
                | df["ai_prediction"].str.lower().str.contains(q, na=False)
                | df["ao_laccp_action"].str.lower().str.contains(q, na=False)
            )
            df = df[mask]
    columns = [
        "event_id", "true_label", "baseline_prediction", "ai_prediction",
        "baseline_correct", "ai_correct", "baseline_risk_score", "ai_risk_score",
        "baseline_action", "ao_laccp_action", "crypto_policy_recommendation",
    ]
    display_df = df[columns].head(250).copy()
    for col in ["baseline_risk_score", "ai_risk_score"]:
        display_df[col] = display_df[col].round(3)
    return html.Div(
        [
            html.Div(f"Showing {len(display_df):,} rows (maximum 250 shown for browser responsiveness). Export includes all held-out test rows.", style={"color": COLORS["muted"], "fontSize": "11px", "marginBottom": "8px"}),
            dash_table.DataTable(
                data=display_df.to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in columns],
                page_size=12,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#1b2b43", "color": COLORS["text"], "fontWeight": "700", "border": "1px solid #31425d"},
                style_cell={"backgroundColor": COLORS["panel"], "color": COLORS["text"], "border": "1px solid #263651", "padding": "8px", "fontSize": "11px", "minWidth": "100px", "maxWidth": "240px", "whiteSpace": "normal"},
                style_data_conditional=[
                    {"if": {"filter_query": "{ai_correct} = true", "column_id": "ai_correct"}, "color": COLORS["accent"], "fontWeight": "700"},
                    {"if": {"filter_query": "{ai_correct} = false", "column_id": "ai_correct"}, "color": COLORS["red"], "fontWeight": "700"},
                ],
            ),
        ]
    )


@app.callback(
    Output("download-csv", "data"),
    Input("export-button", "n_clicks"),
    State("evaluation-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, store):
    if not store:
        return no_update
    df = pd.DataFrame(store["events"])
    return dcc.send_data_frame(df.to_csv, "ao_laccp_heldout_test_results.csv", index=False)


@app.callback(
    Output("download-manifest", "data"),
    Input("manifest-button", "n_clicks"),
    State("evaluation-store", "data"),
    prevent_initial_call=True,
)
def export_manifest(n_clicks, store):
    if not store:
        return no_update
    manifest = {
        "run_info": store["run_info"],
        "metrics": store["metrics"],
        "notes": [
            "Synthetic benchmark only; not production validation.",
            "Baseline and AI predictions are compared on identical held-out events.",
            "Model features exclude event_id and true_label.",
            "Policy actions and crypto posture are recommendations only; no controls are enforced.",
        ],
    }
    return dict(content=json.dumps(manifest, indent=2), filename="ao_laccp_run_manifest.json", type="application/json")


if __name__ == "__main__":
    print("=" * 72)
    print(APP_TITLE)
    print("Synthetic-data research prototype — not for live production enforcement.")
    print("Open http://127.0.0.1:8050 after the server starts.")
    print("=" * 72)
    app.run(debug=False, host="127.0.0.1", port=8050)

