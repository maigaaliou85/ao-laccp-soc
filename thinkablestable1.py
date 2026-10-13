
"""
AO-LACCP SOC Research Prototype
===============================

Features:
- Reproducible synthetic-data benchmark.
- CSV upload and validation.
- Synthetic-trained model inference on uploaded records.
- Optional train/test evaluation on uploaded labeled data.
- Rule-based baseline versus Random Forest comparison.
- Accuracy, macro precision, macro recall, macro F1.
- Confusion matrix, feature importance, risk scores and policy actions.
- CSV exports and JSON run manifest.

Install:
    pip install dash pandas numpy plotly scikit-learn

Run:
    python ao_laccp_soc_research.py

Open:
    http://127.0.0.1:8050

Research integrity:
- No production controls are enforced.
- No actual cryptographic operations are performed.
- Synthetic results are not production validation.
- Uploaded data is processed by the local application process.
- Use only data you are authorized to process.
"""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import (
    Dash, Input, Output, State, dcc, html,
    dash_table, no_update
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "AO-LACCP SOC | Research Prototype"
RANDOM_SEED = 42
MAX_UPLOAD_ROWS = 100_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

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

ACTION_ORDER = [
    "ALLOW_MONITOR",
    "STEP_UP_AUTH",
    "ISOLATE_HOST",
    "BLOCK_AND_ROTATE",
]

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


# ============================================================
# DATA GENERATION
# ============================================================

def generate_synthetic_events(
    n_events: int = 2400,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate reproducible, labeled synthetic SOC events."""

    rng = np.random.default_rng(int(seed))

    labels = rng.choice(
        ATTACK_TYPES,
        size=n_events,
        p=[0.48, 0.12, 0.08, 0.06, 0.09, 0.10, 0.07],
    )

    rows = []

    for label in labels:
        failed = rng.poisson(1.4)
        packet = rng.lognormal(4.15, 0.48)
        latency = np.clip(rng.normal(105, 28), 20, 450)
        entropy = np.clip(rng.normal(0.48, 0.14), 0, 1)
        bytes_out = rng.lognormal(1.8, 0.65)
        geo = rng.binomial(1, 0.035)
        endpoint = rng.poisson(0.25)
        auth_ratio = np.clip(rng.beta(2, 12), 0, 1)

        if label == "brute_force":
            failed += rng.poisson(15) + 5
            auth_ratio = np.clip(rng.beta(8, 2), 0, 1)
            geo = rng.binomial(1, 0.18)

        elif label == "MITM":
            latency += rng.normal(55, 22)
            entropy = np.clip(
                entropy + rng.normal(0.16, 0.12), 0, 1
            )
            endpoint += rng.poisson(1.0)
            geo = rng.binomial(1, 0.16)

        elif label == "zero_day":
            endpoint += rng.poisson(3.0) + 1
            entropy = np.clip(
                entropy + rng.normal(0.12, 0.18), 0, 1
            )
            packet *= rng.uniform(1.15, 2.0)
            geo = rng.binomial(1, 0.14)

        elif label == "ransomware":
            endpoint += rng.poisson(4.0) + 2
            entropy = np.clip(
                entropy + rng.normal(0.25, 0.12), 0, 1
            )
            bytes_out *= rng.uniform(1.2, 3.5)
            packet *= rng.uniform(1.0, 2.0)

        elif label == "phishing":
            failed += rng.poisson(3)
            auth_ratio = np.clip(
                auth_ratio + rng.uniform(0.05, 0.4), 0, 1
            )
            geo = rng.binomial(1, 0.12)
            endpoint += rng.poisson(0.8)

        elif label == "data_exfiltration":
            bytes_out *= rng.uniform(4.0, 10.0)
            packet *= rng.uniform(1.3, 3.0)
            entropy = np.clip(
                entropy + rng.normal(0.12, 0.15), 0, 1
            )
            geo = rng.binomial(1, 0.22)

        failed = max(0, int(failed + rng.integers(-1, 2)))
        packet = max(1.0, packet * rng.lognormal(0, 0.12))
        latency = float(
            np.clip(latency + rng.normal(0, 10), 10, 600)
        )
        bytes_out = max(
            0.01, bytes_out * rng.lognormal(0, 0.15)
        )
        endpoint = max(0, int(endpoint))
        auth_ratio = float(
            np.clip(auth_ratio + rng.normal(0, 0.035), 0, 1)
        )

        rows.append({
            "failed_logins": int(failed),
            "packet_rate": round(float(packet), 3),
            "latency_ms": round(latency, 2),
            "payload_entropy": round(float(entropy), 3),
            "bytes_out_mb": round(float(bytes_out), 3),
            "geo_anomaly": int(geo),
            "endpoint_alerts": int(endpoint),
            "auth_fail_ratio": round(auth_ratio, 3),
            "true_label": label,
        })

    df = pd.DataFrame(rows)

    df.insert(
        0,
        "event_id",
        [
            f"SYN-{int(seed):04d}-{i + 1:05d}"
            for i in range(len(df))
        ],
    )

    return df


# ============================================================
# BASELINE AND POLICY DECISIONS
# ============================================================

def rule_based_baseline(row: pd.Series) -> str:
    """Transparent deterministic rule-based classifier."""

    if (
        row["failed_logins"] >= 12
        or row["auth_fail_ratio"] >= 0.78
    ):
        return "brute_force"

    if (
        row["bytes_out_mb"] >= 35
        and row["packet_rate"] >= 75
    ):
        return "data_exfiltration"

    if (
        row["endpoint_alerts"] >= 5
        and row["payload_entropy"] >= 0.72
    ):
        return "ransomware"

    if (
        row["endpoint_alerts"] >= 3
        or row["payload_entropy"] >= 0.82
    ):
        return "zero_day"

    if (
        row["latency_ms"] >= 185
        and row["geo_anomaly"] == 1
    ):
        return "MITM"

    if (
        row["geo_anomaly"] == 1
        and row["failed_logins"] >= 3
    ):
        return "phishing"

    return "benign"


def risk_score(
    row: pd.Series,
    predicted_label: str,
    confidence: float,
) -> float:
    """
    Bounded policy heuristic, not a calibrated probability
    or validated estimate of actual incident risk.
    """

    score = 0.10
    score += min(float(row["failed_logins"]) / 25.0, 0.22)
    score += min(float(row["endpoint_alerts"]) / 10.0, 0.22)
    score += min(float(row["bytes_out_mb"]) / 100.0, 0.18)
    score += min(float(row["payload_entropy"]), 1.0) * 0.10
    score += 0.08 if int(row["geo_anomaly"]) else 0.0
    score += 0.10 if predicted_label != "benign" else 0.0
    score += 0.08 * float(np.clip(confidence, 0, 1))

    return float(np.clip(score, 0, 1))


def choose_crypto_mode(
    row: pd.Series,
    predicted_label: str,
    score: float,
) -> str:
    """Illustrative policy recommendation; no crypto is performed."""

    if (
        score >= 0.72
        or predicted_label in {
            "ransomware", "data_exfiltration", "zero_day"
        }
    ):
        return "PQC-READY / REKEY REVIEW"

    if (
        predicted_label in {"MITM", "brute_force", "phishing"}
        or row["geo_anomaly"] == 1
    ):
        return "HYBRID / STEP-UP"

    return "CURRENT POLICY / MONITOR"


def choose_action(predicted_label: str, score: float) -> str:
    """Map predicted threat and heuristic risk to a recommendation."""

    if (
        score >= 0.78
        or (
            predicted_label in {"ransomware", "data_exfiltration"}
            and score >= 0.62
        )
    ):
        return "BLOCK_AND_ROTATE"

    if score >= 0.58 or predicted_label in {"zero_day", "MITM"}:
        return "ISOLATE_HOST"

    if score >= 0.34 or predicted_label in {"brute_force", "phishing"}:
        return "STEP_UP_AUTH"

    return "ALLOW_MONITOR"


# ============================================================
# MODEL AND EVALUATION UTILITIES
# ============================================================

def build_model(seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline([
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
    ])


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
    }


def add_policy_outputs(
    records: pd.DataFrame,
    baseline_predictions,
    ai_predictions,
    ai_confidence,
) -> pd.DataFrame:
    """Apply both pipelines to the same observable records."""

    out = records.copy().reset_index(drop=True)

    out["baseline_prediction"] = np.asarray(baseline_predictions)
    out["ai_prediction"] = np.asarray(ai_predictions)
    out["ai_confidence"] = np.asarray(ai_confidence)

    baseline_scores = []
    ai_scores = []
    baseline_actions = []
    ai_actions = []
    crypto_modes = []

    for idx, row in out.iterrows():
        base_label = row["baseline_prediction"]
        ai_label = row["ai_prediction"]

        base_score = risk_score(row, base_label, 0.50)
        ai_score = risk_score(
            row, ai_label, float(row["ai_confidence"])
        )

        baseline_scores.append(base_score)
        ai_scores.append(ai_score)

        baseline_actions.append(
            choose_action(base_label, base_score)
        )
        ai_actions.append(choose_action(ai_label, ai_score))

        crypto_modes.append(
            choose_crypto_mode(row, ai_label, ai_score)
        )

    out["baseline_risk_score"] = baseline_scores
    out["ai_risk_score"] = ai_scores
    out["baseline_action"] = baseline_actions
    out["ao_laccp_action"] = ai_actions
    out["crypto_policy_recommendation"] = crypto_modes

    return out


def evaluate_synthetic(
    seed: int = RANDOM_SEED,
    n_events: int = 2400,
    test_size: float = 0.25,
) -> dict:
    """Stratified held-out evaluation on synthetic events."""

    data = generate_synthetic_events(n_events, seed)

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

    started = time.perf_counter()
    ai_predictions = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])
    inference_ms = (time.perf_counter() - started) * 1000

    confidence = probabilities.max(axis=1)
    baseline_predictions = test.apply(
        rule_based_baseline, axis=1
    ).to_numpy()

    results = add_policy_outputs(
        test,
        baseline_predictions,
        ai_predictions,
        confidence,
    )

    metrics = {
        "Rule baseline": calculate_metrics(
            test["true_label"], baseline_predictions
        ),
        "Random Forest": calculate_metrics(
            test["true_label"], ai_predictions
        ),
    }

    reports = {
        name: classification_report(
            test["true_label"],
            predictions,
            labels=ATTACK_TYPES,
            output_dict=True,
            zero_division=0,
        )
        for name, predictions in [
            ("Rule baseline", baseline_predictions),
            ("Random Forest", ai_predictions),
        ]
    }

    confusion = {
        "Rule baseline": confusion_matrix(
            test["true_label"],
            baseline_predictions,
            labels=ATTACK_TYPES,
        ).tolist(),
        "Random Forest": confusion_matrix(
            test["true_label"],
            ai_predictions,
            labels=ATTACK_TYPES,
        ).tolist(),
    }

    feature_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": (
            model.named_steps["classifier"].feature_importances_
        ),
    }).sort_values("importance", ascending=False)

    return {
        "events": results.to_dict("records"),
        "metrics": metrics,
        "report": reports,
        "confusion": confusion,
        "feature_importance": feature_importance.to_dict("records"),
        "run_info": {
            "evaluation_mode": "synthetic_holdout",
            "seed": int(seed),
            "n_events": int(n_events),
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "test_fraction": float(test_size),
            "model": "RandomForestClassifier",
            "features": FEATURES,
            "labels": ATTACK_TYPES,
            "prediction_elapsed_ms_for_test_batch": round(
                inference_ms, 3
            ),
            "warning": (
                "Synthetic benchmark only; not production validation."
            ),
        },
    }


# ============================================================
# CSV UPLOAD AND VALIDATION
# ============================================================

ALIASES = {
    "failed login": "failed_logins",
    "failed_login": "failed_logins",
    "failed_logins_count": "failed_logins",
    "packet rate": "packet_rate",
    "network_packet_rate": "packet_rate",
    "latency": "latency_ms",
    "network_latency_ms": "latency_ms",
    "entropy": "payload_entropy",
    "outbound_mb": "bytes_out_mb",
    "outbound_bytes_mb": "bytes_out_mb",
    "geographic_anomaly": "geo_anomaly",
    "endpoint_alert_count": "endpoint_alerts",
    "authentication_failure_ratio": "auth_fail_ratio",
    "auth_failure_ratio": "auth_fail_ratio",
}

LABEL_ALIASES = {
    "normal": "benign",
    "normal traffic": "benign",
    "benign": "benign",
    "brute force": "brute_force",
    "brute_force": "brute_force",
    "mitm": "MITM",
    "man-in-the-middle": "MITM",
    "man in the middle": "MITM",
    "zero day": "zero_day",
    "zero_day": "zero_day",
    "zeroday": "zero_day",
    "ransomware": "ransomware",
    "phishing": "phishing",
    "data exfiltration": "data_exfiltration",
    "data_exfiltration": "data_exfiltration",
}


def parse_uploaded_csv(contents: str) -> pd.DataFrame:
    """Decode and validate the size and structure of a CSV upload."""

    if not contents or "," not in contents:
        raise ValueError("No CSV contents were received.")

    try:
        _content_type, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("The uploaded file could not be decoded.") from exc

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(
            "File exceeds the 25 MB upload limit."
        )

    try:
        frame = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if frame.empty:
        raise ValueError("The CSV contains no data rows.")

    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(
            f"Maximum supported upload is {MAX_UPLOAD_ROWS:,} rows."
        )

    frame.columns = [str(c).strip() for c in frame.columns]

    if frame.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found. Rename them and retry."
        )

    rename = {}

    for col in frame.columns:
        normalized = col.strip().lower().replace("-", "_")

        if normalized in ALIASES:
            rename[col] = ALIASES[normalized]
        elif normalized.replace("_", " ") in ALIASES:
            rename[col] = ALIASES[
                normalized.replace("_", " ")
            ]

    frame = frame.rename(columns=rename)

    # Recheck after alias normalization.
    if frame.columns.duplicated().any():
        raise ValueError(
            "Column aliases produced duplicate feature names."
        )

    return frame


def extract_features(uploaded: pd.DataFrame) -> tuple:
    """Validate the eight documented features and return numeric data."""

    missing = [c for c in FEATURES if c not in uploaded.columns]

    if missing:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing)
            + ". Required columns are: "
            + ", ".join(FEATURES)
        )

    features = uploaded[FEATURES].copy()

    for col in FEATURES:
        features[col] = pd.to_numeric(
            features[col], errors="coerce"
        )

        values = features[col].to_numpy(dtype=float)

        if np.isinf(values).any():
            raise ValueError(
                f"Feature '{col}' contains infinite values."
            )

        if features[col].notna().sum() == 0:
            raise ValueError(
                f"Feature '{col}' has no usable numeric values."
            )

    geo_values = features["geo_anomaly"].dropna()

    if not geo_values.isin([0, 1]).all():
        raise ValueError(
            "'geo_anomaly' must contain only 0, 1, or missing values."
        )

    missing_counts = {
        col: int(features[col].isna().sum())
        for col in FEATURES
        if features[col].isna().any()
    }

    return features, missing_counts


def extract_labels(uploaded: pd.DataFrame):
    """Return recognized optional ground-truth labels."""

    label_col = next(
        (
            col for col in uploaded.columns
            if str(col).strip().lower()
            in {"true_label", "label", "attack_type", "class"}
        ),
        None,
    )

    if label_col is None:
        return None, None

    raw = uploaded[label_col]

    if raw.isna().any():
        raise ValueError(
            f"Label column '{label_col}' contains missing labels."
        )

    normalized = raw.astype(str).str.strip().map(
        lambda value: LABEL_ALIASES.get(
            value.lower(), value
        )
    )

    unknown = sorted(set(normalized) - set(ATTACK_TYPES))

    if unknown:
        raise ValueError(
            f"Unrecognized labels in '{label_col}': {unknown[:12]}. "
            f"Supported labels: {ATTACK_TYPES}"
        )

    if normalized.nunique() < 2:
        raise ValueError(
            "At least two different classes are required "
            "for labeled train/test evaluation."
        )

    return label_col, normalized


def evaluate_uploaded_dataset(
    contents: str,
    filename: str,
    seed: int,
    n_events: int,
    test_size: float,
    mode: str,
) -> dict:
    """
    Mode 1: synthetic-trained inference on external records.
    Mode 2: stratified train/test evaluation on labeled external data.

    Both modes use the same observable test/upload rows for the
    baseline and AI policy comparisons.
    """

    uploaded = parse_uploaded_csv(contents)
    features, missing_counts = extract_features(uploaded)

    seed = int(seed)
    n_events = int(n_events)

    label_col, labels = extract_labels(uploaded)

    if mode == "external_holdout":
        if labels is None:
            raise ValueError(
                "External train/test evaluation requires a recognized "
                "ground-truth label column: true_label, label, "
                "attack_type, or class."
            )

        # Each class must have at least two records for a stratified
        # split. This also avoids silently falling back to an
        # unstratified evaluation.
        class_counts = labels.value_counts()

        if (class_counts < 2).any():
            raise ValueError(
                "Each class must contain at least two records for "
                "stratified train/test evaluation. Smallest class "
                f"counts: {class_counts.to_dict()}"
            )

        indices = np.arange(len(uploaded))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(test_size),
            random_state=seed,
            stratify=labels,
        )

        model = build_model(seed)
        model.fit(
            features.iloc[train_idx],
            labels.iloc[train_idx],
        )

        test_features = features.iloc[test_idx].reset_index(drop=True)
        y_test = labels.iloc[test_idx].reset_index(drop=True)
        test_original = uploaded.iloc[test_idx].reset_index(drop=True)

        started = time.perf_counter()
        ai_predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)
        inference_ms = (time.perf_counter() - started) * 1000

        confidence = probabilities.max(axis=1)

        baseline_predictions = test_features.apply(
            rule_based_baseline, axis=1
        ).to_numpy()

        results = add_policy_outputs(
            test_features,
            baseline_predictions,
            ai_predictions,
            confidence,
        )

        results.insert(
            0,
            "upload_row",
            test_idx + 1,
        )

        results["evaluation_label"] = y_test
        results["baseline_correct"] = (
            results["baseline_prediction"] == y_test
        )
        results["ai_correct"] = (
            results["ai_prediction"] == y_test
        )

        # Retain original uploaded columns for traceability, while
        # keeping the model feature matrix separate from the labels.
        for col in test_original.columns:
            if col not in results.columns:
                results[col] = test_original[col].values

        metrics = {
            "Rule baseline": calculate_metrics(
                y_test, results["baseline_prediction"]
            ),
            "Random Forest": calculate_metrics(
                y_test, results["ai_prediction"]
            ),
        }

        confusion = {
            name: confusion_matrix(
                y_test,
                results[pred_col],
                labels=ATTACK_TYPES,
            ).tolist()
            for name, pred_col in [
                ("Rule baseline", "baseline_prediction"),
                ("Random Forest", "ai_prediction"),
            ]
        }

        report = {
            name: classification_report(
                y_test,
                results[pred_col],
                labels=ATTACK_TYPES,
                output_dict=True,
                zero_division=0,
            )
            for name, pred_col in [
                ("Rule baseline", "baseline_prediction"),
                ("Random Forest", "ai_prediction"),
            ]
        }

        feature_importance = pd.DataFrame({
            "feature": FEATURES,
            "importance": (
                model.named_steps["classifier"].feature_importances_
            ),
        }).sort_values("importance", ascending=False)

        run_info = {
            "evaluation_mode": "external_labeled_holdout",
            "filename": filename,
            "seed": seed,
            "total_uploaded_rows": len(uploaded),
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "test_fraction": float(test_size),
            "label_column": label_col,
            "training_data": "Uploaded labeled dataset",
            "model": "RandomForestClassifier",
            "features": FEATURES,
            "labels": ATTACK_TYPES,
            "prediction_elapsed_ms_for_test_batch": round(
                inference_ms, 3
            ),
            "warning": (
                "External-data evaluation; results depend on the "
                "dataset's provenance, labels, representativeness, "
                "and quality."
            ),
        }

        display_results = results

    else:
        # Inference-only mode: train on synthetic data and predict
        # uploaded rows. No accuracy/F1 is claimed unless labels are
        # present, and any such metrics describe the full upload
        # rather than an independent held-out external test set.
        training_data = generate_synthetic_events(
            n_events=n_events,
            seed=seed,
        )

        model = build_model(seed)
        model.fit(
            training_data[FEATURES],
            training_data["true_label"],
        )

        started = time.perf_counter()
        ai_predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        inference_ms = (time.perf_counter() - started) * 1000

        confidence = probabilities.max(axis=1)

        baseline_predictions = features.apply(
            rule_based_baseline, axis=1
        ).to_numpy()

        results = add_policy_outputs(
            features,
            baseline_predictions,
            ai_predictions,
            confidence,
        )

        results.insert(
            0,
            "upload_row",
            np.arange(1, len(results) + 1),
        )

        for col in uploaded.columns:
            if col not in results.columns:
                results[col] = uploaded[col].reset_index(drop=True)

        metrics = None
        report = None
        confusion = None

        if labels is not None:
            y_all = labels.reset_index(drop=True)

            results["evaluation_label"] = y_all
            results["baseline_correct"] = (
                results["baseline_prediction"] == y_all
            )
            results["ai_correct"] = (
                results["ai_prediction"] == y_all
            )

            metrics = {
                "Rule baseline": calculate_metrics(
                    y_all, results["baseline_prediction"]
                ),
                "Random Forest": calculate_metrics(
                    y_all, results["ai_prediction"]
                ),
            }

            report = {
                name: classification_report(
                    y_all,
                    results[pred_col],
                    labels=ATTACK_TYPES,
                    output_dict=True,
                    zero_division=0,
                )
                for name, pred_col in [
                    ("Rule baseline", "baseline_prediction"),
                    ("Random Forest", "ai_prediction"),
                ]
            }

            confusion = {
                name: confusion_matrix(
                    y_all,
                    results[pred_col],
                    labels=ATTACK_TYPES,
                ).tolist()
                for name, pred_col in [
                    ("Rule baseline", "baseline_prediction"),
                    ("Random Forest", "ai_prediction"),
                ]
            }

        feature_importance = pd.DataFrame({
            "feature": FEATURES,
            "importance": (
                model.named_steps["classifier"].feature_importances_
            ),
        }).sort_values("importance", ascending=False)

        run_info = {
            "evaluation_mode": "synthetic_trained_external_inference",
            "filename": filename,
            "seed": seed,
            "uploaded_rows": len(uploaded),
            "synthetic_training_rows": len(training_data),
            "label_column": label_col,
            "training_data": "Synthetic prototype data",
            "model": "RandomForestClassifier",
            "features": FEATURES,
            "labels": ATTACK_TYPES,
            "prediction_elapsed_ms_for_upload": round(
                inference_ms, 3
            ),
            "warning": (
                "Synthetic-trained external inference. If labels are "
                "present, metrics on the upload are descriptive and "
                "are not an independent held-out external test."
            ),
        }

        display_results = results

    # Convert NaN/Inf values to JSON-safe None values before Dash Store.
    display_results = display_results.replace(
        [np.inf, -np.inf], np.nan
    )

    display_results = display_results.astype(object).where(
        pd.notna(display_results), None
    )

    return {
        "filename": filename or "uploaded.csv",
        "rows": display_results.to_dict("records"),
        "row_count": int(len(display_results)),
        "labeled": labels is not None,
        "label_column": label_col,
        "metrics": metrics,
        "report": report,
        "confusion": confusion,
        "feature_importance": feature_importance.to_dict("records"),
        "missing_feature_values_imputed": missing_counts,
        "inference_elapsed_ms": round(float(inference_ms), 3),
        "run_info": run_info,
        "mode": mode,
    }


# ============================================================
# DASHBOARD COMPONENTS
# ============================================================

def card(title: str, value: str, detail: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "color": COLORS["muted"],
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "25px",
                    "fontWeight": "700",
                    "color": COLORS["text"],
                },
            ),
            html.Div(
                detail,
                style={
                    "fontSize": "11px",
                    "color": COLORS["muted"],
                    "marginTop": "6px",
                },
            ),
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


def make_table(df: pd.DataFrame, max_rows: int = 300):
    if df.empty:
        return html.Div(
            "No rows to display.",
            style={"color": COLORS["muted"]},
        )

    display_df = df.head(max_rows).copy()

    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].round(4)

    return html.Div([
        html.Div(
            f"Displaying {len(display_df):,} rows; CSV export contains "
            f"all {len(df):,} available result rows.",
            style={
                "color": COLORS["muted"],
                "fontSize": "11px",
                "margin": "8px 0",
            },
        ),
        dash_table.DataTable(
            data=display_df.to_dict("records"),
            columns=[
                {"name": str(c).replace("_", " ").title(), "id": c}
                for c in display_df.columns
            ],
            page_size=12,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1b2b43",
                "color": COLORS["text"],
                "fontWeight": "700",
                "border": "1px solid #31425d",
            },
            style_cell={
                "backgroundColor": COLORS["panel"],
                "color": COLORS["text"],
                "border": "1px solid #263651",
                "padding": "8px",
                "fontSize": "11px",
                "minWidth": "100px",
                "maxWidth": "240px",
                "whiteSpace": "normal",
            },
        ),
    ])


# ============================================================
# APP LAYOUT
# ============================================================

app = Dash(__name__)
app.title = APP_TITLE

panel_style = {
    "background": COLORS["panel"],
    "padding": "16px",
    "borderRadius": "12px",
    "marginBottom": "14px",
}

button_style = {
    "padding": "10px 14px",
    "borderRadius": "7px",
    "border": "1px solid #405573",
    "background": COLORS["panel2"],
    "color": COLORS["text"],
    "cursor": "pointer",
}

app.layout = html.Div(
    style={
        "background": COLORS["background"],
        "color": COLORS["text"],
        "minHeight": "100vh",
        "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        "padding": "22px",
    },
    children=[
        # HEADER
        html.Div([
            html.Div(
                "AO-LACCP",
                style={
                    "fontSize": "13px",
                    "letterSpacing": "3px",
                    "color": COLORS["accent"],
                    "fontWeight": "700",
                },
            ),
            html.H1(
                "SOC Intelligence & Evaluation Workbench",
                style={"margin": "6px 0", "fontSize": "30px"},
            ),
            html.Div(
                "Reproducible benchmark · Dataset upload · "
                "Baseline versus AI-assisted decision pipeline",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "14px",
                },
            ),
        ], style={"marginBottom": "18px"}),

        # SYNTHETIC EVALUATION CONTROLS
        html.Div([
            html.Div([
                html.Label(
                    "Synthetic events",
                    style={"color": COLORS["muted"]},
                ),
                dcc.Dropdown(
                    id="n-events",
                    options=[
                        {"label": str(x), "value": x}
                        for x in [600, 1200, 2400, 5000]
                    ],
                    value=2400,
                    clearable=False,
                    style={"color": "#152238"},
                ),
            ], style={"minWidth": "160px", "flex": "1"}),

            html.Div([
                html.Label(
                    "Random seed",
                    style={"color": COLORS["muted"]},
                ),
                dcc.Input(
                    id="seed",
                    type="number",
                    value=42,
                    min=0,
                    step=1,
                    style={"width": "100%", "padding": "9px"},
                ),
            ], style={"minWidth": "120px", "flex": "1"}),

            html.Div([
                html.Label(
                    "Test split",
                    style={"color": COLORS["muted"]},
                ),
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
            ], style={"minWidth": "140px", "flex": "1"}),

            html.Button(
                "Run synthetic evaluation",
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
                },
            ),
        ], style={
            **panel_style,
            "display": "flex",
            "gap": "14px",
            "alignItems": "end",
            "flexWrap": "wrap",
        }),

        # CSV UPLOAD
        html.Div([
            html.H2(
                "Upload and evaluate an external CSV dataset",
                style={"marginTop": "0"},
            ),

            html.Div(
                "Required features: " + ", ".join(FEATURES),
                style={
                    "color": COLORS["muted"],
                    "fontSize": "12px",
                    "lineHeight": "1.6",
                },
            ),

            html.Div(
                "Optional ground-truth label column: true_label, label, "
                "attack_type, or class. Label values must map to the "
                "seven supported classes.",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "12px",
                    "marginBottom": "12px",
                },
            ),

            dcc.Upload(
                id="dataset-upload",
                children=html.Div([
                    "Drag and drop a CSV here, or ",
                    html.A(
                        "select a file",
                        style={
                            "color": COLORS["accent"],
                            "fontWeight": "700",
                        },
                    ),
                ]),
                style={
                    "width": "100%",
                    "padding": "24px",
                    "textAlign": "center",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "10px",
                    "borderColor": "#405573",
                    "background": COLORS["panel2"],
                    "cursor": "pointer",
                },
                multiple=False,
                accept=".csv,text/csv",
            ),

            html.Div(
                "Maximum upload: 25 MB and 100,000 rows.",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "11px",
                    "marginTop": "6px",
                },
            ),

            html.Div([
                html.Label(
                    "Dataset evaluation mode",
                    style={
                        "color": COLORS["muted"],
                        "display": "block",
                        "marginBottom": "6px",
                    },
                ),
                dcc.RadioItems(
                    id="upload-mode",
                    options=[
                        {
                            "label": (
                                " Synthetic-trained inference "
                                "(predict uploaded events)"
                            ),
                            "value": "synthetic_inference",
                        },
                        {
                            "label": (
                                " Labeled external train/test evaluation "
                                "(requires ground-truth labels)"
                            ),
                            "value": "external_holdout",
                        },
                    ],
                    value="synthetic_inference",
                    labelStyle={
                        "display": "block",
                        "marginBottom": "8px",
                    },
                    inputStyle={"marginRight": "8px"},
                ),
            ], style={"marginTop": "14px"}),

            html.Button(
                "Process uploaded dataset",
                id="process-upload-button",
                n_clicks=0,
                style={
                    "background": COLORS["accent"],
                    "color": "#09202a",
                    "border": "none",
                    "borderRadius": "8px",
                    "fontWeight": "700",
                    "padding": "12px 18px",
                    "cursor": "pointer",
                    "marginTop": "12px",
                },
            ),

            html.Button(
                "Clear uploaded results",
                id="clear-upload-button",
                n_clicks=0,
                style={**button_style, "marginLeft": "10px"},
            ),

            html.Div(
                id="upload-status",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "12px",
                    "marginTop": "12px",
                    "whiteSpace": "pre-wrap",
                },
            ),

            dcc.Store(id="pending-upload-store"),
            dcc.Store(id="upload-results-store"),

            html.Div(
                id="upload-metrics",
                style={
                    "display": "flex",
                    "gap": "10px",
                    "flexWrap": "wrap",
                    "marginTop": "12px",
                    "marginBottom": "12px",
                },
            ),

            html.Div([
                html.Div(
                    dcc.Graph(id="upload-label-chart"),
                    style={"flex": "1", "minWidth": "280px"},
                ),
                html.Div(
                    dcc.Graph(id="upload-action-chart"),
                    style={"flex": "1", "minWidth": "280px"},
                ),
            ], style={
                "display": "flex",
                "gap": "12px",
                "flexWrap": "wrap",
            }),

            html.Div(
                dcc.Graph(id="upload-confusion-chart"),
                style={"marginTop": "8px"},
            ),

            html.Div(
                dcc.Graph(id="upload-feature-chart"),
                style={"marginTop": "8px"},
            ),

            html.Div(id="upload-table"),

            html.Div([
                html.Button(
                    "Export uploaded predictions CSV",
                    id="export-upload-button",
                    n_clicks=0,
                    style={**button_style, "marginTop": "10px"},
                ),
                dcc.Download(id="download-upload-csv"),

                html.Button(
                    "Download upload evaluation manifest",
                    id="export-upload-manifest-button",
                    n_clicks=0,
                    style={**button_style, "marginTop": "10px",
                           "marginLeft": "8px"},
                ),
                dcc.Download(id="download-upload-manifest"),
            ]),
        ], style=panel_style),

        # RESEARCH NOTICE
        html.Div(
            "RESEARCH BOUNDARY: Synthetic benchmark results are not "
            "production evidence. Uploaded data is processed by this "
            "application process. Do not upload sensitive or regulated "
            "records without authorization and appropriate safeguards. "
            "Risk scores are policy heuristics, not calibrated probabilities. "
            "Actions and cryptographic recommendations are simulated; "
            "no live traffic is collected, no hosts are isolated, and "
            "no cryptographic keys are changed.",
            style={
                "background": "#2c2430",
                "border": "1px solid #68404d",
                "padding": "12px 14px",
                "borderRadius": "8px",
                "color": "#ffd9df",
                "fontSize": "12px",
                "marginBottom": "16px",
            },
        ),

        # SYNTHETIC RESULTS
        dcc.Store(id="evaluation-store"),

        html.Div(
            id="status-line",
            style={
                "color": COLORS["muted"],
                "marginBottom": "12px",
            },
        ),

        html.Div(
            id="metric-cards",
            style={
                "display": "flex",
                "gap": "12px",
                "flexWrap": "wrap",
                "marginBottom": "16px",
            },
        ),

        html.Div([
            html.Div(
                dcc.Graph(id="metric-chart"),
                style={
                    "flex": "1",
                    "minWidth": "300px",
                    "background": COLORS["panel"],
                    "borderRadius": "12px",
                    "padding": "8px",
                },
            ),
            html.Div(
                dcc.Graph(id="action-chart"),
                style={
                    "flex": "1",
                    "minWidth": "300px",
                    "background": COLORS["panel"],
                    "borderRadius": "12px",
                    "padding": "8px",
                },
            ),
        ], style={
            "display": "flex",
            "gap": "14px",
            "flexWrap": "wrap",
            "marginBottom": "14px",
        }),

        html.Div([
            html.Div(
                dcc.Graph(id="confusion-chart"),
                style={
                    "flex": "1",
                    "minWidth": "320px",
                    "background": COLORS["panel"],
                    "borderRadius": "12px",
                    "padding": "8px",
                },
            ),
            html.Div(
                dcc.Graph(id="feature-chart"),
                style={
                    "flex": "1",
                    "minWidth": "320px",
                    "background": COLORS["panel"],
                    "borderRadius": "12px",
                    "padding": "8px",
                },
            ),
        ], style={
            "display": "flex",
            "gap": "14px",
            "flexWrap": "wrap",
            "marginBottom": "14px",
        }),

        html.Div([
            html.H3(
                "Held-out synthetic event decision trace",
                style={"marginTop": "4px"},
            ),
            html.Div(
                "The rule baseline and Random Forest receive the same "
                "held-out feature rows. Ground-truth labels are used "
                "for offline evaluation, not for policy decisions.",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "12px",
                    "marginBottom": "10px",
                },
            ),

            dcc.Dropdown(
                id="label-filter",
                options=[
                    {"label": "All true labels", "value": "ALL"}
                ] + [
                    {"label": x, "value": x}
                    for x in ATTACK_TYPES
                ],
                value="ALL",
                clearable=False,
                style={
                    "color": "#152238",
                    "maxWidth": "300px",
                    "marginBottom": "10px",
                },
            ),

            dcc.Input(
                id="table-search",
                type="text",
                value="",
                placeholder="Filter event ID / prediction...",
                style={
                    "padding": "9px",
                    "borderRadius": "6px",
                    "marginBottom": "10px",
                    "width": "280px",
                    "maxWidth": "100%",
                },
            ),

            html.Button(
                "Export synthetic test CSV",
                id="export-button",
                n_clicks=0,
                style=button_style,
            ),
            dcc.Download(id="download-csv"),

            html.Button(
                "Download synthetic run manifest",
                id="manifest-button",
                n_clicks=0,
                style={**button_style, "marginLeft": "8px"},
            ),
            dcc.Download(id="download-manifest"),

            html.Div(id="event-table"),
        ], style=panel_style),

        html.Details([
            html.Summary(
                "Model, dataset, and decision-policy documentation",
                style={
                    "cursor": "pointer",
                    "fontWeight": "700",
                    "padding": "10px 0",
                },
            ),
            dcc.Markdown("""
**Synthetic data.** Eight observable numerical features are generated
with overlapping class-specific distributions and noise.

**AI model.** A scikit-learn pipeline imputes missing feature values and
fits a Random Forest classifier. Synthetic evaluation uses a stratified
train/test split before model fitting.

**External CSV inference.** The default upload mode trains on synthetic
records and predicts uploaded records. If ground-truth labels are
present, displayed metrics describe the uploaded rows and should not be
presented as independent held-out external validation.

**External labeled evaluation.** The optional external train/test mode
splits uploaded labeled data before model fitting. Both the Random Forest
and rule baseline are evaluated against the same test rows. The split
is stratified, and the model does not use the ground-truth label as a
feature.

**Policy.** Risk scores, actions, and cryptographic recommendations are
illustrative heuristics. The prototype does not perform actual
cryptographic operations or enforce production security controls.

**Scientific reporting.** Record dataset provenance, label definitions,
class distributions, preprocessing, random seed, split ratio, metrics,
uncertainty, and limitations. Avoid claiming operational effectiveness
from synthetic data alone.
            """, style={
                "color": COLORS["muted"],
                "lineHeight": "1.6",
            }),
        ], style=panel_style),
    ],
)


# ============================================================
# SYNTHETIC EVALUATION CALLBACK
# ============================================================

@app.callback(
    Output("evaluation-store", "data"),
    Output("status-line", "children"),
    Input("run-evaluation", "n_clicks"),
    State("n-events", "value"),
    State("seed", "value"),
    State("test-split", "value"),
)
def run_synthetic_callback(n_clicks, n_events, seed, test_split):
    seed = RANDOM_SEED if seed is None else int(seed)
    n_events = 2400 if n_events is None else int(n_events)
    test_split = 0.25 if test_split is None else float(test_split)

    try:
        ev = evaluate_synthetic(
            seed=seed,
            n_events=n_events,
            test_size=test_split,
        )

        info = ev["run_info"]

        status = (
            f"Synthetic evaluation complete. Seed={seed}; "
            f"training rows={info['train_events']:,}; "
            f"test rows={info['test_events']:,}; "
            f"prediction time="
            f"{info['prediction_elapsed_ms_for_test_batch']:.2f} ms. "
            "Inference timing is local batch timing, not network latency."
        )

        return ev, status

    except Exception as exc:
        return no_update, (
            f"Synthetic evaluation failed: "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# SYNTHETIC DASHBOARD RENDERING
# ============================================================

@app.callback(
    Output("metric-cards", "children"),
    Output("metric-chart", "figure"),
    Output("action-chart", "figure"),
    Output("confusion-chart", "figure"),
    Output("feature-chart", "figure"),
    Input("evaluation-store", "data"),
)
def render_synthetic_results(store):
    empty = figure_style(
        go.Figure(),
        "Run evaluation to display results",
    )

    if not store:
        return [], empty, empty, empty, empty

    metrics = store["metrics"]
    events = pd.DataFrame(store["events"])

    cards = [
        card(
            "AI macro F1",
            f"{metrics['Random Forest']['macro_f1']:.3f}",
            "Held-out synthetic test",
        ),
        card(
            "Baseline macro F1",
            f"{metrics['Rule baseline']['macro_f1']:.3f}",
            "Same held-out events",
        ),
        card(
            "AI macro recall",
            f"{metrics['Random Forest']['macro_recall']:.3f}",
            "Across supported classes",
        ),
        card(
            "AI accuracy",
            f"{metrics['Random Forest']['accuracy']:.3f}",
            "Held-out synthetic test",
        ),
        card(
            "Test events",
            f"{len(events):,}",
            "Held-out rows",
        ),
    ]

    metric_rows = []

    for model_name, values in metrics.items():
        for metric_name in [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ]:
            metric_rows.append({
                "Model": model_name,
                "Metric": metric_name.replace("_", " ").title(),
                "Score": values[metric_name],
            })

    fig_metrics = px.bar(
        pd.DataFrame(metric_rows),
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        range_y=[0, 1],
        text_auto=".3f",
    )
    fig_metrics = figure_style(
        fig_metrics,
        "Held-out synthetic classifier comparison",
    )

    action_counts = pd.DataFrame({
        "Action": ACTION_ORDER,
        "Rule baseline": [
            int((events["baseline_action"] == a).sum())
            for a in ACTION_ORDER
        ],
        "AO-LACCP AI-assisted": [
            int((events["ao_laccp_action"] == a).sum())
            for a in ACTION_ORDER
        ],
    }).melt(
        id_vars="Action",
        var_name="Pipeline",
        value_name="Event count",
    )

    fig_actions = px.bar(
        action_counts,
        x="Action",
        y="Event count",
        color="Pipeline",
        barmode="group",
    )
    fig_actions = figure_style(
        fig_actions,
        "Policy actions on identical held-out events",
    )

    cm = np.array(store["confusion"]["Random Forest"])

    fig_cm = px.imshow(
        cm,
        x=ATTACK_TYPES,
        y=ATTACK_TYPES,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Teal",
        labels={
            "x": "Predicted label",
            "y": "True label",
            "color": "Count",
        },
    )
    fig_cm = figure_style(
        fig_cm,
        "Random Forest confusion matrix",
    )
    fig_cm.update_xaxes(tickangle=-35)

    fi = pd.DataFrame(
        store["feature_importance"]
    ).sort_values("importance", ascending=True)

    fig_fi = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        text_auto=".3f",
    )
    fig_fi = figure_style(
        fig_fi,
        "Random Forest feature importance",
    )

    return cards, fig_metrics, fig_actions, fig_cm, fig_fi


@app.callback(
    Output("event-table", "children"),
    Input("evaluation-store", "data"),
    Input("label-filter", "value"),
    Input("table-search", "value"),
)
def render_synthetic_table(store, label_filter, search):
    if not store:
        return html.Div(
            "Run the synthetic evaluation to inspect events.",
            style={"color": COLORS["muted"]},
        )

    df = pd.DataFrame(store["events"])

    if label_filter and label_filter != "ALL":
        df = df[df["true_label"] == label_filter]

    if search:
        query = str(search).strip().lower()

        if query:
            search_cols = [
                "event_id",
                "true_label",
                "baseline_prediction",
                "ai_prediction",
                "ao_laccp_action",
            ]

            mask = pd.Series(False, index=df.index)

            for col in search_cols:
                if col in df.columns:
                    mask |= df[col].astype(str).str.lower().str.contains(
                        query, na=False, regex=False
                    )

            df = df[mask]

    columns = [
        "event_id",
        "true_label",
        "baseline_prediction",
        "ai_prediction",
        "baseline_correct",
        "ai_correct",
        "baseline_risk_score",
        "ai_risk_score",
        "baseline_action",
        "ao_laccp_action",
        "crypto_policy_recommendation",
    ]

    return make_table(df[columns], max_rows=250)


# ============================================================
# CSV UPLOAD CALLBACKS
# ============================================================

@app.callback(
    Output("pending-upload-store", "data"),
    Output("upload-status", "children"),
    Input("dataset-upload", "contents"),
    State("dataset-upload", "filename"),
    prevent_initial_call=True,
)
def stage_upload(contents, filename):
    if not contents:
        return None, "Choose a CSV file."

    try:
        # Validate immediately, but do not process until the user
        # presses the Process button.
        frame = parse_uploaded_csv(contents)

        return {
            "contents": contents,
            "filename": filename or "uploaded.csv",
            "rows": len(frame),
            "columns": list(frame.columns),
        }, (
            f"File selected: {filename or 'uploaded.csv'} · "
            f"{len(frame):,} rows · "
            f"{len(frame.columns)} columns. "
            "Choose an evaluation mode and click Process uploaded dataset."
        )

    except Exception as exc:
        return None, (
            f"Upload validation failed: {type(exc).__name__}: {exc}"
        )


@app.callback(
    Output("upload-results-store", "data"),
    Output("upload-status", "children", allow_duplicate=True),
    Input("process-upload-button", "n_clicks"),
    Input("clear-upload-button", "n_clicks"),
    State("pending-upload-store", "data"),
    State("upload-mode", "value"),
    State("seed", "value"),
    State("n-events", "value"),
    State("test-split", "value"),
    prevent_initial_call=True,
)
def process_or_clear_upload(
    process_clicks,
    clear_clicks,
    pending,
    mode,
    seed,
    n_events,
    test_split,
):
    from dash import ctx

    if ctx.triggered_id == "clear-upload-button":
        return None, "Uploaded results cleared."

    if not pending:
        return no_update, (
            "Select a CSV file before processing."
        )

    try:
        seed = RANDOM_SEED if seed is None else int(seed)
        n_events = 2400 if n_events is None else int(n_events)
        test_split = (
            0.25 if test_split is None else float(test_split)
        )

        result = evaluate_uploaded_dataset(
            contents=pending["contents"],
            filename=pending["filename"],
            seed=seed,
            n_events=n_events,
            test_size=test_split,
            mode=mode,
        )

        info = result["run_info"]

        if mode == "external_holdout":
            status = (
                f"External labeled evaluation complete: "
                f"{result['filename']}. "
                f"Training rows={info['train_rows']:,}; "
                f"held-out test rows={info['test_rows']:,}. "
                f"Prediction time="
                f"{result['inference_elapsed_ms']:.2f} ms. "
                "Metrics describe the held-out portion of this dataset."
            )
        else:
            status = (
                f"External inference complete: {result['filename']}. "
                f"Rows={result['row_count']:,}; "
                f"prediction time="
                f"{result['inference_elapsed_ms']:.2f} ms. "
                "Model trained on synthetic data."
            )

            if result["labeled"]:
                status += (
                    " Ground-truth labels were detected; metrics describe "
                    "the uploaded records and are not an independent "
                    "held-out external test."
                )
            else:
                status += (
                    " No ground-truth labels detected; classification "
                    "performance metrics are unavailable."
                )

        if result["missing_feature_values_imputed"]:
            status += (
                " Missing feature values will be imputed: "
                f"{result['missing_feature_values_imputed']}."
            )

        return result, status

    except Exception as exc:
        return None, (
            f"Dataset processing failed: {type(exc).__name__}: {exc}"
        )


# ============================================================
# UPLOADED DATASET RESULTS
# ============================================================

@app.callback(
    Output("upload-metrics", "children"),
    Output("upload-label-chart", "figure"),
    Output("upload-action-chart", "figure"),
    Output("upload-confusion-chart", "figure"),
    Output("upload-feature-chart", "figure"),
    Output("upload-table", "children"),
    Input("upload-results-store", "data"),
)
def render_uploaded_results(store):
    empty = figure_style(
        go.Figure(),
        "Upload and process a CSV to display results",
    )

    if not store:
        return (
            [],
            empty,
            empty,
            empty,
            empty,
            html.Div(
                "No uploaded dataset has been processed.",
                style={"color": COLORS["muted"]},
            ),
        )

    df = pd.DataFrame(store["rows"])

    cards = [
        card(
            "Evaluated rows",
            f"{store['row_count']:,}",
            store["filename"],
        ),
        card(
            "Prediction time",
            f"{store['inference_elapsed_ms']:.2f} ms",
            "Local batch inference only",
        ),
        card(
            "High-risk events",
            f"{int((pd.to_numeric(df['ai_risk_score'], errors='coerce') >= 0.58).sum()):,}",
            "Heuristic risk score ≥ 0.58",
        ),
    ]

    metrics = store.get("metrics")

    if metrics:
        cards.extend([
            card(
                "AI macro F1",
                f"{metrics['Random Forest']['macro_f1']:.3f}",
                "Current evaluation rows",
            ),
            card(
                "Baseline macro F1",
                f"{metrics['Rule baseline']['macro_f1']:.3f}",
                "Same evaluation rows",
            ),
            card(
                "AI accuracy",
                f"{metrics['Random Forest']['accuracy']:.3f}",
                "Against supplied ground truth",
            ),
        ])

    label_counts = (
        df["ai_prediction"]
        .value_counts()
        .rename_axis("Predicted threat")
        .reset_index(name="Rows")
    )

    fig_labels = px.bar(
        label_counts,
        x="Predicted threat",
        y="Rows",
        text_auto=True,
    )
    fig_labels = figure_style(
        fig_labels,
        "Uploaded data: AI-predicted threat classes",
    )

    action_counts = pd.DataFrame({
        "Action": ACTION_ORDER,
        "Rule baseline": [
            int((df["baseline_action"] == action).sum())
            for action in ACTION_ORDER
        ],
        "AO-LACCP AI-assisted": [
            int((df["ao_laccp_action"] == action).sum())
            for action in ACTION_ORDER
        ],
    }).melt(
        id_vars="Action",
        var_name="Pipeline",
        value_name="Event count",
    )

    fig_actions = px.bar(
        action_counts,
        x="Action",
        y="Event count",
        color="Pipeline",
        barmode="group",
    )
    fig_actions = figure_style(
        fig_actions,
        "Uploaded data: baseline versus AI-assisted actions",
    )

    if store.get("confusion"):
        model_name = "Random Forest"
        cm = np.array(store["confusion"][model_name])

        fig_cm = px.imshow(
            cm,
            x=ATTACK_TYPES,
            y=ATTACK_TYPES,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Teal",
            labels={
                "x": "Predicted label",
                "y": "True label",
                "color": "Count",
            },
        )
        fig_cm = figure_style(
            fig_cm,
            "Uploaded data: AI confusion matrix",
        )
        fig_cm.update_xaxes(tickangle=-35)
    else:
        fig_cm = figure_style(
            go.Figure(),
            "Confusion matrix requires ground-truth labels",
        )

    fi = pd.DataFrame(store["feature_importance"])

    fig_fi = px.bar(
        fi.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        text_auto=".3f",
    )
    fig_fi = figure_style(
        fig_fi,
        "Random Forest feature importance",
    )

    preferred = [
        "upload_row",
        "event_id",
        "evaluation_label",
        "true_label",
        "label",
        "attack_type",
        "class",
        *FEATURES,
        "baseline_prediction",
        "ai_prediction",
        "ai_confidence",
        "baseline_risk_score",
        "ai_risk_score",
        "baseline_action",
        "ao_laccp_action",
        "crypto_policy_recommendation",
        "baseline_correct",
        "ai_correct",
    ]

    columns = [
        col for col in preferred
        if col in df.columns
    ]

    # Preserve any additional uploaded columns.
    columns.extend([
        col for col in df.columns
        if col not in columns
    ])

    table = make_table(df[columns], max_rows=300)

    return cards, fig_labels, fig_actions, fig_cm, fig_fi, table


# ============================================================
# DOWNLOAD CALLBACKS
# ============================================================

@app.callback(
    Output("download-csv", "data"),
    Input("export-button", "n_clicks"),
    State("evaluation-store", "data"),
    prevent_initial_call=True,
)
def export_synthetic_csv(n_clicks, store):
    if not store:
        return no_update

    df = pd.DataFrame(store["events"])

    return dcc.send_data_frame(
        df.to_csv,
        "ao_laccp_synthetic_heldout_results.csv",
        index=False,
    )


@app.callback(
    Output("download-manifest", "data"),
    Input("manifest-button", "n_clicks"),
    State("evaluation-store", "data"),
    prevent_initial_call=True,
)
def export_synthetic_manifest(n_clicks, store):
    if not store:
        return no_update

    manifest = {
        "run_info": store["run_info"],
        "metrics": store["metrics"],
        "notes": [
            "Synthetic benchmark only.",
            "Baseline and AI predictions use identical test events.",
            "Labels and event IDs are excluded from model features.",
            "Actions and cryptographic posture are recommendations only.",
        ],
    }

    return {
        "content": json.dumps(manifest, indent=2),
        "filename": "ao_laccp_synthetic_run_manifest.json",
        "type": "application/json",
    }


@app.callback(
    Output("download-upload-csv", "data"),
    Input("export-upload-button", "n_clicks"),
    State("upload-results-store", "data"),
    prevent_initial_call=True,
)
def export_uploaded_csv(n_clicks, store):
    if not store:
        return no_update

    df = pd.DataFrame(store["rows"])

    return dcc.send_data_frame(
        df.to_csv,
        "ao_laccp_uploaded_dataset_predictions.csv",
        index=False,
    )


@app.callback(
    Output("download-upload-manifest", "data"),
    Input("export-upload-manifest-button", "n_clicks"),
    State("upload-results-store", "data"),
    prevent_initial_call=True,
)
def export_uploaded_manifest(n_clicks, store):
    if not store:
        return no_update

    manifest = {
        "run_info": store["run_info"],
        "metrics": store["metrics"],
        "missing_feature_values_imputed": (
            store["missing_feature_values_imputed"]
        ),
        "notes": [
            "Review dataset provenance and label quality.",
            "Risk scores are policy heuristics, not calibrated probabilities.",
            "Uploaded data is processed locally by the app process.",
            "No live security controls are enforced.",
        ],
    }

    return {
        "content": json.dumps(manifest, indent=2),
        "filename": "ao_laccp_upload_evaluation_manifest.json",
        "type": "application/json",
    }


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print(APP_TITLE)
    print("Research prototype — no live security enforcement.")
    print("Open http://127.0.0.1:8050 in your browser.")
    print("=" * 72)

    app.run(
        debug=False,
        host="127.0.0.1",
        port=8050,
    )
