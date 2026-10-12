    Output("live_feed", "children"), Output("risk_graph", "figure"), Output("latency_graph", "figure"),
    Output("soc_attack_stream", "figure"), Output("soc_risk_gauge", "figure"),
    Output("dataset_name", "children"), Output("crypto_advice", "children"),
    Output("mitigation_table_container", "children"), Output("soc_summary", "children"),
    Input("interval", "n_intervals"), Input("model_selector", "value"), Input("crypto_selector", "value"),
    Input("uploaded-records", "data"), Input("dataset-meta", "data"),
)
def update_dashboard(n, model_type, crypto_mode, uploaded_records, dataset_meta):
    global crypto_mode_global
    crypto_mode_global = crypto_mode or "AUTO"
    dataset_meta = dataset_meta or {"filename": "No dataset uploaded", "source": "synthetic"}
    source = dataset_meta.get("source", "synthetic")
    filename = dataset_meta.get("filename", "No dataset uploaded")

    if source == "upload" and uploaded_records:
        df = pd.DataFrame(uploaded_records)
        if "crypto" not in df.columns:
            df["crypto"] = "UPLOAD"
    elif source == "upload_error":
        df = pd.DataFrame()
    else:
        # Synthetic stream is used only before a valid CSV upload.
        with stream_lock:
            df = pd.DataFrame(list(live_stream))

    if df.empty:
        empty = make_empty_figure("No data available")
        name = f"Dataset: {filename}" if source in ("upload", "upload_error") else "Dataset: No dataset uploaded"
        status = "The selected file could not be processed. Please choose a valid CSV file." if source == "upload_error" else "Waiting for synthetic events..."
        return status, empty, empty, empty, empty, name, "No risk recommendation available yet.", html.Div("No records available."), html.Div("No data available for summary.")

    df = df.copy()
    df["risk_score"] = pd.to_numeric(df.get("risk_score", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
    df["latency"] = pd.to_numeric(df.get("latency", 150.0), errors="coerce").fillna(150.0).clip(lower=0)
    df["attack_type"] = df.get("attack_type", "unknown").fillna("unknown").astype(str)
    if "crypto" not in df.columns:
        df["crypto"] = "UPLOAD" if source == "upload" else "AUTO"
    df = df.tail(MAX_UPLOAD_ROWS).reset_index(drop=True)

    latest = df.iloc[-1]
    avg_risk = float(df["risk_score"].mean())
    avg_latency = float(df["latency"].mean())
    ml_text = "ML model: insufficient records"
    # This is exploratory classification against a risk-score-derived target, not independent ground truth.
    if len(df) >= 13 and df["attack_type"].nunique() > 0:
        try:
            X = pd.get_dummies(df[["attack_type"]], dtype=float)
            y = (df["risk_score"] > 0.65).astype(int)
            if y.nunique() < 2:
                ml_text = "ML model: only one risk class is present"
            else:
                model = (XGBClassifier(n_estimators=80, random_state=42, verbosity=0, eval_metric="logloss")
                         if model_type == "XGB" and XGB_AVAILABLE
                         else RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced"))
                model.fit(X.iloc[:-1], y.iloc[:-1])
                x_last = X.iloc[[-1]].reindex(columns=X.columns, fill_value=0)
                proba_arr = model.predict_proba(x_last)[0]
                classes = list(model.classes_)
                high_idx = classes.index(1) if 1 in classes else None
                proba = float(proba_arr[high_idx]) if high_idx is not None else 0.0
                pred = "HIGH RISK" if proba >= 0.5 else "LOW RISK"
                model_name = "XGBoost" if model_type == "XGB" and XGB_AVAILABLE else "Random Forest"
                ml_text = f"{model_name}: {pred} ({proba:.1%})"
        except Exception:
            ml_text = "ML model: unable to fit this dataset"

    x_values = list(range(1, len(df) + 1))
    risk_fig = px.line(df.assign(record_number=x_values), x="record_number", y="risk_score", title="Risk Score by Record")
    risk_fig.update_layout(template="plotly_dark", height=340, xaxis_title="Record", yaxis_title="Risk score", yaxis_range=[0, 1])
    latency_fig = px.line(df.assign(record_number=x_values), x="record_number", y="latency", title="Latency by Record")
    latency_fig.update_layout(template="plotly_dark", height=340, xaxis_title="Record", yaxis_title="Latency (ms)")
    attack_fig = px.histogram(df, x="attack_type", color="attack_type", title="Threat Categories")
    attack_fig.update_layout(template="plotly_dark", height=380, showlegend=False, xaxis_title="Attack type", yaxis_title="Occurrences")

    gauge = go.Figure(go.Indicator(mode="gauge+number", value=avg_risk, number={"valueformat": ".3f"}, title={"text": "Average Risk Score"},
        gauge={"axis": {"range": [0, 1]}, "steps": [
            {"range": [0, 0.35], "color": "#16a34a"},
            {"range": [0.35, 0.7], "color": "#facc15"},
            {"range": [0.7, 1], "color": "#ef4444"},
        ]}))
    gauge.update_layout(template="plotly_dark", height=380)

    advice = "⚠ High average risk: review PQC readiness and containment controls." if avg_risk > 0.7 else (
        "⚡ Moderate risk: review hybrid cryptography and monitoring." if avg_risk > 0.5 else "✅ Lower observed risk: maintain monitoring and policy enforcement.")
    table_data = generate_mitigation_table(df)
    table = dash_table.DataTable(
        data=table_data, page_size=12, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1e2937", "color": "white", "fontWeight": "bold", "position": "sticky", "top": 0},
        style_cell={"backgroundColor": "#0f172a", "color": "#e2e8f0", "whiteSpace": "normal", "height": "auto", "minWidth": "100px", "maxWidth": "320px"},
    )
    source_label = "Uploaded dataset" if source == "upload" else "Synthetic event generator"
    feed = f"Latest: {str(latest['attack_type']).upper()} | {latest.get('crypto', 'N/A')} | {ml_text}"
    name = f"Dataset: {filename}" if source in ("upload", "upload_error") else "Dataset: No dataset uploaded (synthetic events)"
    summary = html.Div([
        html.H3("SOC Summary"),
        html.P(f"Data source: {source_label}"),
        html.P(f"Records analyzed: {len(df):,}"),
        html.P(f"Mean risk score: {avg_risk:.3f}"),
        html.P(f"Mean latency: {avg_latency:.1f} ms"),
        html.P(f"Distinct threat categories: {df['attack_type'].nunique():,}"),
        html.P("Research note: ML classification uses a risk-score-derived label when no independent ground-truth label is defined."),
    ])
    return feed, risk_fig, latency_fig, attack_fig, gauge, name, advice, table, summary


@app.callback(Output("download-mitigation", "data"), Input("export-btn", "n_clicks"),
              State("uploaded-records", "data"), State("dataset-meta", "data"), prevent_initial_call=True)
def export_mitigation(n_clicks, uploaded_records, dataset_meta):
    if not n_clicks:
        return None
    dataset_meta = dataset_meta or {}
    if dataset_meta.get("source") == "upload" and uploaded_records:
        df = pd.DataFrame(uploaded_records)
    else:
        with stream_lock:
            df = pd.DataFrame(list(live_stream))
    if df.empty:
        return None
    table_data = generate_mitigation_table(df)
    export_df = pd.DataFrame(table_data)
    return dcc.send_data_frame(export_df.to_csv, "AO-LACCP_Mitigation_Plan.csv", index=False)


if __name__ == "__main__":
    print("AO-LACCP SOC Dashboard started")
    print("Open: http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)

