
import base64
import io
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table


# ==========================================================
# AO-LACCP SOC — Uploaded Dataset Only
# ==========================================================

app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

MAX_ROWS = 3000
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

uploaded_dataset_name = "No dataset uploaded"

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
    "padding": "16px",
    "marginBottom": "16px",
}

MITIGATION_DB = {
    "brute_force": {
        "risk": "High",
        "mitigation": "Rate limiting, MFA, account lockout controls, and anomaly monitoring",
    },
    "mitm": {
        "risk": "Critical",
        "mitigation": "TLS enforcement, certificate validation, and network traffic inspection",
    },
    "zero_day": {
        "risk": "Critical",
        "mitigation": "Behavioral detection, threat hunting, isolation, and rapid patching",
    },
    "ransomware": {
        "risk": "Critical",
        "mitigation": "Endpoint detection, immutable backups, and Zero Trust controls",
    },
    "phishing": {
        "risk": "High",
        "mitigation": "Email filtering, MFA, and security awareness controls",
    },
    "data_exfiltration": {
        "risk": "High",
        "mitigation": "Data loss prevention and outbound traffic monitoring",
    },
}


# ==========================================================
# Helpers
# ==========================================================

def normalize_column_name(name):
    """Normalize a column name for matching, not its data values."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_attack_type_column(df):
    """
    Find an existing attack-type column.
    Does not create one if the uploaded dataset lacks it.
    """
    accepted_names = {
        "attacktype",
        "attacktypes",
        "attack",
        "threattype",
        "threatcategory",
    }

    for column in df.columns:
        if normalize_column_name(column) in accepted_names:
            return column

    return None


def parse_upload(contents, filename):
    """
    Read a CSV or Excel file and return only values from its
    existing Attack Type column.

    No random attack types or synthetic values are generated.
    """
    if not contents:
        return None, "Please upload a CSV or Excel dataset."

    try:
        header, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded)

        if len(decoded) > MAX_FILE_SIZE:
            return None, "File is larger than the 5 MB upload limit."

        file_name = (filename or "").lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(
                io.BytesIO(decoded),
                nrows=MAX_ROWS + 1,
            )
        elif file_name.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(
                io.BytesIO(decoded),
                nrows=MAX_ROWS + 1,
                engine="openpyxl",
            )
        else:
            return None, (
                "Unsupported file type. Please upload a .csv, "
                ".xlsx, or .xlsm file."
            )

        if df.empty:
            return None, "The uploaded file contains no data rows."

        attack_column = find_attack_type_column(df)

        if attack_column is None:
            available = ", ".join(map(str, df.columns))
            return None, (
                "No existing Attack Type column was found. "
                f"Columns detected: {available}"
            )

        # Keep only the existing attack-type field.
        # Do not invent missing attack types.
        attack_df = df[[attack_column]].copy()
        attack_df.columns = ["Attack Type"]

        # Exclude empty values instead of replacing them.
        attack_df["Attack Type"] = (
            attack_df["Attack Type"]
            .astype("string")
            .str.strip()
        )

        attack_df = attack_df[
            attack_df["Attack Type"].notna()
            & attack_df["Attack Type"].ne("")
        ].copy()

        if attack_df.empty:
            return None, (
                "The Attack Type column exists, but it contains "
                "no non-empty attack-type values."
            )

        original_rows = len(df)
        valid_rows = len(attack_df)

        message = (
            f"Loaded {filename or 'dataset'}: "
            f"{valid_rows:,} valid Attack Type rows."
        )

        if original_rows > MAX_ROWS:
            message += f" Read limited to the first {MAX_ROWS:,} rows."

        return attack_df.reset_index(drop=True), message

    except Exception as exc:
        return None, f"Could not read the file: {exc}"


def create_empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#e5e7eb", "size": 15},
    )
    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def create_attack_summary(attack_df):
    """
    Summarize only the attack types actually found in the upload.
    Labels are displayed as supplied by the dataset.
    """
    counts = (
        attack_df["Attack Type"]
        .value_counts()
        .rename_axis("Attack Type")
        .reset_index(name="Occurrences")
    )

    counts["Share (%)"] = (
        counts["Occurrences"] / counts["Occurrences"].sum() * 100
    ).round(2)

    return counts


def mitigation_for_attack(attack_name):
    """
    Match known mitigation guidance without changing the
    uploaded attack label or adding new attack categories.
    """
    key = normalize_column_name(attack_name)

    aliases = {
        "mitm": "mitm",
        "maninthemiddle": "mitm",
        "zeroday": "zero_day",
        "dataexfiltration": "data_exfiltration",
        "bruteforce": "brute_force",
    }

    key = aliases.get(key, key)
    info = MITIGATION_DB.get(key)

    if info:
        return info["risk"], info["mitigation"]

    return (
        "Not classified",
        "Review this existing attack type and define an appropriate "
        "mitigation based on your organization's threat model.",
    )


# ==========================================================
# Layout
# ==========================================================

app.layout = html.Div(
    style=STYLE,
    children=[
        html.H1(
            "AO-LACCP SOC Intelligence Platform",
            style={
                "textAlign": "center",
                "color": "#38bdf8",
                "marginBottom": "8px",
            },
        ),

        html.P(
            "Attack-type analysis from the uploaded dataset only. "
            "No simulated attack types are added.",
            style={"textAlign": "center", "color": "#94a3b8"},
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("Upload Dataset"),

                dcc.Upload(
                    id="upload_data",
                    children=html.Div(
                        [
                            "Drag and drop a CSV or Excel file here, ",
                            "or click to select a file.",
                        ]
                    ),
                    accept=".csv,.xlsx,.xlsm",
                    multiple=False,
                    style={
                        "border": "2px dashed #38bdf8",
                        "borderRadius": "10px",
                        "padding": "28px",
                        "textAlign": "center",
                        "cursor": "pointer",
                    },
                ),

                html.Div(
                    id="dataset_name",
                    style={"marginTop": "12px", "fontWeight": "bold"},
                ),

                html.Button(
                    "Clear Dataset",
                    id="clear_dataset",
                    n_clicks=0,
                    style={
                        "marginTop": "12px",
                        "padding": "10px 16px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                "gap": "16px",
            },
            children=[
                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Attack Type Rows"),
                        html.H2(id="total_rows", children="—"),
                    ],
                ),
                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Distinct Attack Types"),
                        html.H2(id="distinct_attacks", children="—"),
                    ],
                ),
                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Most Frequent Attack Type"),
                        html.H2(id="top_attack", children="—"),
                    ],
                ),
            ],
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("Attack Types Found in the Uploaded Dataset"),
                dcc.Graph(id="attack_graph"),
            ],
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("Existing Attack Type Values"),
                html.P(
                    "The labels below come from the uploaded file. "
                    "No additional attack types are generated.",
                    style={"color": "#94a3b8"},
                ),
                html.Div(id="attack_type_list"),
            ],
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("Attack-Type Counts"),
                dash_table.DataTable(
                    id="attack_summary_table",
                    columns=[
                        {"name": "Attack Type", "id": "Attack Type"},
                        {"name": "Occurrences", "id": "Occurrences"},
                        {"name": "Share (%)", "id": "Share (%)"},
                    ],
                    data=[],
                    page_size=15,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1e293b",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_cell={
                        "backgroundColor": "#0f172a",
                        "color": "#e2e8f0",
                        "padding": "10px",
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                ),
            ],
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("SOC Mitigation Guidance"),
                html.P(
                    "Guidance is shown only for attack types present in "
                    "the uploaded dataset. Unknown labels are retained "
                    "and marked as not classified.",
                    style={"color": "#94a3b8"},
                ),
                dash_table.DataTable(
                    id="mitigation_table",
                    columns=[
                        {"name": "Attack Type", "id": "Attack Type"},
                        {"name": "Occurrences", "id": "Occurrences"},
                        {"name": "Risk Classification", "id": "Risk"},
                        {"name": "Recommended Mitigation", "id": "Mitigation"},
                    ],
                    data=[],
                    page_size=10,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1e293b",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_cell={
                        "backgroundColor": "#0f172a",
                        "color": "#e2e8f0",
                        "padding": "10px",
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "minWidth": "120px",
                        "maxWidth": "420px",
                    },
                ),
            ],
        ),
    ],
)


# ==========================================================
# Upload and dashboard callback
# ==========================================================

@app.callback(
    [
        Output("dataset_name", "children"),
        Output("total_rows", "children"),
        Output("distinct_attacks", "children"),
        Output("top_attack", "children"),
        Output("attack_graph", "figure"),
        Output("attack_type_list", "children"),
        Output("attack_summary_table", "data"),
        Output("mitigation_table", "data"),
        Output("upload_data", "contents"),
    ],
    [
        Input("upload_data", "contents"),
        Input("clear_dataset", "n_clicks"),
    ],
    [
        State("upload_data", "filename"),
    ],
)
def update_dashboard(contents, clear_clicks, filename):
    ctx = dash.callback_context

    if ctx.triggered and ctx.triggered[0]["prop_id"].startswith(
        "clear_dataset."
    ):
        return (
            "Dataset cleared. Upload a file to begin.",
            "—",
            "—",
            "—",
            create_empty_figure("Upload a dataset to display attack types."),
            html.Div("No dataset loaded."),
            [],
            [],
            None,
        )

    if not contents:
        return (
            "No dataset uploaded.",
            "—",
            "—",
            "—",
            create_empty_figure("Upload a CSV or Excel dataset."),
            html.Div("No dataset loaded."),
            [],
            [],
            None,
        )

    attack_df, message = parse_upload(contents, filename)

    if attack_df is None or attack_df.empty:
        return (
            message,
            "0",
            "0",
            "—",
            create_empty_figure(message),
            html.Div(message),
            [],
            [],
            contents,
        )

    summary = create_attack_summary(attack_df)

    # Chart contains only labels from the uploaded Attack Type column.
    fig = px.bar(
        summary,
        x="Attack Type",
        y="Occurrences",
        title="Frequency of Existing Attack Types",
        text="Occurrences",
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_title="Attack Type",
        yaxis_title="Number of Rows",
        showlegend=False,
        margin={"l": 30, "r": 20, "t": 60, "b": 100},
    )

    fig.update_traces(textposition="outside")

    # Show the exact existing labels from the upload.
    unique_attacks = attack_df["Attack Type"].drop_duplicates().tolist()

    attack_list = html.Div(
        [
            html.Div(
                str(attack),
                style={
                    "backgroundColor": "#0f172a",
                    "border": "1px solid #334155",
                    "borderRadius": "8px",
                    "padding": "10px 12px",
                    "marginBottom": "8px",
                    "overflowWrap": "anywhere",
                },
            )
            for attack in unique_attacks
        ]
    )

    mitigation_rows = []

    for _, row in summary.iterrows():
        attack = str(row["Attack Type"])
        risk, mitigation = mitigation_for_attack(attack)

        mitigation_rows.append(
            {
                "Attack Type": attack,
                "Occurrences": int(row["Occurrences"]),
                "Risk": risk,
                "Mitigation": mitigation,
            }
        )

    top_attack = str(summary.iloc[0]["Attack Type"])

    return (
        message,
        f"{len(attack_df):,}",
        f"{len(unique_attacks):,}",
        top_attack,
        fig,
        attack_list,
        summary.to_dict("records"),
        mitigation_rows,
        contents,
    )


# ==========================================================
# Start server
# ==========================================================

if __name__ == "__main__":
    print("AO-LACCP SOC dashboard starting...")
    print("Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=8050)
