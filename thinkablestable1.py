
import base64
import io
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table


# ==========================================================
# AO-LACCP SOC | Uploaded Attack Types Only
# ==========================================================

app = dash.Dash(__name__)
app.title = "AO-LACCP SOC | SIEM-X Engine"

MAX_ROWS = 3000
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

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


# ==========================================================
# File upload and Attack Type detection
# ==========================================================

def normalize_column_name(name):
    """Normalize column names for matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_attack_type_column(df):
    """Find an existing Attack Type column without creating one."""
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
    Read CSV or Excel data.
    Return ONLY the existing Attack Type column.
    Do not generate attacks or fabricate missing values.
    """
    if not contents:
        return None, "Please upload a dataset."

    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        if len(decoded) > MAX_FILE_SIZE:
            return None, "File exceeds the 5 MB upload limit."

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
                "Unsupported file format. Upload a CSV, XLSX, or XLSM file."
            )

        if df.empty:
            return None, "The uploaded dataset is empty."

        attack_column = find_attack_type_column(df)

        if attack_column is None:
            available_columns = ", ".join(map(str, df.columns))
            return None, (
                "The uploaded dataset has no recognized Attack Type column. "
                f"Available columns: {available_columns}"
            )

        # Retain only the existing attack-type data.
        attack_df = df[[attack_column]].copy()
        attack_df.columns = ["Attack Type"]

        # Remove blank values without inventing replacements.
        attack_df["Attack Type"] = (
            attack_df["Attack Type"]
            .astype("string")
            .str.strip()
        )

        attack_df = attack_df[
            attack_df["Attack Type"].notna()
            & attack_df["Attack Type"].ne("")
        ]

        if attack_df.empty:
            return None, "No non-empty Attack Type values were found."

        original_rows = len(df)
        valid_rows = len(attack_df)

        message = (
            f"Loaded {filename or 'dataset'} — "
            f"{valid_rows:,} non-empty Attack Type rows."
        )

        if original_rows > MAX_ROWS:
            message += f" Read limited to the first {MAX_ROWS:,} rows."

        return attack_df.reset_index(drop=True), message

    except Exception as exc:
        return None, f"Unable to read the uploaded file: {exc}"


# ==========================================================
# Graph helpers
# ==========================================================

def empty_figure(message):
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
        margin={"l": 25, "r": 25, "t": 55, "b": 35},
    )

    return fig


def build_attack_graphs(attack_df):
    """
    Both graphs use only the attack labels and counts
    present in the uploaded dataset.
    """
    counts = (
        attack_df["Attack Type"]
        .value_counts()
        .rename_axis("Attack Type")
        .reset_index(name="Occurrences")
    )

    bar_fig = px.bar(
        counts,
        x="Attack Type",
        y="Occurrences",
        text="Occurrences",
        title="Attack Types in Uploaded Dataset",
    )

    bar_fig.update_traces(textposition="outside")

    bar_fig.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_title="Existing Attack Type",
        yaxis_title="Number of Records",
        showlegend=False,
        margin={"l": 35, "r": 25, "t": 65, "b": 110},
    )

    pie_fig = px.pie(
        counts,
        names="Attack Type",
        values="Occurrences",
        title="Distribution of Existing Attack Types",
    )

    pie_fig.update_layout(
        template="plotly_dark",
        height=420,
        margin={"l": 25, "r": 25, "t": 65, "b": 35},
    )

    return counts, bar_fig, pie_fig


# ==========================================================
# Dashboard layout
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
            "Uploaded Attack Type Analysis",
            style={
                "textAlign": "center",
                "color": "#94a3b8",
                "fontSize": "17px",
            },
        ),

        html.P(
            "Graphs use only the Attack Type values found in your uploaded "
            "dataset. No synthetic attacks are added.",
            style={
                "textAlign": "center",
                "color": "#cbd5e1",
                "marginBottom": "24px",
            },
        ),

        # Dataset upload
        html.Div(
            style=CARD,
            children=[
                html.H3("Upload Dataset"),

                dcc.Upload(
                    id="upload_data",
                    children=html.Div(
                        [
                            html.Div(
                                "Select or drag and drop your dataset",
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Div(
                                "Supported formats: CSV, XLSX, XLSM | "
                                "Maximum file size: 5 MB",
                                style={
                                    "fontSize": "13px",
                                    "color": "#94a3b8",
                                    "marginTop": "8px",
                                },
                            ),
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
                    children="No dataset uploaded.",
                    style={
                        "marginTop": "14px",
                        "fontWeight": "bold",
                        "overflowWrap": "anywhere",
                    },
                ),

                html.Button(
                    "Clear Dataset",
                    id="clear_dataset",
                    n_clicks=0,
                    style={
                        "marginTop": "14px",
                        "padding": "10px 18px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),

        # Summary cards
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(200px, 1fr))"
                ),
                "gap": "16px",
            },
            children=[
                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Attack Type Records"),
                        html.H2(
                            id="total_rows",
                            children="—",
                            style={"color": "#38bdf8"},
                        ),
                    ],
                ),

                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Distinct Attack Types"),
                        html.H2(
                            id="distinct_attacks",
                            children="—",
                            style={"color": "#38bdf8"},
                        ),
                    ],
                ),

                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Most Frequent Attack Type"),
                        html.H2(
                            id="top_attack",
                            children="—",
                            style={
                                "color": "#facc15",
                                "overflowWrap": "anywhere",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # Bar chart
        html.Div(
            style=CARD,
            children=[
                html.H3("Attack Type Frequency"),
                dcc.Graph(
                    id="attack_bar_graph",
                    figure=empty_figure(
                        "Upload a dataset to display its attack types."
                    ),
                    config={"displaylogo": False, "responsive": True},
                ),
            ],
        ),

        # Pie chart
        html.Div(
            style=CARD,
            children=[
                html.H3("Attack Type Distribution"),
                dcc.Graph(
                    id="attack_pie_graph",
                    figure=empty_figure(
                        "Upload a dataset to display attack distribution."
                    ),
                    config={"displaylogo": False, "responsive": True},
                ),
            ],
        ),

        # Existing values
        html.Div(
            style=CARD,
            children=[
                html.H3("Attack Types Found in the Dataset"),
                html.P(
                    "These labels are taken directly from the uploaded "
                    "Attack Type column.",
                    style={"color": "#94a3b8"},
                ),
                html.Div(id="attack_type_list"),
            ],
        ),

        # Frequency table
        html.Div(
            style=CARD,
            children=[
                html.H3("Attack Type Counts"),
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
                        "padding": "10px",
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
    ],
)


# ==========================================================
# Dashboard callback
# ==========================================================

@app.callback(
    [
        Output("dataset_name", "children"),
        Output("total_rows", "children"),
        Output("distinct_attacks", "children"),
        Output("top_attack", "children"),
        Output("attack_bar_graph", "figure"),
        Output("attack_pie_graph", "figure"),
        Output("attack_type_list", "children"),
        Output("attack_summary_table", "data"),
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

    # Clear the uploaded dataset and reset all visualizations.
    if (
        ctx.triggered
        and ctx.triggered[0]["prop_id"].startswith("clear_dataset.")
    ):
        return (
            "Dataset cleared. Upload a dataset to begin.",
            "—",
            "—",
            "—",
            empty_figure("Upload a dataset to display attack types."),
            empty_figure("Upload a dataset to display attack distribution."),
            html.Div("No dataset loaded."),
            [],
            None,
        )

    if not contents:
        return (
            "No dataset uploaded.",
            "—",
            "—",
            "—",
            empty_figure("Upload a dataset to display attack types."),
            empty_figure("Upload a dataset to display attack distribution."),
            html.Div("No dataset loaded."),
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
            empty_figure(message),
            empty_figure(message),
            html.Div(message),
            [],
            contents,
        )

    counts, bar_fig, pie_fig = build_attack_graphs(attack_df)

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

    total_rows = len(attack_df)
    distinct_count = len(unique_attacks)
    top_attack = str(counts.iloc[0]["Attack Type"])

    return (
        message,
        f"{total_rows:,}",
        f"{distinct_count:,}",
        top_attack,
        bar_fig,
        pie_fig,
        attack_list,
        counts.to_dict("records"),
        contents,
    )


# ==========================================================
# Run application
# ==========================================================

if __name__ == "__main__":
    print("AO-LACCP SOC dashboard starting...")
    print("Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=8050)
