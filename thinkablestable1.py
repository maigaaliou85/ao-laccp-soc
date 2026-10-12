
import base64
import io
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table


# ==========================================================
# AO-LACCP SOC | Uploaded Dataset Attack Analysis
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

# These are candidate column names, not invented attack categories.
ATTACK_COLUMN_HINTS = {
    "attack",
    "attacks",
    "attacktype",
    "attacktypes",
    "attackname",
    "threat",
    "threattype",
    "threatname",
    "category",
    "traffic",
    "traffictype",
    "label",
    "class",
    "target",
    "eventtype",
    "eventname",
    "type",
}


# ==========================================================
# Dataset loading
# ==========================================================

def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def read_dataset(contents, filename):
    """Read a CSV or Excel dataset without generating data."""
    if not contents:
        return None, "Please upload a dataset."

    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded)

        if len(raw) > MAX_FILE_SIZE:
            return None, "The file exceeds the 5 MB upload limit."

        name = (filename or "").lower()

        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw), nrows=MAX_ROWS + 1)
        elif name.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(
                io.BytesIO(raw),
                nrows=MAX_ROWS + 1,
                engine="openpyxl",
            )
        else:
            return None, "Upload a CSV, XLSX, or XLSM file."

        if df.empty:
            return None, "The uploaded dataset contains no rows."

        # Avoid treating completely empty columns as useful fields.
        df = df.dropna(axis=1, how="all")

        if df.empty or len(df.columns) == 0:
            return None, "No usable columns were found."

        # Convert data into browser-safe scalar values.
        df = df.astype(object).where(pd.notna(df), None)

        note = f"Loaded {filename or 'dataset'}."
        if len(df) > MAX_ROWS:
            note += f" Only the first {MAX_ROWS:,} rows were read."

        return df, note

    except Exception as exc:
        return None, f"Unable to read the file: {exc}"


# ==========================================================
# Suggest columns likely to contain attack categories
# ==========================================================

def suggest_attack_columns(df):
    """
    Rank likely category/attack columns using their column names.
    Do not assume a specific name is guaranteed to be correct.
    """
    candidates = []

    for column in df.columns:
        normalized = normalize_name(column)

        if normalized in ATTACK_COLUMN_HINTS:
            candidates.append(str(column))

    # If no likely name exists, offer columns containing text or
    # a manageable number of distinct values for manual selection.
    if not candidates:
        for column in df.columns:
            series = df[column].dropna()

            if series.empty:
                continue

            unique_count = series.nunique()

            if (
                series.dtype == "object"
                or str(series.dtype).startswith("string")
            ) and 1 < unique_count <= 100:
                candidates.append(str(column))

    return candidates


# ==========================================================
# Attack value analysis
# ==========================================================

def get_attack_counts(df, selected_column):
    """
    Count only the actual non-empty values in the selected column.
    No attack labels or values are created.
    """
    if selected_column not in df.columns:
        return pd.DataFrame(
            columns=["Attack / Category", "Occurrences", "Share (%)"]
        )

    values = df[selected_column].dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return pd.DataFrame(
            columns=["Attack / Category", "Occurrences", "Share (%)"]
        )

    counts = (
        values.value_counts()
        .rename_axis("Attack / Category")
        .reset_index(name="Occurrences")
    )

    counts["Share (%)"] = (
        counts["Occurrences"] / counts["Occurrences"].sum() * 100
    ).round(2)

    return counts


def make_empty_figure(message):
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
        height=400,
    )
    return fig


def make_graphs(counts, selected_column):
    if counts.empty:
        empty = make_empty_figure(
            "No non-empty values found in the selected column."
        )
        return empty, empty

    bar = px.bar(
        counts,
        x="Attack / Category",
        y="Occurrences",
        text="Occurrences",
        title=f"Values Found in '{selected_column}'",
    )
    bar.update_traces(textposition="outside")
    bar.update_layout(
        template="plotly_dark",
        height=440,
        xaxis_title=selected_column,
        yaxis_title="Number of Records",
        showlegend=False,
        margin={"l": 35, "r": 25, "t": 65, "b": 110},
    )

    pie = px.pie(
        counts,
        names="Attack / Category",
        values="Occurrences",
        title=f"Distribution of Values in '{selected_column}'",
    )
    pie.update_layout(
        template="plotly_dark",
        height=440,
    )

    return bar, pie


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
            },
        ),

        html.P(
            "Explore attack categories and other labels present in "
            "your uploaded dataset. No simulated records are added.",
            style={
                "textAlign": "center",
                "color": "#94a3b8",
                "marginBottom": "24px",
            },
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("Upload Dataset"),

                dcc.Upload(
                    id="upload_data",
                    children=html.Div(
                        [
                            html.Div(
                                "Click or drag and drop a dataset",
                                style={
                                    "fontSize": "18px",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Div(
                                "CSV, XLSX, or XLSM — maximum 5 MB",
                                style={
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
                    id="dataset_status",
                    children="No dataset uploaded.",
                    style={
                        "marginTop": "12px",
                        "overflowWrap": "anywhere",
                    },
                ),

                html.Label(
                    "Select the column containing attack types or categories:",
                    style={
                        "display": "block",
                        "marginTop": "20px",
                        "marginBottom": "8px",
                    },
                ),

                dcc.Dropdown(
                    id="attack_column",
                    options=[],
                    value=None,
                    placeholder="Upload a dataset first",
                    clearable=True,
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
                        html.H4("Dataset Records"),
                        html.H2(
                            id="total_records",
                            children="—",
                            style={"color": "#38bdf8"},
                        ),
                    ],
                ),
                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Distinct Values"),
                        html.H2(
                            id="distinct_values",
                            children="—",
                            style={"color": "#38bdf8"},
                        ),
                    ],
                ),
                html.Div(
                    style=CARD,
                    children=[
                        html.H4("Most Frequent Value"),
                        html.H2(
                            id="most_frequent",
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

        html.Div(
            style=CARD,
            children=[
                html.H3("Attack / Category Frequency"),
                dcc.Graph(
                    id="attack_bar_graph",
                    figure=make_empty_figure(
                        "Upload a dataset and select a column."
                    ),
                    config={"displaylogo": False},
                ),
            ],
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("Attack / Category Distribution"),
                dcc.Graph(
                    id="attack_pie_graph",
                    figure=make_empty_figure(
                        "Upload a dataset and select a column."
                    ),
                    config={"displaylogo": False},
                ),
            ],
        ),

        html.Div(
            style=CARD,
            children=[
                html.H3("Observed Values"),
                html.P(
                    "Every value shown below comes from the selected "
                    "column in the uploaded dataset.",
                    style={"color": "#94a3b8"},
                ),
                dash_table.DataTable(
                    id="attack_table",
                    columns=[
                        {
                            "name": "Attack / Category",
                            "id": "Attack / Category",
                        },
                        {"name": "Occurrences", "id": "Occurrences"},
                        {"name": "Share (%)", "id": "Share (%)"},
                    ],
                    data=[],
                    page_size=20,
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
# Callbacks
# ==========================================================

@app.callback(
    [
        Output("dataset_status", "children"),
        Output("attack_column", "options"),
        Output("attack_column", "value"),
        Output("total_records", "children"),
        Output("distinct_values", "children"),
        Output("most_frequent", "children"),
        Output("attack_bar_graph", "figure"),
        Output("attack_pie_graph", "figure"),
        Output("attack_table", "data"),
        Output("upload_data", "contents"),
    ],
    [
        Input("upload_data", "contents"),
        Input("attack_column", "value"),
        Input("clear_dataset", "n_clicks"),
    ],
    [
        State("upload_data", "filename"),
        State("upload_data", "contents"),
        State("attack_column", "value"),
    ],
)
def update_dashboard(
    triggered_contents,
    selected_column,
    clear_clicks,
    filename,
    current_contents,
    current_column,
):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if trigger.startswith("clear_dataset."):
        empty = make_empty_figure("Upload a dataset to begin.")
        return (
            "Dataset cleared.",
            [],
            None,
            "—",
            "—",
            "—",
            empty,
            empty,
            [],
            None,
        )

    # Use the currently uploaded file when the user changes columns.
    contents = current_contents

    if not contents:
        empty = make_empty_figure("Upload a dataset to begin.")
        return (
            "No dataset uploaded.",
            [],
            None,
            "—",
            "—",
            "—",
            empty,
            empty,
            [],
            None,
        )

    df, status = read_dataset(contents, filename)

    if df is None:
        empty = make_empty_figure(status)
        return (
            status,
            [],
            None,
            "—",
            "—",
            "—",
            empty,
            empty,
            [],
            contents,
        )

    options = [
        {"label": str(column), "value": str(column)}
        for column in df.columns
    ]

    # When a new file is uploaded, choose a likely attack column.
    if trigger.startswith("upload_data.contents"):
        suggestions = suggest_attack_columns(df)
        chosen = suggestions[0] if suggestions else None
    else:
        chosen = current_column

    # Validate that the selected column still exists.
    if chosen not in [str(column) for column in df.columns]:
        suggestions = suggest_attack_columns(df)
        chosen = suggestions[0] if suggestions else None

    if chosen is None:
        empty = make_empty_figure(
            "Select the column containing attack names or categories."
        )
        return (
            status + " Select the appropriate column below.",
            options,
            None,
            f"{len(df):,}",
            "—",
            "—",
            empty,
            empty,
            [],
            contents,
        )

    # Match the displayed column name to its actual DataFrame column.
    actual_column = next(
        (column for column in df.columns if str(column) == chosen),
        None,
    )

    if actual_column is None:
        empty = make_empty_figure("Please select a valid column.")
        return (
            status,
            options,
            None,
            f"{len(df):,}",
            "—",
            "—",
            empty,
            empty,
            [],
            contents,
        )

    counts = get_attack_counts(df, actual_column)

    if counts.empty:
        empty = make_empty_figure(
            "No non-empty values were found in this column."
        )
        return (
            status + f" Selected column: {chosen}.",
            options,
            chosen,
            f"{len(df):,}",
            "0",
            "—",
            empty,
            empty,
            [],
            contents,
        )

    bar_fig, pie_fig = make_graphs(counts, chosen)

    total_values = int(counts["Occurrences"].sum())
    distinct_values = len(counts)
    most_frequent = str(counts.iloc[0]["Attack / Category"])

    return (
        status + f" Selected column: {chosen}.",
        options,
        chosen,
        f"{len(df):,}",
        f"{distinct_values:,}",
        most_frequent,
        bar_fig,
        pie_fig,
        counts.to_dict("records"),
        contents,
    )


# ==========================================================
# Start server
# ==========================================================

if __name__ == "__main__":
    print("AO-LACCP SOC dashboard starting...")
    print("Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=8050)
