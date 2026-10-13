
# ============================================================
# AO-LACCP SOC — Dataset Upload and Performance Comparison
# ============================================================

import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="AO-LACCP SOC Comparison",
    layout="wide"
)

st.title("AO-LACCP SOC — Performance Comparison")
st.caption("Upload a CSV dataset to compare baseline and AO-LACCP metrics.")

# 1. Upload dataset
uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="Upload a CSV containing baseline and AO-LACCP metric columns."
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# 2. Read dataset safely
try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

st.success(f"Dataset loaded: {len(df)} rows and {len(df.columns)} columns.")

with st.expander("Preview uploaded dataset", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

# 3. Identify numeric columns
numeric_columns = []
for col in df.columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().sum() >= 2:
        numeric_columns.append(col)

if len(numeric_columns) < 2:
    st.error(
        "The dataset must contain at least two columns with "
        "two or more numeric values each."
    )
    st.stop()

# 4. Choose columns
def find_default(options, terms, fallback_index):
    for option in options:
        name = str(option).lower().replace("-", "_").replace(" ", "_")
        if any(term in name for term in terms):
            return option
    return options[min(fallback_index, len(options) - 1)]

default_baseline = find_default(
    numeric_columns, ["baseline", "traditional", "control"], 0
)
default_ao = find_default(
    numeric_columns, ["ao_laccp", "ao-laccp", "aolaccp", "proposed"], 1
)

st.subheader("Select comparison metrics")
col1, col2 = st.columns(2)

with col1:
    bcol = st.selectbox(
        "Baseline metric",
        numeric_columns,
        index=numeric_columns.index(default_baseline)
    )

with col2:
    acol = st.selectbox(
        "AO-LACCP metric",
        numeric_columns,
        index=numeric_columns.index(default_ao)
    )

if bcol == acol:
    st.warning("Select two different columns for comparison.")
    st.stop()

# 5. Clean selected values
baseline_values = (
    pd.to_numeric(df[bcol], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .to_numpy(dtype=float)
)

ao_values = (
    pd.to_numeric(df[acol], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .to_numpy(dtype=float)
)

if len(baseline_values) < 2 or len(ao_values) < 2:
    st.error("Each selected column must contain at least two valid numeric values.")
    st.stop()

# 6. Display boxplot
st.subheader("Baseline vs AO-LACCP boxplot")

fig, ax = plt.subplots(figsize=(9, 5))
params = inspect.signature(ax.boxplot).parameters
label_arg = "tick_labels" if "tick_labels" in params else "labels"

result = ax.boxplot(
    [baseline_values, ao_values],
    positions=[1, 2],
    patch_artist=True,
    showmeans=True,
    **{label_arg: ["Baseline", "AO-LACCP"]}
)

for box, color in zip(result["boxes"], ["lightgray", "lightblue"]):
    box.set_facecolor(color)

ax.set_title(f"Comparison: {bcol} vs {acol}")
ax.set_ylabel("Metric value")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# 7. Descriptive statistics
summary = pd.DataFrame({
    "Group": ["Baseline", "AO-LACCP"],
    "Selected column": [bcol, acol],
    "Valid observations": [len(baseline_values), len(ao_values)],
    "Mean": [baseline_values.mean(), ao_values.mean()],
    "Median": [np.median(baseline_values), np.median(ao_values)],
    "Standard deviation": [
        baseline_values.std(ddof=1),
        ao_values.std(ddof=1)
    ]
})

st.subheader("Descriptive statistics")
st.dataframe(summary, use_container_width=True)

# 8. Download cleaned comparison data
comparison_df = pd.DataFrame({
    "Baseline": pd.Series(baseline_values),
    "AO-LACCP": pd.Series(ao_values)
})

st.download_button(
    "Download selected comparison data",
    data=comparison_df.to_csv(index=False).encode("utf-8"),
    file_name="ao_laccp_comparison.csv",
    mime="text/csv"
)

st.caption(
    "Interpret the chart according to the selected metric. "
    "Lower values may be preferable for risk and latency, while "
    "higher values may be preferable for agility. This app compares "
    "uploaded values; it does not train or execute the SOC AI model."
)
