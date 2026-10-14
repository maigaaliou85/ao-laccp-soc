
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="AO-LACCP SOC Comparison",
    layout="wide"
)

st.title("AO-LACCP SOC Comparison")

st.markdown(
    """
    Compare baseline and AO-LACCP metrics using built-in
    illustrative data.

    **Important:** These values are illustrative examples,
    not actual experimental results.
    """
)

# --------------------------------------------------
# 1. Built-in illustrative comparison data
# --------------------------------------------------
# Replace these example values with verified results
# from your AO-LACCP evaluation when available.

data = pd.DataFrame({
    "Baseline": [
        0.72, 0.75, 0.69, 0.78, 0.74,
        0.71, 0.76, 0.73, 0.70, 0.77
    ],
    "AO-LACCP": [
        0.78, 0.80, 0.74, 0.82, 0.79,
        0.76, 0.81, 0.77, 0.75, 0.83
    ]
})

# --------------------------------------------------
# 2. Select metric
# --------------------------------------------------
metric = st.selectbox(
    "Metric being compared",
    [
        "Illustrative classification score",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Latency (ms)",
        "Risk Score"
    ]
)

if metric != "Illustrative classification score":
    st.warning(
        "The displayed values are illustrative only. "
        "Select a metric only for demonstration; the "
        "values are not verified results for that metric."
    )

# --------------------------------------------------
# 3. Display comparison data
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline")
    st.metric(
        "Mean",
        f"{data['Baseline'].mean():.3f}"
    )

with col2:
    st.subheader("AO-LACCP")
    st.metric(
        "Mean",
        f"{data['AO-LACCP'].mean():.3f}"
    )

st.subheader("Comparison Data")
st.dataframe(data, use_container_width=True)

# --------------------------------------------------
# 4. Boxplot comparison
# --------------------------------------------------
st.subheader("Baseline vs AO-LACCP")

fig, ax = plt.subplots(figsize=(8, 4))

ax.boxplot(
    [
        data["Baseline"].to_numpy(),
        data["AO-LACCP"].to_numpy()
    ],
    labels=["Baseline", "AO-LACCP"],
    showmeans=True
)

ax.set_title("Illustrative Baseline vs AO-LACCP")
ax.set_ylabel("Illustrative metric value")
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 5. Interpretation
# --------------------------------------------------
st.subheader("Interpretation")

difference = (
    data["AO-LACCP"].mean()
    - data["Baseline"].mean()
)

st.write(
    f"Illustrative mean difference (AO-LACCP minus "
    f"baseline): {difference:.3f}"
)

st.info(
    "A difference in the displayed values does not "
    "establish scientific superiority. Use verified "
    "held-out evaluation results, comparable test "
    "conditions, uncertainty intervals, and appropriate "
    "statistical tests before drawing conclusions."
)
