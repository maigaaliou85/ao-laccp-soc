
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="AO-LACCP SOC Comparison",
    layout="wide"
)

st.title("AO-LACCP SOC Comparison")

st.write(
    "Compare baseline and AO-LACCP using built-in "
    "illustrative data."
)

st.warning(
    "Illustrative data only. These values are not "
    "verified experimental results."
)

# Built-in demonstration data
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

# Display comparison statistics
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

# Display data
st.subheader("Comparison Data")
st.dataframe(data, use_container_width=True)

# Plot comparison
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

ax.set_ylabel("Illustrative metric value")
ax.set_title("Baseline vs AO-LACCP Comparison")
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Display mean difference
difference = (
    data["AO-LACCP"].mean()
    - data["Baseline"].mean()
)

st.subheader("Comparison Result")
st.write(f"Mean difference (AO-LACCP − Baseline): {difference:.3f}")

st.info(
    "These demonstration values do not establish "
    "scientific superiority. Replace them with actual "
    "evaluation results before reporting research findings."
)
