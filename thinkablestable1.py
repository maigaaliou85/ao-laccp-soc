
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import inspect

st.title("AO-LACCP SOC — Performance Comparison")

# TEST DATA ONLY: replace with actual measured results when available.
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "Baseline_Risk": rng.normal(0.55, 0.08, 30).clip(0, 1),
    "AO_LACCP_Risk": rng.normal(0.32, 0.05, 30).clip(0, 1),
})

bcol = "Baseline_Risk"
acol = "AO_LACCP_Risk"

baseline = df[bcol].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
ao_values = df[acol].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

if len(baseline) < 2 or len(ao_values) < 2:
    st.error("Not enough valid data to create the boxplot.")
else:
    fig, ax = plt.subplots(figsize=(8, 5))

    params = inspect.signature(ax.boxplot).parameters
    label_arg = "tick_labels" if "tick_labels" in params else "labels"

    result = ax.boxplot(
        [baseline, ao_values],
        positions=[1, 2],
        patch_artist=True,
        showmeans=True,
        **{label_arg: ["Baseline", "AO-LACCP"]}
    )

    for box, color in zip(result["boxes"], ["lightgray", "lightblue"]):
        box.set_facecolor(color)

    ax.set_title("Baseline vs AO-LACCP Risk (Test Data)")
    ax.set_ylabel("Synthetic risk score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "TEST DATA ONLY. These values are generated for chart testing "
        "and must not be presented as empirical research findings."
    )

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )
