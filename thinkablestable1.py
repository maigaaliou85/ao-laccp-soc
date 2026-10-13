
# ============================================================
# AO-LACCP: Robust Baseline vs AO-LACCP Boxplot
# Compatible with newer Matplotlib versions
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect

# Validate that the required columns exist
if bcol not in df.columns or acol not in df.columns:
    st.error(
        f"Cannot create boxplot: missing column(s). "
        f"Required: '{bcol}' and '{acol}'."
    )
else:
    # Convert data to numeric and remove invalid values
    baseline_values = (
        pd.to_numeric(df[bcol], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )

    ao_laccp_values = (
        pd.to_numeric(df[acol], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )

    # Ensure both groups contain valid observations
    if len(baseline_values) == 0 or len(ao_laccp_values) == 0:
        st.warning(
            "The boxplot cannot be generated because "
            "one or both groups contain no valid numeric data."
        )
    else:
        # Create the axes if they do not already exist
        if "ax" not in locals():
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        # Support both newer and older Matplotlib versions
        boxplot_parameters = inspect.signature(ax.boxplot).parameters

        label_parameter = (
            "tick_labels"
            if "tick_labels" in boxplot_parameters
            else "labels"
        )

        boxplot_options = {
            "positions": [1, 2],
            "patch_artist": True,
            "showmeans": True,
            "meanline": False,
        }

        boxplot_options[label_parameter] = [
            "Baseline",
            "AO-LACCP",
        ]

        # Generate the comparison boxplot
        boxplot_result = ax.boxplot(
            [baseline_values, ao_laccp_values],
            **boxplot_options
        )

        # Distinguish the two groups visually
        box_colors = ["lightgray", "lightblue"]

        for patch, color in zip(
            boxplot_result["boxes"], box_colors
        ):
            patch.set_facecolor(color)

        # Configure the chart
        ax.set_title(
            "Baseline vs AO-LACCP Performance Comparison"
        )
        ax.set_ylabel("Metric Value")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Baseline", "AO-LACCP"])
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()

        # Display using Streamlit
        st.pyplot(fig)
