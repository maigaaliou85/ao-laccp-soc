import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("AO-LACCP SOC Comparison")

uploaded_file = st.file_uploader(
    "Upload dataset (CSV)",
    type=["csv"],
    key="ao_laccp_dataset"
)

if uploaded_file is None:
    st.stop()

# Read only the CSV header first
try:
    headers = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
except Exception as e:
    st.error(f"Could not read CSV header: {e}")
    st.stop()

if len(headers) < 2:
    st.error("The CSV needs at least two columns.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    bcol = st.selectbox(
        "Baseline column",
        headers,
        index=0
    )

with col2:
    default_ao_index = next(
        (i for i, c in enumerate(headers)
         if any(k in c.lower() for k in ["ao_laccp", "ao-laccp", "aolaccp"])),
        min(1, len(headers) - 1)
    )
    acol = st.selectbox(
        "AO-LACCP column",
        headers,
        index=default_ao_index
    )

if bcol == acol:
    st.warning("Choose two different columns.")
    st.stop()

# Read only the selected columns, not the entire dataset
if st.button("Load selected metrics"):
    try:
        uploaded_file.seek(0)
        data = pd.read_csv(
            uploaded_file,
            usecols=[bcol, acol]
        )

        for col in [bcol, acol]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) < 2:
            st.error("Not enough valid numeric rows in the selected columns.")
            st.stop()

        st.success(f"Loaded {len(data):,} valid comparison rows.")

        st.dataframe(data.head(10), use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(
            [data[bcol].to_numpy(), data[acol].to_numpy()],
            tick_labels=["Baseline", "AO-LACCP"]
            if "tick_labels" in __import__("inspect").signature(ax.boxplot).parameters
            else ["Baseline", "AO-LACCP"],
            showmeans=True
        )
        ax.set_title("Baseline vs AO-LACCP")
        ax.set_ylabel("Metric value")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Could not load the selected columns: {e}")
