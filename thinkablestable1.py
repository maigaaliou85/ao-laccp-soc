
# ==========================================================
# ORIGINAL AO-LACCP SOC — FLEXIBLE DATASET UPLOAD
# No mandatory Attack Type column
# ==========================================================

def parse_upload(contents):
    if not contents:
        return pd.DataFrame(), ""

    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        # Support CSV uploads
        df = pd.read_csv(io.BytesIO(decoded))

        if df.empty:
            return pd.DataFrame(), " (Dataset is empty)"

        original_rows = len(df)

        # Limit large datasets to avoid memory problems
        if len(df) > 3000:
            df = df.sample(
                n=3000,
                random_state=42
            ).reset_index(drop=True)

            warning = (
                f" (Limited: {original_rows:,} → 3,000 rows)"
            )
        else:
            warning = ""

        # Normalize column names for matching
        normalized_columns = {
            str(col).strip().lower().replace("_", " ").replace("-", " "): col
            for col in df.columns
        }

        # Use an existing attack-related column if one exists.
        # Otherwise use the existing TransactionType column.
        attack_candidates = [
            "attack type",
            "attack",
            "threat type",
            "threat",
            "label",
            "class",
            "category",
            "transaction type",
        ]

        selected_column = None

        for candidate in attack_candidates:
            if candidate in normalized_columns:
                selected_column = normalized_columns[candidate]
                break

        if selected_column is not None:
            df["attack_type"] = (
                df[selected_column]
                .fillna("Unknown")
                .astype(str)
            )
        else:
            # Do not invent attacks.
            # Use the actual transaction type as the event category
            # when that field exists.
            df["attack_type"] = "Unclassified Event"

        # Use actual risk data only if the dataset contains it.
        risk_candidates = [
            "risk_score",
            "risk score",
            "risk",
        ]

        risk_column = next(
            (
                normalized_columns[name]
                for name in risk_candidates
                if name in normalized_columns
            ),
            None,
        )

        if risk_column is not None:
            df["risk_score"] = pd.to_numeric(
                df[risk_column],
                errors="coerce"
            )
        else:
            # No fabricated risk scores.
            df["risk_score"] = np.nan

        # Use actual latency data only if it exists.
        latency_candidates = [
            "latency",
            "latency ms",
            "latency_ms",
            "response time",
            "response time ms",
        ]

        latency_column = next(
            (
                normalized_columns[name]
                for name in latency_candidates
                if name in normalized_columns
            ),
            None,
        )

        if latency_column is not None:
            df["latency"] = pd.to_numeric(
                df[latency_column],
                errors="coerce"
            )
        else:
            # No fabricated latency measurements.
            df["latency"] = np.nan

        return df, warning

    except Exception as e:
        return pd.DataFrame(), f" (Error loading file: {e})"
