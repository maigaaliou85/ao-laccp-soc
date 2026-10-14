"""
AO-LACCP vs. Baseline: transparent paired synthetic simulation
--------------------------------------------------------------
Run:
    pip install numpy pandas scipy
    python ao_laccp_fair_comparison.py

Research integrity:
- All scenarios and outcomes are synthetic.
- Both policies receive identical inputs for each trial.
- The same outcome equations are applied to both policies.
- Policy rules and outcome parameters are illustrative assumptions, not validated
  models of production SOC behavior.
- Statistical results describe this simulation only; they do not establish
  real-world effectiveness or prove the AI model works.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_TRIALS = 500
OUTPUT_DIR = Path("ao_laccp_results")

# Explicit assumptions; justify, calibrate, and sensitivity-test these.
PARAMS = {
    "risk_residual_threat_weight": 0.25,
    "risk_mitigation_per_action": 0.18,
    "latency_mitigation_overhead_ms": 8.0,
    "agility_base": 0.25,
    "agility_per_action": 0.25,
    "agility_compliance_factor": 0.50,
}
# 0 = monitor, 1 = mitigate, 2 = escalate


def make_scenarios(n=N_TRIALS, seed=SEED, scenario_name="Combined Multi-Threat"):
    """Create one shared set of inputs, used unchanged by both policies."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "trial_id": np.arange(1, n + 1),
        "initial_risk": np.clip(rng.normal(0.50, 0.12, n), 0.05, 0.95),
        "base_latency_ms": np.clip(rng.normal(160.0, 30.0, n), 40.0, None),
        "threat_intensity": rng.uniform(0.0, 1.0, n),
        "latency_pressure": rng.uniform(0.0, 1.0, n),
        "compliance_pressure": rng.uniform(0.0, 1.0, n),
    })

    if scenario_name == "Normal Operating Conditions":
        df["threat_intensity"] *= 0.45
        df["latency_pressure"] *= 0.40
        df["compliance_pressure"] *= 0.45
    elif scenario_name == "Latency Spike":
        df["latency_pressure"] = np.clip(df["latency_pressure"] + 0.45, 0, 1)
        df["base_latency_ms"] += 50.0
    elif scenario_name == "Adversarial Signal":
        df["threat_intensity"] = np.clip(df["threat_intensity"] + 0.35, 0, 1)
    elif scenario_name == "Compliance Drift":
        df["compliance_pressure"] = np.clip(df["compliance_pressure"] + 0.40, 0, 1)
    elif scenario_name == "Combined Multi-Threat":
        df["threat_intensity"] = np.clip(df["threat_intensity"] + 0.30, 0, 1)
        df["latency_pressure"] = np.clip(df["latency_pressure"] + 0.35, 0, 1)
        df["compliance_pressure"] = np.clip(df["compliance_pressure"] + 0.35, 0, 1)
        df["base_latency_ms"] += 35.0
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    return df


def baseline_policy(row):
    """Illustrative fixed-threshold baseline; not representative of every SOC."""
    if row["threat_intensity"] >= 0.70:
        return 2
    if row["threat_intensity"] >= 0.40:
        return 1
    return 0


def ao_laccp_policy(row):
    """
    Illustrative adaptive weighted-rule policy.
    This is not a trained ML model or an RL controller. Replace with the
    actual AO-LACCP decision component before claiming to evaluate that model.
    """
    score = (
        0.50 * row["threat_intensity"]
        + 0.25 * row["latency_pressure"]
        + 0.25 * row["compliance_pressure"]
    )
    if score >= 0.68:
        return 2
    if score >= 0.38:
        return 1
    return 0


def calculate_outcomes(inputs, action_column, params=PARAMS):
    """Apply identical outcome equations to each policy's action."""
    out = inputs[[
        "trial_id", "initial_risk", "base_latency_ms", "threat_intensity",
        "latency_pressure", "compliance_pressure"
    ]].copy()
    action = inputs[action_column].astype(float)

    unmitigated_risk = np.clip(
        out["initial_risk"]
        + params["risk_residual_threat_weight"] * out["threat_intensity"],
        0.0, 1.0
    )
    out["risk"] = np.clip(
        unmitigated_risk * (1.0 - params["risk_mitigation_per_action"] * action),
        0.0, 1.0
    )
    out["latency_ms"] = (
        out["base_latency_ms"] * (1.0 + 0.35 * out["latency_pressure"])
        + params["latency_mitigation_overhead_ms"] * action
    )
    out["agility_index"] = np.clip(
        params["agility_base"]
        + params["agility_per_action"] * action
        + params["agility_compliance_factor"] * out["compliance_pressure"],
        0.0, 1.0
    )
    out["action_level"] = action.astype(int)
    return out


def paired_comparison(inputs):
    policy_inputs = inputs.copy()
    policy_inputs["baseline_action"] = policy_inputs.apply(baseline_policy, axis=1)
    policy_inputs["ao_laccp_action"] = policy_inputs.apply(ao_laccp_policy, axis=1)

    base = calculate_outcomes(policy_inputs, "baseline_action")
    ao = calculate_outcomes(policy_inputs, "ao_laccp_action")
    paired = pd.DataFrame({
        "trial_id": inputs["trial_id"],
        "baseline_action": base["action_level"],
        "ao_laccp_action": ao["action_level"],
        "baseline_risk": base["risk"],
        "ao_laccp_risk": ao["risk"],
        "baseline_latency_ms": base["latency_ms"],
        "ao_laccp_latency_ms": ao["latency_ms"],
        "baseline_agility_index": base["agility_index"],
        "ao_laccp_agility_index": ao["agility_index"],
    })
    return policy_inputs, paired


def paired_stats(paired, baseline_col, ao_col, metric_name, higher_is_better):
    base = paired[baseline_col].to_numpy(dtype=float)
    ao = paired[ao_col].to_numpy(dtype=float)
    diff = ao - base  # negative favors AO for lower-is-better measures
    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se = sd_diff / np.sqrt(n)
    if n > 1 and se > 0:
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci_low, ci_high = mean_diff - tcrit * se, mean_diff + tcrit * se
        t_stat, p_value = stats.ttest_rel(ao, base)
        try:
            wilcoxon_p = stats.wilcoxon(ao, base).pvalue
        except ValueError:
            wilcoxon_p = np.nan
    else:
        ci_low = ci_high = mean_diff
        t_stat = p_value = wilcoxon_p = np.nan

    pooled_sd = np.sqrt((np.var(base, ddof=1) + np.var(ao, ddof=1)) / 2)
    standardized_diff = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
    base_mean, ao_mean = float(np.mean(base)), float(np.mean(ao))
    pct_change = ((ao_mean - base_mean) / abs(base_mean) * 100) if base_mean else np.nan

    if np.isclose(mean_diff, 0):
        direction = "no mean difference"
    elif (mean_diff > 0 and higher_is_better) or (mean_diff < 0 and not higher_is_better):
        direction = "favorable to AO-LACCP"
    else:
        direction = "unfavorable to AO-LACCP"

    p_report = (
        "p < .001" if np.isfinite(p_value) and p_value < 0.001
        else f"p = {p_value:.4f}" if np.isfinite(p_value)
        else "not calculated"
    )
    return {
        "Metric": metric_name,
        "N paired scenarios": n,
        "Baseline mean": base_mean,
        "AO-LACCP mean": ao_mean,
        "Mean difference (AO - baseline)": mean_diff,
        "95% CI lower": float(ci_low),
        "95% CI upper": float(ci_high),
        "Paired t statistic": float(t_stat),
        "Paired t-test p-value": float(p_value),
        "Paired t-test p-value (reported)": p_report,
        "Wilcoxon p-value (sensitivity check)": float(wilcoxon_p),
        "Standardized mean difference (descriptive)": float(standardized_diff),
        "Percent change vs baseline (%)": float(pct_change),
        "Direction": direction,
    }


def run_experiment(scenario_name="Combined Multi-Threat", n=N_TRIALS, seed=SEED):
    inputs = make_scenarios(n=n, seed=seed, scenario_name=scenario_name)
    policy_inputs, paired = paired_comparison(inputs)
    results = pd.DataFrame([
        paired_stats(paired, "baseline_risk", "ao_laccp_risk",
                     "Cyber risk (lower is better)", False),
        paired_stats(paired, "baseline_latency_ms", "ao_laccp_latency_ms",
                     "Response latency (lower is better)", False),
        paired_stats(paired, "baseline_agility_index", "ao_laccp_agility_index",
                     "Illustrative agility index (higher is better)", True),
    ])

    OUTPUT_DIR.mkdir(exist_ok=True)
    paired.to_csv(OUTPUT_DIR / "paired_synthetic_results.csv", index=False)
    results.to_csv(OUTPUT_DIR / "paired_statistics.csv", index=False)
    inputs.to_csv(OUTPUT_DIR / "shared_scenario_inputs.csv", index=False)

    print(f"\nScenario: {scenario_name} | Paired scenarios: {n} | Seed: {seed}")
    print("IMPORTANT: synthetic simulation under stated assumptions; not real-world proof.")
    print("\nPolicy action distribution (mean and standard deviation):")
    print(policy_inputs[["baseline_action", "ao_laccp_action"]].agg(["mean", "std"]).round(4))
    print("\nPaired results:")
    print(results.round(4).to_string(index=False))
    print(f"\nCSV files saved in: {OUTPUT_DIR.resolve()}")
    return inputs, policy_inputs, paired, results


if __name__ == "__main__":
    run_experiment("Combined Multi-Threat")
