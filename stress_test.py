"""Macroeconomic stress testing for the credit-default-risk portfolio.

Re-scores the hold-out test portfolio under three CCAR-style scenarios
(baseline, mild recession, severe recession) and reports, per scenario, the
predicted default rate, the approval rate at the production threshold, and the
expected loss per loan.

This is the standalone counterpart to Phase 10 of the project notebook. It
reuses the data pipeline from ``train_model`` (no duplicated loading/splitting)
and loads the serialized model from ``models/`` — train it first with::

    python train_model.py
    python stress_test.py
"""

import os

import joblib
import pandas as pd

from train_model import (
    CLUSTER_FEATURES,
    add_cluster_feature,
    find_project_root,
    load_data,
    preprocess,
    split_data,
)

# Loss-given-default and exposure-at-default (illustrative figures).
LGD = 0.65
EAD = 10_000

# Scenario shocks: income multiplier, utilization multiplier, debt-ratio
# multiplier, and an additive bump to the delinquency counters.
SCENARIOS = {
    "Baseline":         dict(income_mult=1.00, util_mult=1.00, debt_mult=1.00, delinq_add=0),
    "Mild Recession":   dict(income_mult=0.90, util_mult=1.15, debt_mult=1.10, delinq_add=0),
    "Severe Recession": dict(income_mult=0.75, util_mult=1.30, debt_mult=1.25, delinq_add=1),
}

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def load_artifacts(models_dir):
    """Load the serialized scoring pipeline (model, K-Means, scaler, metadata)."""
    paths = {
        "model": "xgb_model.joblib",
        "kmeans": "kmeans.joblib",
        "cluster_scaler": "cluster_scaler.joblib",
        "meta": "metadata.joblib",
    }
    missing = [f for f in paths.values() if not os.path.exists(os.path.join(models_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in {models_dir}: {missing}. "
            "Run `python train_model.py` first to generate them."
        )
    return {k: joblib.load(os.path.join(models_dir, f)) for k, f in paths.items()}


def apply_scenario(Xdf, kmeans, cluster_scaler,
                   income_mult, util_mult, debt_mult, delinq_add):
    """Return a shocked copy of ``Xdf`` with the cluster feature recomputed.

    Income shrinks, utilization/debt rise (utilization capped at 2.0), and the
    delinquency counters climb. The borrower's K-Means cluster is re-derived
    from the shocked segmentation features.
    """
    s = Xdf.copy()
    s["MonthlyIncome"] *= income_mult
    s["RevolvingUtilizationOfUnsecuredLines"] = (
        s["RevolvingUtilizationOfUnsecuredLines"] * util_mult
    ).clip(upper=2.0)
    s["DebtRatio"] *= debt_mult
    for col in DELINQUENCY_COLS:
        s[col] = s[col] + delinq_add
    s["cluster"] = kmeans.predict(cluster_scaler.transform(s[CLUSTER_FEATURES]))
    return s


def run_stress_test(model, X_test, kmeans, cluster_scaler, threshold):
    """Score the portfolio under every scenario; return a summary DataFrame."""
    rows = []
    for name, shock in SCENARIOS.items():
        Xs = apply_scenario(X_test, kmeans, cluster_scaler, **shock)
        proba = model.predict_proba(Xs)[:, 1]
        rows.append({
            "Scenario": name,
            "Pred. Default Rate": proba.mean(),
            f"Approval Rate @{threshold}": (proba < threshold).mean(),
            "Expected Loss / Loan": proba.mean() * LGD * EAD,
        })
    return pd.DataFrame(rows).set_index("Scenario")


def plot_stress_test(stress_df, out_path):
    """Save the three-panel stress-test bar chart used in the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["seagreen", "orange", "firebrick"]
    approval_col = [c for c in stress_df.columns if c.startswith("Approval Rate")][0]

    stress_df["Pred. Default Rate"].plot(kind="bar", ax=axes[0], color=colors)
    axes[0].set_title("Predicted Default Rate"); axes[0].set_ylabel("Rate")
    stress_df[approval_col].plot(kind="bar", ax=axes[1], color=colors)
    axes[1].set_title(approval_col)
    stress_df["Expected Loss / Loan"].plot(kind="bar", ax=axes[2], color=colors)
    axes[2].set_title("Expected Loss per Loan ($)")
    for ax in axes:
        ax.set_xticklabels(stress_df.index, rotation=15, ha="right")
        ax.set_xlabel("")
    plt.suptitle("Portfolio Stress Test — Three Macroeconomic Scenarios", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart to {out_path}")


def main(save_chart=True):
    """Reconstruct the hold-out portfolio and run the stress test."""
    root = find_project_root()
    models_dir = os.path.join(root, "models")
    data_path = os.path.join(root, "data", "cs-training.csv")

    art = load_artifacts(models_dir)

    # Rebuild the exact hold-out portfolio via the shared train_model pipeline.
    df = preprocess(load_data(data_path))
    df, _, _ = add_cluster_feature(df)
    _, X_test, _, _ = split_data(df)

    threshold = art["meta"]["threshold"]
    stress_df = run_stress_test(
        art["model"], X_test, art["kmeans"], art["cluster_scaler"], threshold
    )
    print("\nPortfolio stress test:")
    print(stress_df.round(3).to_string())

    if save_chart:
        plot_stress_test(stress_df, os.path.join(root, "images", "stress_test.png"))
    return stress_df


if __name__ == "__main__":
    main()
