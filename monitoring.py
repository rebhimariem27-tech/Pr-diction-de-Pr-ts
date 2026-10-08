from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# =============================================================================
# FINPREDICT AI PRO — STEP 10: MONITORING & DRIFT DETECTION
# =============================================================================
#
# Purpose
# -------
# Monitor a new production batch against the TRAINING reference data.
#
# Monitored dimensions
# --------------------
# 1. Schema / data quality
# 2. Missing-value drift
# 3. Numeric feature drift
#       - KS statistic
#       - PSI
# 4. Categorical feature drift
#       - PSI
# 5. Prediction drift
#       - default-probability PSI
#       - average P(default)
#       - predicted default/refusal rate at threshold 0.48
# 6. Model performance
#       - evaluated only when loan_status is present in the input
#       - ROC-AUC / PR-AUC / Brier
#
# Important
# ---------
# The TRAINING dataset is the reference population.
# The production CSV is the current monitoring batch.
#
# If no --input is supplied, the script uses the held-out TEST split as a
# demonstration batch. This is useful for development only; it is NOT real
# production monitoring.
#
# Outputs
# -------
# data/reports/monitoring/
#   step10_monitoring_report.json
#   step10_feature_drift.csv
#   step10_data_quality.csv
#   step10_prediction_summary.csv
#   step10_feature_drift.png
#   step10_prediction_distribution.png
#   step10_missingness.png
#   step10_monitoring_report.html
#
# Usage
# -----
# Real production batch:
#
#   python model_ml_step10_monitoring.py \
#       --input data/production/new_batch.csv
#
# Development/demo:
#
#   python model_ml_step10_monitoring.py
# =============================================================================


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATASET_PATH = Path(
    "data/processed/donnees_prets_features.csv"
)

MODEL_ARTIFACT_PATH = Path(
    "data/production/finpredict_production_model.joblib"
)

REPORTS_DIR = Path(
    "data/reports/monitoring"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Model / schema configuration
# -----------------------------------------------------------------------------

TARGET = "loan_status"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "person_emp_length",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "interest_income_ratio",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

PRODUCTION_THRESHOLD = 0.48

RANDOM_STATE = 42
TEST_SIZE = 0.20


# -----------------------------------------------------------------------------
# Monitoring thresholds
# -----------------------------------------------------------------------------
#
# These are monitoring alert thresholds, not model decision thresholds.
# They are deliberately kept configurable.
#
# KS:
#   < 0.10  -> low / normal
#   0.10-0.20 -> moderate
#   >= 0.20 -> high
#
# PSI:
#   < 0.10  -> low / normal
#   0.10-0.25 -> moderate
#   >= 0.25 -> high
#
# Missingness:
#   >= 5 percentage points absolute increase -> alert
# -----------------------------------------------------------------------------

KS_MODERATE = 0.10
KS_HIGH = 0.20

PSI_MODERATE = 0.10
PSI_HIGH = 0.25

MISSINGNESS_MODERATE_PP = 2.0
MISSINGNESS_HIGH_PP = 5.0


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def classify_drift(
    value: float,
    moderate: float,
    high: float,
) -> str:
    if pd.isna(value):
        return "UNKNOWN"

    if value >= high:
        return "HIGH"

    if value >= moderate:
        return "MODERATE"

    return "LOW"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def psi_from_distributions(
    reference: np.ndarray,
    current: np.ndarray,
    bins: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index.

    PSI = sum((current_pct - reference_pct)
              * ln(current_pct / reference_pct))

    Same bins are used for reference/current, so the comparison is consistent.
    """

    ref_hist, _ = np.histogram(
        reference,
        bins=bins,
    )

    cur_hist, _ = np.histogram(
        current,
        bins=bins,
    )

    ref_pct = ref_hist.astype(float)
    cur_pct = cur_hist.astype(float)

    if ref_pct.sum() == 0 or cur_pct.sum() == 0:
        return np.nan

    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(
        np.sum(
            (cur_pct - ref_pct)
            * np.log(cur_pct / ref_pct)
        )
    )


def numeric_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    ref = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    cur = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(ref) < 2 or len(cur) < 2:
        return np.nan

    # Quantile-based bins from the reference population.
    quantiles = np.linspace(
        0.0,
        1.0,
        11,
    )

    bins = np.quantile(
        ref,
        quantiles,
    )

    bins = np.unique(bins)

    if len(bins) < 3:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    return psi_from_distributions(
        ref,
        cur,
        bins,
    )


def categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:

    ref = reference.fillna(
        "__MISSING__"
    ).astype(str)

    cur = current.fillna(
        "__MISSING__"
    ).astype(str)

    categories = sorted(
        set(ref.unique())
        | set(cur.unique())
    )

    if not categories:
        return np.nan

    ref_counts = (
        ref.value_counts()
        .reindex(categories, fill_value=0)
        .astype(float)
    )

    cur_counts = (
        cur.value_counts()
        .reindex(categories, fill_value=0)
        .astype(float)
    )

    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return np.nan

    # IMPORTANT:
    # .copy() garantit des tableaux numpy modifiables.
    ref_pct = ref_counts.to_numpy(
        dtype=float,
        copy=True,
    )

    cur_pct = cur_counts.to_numpy(
        dtype=float,
        copy=True,
    )

    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    ref_pct = np.clip(
        ref_pct,
        1e-6,
        None,
    )

    cur_pct = np.clip(
        cur_pct,
        1e-6,
        None,
    )

    return float(
        np.sum(
            (cur_pct - ref_pct)
            * np.log(cur_pct / ref_pct)
        )
    )
def ks_statistic(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    ref = pd.to_numeric(
    reference,
    errors="coerce",
).dropna().to_numpy(
    dtype=float,
    copy=True,
)

    cur = pd.to_numeric(
    current,
    errors="coerce",
).dropna().to_numpy(
    dtype=float,
    copy=True,
)

    if len(ref) < 2 or len(cur) < 2:
        return np.nan

    return float(
        ks_2samp(
            ref,
            cur,
        ).statistic
    )


def missing_rate(
    series: pd.Series,
) -> float:
    return float(
        series.isna().mean()
    )


def missingness_status(
    delta_pp: float,
) -> str:
    abs_delta = abs(delta_pp)

    if abs_delta >= MISSINGNESS_HIGH_PP:
        return "HIGH"

    if abs_delta >= MISSINGNESS_MODERATE_PP:
        return "MODERATE"

    return "LOW"


def required_columns_check(
    df: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in MODEL_FEATURES
        if column not in df.columns
    ]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_reference_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset de référence introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    if df.empty:
        raise ValueError(
            "Le dataset de référence est vide."
        )

    return df


def load_production_data(
    input_path: Path | None,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    Returns:
        production_df
        demo_mode
    """

    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Fichier production introuvable : {input_path}"
            )

        df = pd.read_csv(
            input_path
        )

        if df.empty:
            raise ValueError(
                "Le fichier production est vide."
            )

        return df, False

    # Demo mode:
    # use the held-out test split as a development batch.
    from sklearn.model_selection import train_test_split

    X = reference[MODEL_FEATURES].copy()
    y = (
        reference[TARGET]
        .astype(int)
        .copy()
    )

    _, X_demo, _, y_demo = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    demo = X_demo.copy()
    demo[TARGET] = y_demo.to_numpy()

    return demo, True


# -----------------------------------------------------------------------------
# Model loading / predictions
# -----------------------------------------------------------------------------


def load_production_artifact() -> dict:
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Artefact production introuvable : "
            f"{MODEL_ARTIFACT_PATH}\n"
            "Lance d'abord : "
            "python model_ml_final_pipeline.py train"
        )

    artifact = joblib.load(
        MODEL_ARTIFACT_PATH
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "L'artefact production doit être un dictionnaire."
        )

    required_keys = [
        "pipeline",
        "production_threshold",
        "model_features",
    ]

    missing = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing:
        raise KeyError(
            f"Clés manquantes dans l'artefact : {missing}"
        )

    return artifact


def add_predictions(
    production_df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    missing = required_columns_check(
        production_df
    )

    if missing:
        raise ValueError(
            "Colonnes nécessaires absentes du batch production : "
            f"{missing}"
        )

    features = artifact["model_features"]

    X_current = production_df[
        features
    ].copy()

    pipeline = artifact["pipeline"]

    threshold = float(
        artifact["production_threshold"]
    )

    probabilities = pipeline.predict_proba(
        X_current
    )[:, 1]

    result = production_df.copy()

    result["proba_defaut_monitoring"] = probabilities

    result["decision_monitoring"] = (
        probabilities >= threshold
    ).astype(int)

    return result


# -----------------------------------------------------------------------------
# Data quality
# -----------------------------------------------------------------------------


def build_data_quality_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in MODEL_FEATURES:

        ref_missing = missing_rate(
            reference[feature]
        )

        cur_missing = missing_rate(
            current[feature]
        )

        delta_pp = (
            cur_missing
            - ref_missing
        ) * 100.0

        row = {
            "feature": feature,
            "reference_missing_pct": 100.0 * ref_missing,
            "current_missing_pct": 100.0 * cur_missing,
            "missing_delta_pp": delta_pp,
            "missingness_status": missingness_status(
                delta_pp
            ),
        }

        if feature in NUMERIC_FEATURES:
            ref_numeric = pd.to_numeric(
                reference[feature],
                errors="coerce",
            )

            cur_numeric = pd.to_numeric(
                current[feature],
                errors="coerce",
            )

            row["reference_min"] = safe_float(
                ref_numeric.min()
            )
            row["current_min"] = safe_float(
                cur_numeric.min()
            )
            row["reference_max"] = safe_float(
                ref_numeric.max()
            )
            row["current_max"] = safe_float(
                cur_numeric.max()
            )

        else:
            row["reference_unique"] = int(
                reference[feature]
                .nunique(
                    dropna=True
                )
            )
            row["current_unique"] = int(
                current[feature]
                .nunique(
                    dropna=True
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Feature drift
# -----------------------------------------------------------------------------


def build_feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in MODEL_FEATURES:

        if feature in NUMERIC_FEATURES:

            ks = ks_statistic(
                reference[feature],
                current[feature],
            )

            psi = numeric_psi(
                reference[feature],
                current[feature],
            )

            drift_status = classify_drift(
                max(
                    ks if pd.notna(ks) else 0.0,
                    psi if pd.notna(psi) else 0.0,
                ),
                moderate=PSI_MODERATE,
                high=PSI_HIGH,
            )

            rows.append(
                {
                    "feature": feature,
                    "type": "numeric",
                    "ks_statistic": ks,
                    "psi": psi,
                    "drift_status": drift_status,
                }
            )

        else:

            psi = categorical_psi(
                reference[feature],
                current[feature],
            )

            drift_status = classify_drift(
                psi,
                moderate=PSI_MODERATE,
                high=PSI_HIGH,
            )

            rows.append(
                {
                    "feature": feature,
                    "type": "categorical",
                    "ks_statistic": np.nan,
                    "psi": psi,
                    "drift_status": drift_status,
                }
            )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        ["drift_status", "psi"],
        ascending=[True, False],
    ).reset_index(
        drop=True
    )


# -----------------------------------------------------------------------------
# Prediction monitoring
# -----------------------------------------------------------------------------


def build_prediction_summary(
    reference_probability: np.ndarray,
    current_probability: np.ndarray,
    current_decisions: np.ndarray,
    threshold: float,
) -> dict:

    # PSI uses reference quantile bins.
    ref_series = pd.Series(
        reference_probability
    )
    cur_series = pd.Series(
        current_probability
    )

    probability_psi = numeric_psi(
        ref_series,
        cur_series,
    )

    ks = float(
        ks_2samp(
            reference_probability,
            current_probability,
        ).statistic
    )

    reference_default_rate = float(
        np.mean(
            reference_probability
            >= threshold
        )
    )

    current_default_rate = float(
        np.mean(
            current_probability
            >= threshold
        )
    )

    return {
        "reference_mean_default_probability": float(
            np.mean(reference_probability)
        ),
        "current_mean_default_probability": float(
            np.mean(current_probability)
        ),
        "reference_median_default_probability": float(
            np.median(reference_probability)
        ),
        "current_median_default_probability": float(
            np.median(current_probability)
        ),
        "prediction_ks_statistic": ks,
        "prediction_psi": probability_psi,
        "reference_predicted_refusal_rate_pct": (
            100.0 * reference_default_rate
        ),
        "current_predicted_refusal_rate_pct": (
            100.0 * current_default_rate
        ),
        "refusal_rate_delta_pp": (
            100.0
            * (
                current_default_rate
                - reference_default_rate
            )
        ),
        "threshold": threshold,
    }


# -----------------------------------------------------------------------------
# Performance monitoring when labels are available
# -----------------------------------------------------------------------------


def build_performance_report(
    current: pd.DataFrame,
    probability_column: str,
    target_column: str,
) -> dict | None:

    if target_column not in current.columns:
        return None

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = (
        current[target_column]
        .astype(int)
        .to_numpy()
    )

    probabilities = current[
        probability_column
    ].to_numpy(
        dtype=float
    )

    predictions = (
        probabilities >= PRODUCTION_THRESHOLD
    ).astype(int)

    if len(np.unique(y)) < 2:
        return {
            "available": False,
            "reason": (
                "Le batch ne contient qu'une seule classe."
            ),
        }

    return {
        "available": True,
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
    }


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------


def save_feature_drift_plot(
    drift: pd.DataFrame,
    path: Path,
) -> None:

    plot_df = drift.copy()

    plot_df["score"] = plot_df[
        "psi"
    ].fillna(
        plot_df[
            "ks_statistic"
        ]
    )

    plot_df = plot_df.sort_values(
        "score",
        ascending=True,
    ).tail(10)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["score"],
    )

    ax.set_xlabel(
        "Drift score"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "FinPredict — Top feature drift"
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_prediction_distribution_plot(
    reference_probability: np.ndarray,
    current_probability: np.ndarray,
    path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        reference_probability,
        bins=30,
        density=True,
        alpha=0.55,
        label="Reference / Train",
    )

    ax.hist(
        current_probability,
        bins=30,
        density=True,
        alpha=0.55,
        label="Current / Production",
    )

    ax.axvline(
        PRODUCTION_THRESHOLD,
        linestyle="--",
        label=f"Threshold = {PRODUCTION_THRESHOLD:.2f}",
    )

    ax.set_xlabel(
        "Predicted probability of default"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "FinPredict — Prediction distribution"
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_missingness_plot(
    quality: pd.DataFrame,
    path: Path,
) -> None:

    plot_df = quality.sort_values(
        "missing_delta_pp",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["missing_delta_pp"],
    )

    ax.axvline(
        0.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Change in missingness (percentage points)"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "FinPredict — Missingness drift"
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# HTML report
# -----------------------------------------------------------------------------


def html_table(
    dataframe: pd.DataFrame,
) -> str:
    return dataframe.to_html(
        index=False,
        border=0,
        classes="table",
        float_format=lambda x: f"{x:.4f}",
    )


def build_html_report(
    report: dict,
    drift: pd.DataFrame,
    quality: pd.DataFrame,
    prediction_summary: dict,
    performance: dict | None,
    feature_plot: Path,
    prediction_plot: Path,
    missing_plot: Path,
    path: Path,
) -> None:

    drift_for_html = drift.copy()

    quality_for_html = quality.copy()

    performance_html = (
        "<p>Not available: target <code>loan_status</code> "
        "is absent from the monitoring batch.</p>"
    )

    if performance is not None:
        performance_html = (
            "<pre>"
            + json.dumps(
                performance,
                indent=2,
            )
            + "</pre>"
        )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FinPredict Monitoring Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.5;
}}
.card {{
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
}}
img {{
    max-width: 100%;
}}
.table {{
    border-collapse: collapse;
    width: 100%;
}}
.table th, .table td {{
    border: 1px solid #ddd;
    padding: 8px;
}}
.table th {{
    text-align: left;
}}
</style>
</head>

<body>

<h1>FinPredict — Monitoring Report</h1>

<div class="card">
<h2>Summary</h2>
<pre>{json.dumps(report["summary"], indent=2)}</pre>
</div>

<div class="card">
<h2>Prediction Monitoring</h2>
<pre>{json.dumps(prediction_summary, indent=2)}</pre>
</div>

<div class="card">
<h2>Model Performance</h2>
{performance_html}
</div>

<div class="card">
<h2>Feature Drift</h2>
{html_table(drift_for_html)}
</div>

<div class="card">
<h2>Data Quality / Missingness</h2>
{html_table(quality_for_html)}
</div>

<div class="card">
<h2>Feature Drift Chart</h2>
<img src="{feature_plot.name}" alt="Feature drift">
</div>

<div class="card">
<h2>Prediction Distribution</h2>
<img src="{prediction_plot.name}" alt="Prediction distribution">
</div>

<div class="card">
<h2>Missingness Drift</h2>
<img src="{missing_plot.name}" alt="Missingness drift">
</div>

</body>
</html>
"""

    path.write_text(
        html,
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "FinPredict — Step 10: Monitoring & Drift Detection"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "CSV du batch production. "
            "Si absent, le TEST est utilisé comme batch de démonstration."
        ),
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "FINPREDICT — STEP 10: MONITORING & DRIFT DETECTION"
    )
    print("=" * 78)

    reference = load_reference_data()

    input_path = (
        Path(args.input)
        if args.input
        else None
    )

    current, demo_mode = load_production_data(
        input_path,
        reference,
    )

    print(
        f"\nReference rows : {len(reference)}"
    )

    print(
        f"Current rows   : {len(current)}"
    )

    if demo_mode:
        print(
            "\nWARNING: DEMO MODE"
        )
        print(
            "Le TEST est utilisé comme batch courant."
        )
        print(
            "Pour du vrai monitoring :"
        )
        print(
            "  python model_ml_step10_monitoring.py "
            "--input <production.csv>"
        )

    # -------------------------------------------------------------------------
    # Model artifact
    # -------------------------------------------------------------------------
    artifact = load_production_artifact()

    threshold = float(
        artifact["production_threshold"]
    )

    print(
        f"\nProduction threshold : {threshold:.2f}"
    )

    # Keep the script aligned with the production artifact.
    global PRODUCTION_THRESHOLD
    PRODUCTION_THRESHOLD = threshold

    # -------------------------------------------------------------------------
    # Data quality and drift
    # -------------------------------------------------------------------------
    quality = build_data_quality_report(
        reference,
        current,
    )

    drift = build_feature_drift_report(
        reference,
        current,
    )

    # -------------------------------------------------------------------------
    # Prediction drift
    #
    # For the reference prediction distribution, use the actual production
    # artifact and generate predictions on the reference TRAINING dataset.
    # This is the distribution FinPredict is expected to resemble.
    # -------------------------------------------------------------------------
    reference_for_prediction = reference[
        MODEL_FEATURES
    ].copy()

    current_with_predictions = add_predictions(
        current,
        artifact,
    )

    reference_with_predictions = add_predictions(
        reference_for_prediction,
        artifact,
    )

    reference_probability = (
        reference_with_predictions[
            "proba_defaut_monitoring"
        ].to_numpy(
            dtype=float
        )
    )

    current_probability = (
        current_with_predictions[
            "proba_defaut_monitoring"
        ].to_numpy(
            dtype=float
        )
    )

    current_decisions = (
        current_with_predictions[
            "decision_monitoring"
        ].to_numpy(
            dtype=int
        )
    )

    prediction_summary = (
        build_prediction_summary(
            reference_probability,
            current_probability,
            current_decisions,
            threshold,
        )
    )

    # -------------------------------------------------------------------------
    # Performance if labels exist
    # -------------------------------------------------------------------------
    performance = build_performance_report(
        current_with_predictions,
        probability_column=(
            "proba_defaut_monitoring"
        ),
        target_column=TARGET,
    )

    # -------------------------------------------------------------------------
    # Status summary
    # -------------------------------------------------------------------------
    high_drift_features = int(
        (
            drift["drift_status"]
            == "HIGH"
        ).sum()
    )

    moderate_drift_features = int(
        (
            drift["drift_status"]
            == "MODERATE"
        ).sum()
    )

    high_missingness_features = int(
        (
            quality["missingness_status"]
            == "HIGH"
        ).sum()
    )

    high_prediction_psi = (
        prediction_summary[
            "prediction_psi"
        ]
        >= PSI_HIGH
    )

    moderate_prediction_psi = (
        prediction_summary[
            "prediction_psi"
        ]
        >= PSI_MODERATE
    )

    if high_drift_features > 0 or high_missingness_features > 0 or high_prediction_psi:
        overall_status = "ALERT"
    elif moderate_drift_features > 0 or moderate_prediction_psi:
        overall_status = "WARNING"
    else:
        overall_status = "NORMAL"

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------
    drift_csv = (
        REPORTS_DIR /
        "step10_feature_drift.csv"
    )

    quality_csv = (
        REPORTS_DIR /
        "step10_data_quality.csv"
    )

    prediction_csv = (
        REPORTS_DIR /
        "step10_prediction_summary.csv"
    )

    drift.to_csv(
        drift_csv,
        index=False,
    )

    quality.to_csv(
        quality_csv,
        index=False,
    )

    pd.DataFrame(
        [prediction_summary]
    ).to_csv(
        prediction_csv,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save plots
    # -------------------------------------------------------------------------
    feature_plot = (
        REPORTS_DIR /
        "step10_feature_drift.png"
    )

    prediction_plot = (
        REPORTS_DIR /
        "step10_prediction_distribution.png"
    )

    missing_plot = (
        REPORTS_DIR /
        "step10_missingness.png"
    )

    save_feature_drift_plot(
        drift,
        feature_plot,
    )

    save_prediction_distribution_plot(
        reference_probability,
        current_probability,
        prediction_plot,
    )

    save_missingness_plot(
        quality,
        missing_plot,
    )

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------
    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "summary": {
            "overall_status": overall_status,
            "demo_mode": demo_mode,
            "reference_rows": len(reference),
            "current_rows": len(current),
            "high_drift_features": high_drift_features,
            "moderate_drift_features": moderate_drift_features,
            "high_missingness_features": high_missingness_features,
            "prediction_psi": prediction_summary[
                "prediction_psi"
            ],
            "prediction_ks": prediction_summary[
                "prediction_ks_statistic"
            ],
            "threshold": threshold,
        },

        "data_quality": {
            "high_missingness_threshold_pp": (
                MISSINGNESS_HIGH_PP
            ),
            "moderate_missingness_threshold_pp": (
                MISSINGNESS_MODERATE_PP
            ),
        },

        "drift_thresholds": {
            "psi_moderate": PSI_MODERATE,
            "psi_high": PSI_HIGH,
            "ks_moderate": KS_MODERATE,
            "ks_high": KS_HIGH,
        },

        "prediction_monitoring": prediction_summary,

        "model_performance": performance,

        "artifacts": {
            "feature_drift_csv": str(
                drift_csv
            ),
            "data_quality_csv": str(
                quality_csv
            ),
            "prediction_summary_csv": str(
                prediction_csv
            ),
            "feature_drift_plot": str(
                feature_plot
            ),
            "prediction_distribution_plot": str(
                prediction_plot
            ),
            "missingness_plot": str(
                missing_plot
            ),
        },
    }

    json_path = (
        REPORTS_DIR /
        "step10_monitoring_report.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # HTML report
    # -------------------------------------------------------------------------
    html_path = (
        REPORTS_DIR /
        "step10_monitoring_report.html"
    )

    build_html_report(
        report,
        drift,
        quality,
        prediction_summary,
        performance,
        feature_plot,
        prediction_plot,
        missing_plot,
        html_path,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MONITORING SUMMARY")
    print("=" * 78)

    print(
        f"Overall status          : {overall_status}"
    )

    print(
        f"High drift features     : {high_drift_features}"
    )

    print(
        f"Moderate drift features : {moderate_drift_features}"
    )

    print(
        f"High missingness        : {high_missingness_features}"
    )

    print(
        f"Prediction PSI          : "
        f"{prediction_summary['prediction_psi']:.4f}"
    )

    print(
        f"Prediction KS           : "
        f"{prediction_summary['prediction_ks_statistic']:.4f}"
    )

    print(
        f"Reference avg P(default): "
        f"{prediction_summary['reference_mean_default_probability']:.4f}"
    )

    print(
        f"Current avg P(default)  : "
        f"{prediction_summary['current_mean_default_probability']:.4f}"
    )

    print(
        f"Refusal rate delta      : "
        f"{prediction_summary['refusal_rate_delta_pp']:+.2f} pp"
    )

    if performance is not None:
        print(
            "\nCurrent batch model performance:"
        )

        if performance.get(
            "available",
            False,
        ):
            print(
                f"  ROC-AUC : "
                f"{performance['roc_auc']:.4f}"
            )
            print(
                f"  PR-AUC  : "
                f"{performance['pr_auc']:.4f}"
            )
            print(
                f"  Brier   : "
                f"{performance['brier_score']:.4f}"
            )
            print(
                f"  F1      : "
                f"{performance['f1']:.4f}"
            )
        else:
            print(
                f"  {performance['reason']}"
            )

    print("\n" + "=" * 78)
    print("ARTEFACTS SAUVEGARDES")
    print("=" * 78)
    print(
        f"Feature drift CSV  : {drift_csv}"
    )
    print(
        f"Data quality CSV   : {quality_csv}"
    )
    print(
        f"Prediction CSV     : {prediction_csv}"
    )
    print(
        f"Feature drift plot : {feature_plot}"
    )
    print(
        f"Prediction plot    : {prediction_plot}"
    )
    print(
        f"Missingness plot   : {missing_plot}"
    )
    print(
        f"JSON report        : {json_path}"
    )
    print(
        f"HTML report        : {html_path}"
    )

    print("\n" + "=" * 78)
    print("STEP 10 TERMINEE")
    print("=" * 78)


if __name__ == "__main__":
    main()