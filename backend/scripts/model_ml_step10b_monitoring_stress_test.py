"""Validate FinPredict monitoring thresholds with deterministic stress scenarios.

The script reuses the production monitoring functions and model artifact. It
creates reproducible batches from the reference population, evaluates each
batch, and writes scenario-level CSV, JSON, and PNG outputs without replacing
the regular Step 10 monitoring report.

Usage:
    python model_ml_step10b_monitoring_stress_test.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.scripts import monitoring


DEFAULT_OUTPUT_DIR = Path("data/reports/monitoring/stress_test")
SAMPLE_SIZE = 6482
RANDOM_STATE = 42


def make_base_batch(reference: pd.DataFrame) -> pd.DataFrame:
    """Return a stable batch size matching the current monitoring demo."""
    size = min(SAMPLE_SIZE, len(reference))
    return reference.sample(
        n=size,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def build_scenarios(reference: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create controlled batches whose changes map to monitoring severities."""
    normal = make_base_batch(reference)

    moderate = normal.copy()
    moderate["person_income"] *= 1.25

    high = normal.copy()
    high["loan_int_rate"] = reference["loan_int_rate"].quantile(0.99)

    missingness = normal.copy()
    missingness.loc[
        missingness.index % 2 == 0,
        ["person_income", "loan_int_rate"],
    ] = np.nan

    prediction = normal.copy()
    prediction["loan_percent_income"] *= 1.35
    prediction["loan_percent_income"] = prediction[
        "loan_percent_income"
    ].clip(lower=0.01, upper=0.95)

    return {
        "Normal": normal,
        "Income drift": moderate,
        "Interest-rate drift": high,
        "Missingness spike": missingness,
        "Prediction drift": prediction,
    }


def classify_overall_status(
    drift: pd.DataFrame,
    quality: pd.DataFrame,
    prediction_summary: dict,
) -> str:
    """Apply the same NORMAL/WARNING/ALERT policy as monitoring.py."""
    high_drift = int((drift["drift_status"] == "HIGH").sum())
    moderate_drift = int((drift["drift_status"] == "MODERATE").sum())
    high_missingness = int(
        (quality["missingness_status"] == "HIGH").sum()
    )
    prediction_psi = float(prediction_summary["prediction_psi"])

    if (
        high_drift > 0
        or high_missingness > 0
        or prediction_psi >= monitoring.PSI_HIGH
    ):
        return "ALERT"

    if (
        moderate_drift > 0
        or prediction_psi >= monitoring.PSI_MODERATE
    ):
        return "WARNING"

    return "NORMAL"


def evaluate_scenario(
    name: str,
    batch: pd.DataFrame,
    reference: pd.DataFrame,
    artifact: dict,
) -> dict:
    """Evaluate one generated batch using the production monitoring contract."""
    current = monitoring.add_predictions(batch, artifact)
    reference_features = reference[monitoring.MODEL_FEATURES].copy()
    current_features = current[monitoring.MODEL_FEATURES].copy()

    drift = monitoring.build_feature_drift_report(
        reference_features,
        current_features,
    )
    quality = monitoring.build_data_quality_report(
        reference_features,
        current_features,
    )

    reference_with_predictions = monitoring.add_predictions(
        reference_features,
        artifact,
    )
    prediction_summary = monitoring.build_prediction_summary(
        reference_with_predictions[
            "proba_defaut_monitoring"
        ].to_numpy(),
        current["proba_defaut_monitoring"].to_numpy(),
        current["decision_monitoring"].to_numpy(),
        float(artifact["production_threshold"]),
    )

    status = classify_overall_status(
        drift,
        quality,
        prediction_summary,
    )

    return {
        "scenario": name,
        "status": status,
        "rows": len(batch),
        "high_drift_features": int(
            (drift["drift_status"] == "HIGH").sum()
        ),
        "moderate_drift_features": int(
            (drift["drift_status"] == "MODERATE").sum()
        ),
        "high_missingness_features": int(
            (quality["missingness_status"] == "HIGH").sum()
        ),
        "prediction_psi": float(
            prediction_summary["prediction_psi"]
        ),
        "prediction_ks": float(
            prediction_summary["prediction_ks_statistic"]
        ),
        "mean_probability_delta_pp": float(
            100.0
            * (
                prediction_summary[
                    "current_mean_default_probability"
                ]
                - prediction_summary[
                    "reference_mean_default_probability"
                ]
            )
        ),
    }


def save_chart(summary: pd.DataFrame, output_path: Path) -> None:
    colors = summary["status"].map(
        {
            "NORMAL": "#2e8b57",
            "WARNING": "#d97706",
            "ALERT": "#b91c1c",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(summary["scenario"], summary["prediction_psi"], color=colors)
    ax.axhline(
        monitoring.PSI_MODERATE,
        color="#d97706",
        linestyle="--",
        label=f"PSI moderate ({monitoring.PSI_MODERATE:.2f})",
    )
    ax.axhline(
        monitoring.PSI_HIGH,
        color="#b91c1c",
        linestyle="--",
        label=f"PSI high ({monitoring.PSI_HIGH:.2f})",
    )
    ax.set_title("FinPredict - Monitoring stress-test scenarios")
    ax.set_ylabel("Prediction PSI")
    ax.set_xlabel("Scenario")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run(output_dir: Path) -> pd.DataFrame:
    reference = monitoring.load_reference_data()
    artifact = monitoring.load_production_artifact()
    scenarios = build_scenarios(reference)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, batch in scenarios.items():
        batch.to_csv(
            output_dir / f"{name.lower().replace(' ', '_')}.csv",
            index=False,
        )

    summary = pd.DataFrame(
        [
            evaluate_scenario(
                name,
                batch,
                reference,
                artifact,
            )
            for name, batch in scenarios.items()
        ]
    )

    summary.to_csv(
        output_dir / "stress_test_summary.csv",
        index=False,
    )
    summary.to_json(
        output_dir / "stress_test_summary.json",
        orient="records",
        indent=2,
    )
    save_chart(
        summary,
        output_dir / "stress_test_status.png",
    )

    expected = {
        "Normal": "NORMAL",
        "Income drift": "WARNING",
        "Interest-rate drift": "ALERT",
        "Missingness spike": "ALERT",
        "Prediction drift": "WARNING",
    }
    actual = dict(
        zip(
            summary["scenario"],
            summary["status"],
        )
    )
    mismatches = {
        scenario: {"expected": expected[scenario], "actual": actual[scenario]}
        for scenario in expected
        if actual.get(scenario) != expected[scenario]
    }
    if mismatches:
        raise AssertionError(
            "Les statuts de stress test ne correspondent pas aux attentes : "
            f"{mismatches}"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère et valide les scénarios de stress Step 10b."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie des batches et rapports.",
    )
    args = parser.parse_args()

    summary = run(args.output_dir)
    print(summary[["scenario", "status"]].to_string(index=False))
    print(f"\nRapports écrits dans : {args.output_dir}")
    print(
        "Validation terminée à : "
        f"{datetime.now(timezone.utc).isoformat()}"
    )


if __name__ == "__main__":
    main()
