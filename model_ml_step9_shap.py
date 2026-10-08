
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# FINPREDICT AI PRO — STEP 9: SHAP EXPLAINABILITY
# =============================================================================
#
# Production model:
#   LightGBM + sigmoid calibration (CV=5)
#   production threshold = 0.48
#
# SHAP explains the underlying LightGBM model before the sigmoid calibration
# layer. Therefore:
#   - production probability = calibrated probability
#   - SHAP values            = contribution to the LightGBM output
#
# Outputs:
#   data/reports/ml/step9_shap_summary.png
#   data/reports/ml/step9_shap_bar.png
#   data/reports/ml/step9_shap_global_importance.csv
#   data/reports/ml/step9_shap_test_explanations.csv
#   data/reports/ml/step9_shap_examples.json
#   data/reports/ml/step9_shap_report.json
#   data/reports/ml/step9_explainability_model.joblib
# =============================================================================


DATASET_PATH = Path("data/processed/donnees_prets_features.csv")
REPORTS_DIR = Path("data/reports/ml")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5
PRODUCTION_THRESHOLD = 0.48

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_SHAP_SAMPLE_SIZE = 3000
TOP_FACTORS = 5


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_base_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", lgb.LGBMClassifier(**LGB_PARAMS)),
        ]
    )


def build_calibrated_pipeline() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
    )


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = MODEL_FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    return (
        df[MODEL_FEATURES].copy(),
        df[TARGET].astype(int).copy(),
    )


def clean_feature_names(names) -> list[str]:
    return [
        str(name)
        .replace("numeric__", "")
        .replace("categorical__", "")
        for name in names
    ]


def normalize_shap_values(
    shap_values,
    expected_rows: int,
    expected_features: int,
) -> np.ndarray:
    """
    Handles SHAP formats used by different SHAP versions:
      - ndarray
      - list[class0, class1]
      - shap.Explanation
    """

    values = shap_values

    if isinstance(values, shap.Explanation):
        values = values.values

    if isinstance(values, list):
        if len(values) == 2:
            values = values[1]
        elif len(values) == 1:
            values = values[0]
        else:
            raise ValueError(
                f"Format SHAP inattendu avec {len(values)} éléments."
            )

    values = np.asarray(values)

    if values.ndim == 3:
        # Common possibilities:
        # (rows, features, classes)
        # (classes, rows, features)
        if values.shape[0] == expected_rows:
            values = values[:, :, -1]
        elif values.shape[1] == expected_rows:
            values = values[-1, :, :]
        else:
            raise ValueError(
                f"Shape SHAP 3D inattendue : {values.shape}"
            )

    if values.shape != (expected_rows, expected_features):
        raise ValueError(
            f"Shape SHAP inattendue : {values.shape}; "
            f"attendu {(expected_rows, expected_features)}"
        )

    return values


def positive_expected_value(explainer) -> float:
    value = explainer.expected_value

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return float(arr[-1])

    return float(value)


def save_summary_plots(
    shap_values: np.ndarray,
    X_shap: pd.DataFrame,
    summary_path: Path,
    bar_path: Path,
) -> None:
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=list(X_shap.columns),
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=list(X_shap.columns),
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()


def get_individual_explanation(
    explainer,
    preprocessor: ColumnTransformer,
    X_row: pd.DataFrame,
    feature_names: list[str],
    production_probability: float,
    actual_status: int,
    row_index: int,
) -> dict:
    processed = preprocessor.transform(X_row)

    if not isinstance(processed, pd.DataFrame):
        processed = pd.DataFrame(
            processed,
            columns=feature_names,
        )

    processed.columns = feature_names

    raw_values = explainer.shap_values(processed)

    values = normalize_shap_values(
        raw_values,
        expected_rows=1,
        expected_features=len(feature_names),
    )[0]

    details = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": [
                processed.iloc[0][col]
                for col in feature_names
            ],
            "shap_value": values,
        }
    )

    details["abs_shap"] = details["shap_value"].abs()
    details["direction"] = np.where(
        details["shap_value"] > 0,
        "increases_risk",
        "decreases_risk",
    )

    top = details.nlargest(
        TOP_FACTORS,
        "abs_shap",
    )

    return {
        "row_index": int(row_index),
        "actual_loan_status": int(actual_status),
        "production_probability_default": float(
            production_probability
        ),
        "production_decision": (
            "REFUSE"
            if production_probability >= PRODUCTION_THRESHOLD
            else "ACCEPT"
        ),
        "top_factors": [
            {
                "feature": str(row["feature"]),
                "feature_value": (
                    None
                    if pd.isna(row["feature_value"])
                    else str(row["feature_value"])
                ),
                "shap_value": float(row["shap_value"]),
                "direction": str(row["direction"]),
            }
            for _, row in top.iterrows()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinPredict — Step 9: SHAP explainability"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SHAP_SAMPLE_SIZE,
        help="Nombre de lignes TEST utilisées pour les graphiques SHAP.",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("FINPREDICT — STEP 9: SHAP EXPLAINABILITY")
    print("=" * 78)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  : {len(X_test)}")

    # -------------------------------------------------------------------------
    # 1. Production pipeline: calibrated probabilities
    # -------------------------------------------------------------------------
    print("\nEntraînement du pipeline de production calibré...")

    production_pipeline = build_calibrated_pipeline()
    production_pipeline.fit(X_train, y_train)

    test_proba = production_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= PRODUCTION_THRESHOLD).astype(int)

    test_auc = roc_auc_score(y_test, test_proba)
    test_brier = brier_score_loss(y_test, test_proba)

    print(f"Production threshold : {PRODUCTION_THRESHOLD:.2f}")
    print(f"TEST ROC-AUC         : {test_auc:.4f}")
    print(f"TEST Brier           : {test_brier:.4f}")

    # -------------------------------------------------------------------------
    # 2. Uncalibrated LightGBM pipeline for SHAP
    # -------------------------------------------------------------------------
    print(
        "\nEntraînement du modèle LightGBM "
        "pour l'explicabilité..."
    )

    shap_pipeline = build_base_pipeline()
    shap_pipeline.fit(X_train, y_train)

    preprocessor = shap_pipeline.named_steps["preprocessor"]
    model = shap_pipeline.named_steps["model"]

    feature_names = clean_feature_names(
        preprocessor.get_feature_names_out()
    )

    X_test_processed = preprocessor.transform(X_test)

    if not isinstance(X_test_processed, pd.DataFrame):
        X_test_processed = pd.DataFrame(
            X_test_processed,
            columns=feature_names,
            index=X_test.index,
        )

    X_test_processed.columns = feature_names

    print(
        f"Features après preprocessing : "
        f"{X_test_processed.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # 3. Deterministic SHAP sample
    # -------------------------------------------------------------------------
    sample_size = min(
        max(args.sample_size, 1),
        len(X_test_processed),
    )

    X_shap = X_test_processed.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    print(f"SHAP sample : {len(X_shap)} lignes")

    # -------------------------------------------------------------------------
    # 4. SHAP TreeExplainer
    # -------------------------------------------------------------------------
    print("\nCalcul des valeurs SHAP...")

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_shap)

    shap_values = normalize_shap_values(
        raw_shap,
        expected_rows=len(X_shap),
        expected_features=len(feature_names),
    )

    expected_value = positive_expected_value(explainer)

    # -------------------------------------------------------------------------
    # 5. Global importance
    # -------------------------------------------------------------------------
    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(
                np.abs(shap_values),
                axis=0,
            ),
            "mean_signed_shap": np.mean(
                shap_values,
                axis=0,
            ),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 78)
    print("GLOBAL SHAP IMPORTANCE — TOP 15")
    print("=" * 78)

    print(
        global_importance.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # -------------------------------------------------------------------------
    # 6. Global plots
    # -------------------------------------------------------------------------
    summary_png = REPORTS_DIR / "step9_shap_summary.png"
    bar_png = REPORTS_DIR / "step9_shap_bar.png"

    save_summary_plots(
        shap_values,
        X_shap,
        summary_png,
        bar_png,
    )

    # -------------------------------------------------------------------------
    # 7. Individual explanations
    # -------------------------------------------------------------------------
    highest_risk_pos = int(np.argmax(test_proba))
    lowest_risk_pos = int(np.argmin(test_proba))
    closest_threshold_pos = int(
        np.argmin(
            np.abs(test_proba - PRODUCTION_THRESHOLD)
        )
    )

    examples = []

    for label, position in [
        ("highest_risk", highest_risk_pos),
        ("closest_to_threshold", closest_threshold_pos),
        ("lowest_risk", lowest_risk_pos),
    ]:
        explanation = get_individual_explanation(
            explainer=explainer,
            preprocessor=preprocessor,
            X_row=X_test.iloc[[position]],
            feature_names=feature_names,
            production_probability=float(test_proba[position]),
            actual_status=int(y_test.iloc[position]),
            row_index=int(X_test.index[position]),
        )

        explanation["example"] = label
        examples.append(explanation)

    print("\n" + "=" * 78)
    print("INDIVIDUAL SHAP EXPLANATIONS")
    print("=" * 78)

    for example in examples:
        print(f"\n[{example['example']}]")
        print(f"  row index  : {example['row_index']}")
        print(
            f"  actual     : {example['actual_loan_status']}"
        )
        print(
            f"  P(default) : "
            f"{example['production_probability_default']:.4f}"
        )
        print(
            f"  decision   : "
            f"{example['production_decision']}"
        )
        print("  Top factors:")

        for factor in example["top_factors"]:
            arrow = (
                "↑ risk"
                if factor["shap_value"] > 0
                else "↓ risk"
            )
            print(
                f"    {factor['feature']:<40} "
                f"{factor['shap_value']:+.6f} "
                f"({arrow})"
            )

    # -------------------------------------------------------------------------
    # 8. Save artefacts
    # -------------------------------------------------------------------------
    global_csv = REPORTS_DIR / "step9_shap_global_importance.csv"
    examples_csv = REPORTS_DIR / "step9_shap_test_explanations.csv"
    examples_json = REPORTS_DIR / "step9_shap_examples.json"
    model_path = REPORTS_DIR / "step9_explainability_model.joblib"
    report_path = REPORTS_DIR / "step9_shap_report.json"

    global_importance.to_csv(
        global_csv,
        index=False,
    )

    examples_rows = []
    for example in examples:
        for factor in example["top_factors"]:
            examples_rows.append(
                {
                    "example": example["example"],
                    "row_index": example["row_index"],
                    "actual_loan_status": example[
                        "actual_loan_status"
                    ],
                    "production_probability_default": example[
                        "production_probability_default"
                    ],
                    "production_decision": example[
                        "production_decision"
                    ],
                    **factor,
                }
            )

    pd.DataFrame(examples_rows).to_csv(
        examples_csv,
        index=False,
    )

    examples_json.write_text(
        json.dumps(
            {
                "production_threshold": PRODUCTION_THRESHOLD,
                "examples": examples,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    joblib.dump(
        shap_pipeline,
        model_path,
    )

    report = {
        "step": "9",
        "title": "SHAP Explainability",
        "dataset": str(DATASET_PATH),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "shap_sample_size": int(len(X_shap)),
        "preprocessed_feature_count": int(
            len(feature_names)
        ),
        "production_model": {
            "lightgbm_params": LGB_PARAMS,
            "calibration": CALIBRATION_METHOD,
            "calibration_cv": CALIBRATION_CV,
            "threshold": PRODUCTION_THRESHOLD,
            "test_roc_auc": float(test_auc),
            "test_brier": float(test_brier),
        },
        "shap_model": {
            "explainer": "TreeExplainer",
            "explains": (
                "underlying LightGBM before sigmoid calibration"
            ),
            "expected_value_positive_class": float(
                expected_value
            ),
            "top_features": global_importance.head(
                15
            ).to_dict(orient="records"),
        },
        "individual_examples": examples,
        "artifacts": {
            "summary_plot": str(summary_png),
            "bar_plot": str(bar_png),
            "global_importance_csv": str(global_csv),
            "examples_csv": str(examples_csv),
            "examples_json": str(examples_json),
            "explainability_model": str(model_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("ARTEFACTS SAUVEGARDES")
    print("=" * 78)
    print(f"SHAP summary       : {summary_png}")
    print(f"SHAP bar           : {bar_png}")
    print(f"Global importance  : {global_csv}")
    print(f"Individual SHAP    : {examples_csv}")
    print(f"Examples JSON      : {examples_json}")
    print(f"SHAP model         : {model_path}")
    print(f"Report             : {report_path}")

    print("\n" + "=" * 78)
    print("STEP 9 TERMINEE")
    print("=" * 78)


if __name__ == "__main__":
    main()
