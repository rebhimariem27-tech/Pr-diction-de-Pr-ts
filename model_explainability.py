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
# FINPREDICT AI PRO — STEP 9b
# PRODUCTION-ALIGNED SHAP EXPLAINABILITY
# =============================================================================
#
# Production model
# ----------------
#   LightGBM
#   + Sigmoid calibration
#   + CV=5
#   + Production threshold = 0.48
#
# Why Step 9b?
# ------------
# Step 9 explained a separately trained LightGBM.
# Step 9b instead extracts the 5 LightGBM estimators INSIDE the
# CalibratedClassifierCV actually used by the production pipeline.
#
# For each test row:
#   1. production probability = calibrated pipeline probability
#   2. SHAP values are computed for each of the 5 internal LightGBM models
#   3. SHAP values are aggregated across folds
#   4. one-hot columns are regrouped into the original business feature
#
# IMPORTANT
# ---------
# SHAP still explains the underlying LightGBM score BEFORE sigmoid calibration.
# The sigmoid calibration transforms the model score into the production
# probability. Therefore, SHAP contributions must not be interpreted as
# direct percentage-point changes in P(default).
#
# Outputs
# -------
# data/reports/ml/
#   step9b_shap_global_business_features.csv
#   step9b_shap_individual_explanations.csv
#   step9b_shap_examples.json
#   step9b_shap_summary_business_features.png
#   step9b_shap_bar_business_features.png
#   step9b_shap_report.json
#   step9b_production_calibrated_pipeline.joblib
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
        raise ValueError(f"Colonnes manquantes : {missing}")

    return (
        df[MODEL_FEATURES].copy(),
        df[TARGET].astype(int).copy(),
    )


def normalize_shap_values(
    shap_values,
    expected_rows: int,
    expected_features: int,
) -> np.ndarray:
    """
    Supports SHAP output formats from several SHAP versions:
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
        # Common formats:
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


def clean_name(name: str) -> str:
    return (
        str(name)
        .replace("numeric__", "")
        .replace("categorical__", "")
    )


def build_business_feature_mapping(
    processed_feature_names: list[str],
) -> dict[str, str]:
    """
    Maps one-hot encoded columns back to the original business feature.
    """

    mapping = {}

    for feature in processed_feature_names:
        clean = clean_name(feature)

        if clean.startswith("person_home_ownership_"):
            business = "person_home_ownership"

        elif clean.startswith("loan_intent_"):
            business = "loan_intent"

        elif clean.startswith("loan_grade_"):
            business = "loan_grade"

        elif clean.startswith("cb_person_default_on_file_"):
            business = "cb_person_default_on_file"

        else:
            business = clean

        mapping[feature] = business

    return mapping


def aggregate_business_shap(
    shap_values: np.ndarray,
    processed_feature_names: list[str],
) -> pd.DataFrame:
    """
    Aggregate one-hot SHAP contributions back to the original feature.

    For each row:
        business_shap = sum(one_hot_shaps)

    Then:
        global importance = mean(abs(business_shap))
    """

    mapping = build_business_feature_mapping(
        processed_feature_names
    )

    business_features = sorted(
        set(mapping.values())
    )

    result = np.zeros(
        (
            shap_values.shape[0],
            len(business_features),
        ),
        dtype=float,
    )

    feature_to_position = {
        feature: idx
        for idx, feature in enumerate(business_features)
    }

    for column_idx, processed_feature in enumerate(
        processed_feature_names
    ):
        business_feature = mapping[processed_feature]

        result[:, feature_to_position[business_feature]] += (
            shap_values[:, column_idx]
        )

    return pd.DataFrame(
        result,
        columns=business_features,
    )


def extract_calibrated_estimators(
    calibrated_pipeline: CalibratedClassifierCV,
) -> list[Pipeline]:
    """
    Retrieve the 5 underlying fitted pipelines from the production
    CalibratedClassifierCV.

    This is written defensively because sklearn versions can expose
    calibrated estimators through the calibrated_classifiers_ objects.
    """

    fitted = getattr(
        calibrated_pipeline,
        "calibrated_classifiers_",
        None,
    )

    if fitted is None:
        raise AttributeError(
            "Impossible de trouver calibrated_classifiers_. "
            "Vérifiez votre version de scikit-learn."
        )

    pipelines = []

    for idx, calibrated_classifier in enumerate(fitted):
        estimator = getattr(
            calibrated_classifier,
            "estimator",
            None,
        )

        if estimator is None:
            # Compatibility fallback for versions exposing base_estimator.
            estimator = getattr(
                calibrated_classifier,
                "base_estimator",
                None,
            )

        if estimator is None:
            raise AttributeError(
                "Impossible de récupérer l'estimateur du fold "
                f"{idx} de CalibratedClassifierCV."
            )

        if not isinstance(estimator, Pipeline):
            raise TypeError(
                f"L'estimateur du fold {idx} n'est pas un Pipeline."
            )

        pipelines.append(estimator)

    return pipelines


def compute_fold_shap(
    estimator_pipeline: Pipeline,
    X_input: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP for one of the actual internal LightGBM pipelines.
    """

    preprocessor = estimator_pipeline.named_steps["preprocessor"]
    model = estimator_pipeline.named_steps["model"]

    processed = preprocessor.transform(X_input)

    feature_names = [
        clean_name(name)
        for name in preprocessor.get_feature_names_out()
    ]

    if not isinstance(processed, pd.DataFrame):
        processed = pd.DataFrame(
            processed,
            columns=feature_names,
            index=X_input.index,
        )
    else:
        processed.columns = feature_names

    explainer = shap.TreeExplainer(model)

    raw_shap = explainer.shap_values(processed)

    values = normalize_shap_values(
        raw_shap,
        expected_rows=len(processed),
        expected_features=len(feature_names),
    )

    return values, feature_names


def compute_ensemble_business_shap(
    calibrated_pipeline: CalibratedClassifierCV,
    X_input: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute SHAP on all 5 internal production estimators and average them.
    """

    estimators = extract_calibrated_estimators(
        calibrated_pipeline
    )

    aggregated_folds = []

    reference_business_features = None

    for fold_idx, estimator in enumerate(estimators, start=1):
        print(
            f"  SHAP fold {fold_idx}/{len(estimators)}..."
        )

        shap_values, processed_names = compute_fold_shap(
            estimator,
            X_input,
        )

        business_shap = aggregate_business_shap(
            shap_values,
            processed_names,
        )

        if reference_business_features is None:
            reference_business_features = list(
                business_shap.columns
            )
        else:
            business_shap = business_shap[
                reference_business_features
            ]

        aggregated_folds.append(
            business_shap.to_numpy()
        )

    average_values = np.mean(
        np.stack(aggregated_folds, axis=0),
        axis=0,
    )

    return (
        pd.DataFrame(
            average_values,
            columns=reference_business_features,
            index=X_input.index,
        ),
        reference_business_features,
    )


def get_global_importance(
    business_shap: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "feature": business_shap.columns,
            "mean_abs_shap": np.mean(
                np.abs(business_shap.to_numpy()),
                axis=0,
            ),
            "mean_signed_shap": np.mean(
                business_shap.to_numpy(),
                axis=0,
            ),
        }
    )

    return result.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)


def get_expected_value_for_reporting(
    estimator_pipeline: Pipeline,
) -> float:
    """
    Retrieve the positive-class expected SHAP base value from one internal
    LightGBM estimator. This is only a reference value for audit/debugging;
    SHAP is not interpreted as probability points.
    """

    model = estimator_pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)

    value = explainer.expected_value

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return float(arr[-1])

    return float(value)


def explain_single_row(
    calibrated_pipeline: CalibratedClassifierCV,
    X_row: pd.DataFrame,
    production_probability: float,
    actual_status: int,
) -> dict:
    """
    Production-aligned individual explanation:
      - production probability comes from calibrated pipeline
      - SHAP comes from all internal LightGBM estimators
      - one-hot features are grouped into business features
    """

    business_shap, business_features = (
        compute_ensemble_business_shap(
            calibrated_pipeline,
            X_row,
        )
    )

    row = business_shap.iloc[0]

    details = pd.DataFrame(
        {
            "feature": business_features,
            "shap_value": [
                float(row[feature])
                for feature in business_features
            ],
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

    risk_factors = (
        top[top["shap_value"] > 0]
        .sort_values(
            "shap_value",
            ascending=False,
        )
    )

    protective_factors = (
        top[top["shap_value"] < 0]
        .sort_values(
            "shap_value",
            ascending=True,
        )
    )

    return {
        "production_probability_default": float(
            production_probability
        ),
        "production_threshold": PRODUCTION_THRESHOLD,
        "production_decision": (
            "REFUSE"
            if production_probability >= PRODUCTION_THRESHOLD
            else "ACCEPT"
        ),
        "actual_loan_status": int(actual_status),
        "risk_factors": [
            {
                "feature": str(row["feature"]),
                "shap_value": float(row["shap_value"]),
                "direction": "increases_risk",
            }
            for _, row in risk_factors.iterrows()
        ],
        "protective_factors": [
            {
                "feature": str(row["feature"]),
                "shap_value": float(row["shap_value"]),
                "direction": "decreases_risk",
            }
            for _, row in protective_factors.iterrows()
        ],
        "top_factors": [
            {
                "feature": str(row["feature"]),
                "shap_value": float(row["shap_value"]),
                "direction": str(row["direction"]),
            }
            for _, row in top.iterrows()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FinPredict — Step 9b: "
            "production-aligned SHAP explainability"
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SHAP_SAMPLE_SIZE,
        help="Nombre de lignes TEST utilisées pour le SHAP global.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("FINPREDICT — STEP 9b: PRODUCTION-ALIGNED SHAP")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Load data / identical split
    # -------------------------------------------------------------------------
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
    # 2. Fit the REAL production calibrated pipeline
    # -------------------------------------------------------------------------
    print(
        "\nEntraînement du pipeline de production "
        "LightGBM + sigmoid..."
    )

    production_pipeline = build_calibrated_pipeline()
    production_pipeline.fit(
        X_train,
        y_train,
    )

    production_proba = (
        production_pipeline.predict_proba(X_test)[:, 1]
    )

    print(
        f"Production threshold : {PRODUCTION_THRESHOLD:.2f}"
    )
    print(
        f"TEST ROC-AUC         : "
        f"{roc_auc_score(y_test, production_proba):.4f}"
    )
    print(
        f"TEST Brier           : "
        f"{brier_score_loss(y_test, production_proba):.4f}"
    )

    internal_estimators = extract_calibrated_estimators(
        production_pipeline
    )

    print(
        f"Internal LightGBM estimators : "
        f"{len(internal_estimators)}"
    )

    # -------------------------------------------------------------------------
    # 3. Deterministic SHAP sample
    # -------------------------------------------------------------------------
    sample_size = min(
        max(args.sample_size, 1),
        len(X_test),
    )

    sample_indices = (
        X_test.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        ).index
    )

    X_shap = X_test.loc[sample_indices]

    print(
        f"SHAP sample : {len(X_shap)} lignes"
    )

    # -------------------------------------------------------------------------
    # 4. Production-aligned SHAP
    # -------------------------------------------------------------------------
    print(
        "\nCalcul des SHAP agrégés sur les 5 estimateurs..."
    )

    business_shap, business_features = (
        compute_ensemble_business_shap(
            production_pipeline,
            X_shap,
        )
    )

    global_importance = get_global_importance(
        business_shap
    )

    print("\n" + "=" * 78)
    print("GLOBAL BUSINESS-FEATURE SHAP — TOP FEATURES")
    print("=" * 78)

    print(
        global_importance.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # -------------------------------------------------------------------------
    # 5. Global summary bar plot
    #
    # beeswarm is not used here because SHAP values have been aggregated
    # across one-hot variables into business-level features.
    # -------------------------------------------------------------------------
    top_global = global_importance.head(12).copy()

    summary_png = (
        REPORTS_DIR /
        "step9b_shap_summary_business_features.png"
    )

    plt.figure(figsize=(10, 7))

    ordered = top_global.sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    plt.barh(
        ordered["feature"],
        ordered["mean_abs_shap"],
    )

    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Business feature")
    plt.title(
        "FinPredict — SHAP importance by business feature"
    )

    plt.tight_layout()
    plt.savefig(
        summary_png,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    bar_png = (
        REPORTS_DIR /
        "step9b_shap_bar_business_features.png"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        ordered["feature"],
        ordered["mean_abs_shap"],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Business feature")
    plt.title(
        "FinPredict — Global feature importance"
    )

    plt.tight_layout()
    plt.savefig(
        bar_png,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------------------
    # 6. Individual examples
    # -------------------------------------------------------------------------
    highest_risk_pos = int(
        np.argmax(production_proba)
    )

    lowest_risk_pos = int(
        np.argmin(production_proba)
    )

    closest_threshold_pos = int(
        np.argmin(
            np.abs(
                production_proba
                - PRODUCTION_THRESHOLD
            )
        )
    )

    example_definitions = [
        ("highest_risk", highest_risk_pos),
        ("closest_to_threshold", closest_threshold_pos),
        ("lowest_risk", lowest_risk_pos),
    ]

    examples = []

    print("\n" + "=" * 78)
    print("INDIVIDUAL PRODUCTION-ALIGNED EXPLANATIONS")
    print("=" * 78)

    for label, position in example_definitions:
        row_index = X_test.index[position]

        explanation = explain_single_row(
            calibrated_pipeline=production_pipeline,
            X_row=X_test.iloc[[position]],
            production_probability=float(
                production_proba[position]
            ),
            actual_status=int(
                y_test.iloc[position]
            ),
        )

        explanation["example"] = label
        explanation["row_index"] = int(row_index)

        examples.append(
            explanation
        )

        print(f"\n[{label}]")
        print(
            f"  row index  : {row_index}"
        )
        print(
            f"  actual     : "
            f"{explanation['actual_loan_status']}"
        )
        print(
            f"  P(default) : "
            f"{explanation['production_probability_default']:.4f}"
        )
        print(
            f"  decision   : "
            f"{explanation['production_decision']}"
        )

        print("  Risk factors:")
        for factor in explanation["risk_factors"]:
            print(
                f"    ↑ {factor['feature']:<35} "
                f"{factor['shap_value']:+.6f}"
            )

        print("  Protective factors:")
        for factor in explanation["protective_factors"]:
            print(
                f"    ↓ {factor['feature']:<35} "
                f"{factor['shap_value']:+.6f}"
            )

    # -------------------------------------------------------------------------
    # 7. Save individual CSV
    # -------------------------------------------------------------------------
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
                    "production_threshold": (
                        PRODUCTION_THRESHOLD
                    ),
                    "production_decision": example[
                        "production_decision"
                    ],
                    "feature": factor["feature"],
                    "shap_value": factor["shap_value"],
                    "direction": factor["direction"],
                }
            )

    examples_csv = (
        REPORTS_DIR /
        "step9b_shap_individual_explanations.csv"
    )

    pd.DataFrame(examples_rows).to_csv(
        examples_csv,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 8. Save global CSV
    # -------------------------------------------------------------------------
    global_csv = (
        REPORTS_DIR /
        "step9b_shap_global_business_features.csv"
    )

    global_importance.to_csv(
        global_csv,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 9. Save JSON
    # -------------------------------------------------------------------------
    examples_json = (
        REPORTS_DIR /
        "step9b_shap_examples.json"
    )

    examples_json.write_text(
        json.dumps(
            {
                "production_threshold": (
                    PRODUCTION_THRESHOLD
                ),
                "production_probability_source": (
                    "CalibratedClassifierCV"
                ),
                "shap_source": (
                    "Average SHAP across the 5 internal "
                    "LightGBM estimators"
                ),
                "business_feature_aggregation": (
                    "sum one-hot SHAP contributions into "
                    "original business feature"
                ),
                "examples": examples,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 10. Save production pipeline for downstream app integration
    # -------------------------------------------------------------------------
    pipeline_path = (
        REPORTS_DIR /
        "step9b_production_calibrated_pipeline.joblib"
    )

    joblib.dump(
        production_pipeline,
        pipeline_path,
    )

    # -------------------------------------------------------------------------
    # 11. Report
    # -------------------------------------------------------------------------
    central_estimator = internal_estimators[0]

    try:
        base_expected_value = (
            get_expected_value_for_reporting(
                central_estimator
            )
        )
    except Exception:
        base_expected_value = None

    report = {
        "step": "9b",
        "title": "Production-aligned SHAP Explainability",
        "dataset": str(DATASET_PATH),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "shap_sample_size": int(len(X_shap)),
        "production": {
            "model": "LightGBM + sigmoid calibration",
            "calibration_method": CALIBRATION_METHOD,
            "calibration_cv": CALIBRATION_CV,
            "threshold": PRODUCTION_THRESHOLD,
            "test_roc_auc": float(
                roc_auc_score(
                    y_test,
                    production_proba,
                )
            ),
            "test_brier": float(
                brier_score_loss(
                    y_test,
                    production_proba,
                )
            ),
            "internal_lightgbm_count": len(
                internal_estimators
            ),
        },
        "shap": {
            "explainer": "TreeExplainer",
            "aggregation": (
                "average SHAP across the 5 internal "
                "LightGBM estimators"
            ),
            "categorical_aggregation": (
                "one-hot columns grouped back to original "
                "business feature"
            ),
            "interpretation": (
                "SHAP explains the underlying LightGBM "
                "score before sigmoid calibration; SHAP "
                "values are not percentage-point changes "
                "in calibrated P(default)."
            ),
            "reference_expected_value": (
                base_expected_value
            ),
            "global_importance": (
                global_importance.to_dict(
                    orient="records"
                )
            ),
        },
        "examples": examples,
        "artifacts": {
            "global_csv": str(global_csv),
            "individual_csv": str(examples_csv),
            "examples_json": str(examples_json),
            "summary_plot": str(summary_png),
            "bar_plot": str(bar_png),
            "production_pipeline": str(pipeline_path),
        },
    }

    report_path = (
        REPORTS_DIR /
        "step9b_shap_production_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("ARTEFACTS SAUVEGARDES")
    print("=" * 78)
    print(f"Global SHAP CSV    : {global_csv}")
    print(f"Individual CSV     : {examples_csv}")
    print(f"Examples JSON      : {examples_json}")
    print(f"Summary plot       : {summary_png}")
    print(f"Bar plot           : {bar_png}")
    print(f"Production pipeline: {pipeline_path}")
    print(f"Report             : {report_path}")

    print("\n" + "=" * 78)
    print("STEP 9b TERMINEE")
    print("=" * 78)


if __name__ == "__main__":
    main()