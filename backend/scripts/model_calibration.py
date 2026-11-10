"""
FinPredict AI Pro - Model Calibration
======================================

Objectif :
    Évaluer et améliorer la calibration des probabilités produites
    par le modèle LightGBM.

Méthodes comparées :
    1. Modèle brut (non calibré)
    2. Sigmoid / Platt Scaling
    3. Isotonic Regression

Principe :
    TRAIN      -> entraîne LightGBM
    VALIDATION -> apprend la calibration
    TEST       -> évaluation finale

IMPORTANT :
    Le TEST ne sert pas à apprendre la calibration.

Sorties :
    data/reports/ml/calibration/
        calibration_comparison.json
        calibration_curve_comparison.png
        brier_comparison.png
        ece_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/processed/donnees_prets_features.csv"
)

REPORT_DIR = Path(
    "data/reports/ml/calibration"
)

TARGET = "loan_status"


# ============================================================
# 2. FEATURES
# ============================================================

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


# ============================================================
# 3. CHARGEMENT
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    if df.empty:
        raise ValueError(
            "Le dataset est vide."
        )

    return df


# ============================================================
# 4. X / y
# ============================================================

def build_xy(
    df: pd.DataFrame,
):

    X = df[
        MODEL_FEATURES
    ].copy()

    y = df[
        TARGET
    ].copy()

    return X, y


# ============================================================
# 5. TRAIN / VALIDATION / TEST
# ============================================================

def create_splits(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Split :

        64 % TRAIN
        16 % VALIDATION
        20 % TEST
    """

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        random_state=42,
        stratify=y_development,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# 6. PREPROCESSOR
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# 7. LIGHTGBM
# ============================================================

def build_model():

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


# ============================================================
# 8. ECE
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10,
):

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for lower, upper in zip(
        bins[:-1],
        bins[1:],
    ):

        if upper == 1.0:

            mask = (
                (probabilities >= lower)
                &
                (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                &
                (probabilities < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        confidence = float(
            probabilities[mask].mean()
        )

        actual_rate = float(
            y_true[mask].mean()
        )

        ece += (
            count
            / len(probabilities)
        ) * abs(
            confidence
            - actual_rate
        )

    return float(ece)


# ============================================================
# 9. MÉTRIQUES
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
        ),
    }


# ============================================================
# 10. COURBE DE CALIBRATION
# ============================================================

def save_calibration_curve(
    y_test,
    probability_dict,
):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Calibration parfaite",
    )

    for name, probabilities in probability_dict.items():

        fraction_positive, mean_predicted = (
            calibration_curve(
                y_test,
                probabilities,
                n_bins=10,
                strategy="uniform",
            )
        )

        ax.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=name,
        )

    ax.set_xlabel(
        "Probabilité moyenne prédite"
    )

    ax.set_ylabel(
        "Fréquence réellement positive"
    )

    ax.set_title(
        "FinPredict — Comparaison des calibrations"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "calibration_curve_comparison.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# 11. COMPARAISON BRIER
# ============================================================

def save_brier_comparison(
    metrics_dict,
):

    names = list(
        metrics_dict.keys()
    )

    values = [
        metrics_dict[name][
            "brier_score"
        ]
        for name in names
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        names,
        values,
    )

    ax.set_ylabel(
        "Brier Score"
    )

    ax.set_title(
        "FinPredict — Comparaison Brier Score"
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "brier_comparison.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# 12. COMPARAISON ECE
# ============================================================

def save_ece_comparison(
    metrics_dict,
):

    names = list(
        metrics_dict.keys()
    )

    values = [
        metrics_dict[name][
            "ece"
        ]
        for name in names
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        names,
        values,
    )

    ax.set_ylabel(
        "ECE"
    )

    ax.set_title(
        "FinPredict — Comparaison ECE"
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "ece_comparison.png",
        dpi=150,
    )

    plt.close(fig)
class SigmoidCalibrator:
    """
    Calibration Sigmoid / Platt.

    On transforme la probabilité brute p en logit(p),
    puis une régression logistique apprend la correction.
    """

    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs",
            random_state=42,
        )

    def fit(self, probabilities, y):
        p = np.clip(
            np.asarray(probabilities, dtype=float),
            1e-6,
            1 - 1e-6,
        )

        logit = np.log(
            p / (1.0 - p)
        ).reshape(-1, 1)

        self.model.fit(
            logit,
            y,
        )

        return self

    def predict_proba(self, probabilities):
        p = np.clip(
            np.asarray(probabilities, dtype=float),
            1e-6,
            1 - 1e-6,
        )

        logit = np.log(
            p / (1.0 - p)
        ).reshape(-1, 1)

        return self.model.predict_proba(
            logit
        )[:, 1]


class IsotonicCalibrator:
    """
    Calibration isotone.

    Apprend une fonction monotone qui transforme
    les probabilités brutes en probabilités calibrées.
    """

    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

    def fit(self, probabilities, y):
        self.model.fit(
            probabilities,
            y,
        )

        return self

    def predict_proba(self, probabilities):
        return self.model.predict(
            probabilities
        )

# ============================================================
# 13. MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "FINPREDICT — "
        "MODEL CALIBRATION"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    X, y = build_xy(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_splits(
        X,
        y,
    )

    print(
        f"\nTrain      : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_validation)}"
    )

    print(
        f"Test       : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Préprocessing
    # --------------------------------------------------------

    preprocessor = (
        build_preprocessor()
    )

    X_train_p = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_validation_p = (
        preprocessor.transform(
            X_validation
        )
    )

    X_test_p = (
        preprocessor.transform(
            X_test
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    X_train_p = pd.DataFrame(
        X_train_p,
        columns=feature_names,
        index=X_train.index,
    )

    X_validation_p = pd.DataFrame(
        X_validation_p,
        columns=feature_names,
        index=X_validation.index,
    )

    X_test_p = pd.DataFrame(
        X_test_p,
        columns=feature_names,
        index=X_test.index,
    )

    print(
        f"\nFeatures après preprocessing : "
        f"{X_train_p.shape[1]}"
    )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    base_model = build_model()

    print(
        "\nEntraînement LightGBM..."
    )

    base_model.fit(
        X_train_p,
        y_train,
    )

    print(
        "✅ LightGBM entraîné."
    )

    # --------------------------------------------------------
    # Probabilités du modèle brut
    # --------------------------------------------------------

    p_validation_raw = (
        base_model.predict_proba(
            X_validation_p
        )[:, 1]
    )

    p_test_raw = (
        base_model.predict_proba(
            X_test_p
        )[:, 1]
    )

    # --------------------------------------------------------
    # Calibration Sigmoid
    # --------------------------------------------------------

    print(
        "\nApprentissage calibration SIGMOID..."
    )

    sigmoid = (
    SigmoidCalibrator()
    .fit(
        p_validation_raw,
        y_validation,
    )
)

    p_test_sigmoid = (
    sigmoid.predict_proba(
        p_test_raw
    )
)
    print(
    "✅ Sigmoid terminée."
)

    # --------------------------------------------------------
    # Calibration Isotonic
    # --------------------------------------------------------

    print(
        "Apprentissage calibration ISOTONIC..."
    )

    isotonic = (
    IsotonicCalibrator()
    .fit(
        p_validation_raw,
        y_validation,
    )
)

    p_test_isotonic = (
        isotonic.predict_proba(
        p_test_raw
    )
)

    print(
    "✅ Isotonic terminée."
)
    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    results = {

        "raw_lightgbm": calculate_metrics(
            y_test,
            p_test_raw,
        ),

        "sigmoid": calculate_metrics(
            y_test,
            p_test_sigmoid,
        ),

        "isotonic": calculate_metrics(
            y_test,
            p_test_isotonic,
        ),
    }

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "COMPARAISON DES MODÈLES"
    )

    print(
        "=" * 72
    )

    for name, metrics in results.items():

        print(
            f"\n{name.upper()}"
        )

        print(
            f"  ROC-AUC   : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC    : "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"  Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"  Brier     : "
            f"{metrics['brier_score']:.4f}"
        )

        print(
            f"  ECE       : "
            f"{metrics['ece']:.4f}"
        )

    # --------------------------------------------------------
    # Graphiques
    # --------------------------------------------------------

    probability_dict = {
        "LightGBM brut": p_test_raw,
        "Sigmoid": p_test_sigmoid,
        "Isotonic": p_test_isotonic,
    }

    save_calibration_curve(
        y_test,
        probability_dict,
    )

    save_brier_comparison(
        results
    )

    save_ece_comparison(
        results
    )

    # --------------------------------------------------------
    # Rapport
    # --------------------------------------------------------

    report = {
        "test_metrics": results,
        "calibration_methods": [
            "raw_lightgbm",
            "sigmoid",
            "isotonic",
        ],
        "methodology": {
            "lightgbm_training": "train",
            "calibration_training": "validation",
            "final_evaluation": "test",
        },
    }

    report_path = (
        REPORT_DIR
        / "calibration_comparison.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n✅ Calibration terminée."
    )

    print(
        f"Rapport : {report_path}"
    )

    print(
        f"Graphiques : {REPORT_DIR}"
    )

    print(
        "\nLe TEST n'a pas servi à apprendre "
        "la calibration."
    )


if __name__ == "__main__":
    main()