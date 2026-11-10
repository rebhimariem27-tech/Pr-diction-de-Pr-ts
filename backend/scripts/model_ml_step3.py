"""
FinPredict AI Pro - ML Pipeline | Step 3
========================================

Objectifs :
- construire le preprocessing
- entraîner LightGBM
- évaluer sur le TEST uniquement
- effectuer une cross-validation sur TRAIN
- générer les premières visualisations du modèle

Important :
- Le TEST reste invisible pendant l'entraînement.
- Les métriques finales sont calculées uniquement sur TEST.
- Cette version constitue le BASELINE MODEL.
- Aucun tuning avancé n'est effectué ici.

Usage:
    python model_ml_step3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/processed/donnees_prets_features.csv"
)

REPORTS_DIR = Path(
    "data/reports/ml"
)

TARGET = "loan_status"


# ------------------------------------------------------------
# Features numériques
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Features catégorielles
# ------------------------------------------------------------

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
# 2. CHARGEMENT
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
# 3. X / y
# ============================================================

def build_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[
        MODEL_FEATURES
    ].copy()

    y = df[
        TARGET
    ].copy()

    return X, y


# ============================================================
# 4. TRAIN / TEST
# ============================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


# ============================================================
# 5. PREPROCESSING
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
    )


# ============================================================
# 6. MODÈLE LIGHTGBM
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
# 7. PIPELINE COMPLÈTE
# ============================================================

def build_pipeline():

    preprocessor = build_preprocessor()

    model = build_model()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

def cross_validate_model(
    pipeline,
    X_train,
    y_train,
):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=None,
    )

    return scores


# ============================================================
# 9. MATRICE DE CONFUSION
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Classe 0",
            "Classe 1",
        ],
    )

    display.plot(
        ax=ax
    )

    ax.set_title(
        "FinPredict — Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR
        / "confusion_matrix.png",
        dpi=150,
    )

    plt.close(fig)

    return cm


# ============================================================
# 10. ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    y_proba,
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_proba,
    )

    auc = roc_auc_score(
        y_true,
        y_proba,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "FinPredict — ROC Curve"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR
        / "roc_curve.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# 11. PRECISION-RECALL CURVE
# ============================================================

def save_precision_recall_curve(
    y_true,
    y_proba,
):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            y_proba,
        )
    )

    ap = average_precision_score(
        y_true,
        y_proba,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        recall,
        precision,
        label=f"AP = {ap:.4f}",
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "FinPredict — Precision-Recall Curve"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR
        / "precision_recall_curve.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# 12. RAPPORT MÉTRIQUES
# ============================================================

def build_metrics(
    y_test,
    y_pred,
    y_proba,
    cv_scores,
):

    return {
        "test": {
            "roc_auc": float(
                roc_auc_score(
                    y_test,
                    y_proba,
                )
            ),
            "pr_auc": float(
                average_precision_score(
                    y_test,
                    y_proba,
                )
            ),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
        },
        "cross_validation": {
            "folds": 5,
            "roc_auc_scores": [
                float(score)
                for score in cv_scores
            ],
            "mean_roc_auc": float(
                cv_scores.mean()
            ),
            "std_roc_auc": float(
                cv_scores.std()
            ),
        },
    }


# ============================================================
# 13. MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "FINPREDICT — "
        "ML PIPELINE STEP 3"
    )

    print(
        "BASELINE LIGHTGBM + EVALUATION"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    X, y = build_xy(
        df
    )

    print(
        f"\nDataset : {len(df)} lignes"
    )

    print(
        f"Features : {len(MODEL_FEATURES)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    print(
        f"\nTRAIN : {len(X_train)}"
    )

    print(
        f"TEST  : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = build_pipeline()

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print(
        "\nCross-validation 5 folds..."
    )

    cv_scores = cross_validate_model(
        pipeline,
        X_train,
        y_train,
    )

    print(
        "\nROC-AUC par fold :"
    )

    for i, score in enumerate(
        cv_scores,
        start=1,
    ):
        print(
            f"  Fold {i}: "
            f"{score:.4f}"
        )

    print(
        f"\nROC-AUC moyen : "
        f"{cv_scores.mean():.4f}"
    )

    print(
        f"Écart-type    : "
        f"{cv_scores.std():.4f}"
    )

    # --------------------------------------------------------
    # Entraînement final sur TRAIN
    # --------------------------------------------------------

    print(
        "\nEntraînement LightGBM..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "✅ Modèle entraîné."
    )

    # --------------------------------------------------------
    # Prédictions TEST
    # --------------------------------------------------------

    y_proba = pipeline.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_proba >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    metrics = build_metrics(
        y_test,
        y_pred,
        y_proba,
        cv_scores,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "PERFORMANCES SUR TEST"
    )

    print(
        "=" * 72
    )

    print(
        f"ROC-AUC   : "
        f"{metrics['test']['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{metrics['test']['pr_auc']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['test']['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['test']['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['test']['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report :\n"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Classe 0",
                "Classe 1",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Graphiques
    # --------------------------------------------------------

    print(
        "Génération des graphiques..."
    )

    cm = save_confusion_matrix(
        y_test,
        y_pred,
    )

    save_roc_curve(
        y_test,
        y_proba,
    )

    save_precision_recall_curve(
        y_test,
        y_proba,
    )

    # --------------------------------------------------------
    # Rapport JSON
    # --------------------------------------------------------

    report_path = (
        REPORTS_DIR
        / "baseline_metrics.json"
    )

    report_path.write_text(
        __import__("json").dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print(
        "\n✅ Évaluation terminée."
    )

    print(
        f"Rapport : {report_path}"
    )

    print(
        f"Graphiques : {REPORTS_DIR}"
    )

    print(
        "\nIMPORTANT :"
    )

    print(
        "Ces résultats constituent le BASELINE."
    )

    print(
        "Nous ne considérons pas encore le modèle "
        "comme optimisé ou calibré."
    )


if __name__ == "__main__":
    main()