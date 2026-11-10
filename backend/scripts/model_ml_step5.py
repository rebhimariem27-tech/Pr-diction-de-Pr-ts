"""
FinPredict AI Pro - ML Pipeline | Step 5
=========================================

Objectif :
- Comparer de façon PARFAITEMENT EQUITABLE deux pipelines de preprocessing :

    Pipeline A - BASELINE
        Median Imputation (numeric) + Most-frequent Imputation (categorical)
        + OneHotEncoder

    Pipeline B - SMART (KNN)
        KNN Imputation (k=7) + Missing Indicators (numeric)
        + Most-frequent Imputation (categorical)
        + OneHotEncoder

- Les deux pipelines sont évalués :
    * en 5-fold StratifiedKFold CV sur TRAIN, avec EXACTEMENT les mêmes folds
    * sur le même TEST final (jamais utilisé pour choisir quoi que ce soit)

- Le TEST n'est utilisé qu'une seule fois, à la toute fin, pour les deux
  pipelines, une fois que la CV a déjà été faite.

Important :
- Split identique à Step 3 / Step 4 : test_size=0.20, random_state=42, stratify=y
- Même modèle LightGBM (mêmes hyperparamètres) que Step 3 / Step 4
- Rien n'est optimisé ici : c'est une comparaison, pas un tuning.

Usage:
    python model_ml_step5.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION (identique à Step 3 / Step 4)
# ============================================================

DATASET_PATH = Path("data/processed/donnees_prets_features.csv")
REPORTS_DIR = Path("data/reports/ml")

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

KNN_K = 7  # même k que smart_preprocessor.joblib (Step 4)


# ============================================================
# 2. CHARGEMENT
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("Le dataset est vide.")

    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


# ============================================================
# 3. PREPROCESSORS
# ============================================================

def build_baseline_preprocessor() -> ColumnTransformer:
    """Pipeline A : identique à Step 3."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_knn_preprocessor() -> ColumnTransformer:
    """Pipeline B : KNN Imputation (k=7) + Missing Indicators + OneHot."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                KNNImputer(n_neighbors=KNN_K, add_indicator=True),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ============================================================
# 4. MODELE (identique à Step 3 / Step 4)
# ============================================================

def build_model() -> lgb.LGBMClassifier:
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


def build_full_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model()),
        ]
    )


# ============================================================
# 5. CROSS-VALIDATION (mêmes folds pour A et B)
# ============================================================

def cross_validate_pipeline(pipeline: Pipeline, X_train, y_train) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
# 6. EVALUATION SUR TEST
# ============================================================

def evaluate_on_test(pipeline: Pipeline, X_train, y_train, X_test, y_test, threshold: float = 0.50) -> dict:
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Classe 0", "Classe 1"], zero_division=0
        ),
    }


# ============================================================
# 7. AFFICHAGE COMPARATIF
# ============================================================

def print_cv_table(cv_baseline: np.ndarray, cv_knn: np.ndarray) -> None:
    print("\n" + "=" * 78)
    print("5-FOLD CROSS VALIDATION (ROC-AUC) — MÊMES FOLDS POUR LES DEUX PIPELINES")
    print("=" * 78)

    header = f"{'Fold':<8}{'BASELINE':<15}{'KNN':<15}"
    print(header)

    for i, (a, b) in enumerate(zip(cv_baseline, cv_knn), start=1):
        print(f"{i:<8}{a:<15.4f}{b:<15.4f}")

    print("-" * 38)
    print(f"{'Mean':<8}{cv_baseline.mean():<15.4f}{cv_knn.mean():<15.4f}")
    print(f"{'Std':<8}{cv_baseline.std():<15.4f}{cv_knn.std():<15.4f}")


def print_test_table(metrics_baseline: dict, metrics_knn: dict) -> None:
    print("\n" + "=" * 78)
    print("FINAL TEST (même test, jamais utilisé pour choisir la pipeline)")
    print("=" * 78)

    rows = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    labels = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }

    print(f"{'Metric':<12}{'BASELINE':<15}{'KNN':<15}{'Diff (KNN-Base)':<18}")
    for key in rows:
        a = metrics_baseline[key]
        b = metrics_knn[key]
        print(f"{labels[key]:<12}{a:<15.4f}{b:<15.4f}{b - a:+.4f}")

    print("\nConfusion Matrix — BASELINE:")
    print(np.array(metrics_baseline["confusion_matrix"]))

    print("\nConfusion Matrix — KNN:")
    print(np.array(metrics_knn["confusion_matrix"]))


# ============================================================
# 8. MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINPREDICT — STEP 5: BASELINE VS SMART IMPUTATION (FAIR COMPARISON)")
    print("=" * 78)

    # --------------------------------------------------------
    # Dataset + split (identique à Step 3 / Step 4)
    # --------------------------------------------------------

    df = load_dataset()
    X, y = build_xy(df)

    print(f"\nDataset : {len(df)} lignes")
    print(f"Features : {len(MODEL_FEATURES)}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  : {len(X_test)}")

    # --------------------------------------------------------
    # Pipelines
    # --------------------------------------------------------

    pipeline_baseline = build_full_pipeline(build_baseline_preprocessor())
    pipeline_knn = build_full_pipeline(build_knn_preprocessor())

    # --------------------------------------------------------
    # Cross-validation (mêmes folds pour les deux)
    # --------------------------------------------------------

    print("\nCross-validation BASELINE (5 folds)...")
    cv_baseline = cross_validate_pipeline(pipeline_baseline, X_train, y_train)

    print("Cross-validation KNN (5 folds, mêmes splits)...")
    cv_knn = cross_validate_pipeline(pipeline_knn, X_train, y_train)

    print_cv_table(cv_baseline, cv_knn)

    # --------------------------------------------------------
    # Evaluation finale sur TEST (une seule fois, pour les deux)
    # --------------------------------------------------------

    print("\nEntraînement final + évaluation TEST — BASELINE...")
    metrics_baseline = evaluate_on_test(pipeline_baseline, X_train, y_train, X_test, y_test)

    print("Entraînement final + évaluation TEST — KNN...")
    metrics_knn = evaluate_on_test(pipeline_knn, X_train, y_train, X_test, y_test)

    print_test_table(metrics_baseline, metrics_knn)

    print("\nClassification Report — BASELINE:\n")
    print(metrics_baseline["classification_report"])

    print("Classification Report — KNN:\n")
    print(metrics_knn["classification_report"])

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    roc_gain = metrics_knn["roc_auc"] - metrics_baseline["roc_auc"]
    cv_gain = cv_knn.mean() - cv_baseline.mean()

    print("\n" + "=" * 78)
    print("VERDICT (purement descriptif — pas de décision automatique)")
    print("=" * 78)
    print(f"Gain ROC-AUC (CV)   : {cv_gain:+.4f}")
    print(f"Gain ROC-AUC (test) : {roc_gain:+.4f}")

    if abs(roc_gain) < 0.002 and abs(cv_gain) < 0.002:
        print(
            "\n=> Les deux pipelines sont statistiquement quasi équivalentes "
            "en discrimination (ROC-AUC). Le choix entre BASELINE et KNN doit "
            "se faire sur d'autres critères : stabilité (std CV), calibration "
            "des probabilités, comportement sur les valeurs manquantes."
        )
    elif roc_gain > 0:
        print("\n=> KNN apporte un gain mesurable sur le test par rapport au BASELINE.")
    else:
        print("\n=> BASELINE reste au moins aussi performant que KNN sur ce test.")

    # --------------------------------------------------------
    # Sauvegarde JSON
    # --------------------------------------------------------

    report = {
        "cross_validation": {
            "folds": 5,
            "baseline": {
                "scores": [float(s) for s in cv_baseline],
                "mean": float(cv_baseline.mean()),
                "std": float(cv_baseline.std()),
            },
            "knn": {
                "scores": [float(s) for s in cv_knn],
                "mean": float(cv_knn.mean()),
                "std": float(cv_knn.std()),
            },
        },
        "test": {
            "baseline": {
                k: v
                for k, v in metrics_baseline.items()
                if k != "classification_report"
            },
            "knn": {
                k: v
                for k, v in metrics_knn.items()
                if k != "classification_report"
            },
        },
    }

    report_path = REPORTS_DIR / "step5_baseline_vs_knn.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nRapport sauvegardé : {report_path}")

    print("\n" + "=" * 78)
    print("STEP 5 TERMINÉE")
    print("=" * 78)


if __name__ == "__main__":
    main()