"""
FinPredict AI Pro - ML Pipeline | Step 6
=========================================

Objectif :
- Optimiser les hyperparamètres de LightGBM SANS jamais toucher au TEST.
- Méthodologie : TRAIN -> TRAIN' / VALIDATION -> sélection -> TEST (une fois).

Pipeline retenue (voir Step 5) :
    BASELINE = Median Imputation + Most-frequent Imputation + OneHotEncoder

    (Step 5 a montré que KNN Smart Imputation n'apporte pas de gain de
    discrimination mesurable ; on garde donc le BASELINE, plus simple et
    plus rapide, pour le tuning. KNN reste candidat pour Step 8 calibration.)

Déroulé :
1. Charger les données, split TRAIN/TEST identique à Step 3/4/5 (80/20, seed=42)
2. Split TRAIN -> TRAIN' (80%) / VALIDATION (20%), stratifié, seed=42
3. Tester une grille de configurations LightGBM, entraînées sur TRAIN',
   évaluées sur VALIDATION (ROC-AUC)
4. Sélectionner la meilleure configuration sur VALIDATION
5. Réentraîner cette configuration sur TRAIN complet (TRAIN' + VALIDATION)
6. Évaluer UNE SEULE FOIS sur TEST
7. Comparer au baseline Step 3/5 (défaut LightGBM)

Usage:
    python model_ml_step6.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION (identique à Step 3 / 4 / 5)
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

# Baseline de référence (défauts utilisés en Step 3/4/5), pour comparaison finale
BASELINE_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

# Grille d'hyperparamètres à tester sur la VALIDATION
# (volontairement raisonnable : on cherche un mieux, pas une explosion combinatoire)
PARAM_GRID = [
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "reg_lambda": 0.0},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "reg_lambda": 0.0},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1, "min_child_samples": 20, "reg_lambda": 1.0},
    {"n_estimators": 800, "learning_rate": 0.02, "num_leaves": 63, "max_depth": -1, "min_child_samples": 30, "reg_lambda": 1.0},
    {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 6, "min_child_samples": 20, "reg_lambda": 0.5},
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_child_samples": 50, "reg_lambda": 2.0},
    {"n_estimators": 1000, "learning_rate": 0.01, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "reg_lambda": 1.0},
]


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


# ============================================================
# 3. PREPROCESSOR (BASELINE, retenu suite à Step 5)
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
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


def build_pipeline(lgb_params: dict) -> Pipeline:
    params = {
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        **lgb_params,
    }

    model = lgb.LGBMClassifier(**params)

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# ============================================================
# 4. METRIQUES
# ============================================================

def compute_metrics(y_true, y_proba, threshold: float = 0.50) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINPREDICT — STEP 6: HYPERPARAMETER TUNING (TRAIN / VALIDATION)")
    print("=" * 78)

    # --------------------------------------------------------
    # Dataset + split TRAIN/TEST (identique à Step 3/4/5)
    # --------------------------------------------------------

    df = load_dataset()
    X, y = build_xy(df)

    print(f"\nDataset : {len(df)} lignes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"TRAIN (avant split val) : {len(X_train)}")
    print(f"TEST  (intouché)        : {len(X_test)}")

    # --------------------------------------------------------
    # Split TRAIN -> TRAIN' / VALIDATION
    # --------------------------------------------------------

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    print(f"\nTRAIN' (fit)  : {len(X_train_fit)}")
    print(f"VALIDATION    : {len(X_val)}")

    # --------------------------------------------------------
    # Recherche d'hyperparamètres sur VALIDATION uniquement
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print(f"RECHERCHE SUR VALIDATION — {len(PARAM_GRID)} configurations")
    print("=" * 78)

    results = []

    for i, params in enumerate(PARAM_GRID, start=1):
        pipeline = build_pipeline(params)
        pipeline.fit(X_train_fit, y_train_fit)

        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_proba)

        results.append({"config_id": i, "params": params, "val_metrics": val_metrics})

        print(
            f"[{i}/{len(PARAM_GRID)}] "
            f"n_estimators={params['n_estimators']:<5} "
            f"lr={params['learning_rate']:<6} "
            f"num_leaves={params['num_leaves']:<4} "
            f"max_depth={params['max_depth']:<3} "
            f"min_child={params['min_child_samples']:<3} "
            f"reg_lambda={params['reg_lambda']:<4} "
            f"-> VAL ROC-AUC = {val_metrics['roc_auc']:.4f}  "
            f"PR-AUC = {val_metrics['pr_auc']:.4f}"
        )

    # --------------------------------------------------------
    # Sélection du meilleur sur VALIDATION (ROC-AUC)
    # --------------------------------------------------------

    best = max(results, key=lambda r: r["val_metrics"]["roc_auc"])

    print("\n" + "=" * 78)
    print("MEILLEURE CONFIGURATION (choisie sur VALIDATION)")
    print("=" * 78)
    print(f"Config #{best['config_id']} : {best['params']}")
    print(f"VAL ROC-AUC : {best['val_metrics']['roc_auc']:.4f}")
    print(f"VAL PR-AUC  : {best['val_metrics']['pr_auc']:.4f}")

    # --------------------------------------------------------
    # Réentraînement sur TRAIN complet avec la meilleure config
    # --------------------------------------------------------

    print("\nRéentraînement de la meilleure config sur TRAIN complet...")

    best_pipeline = build_pipeline(best["params"])
    best_pipeline.fit(X_train, y_train)

    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    tuned_test_metrics = compute_metrics(y_test, y_test_proba)

    # --------------------------------------------------------
    # Baseline de référence (défauts Step 3/4/5) sur le même TEST
    # --------------------------------------------------------

    print("Entraînement du BASELINE (hyperparamètres par défaut) pour comparaison...")

    default_pipeline = build_pipeline(
        {k: v for k, v in BASELINE_PARAMS.items() if k not in ("objective", "class_weight", "random_state", "n_jobs", "verbosity")}
    )
    default_pipeline.fit(X_train, y_train)

    y_test_proba_default = default_pipeline.predict_proba(X_test)[:, 1]
    default_test_metrics = compute_metrics(y_test, y_test_proba_default)

    # --------------------------------------------------------
    # Comparaison finale sur TEST
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("COMPARAISON FINALE SUR TEST (utilisé une seule fois)")
    print("=" * 78)

    rows = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    labels = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }

    print(f"{'Metric':<12}{'DEFAULT':<15}{'TUNED':<15}{'Diff':<12}")
    for key in rows:
        a = default_test_metrics[key]
        b = tuned_test_metrics[key]
        print(f"{labels[key]:<12}{a:<15.4f}{b:<15.4f}{b - a:+.4f}")

    print("\nClassification Report — TUNED:\n")
    y_test_pred_tuned = (y_test_proba >= 0.50).astype(int)
    print(
        classification_report(
            y_test, y_test_pred_tuned, target_names=["Classe 0", "Classe 1"], zero_division=0
        )
    )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    import joblib

    model_path = REPORTS_DIR.parent / "lightgbm_tuned.joblib"
    joblib.dump(best_pipeline, model_path)

    report = {
        "validation_search": results,
        "best_config": best["params"],
        "test": {
            "default": default_test_metrics,
            "tuned": tuned_test_metrics,
        },
    }

    report_path = REPORTS_DIR / "step6_tuning_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nModèle tuné sauvegardé : {model_path}")
    print(f"Rapport sauvegardé     : {report_path}")

    print("\n" + "=" * 78)
    print("STEP 6 TERMINÉE")
    print("=" * 78)
    print(
        "\nRappel : le TEST n'a été utilisé qu'une seule fois, à la toute fin, "
        "pour comparer DEFAULT vs TUNED. Aucune décision n'a été prise en "
        "regardant le TEST pendant la recherche d'hyperparamètres."
    )


if __name__ == "__main__":
    main()