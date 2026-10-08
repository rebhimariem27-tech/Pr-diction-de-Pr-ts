"""
FinPredict AI Pro - ML Pipeline | Step 8
=========================================

Objectif :
- Calibrer les probabilités du modèle LightGBM (déformées par
  class_weight="balanced"), puis refaire la recherche de seuil (Step 7)
  sur des probabilités calibrées, pour vérifier si le seuil optimal se
  rapproche du seuil Bayes théorique (0.167) et devient plus stable.

Méthodologie :
1. Split TRAIN/TEST identique aux steps précédents (80/20, seed=42)
2. Split TRAIN -> TRAIN' / VALIDATION (80/20, seed=42)
3. Comparer 3 versions des probabilités sur VALIDATION :
     - RAW        : pipeline standard, non calibré
     - SIGMOID    : CalibratedClassifierCV(method="sigmoid", cv=5)
     - ISOTONIC   : CalibratedClassifierCV(method="isotonic", cv=5)
   via le Brier score (plus bas = mieux calibré) + courbe de calibration
4. Choisir automatiquement la meilleure méthode (Brier le plus bas)
5. Calculer des probabilités OUT-OF-FOLD calibrées sur TOUT le TRAIN
   (cross_val_predict avec la méthode choisie)
6. Refaire le scan de seuils (coût métier FN=5xFP) sur ces probabilités
   calibrées
7. Réentraîner le modèle calibré final sur TRAIN complet, appliquer le
   seuil choisi UNE SEULE FOIS au TEST

Usage:
    python model_ml_step8.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION (identique aux steps précédents)
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

DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

COST_FN = 5.0
COST_FP = 1.0
THEORETICAL_THRESHOLD = COST_FP / (COST_FP + COST_FN)

THRESHOLDS_TO_SCAN = np.round(np.arange(0.05, 0.96, 0.01), 2)


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
# 3. PIPELINE DE BASE
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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


def build_base_pipeline() -> Pipeline:
    model = lgb.LGBMClassifier(**DEFAULT_LGB_PARAMS)
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def build_calibrated_pipeline(method: str) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method=method,
        cv=5,
    )


# ============================================================
# 4. COUT METIER (identique à Step 7)
# ============================================================

def cost_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_cost = fp * COST_FP + fn * COST_FN
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total_cost": float(total_cost),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def scan_thresholds(y_true, y_proba) -> list[dict]:
    return [cost_at_threshold(y_true, y_proba, t) for t in THRESHOLDS_TO_SCAN]


# ============================================================
# 5. GRAPHIQUES
# ============================================================

def save_calibration_curves(y_val, proba_raw, proba_sigmoid, proba_isotonic, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Calibration parfaite")

    for label, proba in [
        ("RAW (non calibré)", proba_raw),
        ("SIGMOID", proba_sigmoid),
        ("ISOTONIC", proba_isotonic),
    ]:
        frac_pos, mean_pred = calibration_curve(y_val, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)

    ax.set_xlabel("Probabilité prédite moyenne")
    ax.set_ylabel("Fraction de vrais positifs observée")
    ax.set_title("FinPredict — Courbes de calibration (VALIDATION)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cost_curve(scan_results: list[dict], best_threshold: float, path: Path) -> None:
    thresholds = [r["threshold"] for r in scan_results]
    costs = [r["total_cost"] for r in scan_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, costs, label="Coût total OOF (TRAIN, probabilités calibrées)")
    ax.axvline(best_threshold, color="red", linestyle="--", label=f"Seuil optimal = {best_threshold:.2f}")
    ax.axvline(0.50, color="gray", linestyle=":", label="Seuil par défaut = 0.50")
    ax.axvline(
        THEORETICAL_THRESHOLD, color="green", linestyle="-.",
        label=f"Seuil Bayes théorique = {THEORETICAL_THRESHOLD:.2f}",
    )
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Coût total (unités arbitraires)")
    ax.set_title("FinPredict — Coût métier vs Seuil (probabilités CALIBREES, OOF sur TRAIN)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINPREDICT — STEP 8: PROBABILITY CALIBRATION + THRESHOLD REVISITED")
    print("=" * 78)

    df = load_dataset()
    X, y = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  (intouché jusqu'à la fin) : {len(X_test)}")

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    print(f"TRAIN' (fit)  : {len(X_train_fit)}")
    print(f"VALIDATION    : {len(X_val)}")

    # --------------------------------------------------------
    # Comparaison RAW vs SIGMOID vs ISOTONIC sur VALIDATION
    # --------------------------------------------------------

    print("\nEntraînement RAW (non calibré) sur TRAIN'...")
    raw_pipeline = build_base_pipeline()
    raw_pipeline.fit(X_train_fit, y_train_fit)
    proba_raw_val = raw_pipeline.predict_proba(X_val)[:, 1]

    print("Entraînement SIGMOID (Platt scaling, cv=5) sur TRAIN'...")
    sigmoid_pipeline = build_calibrated_pipeline("sigmoid")
    sigmoid_pipeline.fit(X_train_fit, y_train_fit)
    proba_sigmoid_val = sigmoid_pipeline.predict_proba(X_val)[:, 1]

    print("Entraînement ISOTONIC (cv=5) sur TRAIN'...")
    isotonic_pipeline = build_calibrated_pipeline("isotonic")
    isotonic_pipeline.fit(X_train_fit, y_train_fit)
    proba_isotonic_val = isotonic_pipeline.predict_proba(X_val)[:, 1]

    brier_raw = brier_score_loss(y_val, proba_raw_val)
    brier_sigmoid = brier_score_loss(y_val, proba_sigmoid_val)
    brier_isotonic = brier_score_loss(y_val, proba_isotonic_val)

    roc_raw = roc_auc_score(y_val, proba_raw_val)
    roc_sigmoid = roc_auc_score(y_val, proba_sigmoid_val)
    roc_isotonic = roc_auc_score(y_val, proba_isotonic_val)

    print("\n" + "=" * 78)
    print("COMPARAISON DE CALIBRATION (VALIDATION)")
    print("=" * 78)
    print(f"{'Méthode':<12}{'Brier (bas=mieux)':<20}{'ROC-AUC':<10}")
    print(f"{'RAW':<12}{brier_raw:<20.5f}{roc_raw:<10.4f}")
    print(f"{'SIGMOID':<12}{brier_sigmoid:<20.5f}{roc_sigmoid:<10.4f}")
    print(f"{'ISOTONIC':<12}{brier_isotonic:<20.5f}{roc_isotonic:<10.4f}")

    calib_curve_path = REPORTS_DIR / "step8_calibration_curves.png"
    save_calibration_curves(y_val, proba_raw_val, proba_sigmoid_val, proba_isotonic_val, calib_curve_path)
    print(f"\nCourbes de calibration sauvegardées : {calib_curve_path}")

    candidates = {"sigmoid": brier_sigmoid, "isotonic": brier_isotonic}
    best_method = min(candidates, key=candidates.get)

    print(f"\n>>> Méthode de calibration retenue : {best_method.upper()} (Brier le plus bas) <<<")

    # --------------------------------------------------------
    # Probabilités OUT-OF-FOLD calibrées sur TOUT le TRAIN
    # --------------------------------------------------------

    print(f"\nCalcul des probabilités out-of-fold CALIBREES ({best_method}) sur tout TRAIN...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_calibrated_proba = cross_val_predict(
        build_calibrated_pipeline(best_method),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    print(f"Probabilités out-of-fold calibrées obtenues pour {len(oof_calibrated_proba)} échantillons.")

    # --------------------------------------------------------
    # Scan des seuils sur probabilités calibrées
    # --------------------------------------------------------

    print(f"\nBalayage de {len(THRESHOLDS_TO_SCAN)} seuils (probabilités calibrées, OOF TRAIN)...")

    scan_results = scan_thresholds(y_train, oof_calibrated_proba)
    best = min(scan_results, key=lambda r: r["total_cost"])
    default_050 = next(r for r in scan_results if abs(r["threshold"] - 0.50) < 1e-9)

    print("\n" + "=" * 78)
    print("RESULTATS DU SCAN (OOF TRAIN, CALIBRE) — seuils autour de l'optimum")
    print("=" * 78)

    around_best = [r for r in scan_results if abs(r["threshold"] - best["threshold"]) <= 0.05]
    for r in around_best:
        marker = "  <-- OPTIMAL" if r["threshold"] == best["threshold"] else ""
        print(
            f"seuil={r['threshold']:.2f}  FP={r['fp']:<5} FN={r['fn']:<5} "
            f"coût={r['total_cost']:<8.0f} "
            f"precision={r['precision']:.4f} recall={r['recall']:.4f}{marker}"
        )

    print("\n" + "-" * 78)
    print(f"Seuil 0.50 (OOF calibré)         : coût={default_050['total_cost']:.0f}")
    print(f"Seuil optimal {best['threshold']:.2f} (OOF calibré) : coût={best['total_cost']:.0f}")
    print(f"Distance au seuil Bayes théorique ({THEORETICAL_THRESHOLD:.3f}) : {abs(best['threshold'] - THEORETICAL_THRESHOLD):.3f}")

    cost_reduction = default_050["total_cost"] - best["total_cost"]
    cost_reduction_pct = 100 * cost_reduction / default_050["total_cost"]
    print(f"\nRéduction de coût vs 0.50 (OOF calibré) : {cost_reduction:.0f} ({cost_reduction_pct:+.1f}%)")

    cost_curve_path = REPORTS_DIR / "step8_cost_vs_threshold_calibrated.png"
    save_cost_curve(scan_results, best["threshold"], cost_curve_path)
    print(f"Graphique sauvegardé : {cost_curve_path}")

    # --------------------------------------------------------
    # Application UNE SEULE FOIS au TEST
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("APPLICATION FINALE SUR TEST (modèle calibré, une seule fois)")
    print("=" * 78)

    final_calibrated_pipeline = build_calibrated_pipeline(best_method)
    final_calibrated_pipeline.fit(X_train, y_train)

    y_test_proba_calibrated = final_calibrated_pipeline.predict_proba(X_test)[:, 1]

    test_default = cost_at_threshold(y_test, y_test_proba_calibrated, 0.50)
    test_optimal = cost_at_threshold(y_test, y_test_proba_calibrated, best["threshold"])
    test_brier = brier_score_loss(y_test, y_test_proba_calibrated)
    test_roc_auc = roc_auc_score(y_test, y_test_proba_calibrated)

    print(f"\nBrier score sur TEST (modèle calibré) : {test_brier:.5f}")
    print(f"ROC-AUC sur TEST (modèle calibré)      : {test_roc_auc:.4f}")

    print(f"\n{'':15}{'Seuil 0.50':<20}{'Seuil optimal ' + str(best['threshold']):<20}")
    print(f"{'FP':<15}{test_default['fp']:<20}{test_optimal['fp']:<20}")
    print(f"{'FN':<15}{test_default['fn']:<20}{test_optimal['fn']:<20}")
    print(f"{'Coût total':<15}{test_default['total_cost']:<20.0f}{test_optimal['total_cost']:<20.0f}")
    print(f"{'Precision':<15}{test_default['precision']:<20.4f}{test_optimal['precision']:<20.4f}")
    print(f"{'Recall':<15}{test_default['recall']:<20.4f}{test_optimal['recall']:<20.4f}")
    print(f"{'F1':<15}{test_default['f1']:<20.4f}{test_optimal['f1']:<20.4f}")

    test_cost_reduction = test_default["total_cost"] - test_optimal["total_cost"]
    test_cost_reduction_pct = (
        100 * test_cost_reduction / test_default["total_cost"] if test_default["total_cost"] > 0 else 0.0
    )
    print(f"\nRéduction de coût vs seuil 0.50 (sur TEST, calibré) : {test_cost_reduction:.0f} unités ({test_cost_reduction_pct:+.1f}%)")

    print("\nClassification Report — TEST @ seuil optimal (calibré) :\n")
    y_test_pred_optimal = (y_test_proba_calibrated >= best["threshold"]).astype(int)
    print(classification_report(y_test, y_test_pred_optimal, target_names=["Classe 0", "Classe 1"], zero_division=0))

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    import joblib

    model_path = REPORTS_DIR.parent / "lightgbm_calibrated_final.joblib"
    joblib.dump(final_calibrated_pipeline, model_path)

    report = {
        "calibration_comparison_validation": {
            "raw": {"brier": brier_raw, "roc_auc": roc_raw},
            "sigmoid": {"brier": brier_sigmoid, "roc_auc": roc_sigmoid},
            "isotonic": {"brier": brier_isotonic, "roc_auc": roc_isotonic},
            "chosen_method": best_method,
        },
        "cost_assumption": {"cost_fn": COST_FN, "cost_fp": COST_FP},
        "theoretical_bayes_threshold": THEORETICAL_THRESHOLD,
        "oof_scan_calibrated": scan_results,
        "best_threshold": best["threshold"],
        "oof_at_best": best,
        "oof_at_default_050": default_050,
        "test_brier_score": float(test_brier),
        "test_roc_auc": float(test_roc_auc),
        "test_at_default_050": test_default,
        "test_at_best_threshold": test_optimal,
    }

    report_path = REPORTS_DIR / "step8_calibration_and_threshold.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nModèle final calibré sauvegardé : {model_path}")
    print(f"Rapport sauvegardé               : {report_path}")
    print(f"\n>>> METHODE DE CALIBRATION : {best_method.upper()} <<<")
    print(f">>> SEUIL DE PRODUCTION RETENU : {best['threshold']:.2f} <<<")

    print("\n" + "=" * 78)
    print("STEP 8 TERMINÉE")
    print("=" * 78)


if __name__ == "__main__":
    main()