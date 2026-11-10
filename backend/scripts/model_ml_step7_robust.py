"""
FinPredict AI Pro - ML Pipeline | Step 7 (version robuste)
=============================================================

Pourquoi cette version ?
-------------------------
La première version de Step 7 choisissait le seuil sur un seul split
TRAIN' / VALIDATION (5186 échantillons, ~350 positifs). La courbe de coût
était quasi plate entre 0.30 et 0.40 : le minimum exact (0.34) était en
grande partie du bruit d'échantillonnage. Résultat : le seuil "optimal"
sur validation (-6.9% de coût) s'est dégradé sur test (+2.5% de coût).

Correction :
- On calcule des probabilités OUT-OF-FOLD sur TOUT le TRAIN (25927
  échantillons) via une validation croisée à 5 folds (cross_val_predict).
- Chaque échantillon de TRAIN reçoit une probabilité prédite par un modèle
  qui ne l'a PAS vu pendant l'entraînement (comme du out-of-sample), mais
  on utilise 5x plus de données que l'ancienne VALIDATION -> courbe de
  coût beaucoup plus stable, minimum beaucoup moins sensible au bruit.
- Le seuil optimal est choisi sur cette courbe out-of-fold.
- Le modèle final est réentraîné sur TRAIN complet, et le seuil déjà
  choisi est appliqué UNE SEULE FOIS au TEST.

Usage:
    python model_ml_step7_robust.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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

THRESHOLDS_TO_SCAN = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Seuil théorique Bayes-optimal SI le modèle était parfaitement calibré :
# p* = cost_FP / (cost_FP + cost_FN)
THEORETICAL_THRESHOLD = COST_FP / (COST_FP + COST_FN)


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
# 3. PIPELINE
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


def build_pipeline() -> Pipeline:
    model = lgb.LGBMClassifier(**DEFAULT_LGB_PARAMS)
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


# ============================================================
# 4. COUT METIER
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


def save_cost_curve(scan_results: list[dict], best_threshold: float, path: Path) -> None:
    thresholds = [r["threshold"] for r in scan_results]
    costs = [r["total_cost"] for r in scan_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, costs, label="Coût total OUT-OF-FOLD (TRAIN, 5 folds)")
    ax.axvline(best_threshold, color="red", linestyle="--", label=f"Seuil optimal = {best_threshold:.2f}")
    ax.axvline(0.50, color="gray", linestyle=":", label="Seuil par défaut = 0.50")
    ax.axvline(
        THEORETICAL_THRESHOLD, color="green", linestyle="-.",
        label=f"Seuil Bayes théorique = {THEORETICAL_THRESHOLD:.2f}",
    )
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Coût total (unités arbitraires)")
    ax.set_title("FinPredict — Coût métier vs Seuil (OUT-OF-FOLD sur TRAIN)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# 5. MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINPREDICT — STEP 7 (ROBUSTE): THRESHOLD VIA OUT-OF-FOLD CV")
    print(f"Coût FN = {COST_FN}x coût FP  |  Seuil Bayes théorique = {THEORETICAL_THRESHOLD:.3f}")
    print("=" * 78)

    df = load_dataset()
    X, y = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  (intouché jusqu'à la fin) : {len(X_test)}")

    # --------------------------------------------------------
    # Probabilités OUT-OF-FOLD sur TOUT le TRAIN (5 folds)
    # --------------------------------------------------------

    print("\nCalcul des probabilités out-of-fold (5 folds, tout le TRAIN)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_proba = cross_val_predict(
        build_pipeline(),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    print(f"Probabilités out-of-fold obtenues pour {len(oof_proba)} échantillons TRAIN.")

    # --------------------------------------------------------
    # Scan des seuils sur les probabilités OOF (bien plus stable)
    # --------------------------------------------------------

    print(f"\nBalayage de {len(THRESHOLDS_TO_SCAN)} seuils sur OUT-OF-FOLD TRAIN...")

    scan_results = scan_thresholds(y_train, oof_proba)

    best = min(scan_results, key=lambda r: r["total_cost"])
    default_050 = next(r for r in scan_results if abs(r["threshold"] - 0.50) < 1e-9)

    print("\n" + "=" * 78)
    print("RESULTATS DU SCAN (OUT-OF-FOLD TRAIN) — seuils autour de l'optimum")
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
    print(
        f"Seuil 0.50 (OOF)     : FP={default_050['fp']}  FN={default_050['fn']}  "
        f"coût={default_050['total_cost']:.0f}"
    )
    print(
        f"Seuil optimal {best['threshold']:.2f} (OOF) : FP={best['fp']}  FN={best['fn']}  "
        f"coût={best['total_cost']:.0f}"
    )

    cost_reduction = default_050["total_cost"] - best["total_cost"]
    cost_reduction_pct = 100 * cost_reduction / default_050["total_cost"]
    print(f"\nRéduction de coût vs 0.50 (OOF, sur ~26000 échantillons) : {cost_reduction:.0f} ({cost_reduction_pct:+.1f}%)")

    curve_path = REPORTS_DIR / "step7_robust_cost_vs_threshold.png"
    save_cost_curve(scan_results, best["threshold"], curve_path)
    print(f"Graphique sauvegardé : {curve_path}")

    # --------------------------------------------------------
    # Application UNE SEULE FOIS au TEST
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("APPLICATION DU SEUIL OPTIMAL SUR TEST (une seule fois)")
    print("=" * 78)

    final_pipeline = build_pipeline()
    final_pipeline.fit(X_train, y_train)

    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

    test_default = cost_at_threshold(y_test, y_test_proba, 0.50)
    test_optimal = cost_at_threshold(y_test, y_test_proba, best["threshold"])

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
    print(f"\nRéduction de coût vs seuil 0.50 (sur TEST) : {test_cost_reduction:.0f} unités ({test_cost_reduction_pct:+.1f}%)")

    print("\nClassification Report — TEST @ seuil optimal :\n")
    y_test_pred_optimal = (y_test_proba >= best["threshold"]).astype(int)
    print(classification_report(y_test, y_test_pred_optimal, target_names=["Classe 0", "Classe 1"], zero_division=0))

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    import joblib

    model_path = REPORTS_DIR.parent / "lightgbm_final_thresholded_robust.joblib"
    joblib.dump(final_pipeline, model_path)

    report = {
        "cost_assumption": {"cost_fn": COST_FN, "cost_fp": COST_FP},
        "theoretical_bayes_threshold": THEORETICAL_THRESHOLD,
        "oof_scan": scan_results,
        "best_threshold": best["threshold"],
        "oof_at_best": best,
        "oof_at_default_050": default_050,
        "test_at_default_050": test_default,
        "test_at_best_threshold": test_optimal,
    }

    report_path = REPORTS_DIR / "step7_robust_threshold_optimization.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nModèle final sauvegardé : {model_path}")
    print(f"Rapport sauvegardé      : {report_path}")
    print(f"\n>>> SEUIL DE PRODUCTION RETENU (robuste) : {best['threshold']:.2f} <<<")

    print("\n" + "=" * 78)
    print("STEP 7 (ROBUSTE) TERMINÉE")
    print("=" * 78)


if __name__ == "__main__":
    main()