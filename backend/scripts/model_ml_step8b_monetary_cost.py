"""
FinPredict AI Pro - ML Pipeline | Step 8b
===========================================

Objectif :
- Remplacer le ratio fixe "FN = 5x FP" (compté PAR CAS, indépendamment du
  montant) par un COUT MONETAIRE REEL, pondéré par le montant du prêt.

Pourquoi ce changement ?
- Un défaut sur un prêt de 2 000$ et un défaut sur un prêt de 50 000$ ne
  coûtent pas la même chose à la banque. Le ratio fixe l'ignorait.
- Un défaut ne fait pas perdre 100% du capital : il y a du recouvrement.
  On utilise LGD (Loss Given Default) = 40% (validé avec l'utilisateur) :
  perte réelle sur un FN = 40% x loan_amnt.
- Un FP (bon client refusé) ne coûte pas un montant fixe : la banque perd
  la marge nette qu'elle aurait gagnée sur la durée du prêt.

Hypothèses (modifiables ci-dessous) :
    LGD                  = 0.40   (recouvrement fort, validé utilisateur)
    COST_OF_FUNDS        = 0.03   (coût de financement de la banque, 3%)
    AVERAGE_LOAN_DURATION_YEARS = 3.0  (pas de colonne durée dans le
                                          dataset -> hypothèse standard
                                          prêt personnel)

Formules par ligne i :
    cout_FN_i = LGD x loan_amnt_i
    marge_annuelle_i = max(loan_int_rate_i / 100 - COST_OF_FUNDS, 0) x loan_amnt_i
    cout_FP_i = marge_annuelle_i x AVERAGE_LOAN_DURATION_YEARS

Méthodologie (comme Step 7/8) :
1. Split TRAIN/TEST identique (80/20, seed=42)
2. Probabilités calibrées OUT-OF-FOLD sur tout TRAIN (sigmoid, cv=5,
   méthode retenue en Step 8)
3. Scan de seuils : coût MONETAIRE total (pas un simple comptage) sur OOF
4. Sélection du seuil qui minimise le coût monétaire total sur OOF
5. Application UNE SEULE FOIS au TEST avec un modèle calibré réentraîné
   sur TRAIN complet
6. Comparaison avec le seuil retenu en Step 8 (0.18, ratio fixe 5x) pour
   voir si la pondération par montant change la conclusion

Usage:
    python model_ml_step8b_monetary_cost.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
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

CALIBRATION_METHOD = "sigmoid"  # retenu en Step 8

# --- HYPOTHESES DE COUT (modifiables) --------------------------------------
LGD = 0.40                          # validé avec l'utilisateur
COST_OF_FUNDS = 0.03                # coût de financement de la banque
AVERAGE_LOAN_DURATION_YEARS = 3.0   # pas de colonne durée dans le dataset
# ----------------------------------------------------------------------------

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
# 3. COUTS MONETAIRES PAR LIGNE
# ============================================================

def compute_row_costs(
    X: pd.DataFrame,
    loan_amnt_fill_value: float,
    loan_int_rate_fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (cout_fn, cout_fp) par ligne, alignés sur l'index de X.

    IMPORTANT : loan_amnt et loan_int_rate peuvent contenir des NaN dans le
    dataset brut (c'est pour ça que le pipeline de preprocessing utilise un
    SimpleImputer(strategy="median")). Si on ne les impute pas ici aussi,
    un seul NaN dans loan_int_rate rend cost_fp = NaN pour cette ligne, puis
    total_cost = NaN dès qu'un FP tombe sur cette ligne, et min() sur une
    liste contenant des NaN renvoie un résultat arbitraire (le premier
    élément), pas le vrai minimum. D'où le "seuil optimal = 0.05" aberrant.

    On impute avec la médiane du TRAIN (fill values passés en argument),
    exactement comme le fait le pipeline de preprocessing, pour rester
    cohérent et ne pas fuiter d'information du test vers le train.
    """

    loan_amnt = X["loan_amnt"].fillna(loan_amnt_fill_value).to_numpy(dtype=float)
    loan_int_rate = X["loan_int_rate"].fillna(loan_int_rate_fill_value).to_numpy(dtype=float)

    n_missing_amnt = X["loan_amnt"].isna().sum()
    n_missing_rate = X["loan_int_rate"].isna().sum()
    if n_missing_amnt or n_missing_rate:
        print(
            f"  [compute_row_costs] NaN imputés : loan_amnt={n_missing_amnt}, "
            f"loan_int_rate={n_missing_rate}"
        )

    cost_fn = LGD * loan_amnt

    margin_rate = np.maximum(loan_int_rate / 100.0 - COST_OF_FUNDS, 0.0)
    annual_margin = margin_rate * loan_amnt
    cost_fp = annual_margin * AVERAGE_LOAN_DURATION_YEARS

    return cost_fn, cost_fp


# ============================================================
# 4. PIPELINE
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


def build_calibrated_pipeline() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method=CALIBRATION_METHOD,
        cv=5,
    )


# ============================================================
# 5. COUT MONETAIRE TOTAL POUR UN SEUIL DONNE
# ============================================================

def monetary_cost_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    is_fn = (y_pred == 0) & (y_true == 1)
    is_fp = (y_pred == 1) & (y_true == 0)

    total_fn_cost = float(cost_fn[is_fn].sum())
    total_fp_cost = float(cost_fp[is_fp].sum())
    total_cost = total_fn_cost + total_fp_cost

    return {
        "threshold": float(threshold),
        "n_fn": int(is_fn.sum()),
        "n_fp": int(is_fp.sum()),
        "fn_cost": total_fn_cost,
        "fp_cost": total_fp_cost,
        "total_cost": total_cost,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def scan_thresholds(y_true, y_proba, cost_fn, cost_fp) -> list[dict]:
    return [
        monetary_cost_at_threshold(y_true, y_proba, cost_fn, cost_fp, t)
        for t in THRESHOLDS_TO_SCAN
    ]


def save_cost_curve(scan_results: list[dict], best_threshold: float, fixed_ratio_threshold: float, path: Path) -> None:
    thresholds = [r["threshold"] for r in scan_results]
    costs = [r["total_cost"] for r in scan_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, costs, label="Coût monétaire total OOF (TRAIN)")
    ax.axvline(best_threshold, color="red", linestyle="--", label=f"Seuil optimal (monétaire) = {best_threshold:.2f}")
    ax.axvline(0.50, color="gray", linestyle=":", label="Seuil par défaut = 0.50")
    ax.axvline(
        fixed_ratio_threshold, color="purple", linestyle="-.",
        label=f"Seuil Step 8 (ratio fixe 5x) = {fixed_ratio_threshold:.2f}",
    )
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Coût monétaire total ($)")
    ax.set_title("FinPredict — Coût monétaire réel vs Seuil (pondéré par montant)")
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
    print("FINPREDICT — STEP 8b: MONETARY COST MODEL (pondéré par montant)")
    print(f"LGD={LGD}  |  Coût du capital={COST_OF_FUNDS}  |  Durée moyenne={AVERAGE_LOAN_DURATION_YEARS} ans")
    print("=" * 78)

    df = load_dataset()
    X, y = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  (intouché jusqu'à la fin) : {len(X_test)}")

    # Coûts monétaires par ligne (train et test)
    # Médianes calculées sur TRAIN uniquement (même logique que le
    # SimpleImputer du pipeline), puis appliquées à train ET test.
    loan_amnt_median = X_train["loan_amnt"].median()
    loan_int_rate_median = X_train["loan_int_rate"].median()

    print(
        f"\nMédianes (TRAIN) utilisées pour imputer les coûts : "
        f"loan_amnt={loan_amnt_median:.0f}, loan_int_rate={loan_int_rate_median:.2f}"
    )

    cost_fn_train, cost_fp_train = compute_row_costs(X_train, loan_amnt_median, loan_int_rate_median)
    cost_fn_test, cost_fp_test = compute_row_costs(X_test, loan_amnt_median, loan_int_rate_median)

    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    print(f"\nCoût FN moyen (TRAIN)  : {cost_fn_train.mean():.0f} $")
    print(f"Coût FP moyen (TRAIN)  : {cost_fp_train.mean():.0f} $")
    print(f"Ratio moyen FN/FP implicite : {cost_fn_train.mean() / cost_fp_train.mean():.2f}x")

    # --------------------------------------------------------
    # Probabilités OUT-OF-FOLD calibrées sur tout TRAIN
    # --------------------------------------------------------

    print("\nCalcul des probabilités out-of-fold calibrées (sigmoid, 5 folds)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_proba = cross_val_predict(
        build_calibrated_pipeline(),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    # --------------------------------------------------------
    # Scan des seuils : coût MONETAIRE sur OOF
    # --------------------------------------------------------

    print(f"\nBalayage de {len(THRESHOLDS_TO_SCAN)} seuils (coût monétaire, OOF TRAIN)...")

    scan_results = scan_thresholds(y_train_arr, oof_proba, cost_fn_train, cost_fp_train)

    best = min(scan_results, key=lambda r: r["total_cost"])
    at_050 = next(r for r in scan_results if abs(r["threshold"] - 0.50) < 1e-9)
    at_018 = min(scan_results, key=lambda r: abs(r["threshold"] - 0.18))

    print("\n" + "=" * 78)
    print("RESULTATS DU SCAN (OOF TRAIN) — seuils autour de l'optimum")
    print("=" * 78)

    around_best = [r for r in scan_results if abs(r["threshold"] - best["threshold"]) <= 0.05]
    for r in around_best:
        marker = "  <-- OPTIMAL (monétaire)" if r["threshold"] == best["threshold"] else ""
        print(
            f"seuil={r['threshold']:.2f}  n_FP={r['n_fp']:<5} n_FN={r['n_fn']:<5} "
            f"coût=${r['total_cost']:<12,.0f} "
            f"precision={r['precision']:.4f} recall={r['recall']:.4f}{marker}"
        )

    print("\n" + "-" * 78)
    print(f"Seuil 0.50                          : coût=${at_050['total_cost']:,.0f}")
    print(f"Seuil 0.18 (retenu en Step 8, ratio 5x) : coût=${at_018['total_cost']:,.0f}")
    print(f"Seuil optimal (coût monétaire) = {best['threshold']:.2f} : coût=${best['total_cost']:,.0f}")

    print(f"\nÉcart entre seuil monétaire optimal et seuil Step 8 (0.18) : {abs(best['threshold'] - 0.18):.2f}")

    cost_curve_path = REPORTS_DIR / "step8b_monetary_cost_vs_threshold.png"
    save_cost_curve(scan_results, best["threshold"], 0.18, cost_curve_path)
    print(f"Graphique sauvegardé : {cost_curve_path}")

    # --------------------------------------------------------
    # Application UNE SEULE FOIS au TEST
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("APPLICATION SUR TEST (une seule fois) — comparaison des 3 seuils")
    print("=" * 78)

    final_pipeline = build_calibrated_pipeline()
    final_pipeline.fit(X_train, y_train)
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

    test_050 = monetary_cost_at_threshold(y_test_arr, y_test_proba, cost_fn_test, cost_fp_test, 0.50)
    test_018 = monetary_cost_at_threshold(y_test_arr, y_test_proba, cost_fn_test, cost_fp_test, 0.18)
    test_optimal = monetary_cost_at_threshold(y_test_arr, y_test_proba, cost_fn_test, cost_fp_test, best["threshold"])

    print(f"\n{'':20}{'Seuil 0.50':<18}{'Seuil 0.18 (Step8)':<20}{'Seuil ' + str(best['threshold']) + ' (monétaire)':<22}")
    print(f"{'n_FP':<20}{test_050['n_fp']:<18}{test_018['n_fp']:<20}{test_optimal['n_fp']:<22}")
    print(f"{'n_FN':<20}{test_050['n_fn']:<18}{test_018['n_fn']:<20}{test_optimal['n_fn']:<22}")
    print(f"{'Coût total ($)':<20}{test_050['total_cost']:<18,.0f}{test_018['total_cost']:<20,.0f}{test_optimal['total_cost']:<22,.0f}")
    print(f"{'Precision':<20}{test_050['precision']:<18.4f}{test_018['precision']:<20.4f}{test_optimal['precision']:<22.4f}")
    print(f"{'Recall':<20}{test_050['recall']:<18.4f}{test_018['recall']:<20.4f}{test_optimal['recall']:<22.4f}")

    savings_vs_050 = test_050["total_cost"] - test_optimal["total_cost"]
    savings_vs_050_pct = 100 * savings_vs_050 / test_050["total_cost"] if test_050["total_cost"] > 0 else 0.0

    savings_vs_018 = test_018["total_cost"] - test_optimal["total_cost"]
    savings_vs_018_pct = 100 * savings_vs_018 / test_018["total_cost"] if test_018["total_cost"] > 0 else 0.0

    print(f"\nÉconomie du seuil monétaire optimal vs 0.50 : ${savings_vs_050:,.0f} ({savings_vs_050_pct:+.1f}%)")
    print(f"Économie du seuil monétaire optimal vs 0.18  : ${savings_vs_018:,.0f} ({savings_vs_018_pct:+.1f}%)")

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    report = {
        "assumptions": {
            "LGD": LGD,
            "cost_of_funds": COST_OF_FUNDS,
            "average_loan_duration_years": AVERAGE_LOAN_DURATION_YEARS,
        },
        "implicit_fn_fp_ratio_train_mean": float(cost_fn_train.mean() / cost_fp_train.mean()),
        "oof_scan": scan_results,
        "best_threshold_monetary": best["threshold"],
        "test_at_050": test_050,
        "test_at_018_step8": test_018,
        "test_at_optimal_monetary": test_optimal,
    }

    report_path = REPORTS_DIR / "step8b_monetary_cost_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nRapport sauvegardé : {report_path}")
    print(f"\n>>> SEUIL RECOMMANDÉ (coût monétaire réel) : {best['threshold']:.2f} <<<")

    print("\n" + "=" * 78)
    print("STEP 8b TERMINÉE")
    print("=" * 78)


if __name__ == "__main__":
    main()