"""
FinPredict AI Pro - PRODUCTION PIPELINE
=========================================

Ce script consolide toutes les décisions validées à travers le
développement (Steps 1 à 8) en un seul pipeline de production :

    Preprocessing : BASELINE
        Median Imputation (numérique) + Most-frequent (catégoriel)
        + OneHotEncoder
        -> retenu en Step 5 : KNN Smart Imputation n'apportait pas de
           gain de discrimination mesurable (ROC-AUC quasi identique,
           écart < 0.002), donc on garde le plus simple et le plus rapide.

    Modèle : LightGBM, hyperparamètres PAR DEFAUT
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight="balanced"
        -> retenu en Step 6 : le tuning (grid search sur validation)
           améliorait le ROC-AUC de façon négligeable (+0.0005) mais
           faisait BAISSER le recall (-0.0092), ce qui va dans le
           mauvais sens vu que FN coûte plus cher qu'un FP. On garde
           donc les hyperparamètres par défaut.

    Calibration : SIGMOID (Platt scaling, cv=5)
        -> retenu en Step 8 : le modèle brut (class_weight="balanced")
           est mal calibré (Brier=0.068, courbe très éloignée de la
           diagonale). SIGMOID a le meilleur Brier score (0.0564) sur
           validation, proche d'ISOTONIC (0.0566) mais plus simple.

    Seuil de décision : 0.48
        -> Step 8 avait d'abord proposé 0.18, sous l'hypothèse
           SIMPLIFIEE d'un ratio de coût fixe FN = 5x FP (compté par
           CAS, indépendamment du montant du prêt).

           Step 8b a remplacé cette hypothèse par un coût MONETAIRE
           réel, pondéré par loan_amnt et loan_int_rate, avec LGD=40%
           (recouvrement sur défaut), coût du capital=3%, durée
           moyenne=3 ans. Résultat : le ratio FN/FP IMPLICITE réel du
           portefeuille est ~1.60x, pas 5x — le seuil optimal qui en
           découle est 0.48, quasiment identique au seuil neutre 0.50.

           Vérification sur TEST (jamais utilisé pour choisir le
           seuil) : le seuil 0.18 (ratio 5x) coûte 35.7% de PLUS
           ($764k) que le seuil monétaire 0.48. Le seuil 0.18 était
           donc une erreur économique, corrigée ici.

        !! Ces hypothèses (LGD=40%, coût du capital=3%, durée=3 ans)
        restent des estimations. Si les vraies valeurs de la banque
        diffèrent sensiblement, refaites tourner Step 8b avec les
        bonnes valeurs et mettez à jour PRODUCTION_THRESHOLD. !!

Deux modes d'utilisation :

    1. Entraînement + export de l'artefact de production :
           python model_ml_final_pipeline.py train

    2. Prédiction sur un nouveau CSV avec l'artefact déjà entraîné :
           python model_ml_final_pipeline.py predict --input nouveaux_prets.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
# 1. CONFIGURATION DE PRODUCTION
# ============================================================

DATASET_PATH = Path("data/processed/donnees_prets_features.csv")
ARTIFACTS_DIR = Path("data/production")
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
    "n_jobs": -1,
    "verbosity": -1,
}

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5

# >>> PRODUCTION THRESHOLD <<<
# Selected through Step 8d robust economic sensitivity analysis.
#
# 60 scenarios were evaluated:
#   - 5 LGD scenarios
#   - 4 cost-of-funds scenarios
#   - 3 loan-duration scenarios
#
# Threshold selection was performed exclusively on OOF TRAIN
# probabilities using minimum maximum normalized regret.
#
# Result:
#   robust threshold = 0.48
#   mean regret      = 0.92%
#   max regret       = 5.08%
#   optimal in       = 27/60 scenarios
#
# The TEST set was used only once for final evaluation.
PRODUCTION_THRESHOLD = 0.48

MODEL_VERSION = "1.0.0"


# ============================================================
# 2. PREPROCESSING + MODELE (reproduction exacte de Step 3-6-8)
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
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def build_calibrated_pipeline() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
    )


# ============================================================
# 3. CHARGEMENT
# ============================================================

def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Le dataset est vide.")
    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in MODEL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


# ============================================================
# 4. METRIQUES DE VALIDATION FINALE
# ============================================================

def evaluate(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 5. ENTRAINEMENT + EXPORT DE L'ARTEFACT DE PRODUCTION
# ============================================================

def train_and_export() -> None:

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINPREDICT — PRODUCTION PIPELINE: TRAIN + EXPORT")
    print("=" * 78)

    df = load_dataset()
    X, y = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  (utilisé une seule fois, pour rapport final) : {len(X_test)}")

    print(f"\nEntraînement du pipeline de production...")
    print(f"  Preprocessing : median + most_frequent + OneHot (BASELINE)")
    print(f"  Modèle        : LightGBM {LGB_PARAMS}")
    print(f"  Calibration   : {CALIBRATION_METHOD} (cv={CALIBRATION_CV})")
    print(f"  Seuil         : {PRODUCTION_THRESHOLD}")

    production_pipeline = build_calibrated_pipeline()
    production_pipeline.fit(X_train, y_train)

    print("\n✅ Entraînement terminé.")

    # --------------------------------------------------------
    # Evaluation finale (une seule fois) sur TEST
    # --------------------------------------------------------

    y_test_proba = production_pipeline.predict_proba(X_test)[:, 1]
    metrics_at_050 = evaluate(y_test, y_test_proba, 0.50)
    metrics_at_production = evaluate(y_test, y_test_proba, PRODUCTION_THRESHOLD)

    print("\n" + "=" * 78)
    print("EVALUATION FINALE SUR TEST")
    print("=" * 78)
    print(f"{'Metric':<15}{'Seuil 0.50':<15}{'Seuil ' + str(PRODUCTION_THRESHOLD) + ' (prod)':<20}")
    for key in ["roc_auc", "pr_auc", "brier_score", "precision", "recall", "f1"]:
        print(f"{key:<15}{metrics_at_050[key]:<15.4f}{metrics_at_production[key]:<20.4f}")

    print("\nClassification Report — TEST @ seuil de production :\n")
    y_test_pred = (y_test_proba >= PRODUCTION_THRESHOLD).astype(int)
    print(
        classification_report(
            y_test, y_test_pred, target_names=["Classe 0", "Classe 1"], zero_division=0
        )
    )

    # --------------------------------------------------------
    # Export de l'artefact de production
    # --------------------------------------------------------

    artifact = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": production_pipeline,
        "production_threshold": PRODUCTION_THRESHOLD,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "lgb_params": LGB_PARAMS,
        "calibration_method": CALIBRATION_METHOD,
        "test_metrics_at_production_threshold": metrics_at_production,
        "test_metrics_at_050": metrics_at_050,
    }

    artifact_path = ARTIFACTS_DIR / "finpredict_production_model.joblib"
    joblib.dump(artifact, artifact_path)

    metadata_path = ARTIFACTS_DIR / "finpredict_production_metadata.json"
    metadata = {k: v for k, v in artifact.items() if k != "pipeline"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n✅ Artefact de production sauvegardé : {artifact_path}")
    print(f"✅ Métadonnées sauvegardées            : {metadata_path}")

    print("\n" + "=" * 78)
    print("PRODUCTION PIPELINE PRÊT")
    print("=" * 78)


# ============================================================
# 6. CHARGEMENT DE L'ARTEFACT + PREDICTION SUR NOUVELLES DONNEES
# ============================================================

def load_production_artifact(
    artifact_path: Path = ARTIFACTS_DIR / "finpredict_production_model.joblib",
) -> dict:
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artefact de production introuvable : {artifact_path}\n"
            "Lancez d'abord : python model_ml_final_pipeline.py train"
        )
    return joblib.load(artifact_path)


def predict(df_new: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Prédit sur de nouvelles données. Retourne un DataFrame avec
    proba_defaut et decision (0=accepter, 1=refuser) au seuil de production."""

    if artifact is None:
        artifact = load_production_artifact()

    features = artifact["model_features"]
    missing = [c for c in features if c not in df_new.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans les nouvelles données : {missing}")

    X_new = df_new[features].copy()

    pipeline = artifact["pipeline"]
    threshold = artifact["production_threshold"]

    proba = pipeline.predict_proba(X_new)[:, 1]
    decision = (proba >= threshold).astype(int)

    result = df_new.copy()
    result["proba_defaut"] = proba
    result["decision"] = decision  # 0 = accepter le prêt, 1 = refuser (risque trop élevé)
    result["decision_label"] = np.where(decision == 1, "REFUSE (risque élevé)", "ACCEPTE")

    return result


def run_predict_cli(input_path: Path, output_path: Path) -> None:
    print("=" * 78)
    print("FINPREDICT — PRODUCTION PIPELINE: PREDICT")
    print("=" * 78)

    artifact = load_production_artifact()

    print(f"\nModèle chargé : version {artifact['model_version']}, entraîné le {artifact['trained_at_utc']}")
    print(f"Seuil de production : {artifact['production_threshold']}")

    df_new = pd.read_csv(input_path)
    print(f"\nDonnées à prédire : {len(df_new)} lignes")

    result = predict(df_new, artifact=artifact)

    n_refused = int((result["decision"] == 1).sum())
    n_accepted = int((result["decision"] == 0).sum())

    print(f"\nRésultats :")
    print(f"  Acceptés : {n_accepted} ({100 * n_accepted / len(result):.1f}%)")
    print(f"  Refusés  : {n_refused} ({100 * n_refused / len(result):.1f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\n✅ Prédictions sauvegardées : {output_path}")


# ============================================================
# 7. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FinPredict — Production Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Entraîne et exporte l'artefact de production")

    predict_parser = subparsers.add_parser("predict", help="Prédit sur un nouveau CSV")
    predict_parser.add_argument("--input", type=str, required=True, help="CSV en entrée")
    predict_parser.add_argument("--output", type=str, required=True, help="CSV de sortie (avec prédictions)")

    args = parser.parse_args()

    if args.command == "train":
        train_and_export()
    elif args.command == "predict":
        run_predict_cli(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()