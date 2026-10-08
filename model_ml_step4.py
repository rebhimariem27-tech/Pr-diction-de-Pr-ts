import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FINPREDICT — STEP 4: LIGHTGBM TRAINING WITH SMART IMPUTATION")

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/donnees_prets_features.csv",
        help="Input feature dataset"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 78)
    print("FINPREDICT — STEP 4: LIGHTGBM TRAINING WITH SMART IMPUTATION")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {input_path}"
        )

    df = pd.read_csv(input_path)

    target = "loan_status"

    if target not in df.columns:
        raise ValueError(
            f"La colonne cible '{target}' est absente du dataset."
        )

    print(f"\nDataset : {input_path}")
    print(f"Shape   : {df.shape}")

    # ------------------------------------------------------------------
    # 2. Features
    # ------------------------------------------------------------------

    features = [
        "person_age",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "person_emp_length",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "interest_income_ratio",
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    missing_features = [
        col for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features manquantes : {missing_features}"
        )

    X = df[features].copy()
    y = df[target].astype(int).copy()

    # ------------------------------------------------------------------
    # 3. SAME train/test split as Step 1
    # ------------------------------------------------------------------

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nSplit :")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    # ------------------------------------------------------------------
    # 4. Load smart preprocessor
    # ------------------------------------------------------------------

    preprocessor_path = Path(
        "data/reports/smart_preprocessor.joblib"
    )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor introuvable : {preprocessor_path}"
        )

    preprocessor_bundle = joblib.load(preprocessor_path)

    print("\nSmart preprocessor chargé.")

# Le fichier contient un dictionnaire avec le vrai preprocessor
    if isinstance(preprocessor_bundle, dict):
        if "preprocessor" not in preprocessor_bundle:
            raise KeyError(
            "Le fichier smart_preprocessor.joblib est un dictionnaire "
            "mais la clé 'preprocessor' est absente."
        )

        preprocessor = preprocessor_bundle["preprocessor"]

        print(
        f"Type du bundle       : {type(preprocessor_bundle).__name__}"
    )
        print(
        f"KNN k sélectionné    : "
        f"{preprocessor_bundle.get('selected_k', 'N/A')}"
    )
    else:
    # Compatibilité avec un ancien format
        preprocessor = preprocessor_bundle

    print(
    f"Type du preprocessor : {type(preprocessor).__name__}"
)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    # Convert to numpy if necessary
    X_train_processed = np.asarray(X_train_processed)
    X_test_processed = np.asarray(X_test_processed)

    print("\nAprès preprocessing :")
    print(f"X_train processed : {X_train_processed.shape}")
    print(f"X_test processed  : {X_test_processed.shape}")

    print(f"NaN train : {np.isnan(X_train_processed).sum()}")
    print(f"NaN test  : {np.isnan(X_test_processed).sum()}")

    # ------------------------------------------------------------------
    # 6. Train LightGBM
    # ------------------------------------------------------------------

    print("\nTraining LightGBM...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train_processed,
        y_train
    )

    print("LightGBM entraîné.")

    # ------------------------------------------------------------------
    # 7. Predictions
    # ------------------------------------------------------------------

    y_proba = model.predict_proba(X_test_processed)[:, 1]

    # Standard threshold for initial evaluation
    threshold = 0.50

    y_pred = (y_proba >= threshold).astype(int)

    # ------------------------------------------------------------------
    # 8. Metrics
    # ------------------------------------------------------------------

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 78)
    print("TEST METRICS")
    print("=" * 78)

    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    # ------------------------------------------------------------------
    # 9. Save model
    # ------------------------------------------------------------------

    model_path = Path(
        "data/reports/lightgbm_smart_imputation.joblib"
    )

    joblib.dump(model, model_path)

    # ------------------------------------------------------------------
    # 10. Save metrics
    # ------------------------------------------------------------------

    metrics = {
        "pipeline": "KNN imputation + missing indicators + LightGBM",
        "dataset": str(input_path),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "processed_features": int(X_train_processed.shape[1]),
        "threshold": threshold,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = Path(
        "data/reports/step4_smart_model_metrics.json"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    print("\nArtefacts sauvegardés :")
    print(f"  Model   : {model_path}")
    print(f"  Metrics : {metrics_path}")

    print("\n" + "=" * 78)
    print("STEP 4 TERMINÉE")
    print("=" * 78)


if __name__ == "__main__":
    main()