"""
FinPredict AI Pro - Model Diagnostics
=====================================
Baseline diagnostics: train/validation/test, threshold tuning on validation,
ROC/PR curves, calibration, Brier, ECE, feature importance and optional SHAP.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = Path("data/processed/donnees_prets_features.csv")
REPORT_DIR = Path("data/reports/ml/diagnostics")
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


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise ValueError("Le dataset est vide.")
    return df


def build_xy(df: pd.DataFrame):
    return df[MODEL_FEATURES].copy(), df[TARGET].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


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


def create_splits(X, y):
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(mask.sum())
        if count == 0:
            continue

        confidence = float(probabilities[mask].mean())
        actual_rate = float(y_true[mask].mean())
        ece += (count / len(probabilities)) * abs(confidence - actual_rate)

    return float(ece)


def metrics(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(np.mean((probabilities - np.asarray(y_true)) ** 2)),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def optimize_threshold(y_val, p_val):
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        pred = (p_val >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
        })

    df = pd.DataFrame(rows)
    best_idx = df["f1"].idxmax()
    return float(df.loc[best_idx, "threshold"]), df


def save_roc(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    auc = roc_auc_score(y_true, p)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("FinPredict — ROC Curve (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curve_test.png", dpi=150)
    plt.close(fig)


def save_pr(y_true, p):
    precision, recall, _ = precision_recall_curve(y_true, p)
    ap = average_precision_score(y_true, p)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("FinPredict — Precision-Recall Curve (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "precision_recall_test.png", dpi=150)
    plt.close(fig)


def save_confusion(y_true, p, threshold):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Classe 0", "Classe 1"],
    ).plot(ax=ax)
    ax.set_title(f"Confusion Matrix — threshold={threshold:.2f}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    return cm


def save_calibration(y_true, p):
    frac_pos, mean_pred = calibration_curve(
        y_true, p, n_bins=10, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Calibration parfaite")
    ax.plot(mean_pred, frac_pos, marker="o", label="FinPredict")
    ax.set_xlabel("Probabilité moyenne prédite")
    ax.set_ylabel("Fréquence réellement positive")
    ax.set_title("FinPredict — Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "calibration_curve_test.png", dpi=150)
    plt.close(fig)


def save_threshold_plot(df, best_threshold):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["threshold"], df["precision"], label="Precision")
    ax.plot(df["threshold"], df["recall"], label="Recall")
    ax.plot(df["threshold"], df["f1"], label="F1")
    ax.axvline(best_threshold, linestyle="--", label=f"Seuil = {best_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("FinPredict — Threshold Analysis (Validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "threshold_analysis_validation.png", dpi=150)
    plt.close(fig)


def save_split_plot(metric_dict):
    labels = ["Train", "Validation", "Test"]
    aucs = [metric_dict[k]["roc_auc"] for k in ["train", "validation", "test"]]
    f1s = [metric_dict[k]["f1"] for k in ["train", "validation", "test"]]
    x = np.arange(3)
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - w / 2, aucs, w, label="ROC-AUC")
    ax.bar(x + w / 2, f1s, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("FinPredict — Train / Validation / Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "train_validation_test.png", dpi=150)
    plt.close(fig)


def save_feature_importance(preprocessor, model):
    names = preprocessor.get_feature_names_out()
    values = model.feature_importances_

    df = pd.DataFrame({
        "feature": names,
        "importance": values,
    }).sort_values("importance", ascending=False)

    df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    top = df.head(15).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance LightGBM")
    ax.set_title("FinPredict — Top Feature Importance")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    return df


def save_shap(preprocessor, model, X_test):
    try:
        import shap
    except ImportError:
        print("⚠️ SHAP non installé : graphique SHAP ignoré.")
        return False

    try:
        Xt = preprocessor.transform(X_test)
        names = preprocessor.get_feature_names_out()
        sample = pd.DataFrame(Xt, columns=names, index=X_test.index)

        sample = sample.sample(
            n=min(2000, len(sample)),
            random_state=42,
        )

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)

        if isinstance(values, list):
            values = values[1]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            values,
            sample,
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / "shap_summary.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("✅ SHAP summary généré.")
        return True

    except Exception as exc:
        print(f"⚠️ SHAP non généré : {exc}")
        return False


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FINPREDICT — MODEL DIAGNOSTICS")
    print("=" * 72)

    df = load_dataset()
    X, y = build_xy(df)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    print(f"Train      : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Test       : {len(X_test)}")

    preprocessor = build_preprocessor()

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_p = pd.DataFrame(X_train_p, columns=feature_names, index=X_train.index)
    X_val_p = pd.DataFrame(X_val_p, columns=feature_names, index=X_val.index)
    X_test_p = pd.DataFrame(X_test_p, columns=feature_names, index=X_test.index)

    print(f"Features après encodage : {X_train_p.shape[1]}")
    print(f"NaN train      : {int(X_train_p.isna().sum().sum())}")
    print(f"NaN validation : {int(X_val_p.isna().sum().sum())}")
    print(f"NaN test       : {int(X_test_p.isna().sum().sum())}")

    model = build_model()
    model.fit(X_train_p, y_train)

    p_train = model.predict_proba(X_train_p)[:, 1]
    p_val = model.predict_proba(X_val_p)[:, 1]
    p_test = model.predict_proba(X_test_p)[:, 1]

    best_threshold, threshold_df = optimize_threshold(y_val, p_val)

    print(f"Seuil choisi sur VALIDATION : {best_threshold:.2f}")

    all_metrics = {
        "train": metrics(y_train, p_train, best_threshold),
        "validation": metrics(y_val, p_val, best_threshold),
        "test": metrics(y_test, p_test, best_threshold),
    }

    print("" + "=" * 72)
    print("PERFORMANCES")
    print("=" * 72)

    for name in ["train", "validation", "test"]:
        m = all_metrics[name]
        print(f"{name.upper()}")
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"  PR-AUC    : {m['pr_auc']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1        : {m['f1']:.4f}")
        print(f"  Brier     : {m['brier_score']:.4f}")
        print(f"  ECE       : {m['ece']:.4f}")

    cm = save_confusion(y_test, p_test, best_threshold)
    save_roc(y_test, p_test)
    save_pr(y_test, p_test)
    save_calibration(y_test, p_test)
    save_threshold_plot(threshold_df, best_threshold)
    save_split_plot(all_metrics)

    importance = save_feature_importance(preprocessor, model)
    shap_ok = save_shap(preprocessor, model, X_test)

    report = {
        "dataset": {
            "rows": len(df),
            "features_before_encoding": len(MODEL_FEATURES),
            "features_after_encoding": X_train_p.shape[1],
        },
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "threshold": {
            "selected_on": "validation",
            "criterion": "maximum_f1",
            "value": best_threshold,
        },
        "metrics": all_metrics,
        "confusion_matrix_test": cm.tolist(),
        "shap_generated": shap_ok,
        "top_features": importance.head(15).to_dict(orient="records"),
    }

    (REPORT_DIR / "model_diagnostics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("✅ Diagnostics terminés.")
    print(f"Rapport : {REPORT_DIR / 'model_diagnostics.json'}")
    print(f"Graphiques : {REPORT_DIR}")
    print("Le seuil a été choisi sur VALIDATION.")
    print("Le TEST reste réservé à l'évaluation finale.")


if __name__ == "__main__":
    main()
