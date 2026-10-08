from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

K_CANDIDATES = [3, 5, 7, 10, 15]
RANDOM_STATE = 42


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    df = pd.read_csv(path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Colonnes manquantes: " + ", ".join(missing))
    return df


def _normalized_mae(y_true: np.ndarray, y_pred: np.ndarray, iqr: float) -> float:
    scale = iqr if np.isfinite(iqr) and iqr > 1e-12 else float(np.std(y_true))
    scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    return float(mean_absolute_error(y_true, y_pred) / scale)


def evaluate_knn_k(
    train_X_df: pd.DataFrame,
    train_y: pd.Series,
    k: int,
    validation_fraction: float = 0.20,
    mask_fraction: float = 0.10,
) -> dict:
    """Choose K without touching the final test set.

    A validation subset is created from TRAIN only. Additional values are
    artificially masked in that subset, then reconstructed with KNN.
    Error is measured only on the artificially masked cells.
    """
    inner_indices_train, inner_indices_val = train_test_split(
        np.arange(len(train_X_df)),
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=train_y.to_numpy(),
    )

    inner_train = train_X_df.iloc[inner_indices_train]
    inner_val = train_X_df.iloc[inner_indices_val]

    X_train = inner_train[NUMERIC_FEATURES].copy()
    X_val = inner_val[NUMERIC_FEATURES].copy()

    # Scale before KNN so income does not dominate smaller-scale variables.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    rng = np.random.default_rng(RANDOM_STATE + k)
    X_val_masked = X_val_scaled.copy()
    masked_positions: dict[str, np.ndarray] = {}

    for j, col in enumerate(NUMERIC_FEATURES):
        observed = np.flatnonzero(X_val[col].notna().to_numpy())
        if len(observed) == 0:
            masked_positions[col] = np.array([], dtype=int)
            continue
        n_mask = max(1, int(len(observed) * mask_fraction))
        chosen = rng.choice(observed, size=min(n_mask, len(observed)), replace=False)
        X_val_masked[chosen, j] = np.nan
        masked_positions[col] = chosen

    imputer = KNNImputer(n_neighbors=k, weights="distance", add_indicator=False)
    imputer.fit(X_train_scaled)
    X_val_imputed_scaled = imputer.transform(X_val_masked)
    X_val_imputed = scaler.inverse_transform(X_val_imputed_scaled)
    X_val_original = X_val[NUMERIC_FEATURES].to_numpy(dtype=float)

    per_feature: dict[str, dict[str, float | int]] = {}
    normalized_maes: list[float] = []

    for j, col in enumerate(NUMERIC_FEATURES):
        rows = masked_positions[col]
        if len(rows) == 0:
            continue
        y_true = X_val_original[rows, j]
        y_pred = X_val_imputed[rows, j]
        iqr = float(np.nanpercentile(X_train[col], 75) - np.nanpercentile(X_train[col], 25))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        nmae = _normalized_mae(y_true, y_pred, iqr)
        normalized_maes.append(nmae)
        per_feature[col] = {
            "masked_values": int(len(rows)),
            "MAE": round(mae, 6),
            "RMSE": round(rmse, 6),
            "normalized_MAE": round(nmae, 6),
        }

    return {
        "k": k,
        "mean_normalized_MAE": round(float(np.mean(normalized_maes)), 6),
        "per_feature": per_feature,
    }


def select_best_k(train_X: pd.DataFrame, train_y: pd.Series, max_rows: int = 6000) -> tuple[int, list[dict]]:
    # KNN imputation is O(n^2) in the reference set. For hyperparameter selection,
    # use a stratified subset of TRAIN only; the final imputer is still fitted on all TRAIN rows.
    if len(train_X) > max_rows:
        idx_train, _ = train_test_split(
            np.arange(len(train_X)),
            train_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=train_y.to_numpy(),
        )
        selection_X = train_X.iloc[idx_train].copy()
        selection_y = train_y.iloc[idx_train].copy()
    else:
        selection_X = train_X.copy()
        selection_y = train_y.copy()
    results = [evaluate_knn_k(selection_X, selection_y, k) for k in K_CANDIDATES]
    results.sort(key=lambda x: x["mean_normalized_MAE"])
    return int(results[0]["k"]), results


def smart_categorical_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Confidence-aware categorical imputation.

    For categorical columns with missing values, a small CatBoost classifier
    is trained using only the training set. If confidence is low, the value is
    kept as __MISSING__ rather than forcing a weak guess.

    Current FinPredict data has no missing categorical values, so this layer
    remains inactive today while keeping the pipeline robust to future data.
    """
    train = X_train.copy()
    test = X_test.copy()
    report: dict[str, dict] = {}

    for target_col in CATEGORICAL_FEATURES:
        train_missing = train[target_col].isna()
        test_missing = test[target_col].isna()
        report[target_col] = {
            "train_missing": int(train_missing.sum()),
            "test_missing": int(test_missing.sum()),
            "method": "catboost_confidence_aware" if (train_missing.any() or test_missing.any()) else "not_needed",
        }

        if not train_missing.any() and not test_missing.any():
            continue

        predictor_cols = NUMERIC_FEATURES + [c for c in CATEGORICAL_FEATURES if c != target_col]
        cat_predictor_positions = [i for i, c in enumerate(predictor_cols) if c in CATEGORICAL_FEATURES]

        fit_rows = ~train[target_col].isna()
        X_fit = train.loc[fit_rows, predictor_cols].copy()
        y_fit = train.loc[fit_rows, target_col].astype(str)

        for col in CATEGORICAL_FEATURES:
            if col != target_col:
                X_fit[col] = X_fit[col].astype("string").fillna("__MISSING__").astype(str)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=200,
            depth=5,
            learning_rate=0.05,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_fit, y_fit, cat_features=cat_predictor_positions)

        classes = np.asarray(model.classes_)

        def predict_missing(frame: pd.DataFrame, missing_mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
            if not missing_mask.any():
                return np.array([], dtype=object), np.array([], dtype=float)
            X_pred = frame.loc[missing_mask, predictor_cols].copy()
            for col in CATEGORICAL_FEATURES:
                if col != target_col:
                    X_pred[col] = X_pred[col].astype("string").fillna("__MISSING__").astype(str)
            probs = model.predict_proba(X_pred)
            idx = probs.argmax(axis=1)
            return classes[idx], probs.max(axis=1)

        train_preds, train_conf = predict_missing(train, train_missing)
        test_preds, test_conf = predict_missing(test, test_missing)

        # Conservative rule: only accept model-based guesses at >= 0.60 confidence.
        train_assign = np.where(train_conf >= 0.60, train_preds, "__MISSING__")
        test_assign = np.where(test_conf >= 0.60, test_preds, "__MISSING__")

        train.loc[train_missing, target_col] = train_assign
        test.loc[test_missing, target_col] = test_assign

        report[target_col].update({
            "confidence_threshold": 0.60,
            "train_imputed_high_confidence": int(np.sum(train_conf >= 0.60)),
            "train_kept_missing_low_confidence": int(np.sum(train_conf < 0.60)),
            "test_imputed_high_confidence": int(np.sum(test_conf >= 0.60)),
            "test_kept_missing_low_confidence": int(np.sum(test_conf < 0.60)),
        })

    for col in CATEGORICAL_FEATURES:
        train[col] = train[col].astype("string")
        test[col] = test[col].astype("string")

    return train, test, report


def build_preprocessor(best_k: int) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("scale_for_knn", StandardScaler()),
            (
                "knn_imputer",
                KNNImputer(
                    n_neighbors=best_k,
                    weights="distance",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "missing_category",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    add_indicator=True,
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FinPredict — preprocessing V2: KNN + smart categorical handling")
    parser.add_argument("--input", required=True)
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    df = load_features(Path(args.input))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 78)
    print("FINPREDICT — STEP 2: SMART IMPUTATION")
    print("=" * 78)
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    best_k, k_results = select_best_k(X_train, y_train)
    print("\nKNN selection (TRAIN only):")
    for result in sorted(k_results, key=lambda x: x["k"]):
        marker = " <-- selected" if result["k"] == best_k else ""
        print(f"  k={result['k']:>2} | normalized MAE={result['mean_normalized_MAE']:.6f}{marker}")

    X_train_cat, X_test_cat, categorical_report = smart_categorical_imputation(X_train, X_test)

    preprocessor = build_preprocessor(best_k)
    X_train_processed = preprocessor.fit_transform(X_train_cat)
    X_test_processed = preprocessor.transform(X_test_cat)

    feature_names = preprocessor.get_feature_names_out().tolist()

    print("\nProcessed shapes:")
    print(f"X_train processed : {X_train_processed.shape}")
    print(f"X_test processed  : {X_test_processed.shape}")
    print(f"NaN train         : {int(np.isnan(X_train_processed).sum())}")
    print(f"NaN test          : {int(np.isnan(X_test_processed).sum())}")
    print(f"Selected k        : {best_k}")
    print("\nCategorical imputation:")
    for col, info in categorical_report.items():
        print(f"  {col}: {info}")

    report = {
        "method": {
            "numeric": "StandardScaler -> KNNImputer(weights=distance, add_indicator=True)",
            "categorical": "CatBoost confidence-aware when missing; otherwise explicit __MISSING__ category + indicator",
            "leakage_control": "All preprocessing is fitted on TRAIN only; K selection uses an internal validation split of TRAIN.",
        },
        "selected_k": best_k,
        "k_candidates": k_results,
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
        },
        "categorical_imputation": categorical_report,
        "processed_shape": {
            "train": list(X_train_processed.shape),
            "test": list(X_test_processed.shape),
        },
        "processed_nan_count": {
            "train": int(np.isnan(X_train_processed).sum()),
            "test": int(np.isnan(X_test_processed).sum()),
        },
        "feature_names": feature_names,
    }

    out = Path("data/reports")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "smart_imputation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save the fitted preprocessor + feature schema for downstream steps.
    import joblib

    artifact = {
        "preprocessor": preprocessor,
        "best_k": best_k,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features_out": feature_names,
    }
    joblib.dump(artifact, out / "smart_preprocessor.joblib")

    print(f"\nRapport : {report_path}")
    print(f"Artefact : {out / 'smart_preprocessor.joblib'}")
    print("Étape 2 terminée.")


if __name__ == "__main__":
    main()
