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

DATASET_PATH = Path("data/processed/donnees_prets_features.csv")
REPORTS_DIR = Path("data/reports/ml")
TARGET = "loan_status"

NUMERIC_FEATURES = [
    "person_age", "person_income", "loan_amnt", "loan_int_rate",
    "person_emp_length", "loan_percent_income", "cb_person_cred_hist_length",
    "interest_income_ratio",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LGB_PARAMS = {
    "objective": "binary", "n_estimators": 300, "learning_rate": 0.05,
    "num_leaves": 31, "class_weight": "balanced", "random_state": 42,
    "n_jobs": -1, "verbosity": -1,
}

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Publicly-informed LGD scenarios used for sensitivity analysis.
# Treat these as scenario values, not universal bank parameters.
LGD_SCENARIOS = [
    ("LGD_30pct", 0.30),
    ("LGD_36pct", 0.36),
    ("LGD_40pct_FinPredict", 0.40),
    ("LGD_42_53pct", 0.4253),
    ("LGD_54_8pct", 0.548),
]

COST_OF_FUNDS_SCENARIOS = [0.025, 0.030, 0.035, 0.040]
DURATION_SCENARIOS = [2.0, 3.0, 4.0]

# Pre-declared central scenario. The TEST is not used to choose this.
CENTRAL_LGD = 0.40
CENTRAL_COST_OF_FUNDS = 0.03
CENTRAL_DURATION = 3.0


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", num, NUMERIC_FEATURES),
        ("categorical", cat, CATEGORICAL_FEATURES),
    ])

    # Conserve un DataFrame pandas après transformation,
    # avec les noms des features.
    preprocessor.set_output(transform="pandas")

    return preprocessor


def build_pipeline() -> CalibratedClassifierCV:
    base = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", lgb.LGBMClassifier(**LGB_PARAMS)),
    ])
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)


def row_costs(X, loan_amnt_median, rate_median, lgd, cof, duration):
    amount = X["loan_amnt"].fillna(loan_amnt_median).to_numpy(float)
    rate = X["loan_int_rate"].fillna(rate_median).to_numpy(float)
    cost_fn = lgd * amount
    margin_rate = np.maximum(rate / 100.0 - cof, 0.0)
    cost_fp = margin_rate * amount * duration
    return cost_fn, cost_fp


def cost_at_threshold(y, p, cost_fn, cost_fp, threshold):
    pred = (p >= threshold).astype(int)
    is_fn = (pred == 0) & (y == 1)
    is_fp = (pred == 1) & (y == 0)
    return {
        "threshold": float(threshold),
        "n_fn": int(is_fn.sum()),
        "n_fp": int(is_fp.sum()),
        "fn_cost": float(cost_fn[is_fn].sum()),
        "fp_cost": float(cost_fp[is_fp].sum()),
        "total_cost": float(cost_fn[is_fn].sum() + cost_fp[is_fp].sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def optimal_threshold(y, p, cost_fn, cost_fp):
    results = [cost_at_threshold(y, p, cost_fn, cost_fp, t) for t in THRESHOLDS]
    return min(results, key=lambda r: r["total_cost"])


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINPREDICT — STEP 8c: ECONOMIC SENSITIVITY ANALYSIS")
    print("=" * 78)

    df = pd.read_csv(DATASET_PATH)
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  : {len(X_test)}")
    print("TEST is a holdout: thresholds are selected only on OOF TRAIN.")

    # One calibrated OOF probability set is sufficient for all cost scenarios.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nCalcul des probabilités OOF calibrées (sigmoid, 5 folds)...")
    oof_p = cross_val_predict(
        build_pipeline(), X_train, y_train, cv=cv,
        method="predict_proba", n_jobs=None
    )[:, 1]

    # Final calibrated model for the single central TEST evaluation.
    final_model = build_pipeline()
    final_model.fit(X_train, y_train)
    test_p = final_model.predict_proba(X_test)[:, 1]

    amount_median = X_train["loan_amnt"].median()
    rate_median = X_train["loan_int_rate"].median()
    ytr = y_train.to_numpy()

    # ------------------------------------------------------------------
    # A) LGD benchmark sensitivity, fixed CoF=3%, duration=3 years.
    # ------------------------------------------------------------------
    lgd_rows = []
    for name, lgd in LGD_SCENARIOS:
        cfn, cfp = row_costs(
            X_train, amount_median, rate_median,
            lgd, 0.03, 3.0
        )
        best = optimal_threshold(ytr, oof_p, cfn, cfp)
        lgd_rows.append({
            "scenario": name,
            "lgd": lgd,
            "cost_of_funds": 0.03,
            "duration_years": 3.0,
            **best,
        })

    lgd_df = pd.DataFrame(lgd_rows)
    print("\n" + "=" * 78)
    print("A) SENSIBILITE LGD")
    print("=" * 78)
    print(lgd_df[[
        "scenario", "lgd", "threshold", "total_cost", "n_fn", "n_fp",
        "precision", "recall", "f1"
    ]].to_string(index=False))

    # ------------------------------------------------------------------
    # B) CoF x duration grid, fixed LGD=40%.
    # ------------------------------------------------------------------
    grid_rows = []
    for cof in COST_OF_FUNDS_SCENARIOS:
        for duration in DURATION_SCENARIOS:
            cfn, cfp = row_costs(
                X_train, amount_median, rate_median,
                CENTRAL_LGD, cof, duration
            )
            best = optimal_threshold(ytr, oof_p, cfn, cfp)
            grid_rows.append({
                "lgd": CENTRAL_LGD,
                "cost_of_funds": cof,
                "duration_years": duration,
                **best,
            })

    grid_df = pd.DataFrame(grid_rows)
    print("\n" + "=" * 78)
    print("B) SENSIBILITE COST OF FUNDS x DUREE")
    print("=" * 78)
    print(grid_df[[
        "cost_of_funds", "duration_years", "threshold",
        "total_cost", "n_fn", "n_fp"
    ]].sort_values(["duration_years", "cost_of_funds"]).to_string(index=False))

    # ------------------------------------------------------------------
    # C) Central scenario: select threshold on OOF TRAIN, evaluate once on TEST.
    # ------------------------------------------------------------------
    cfn_tr, cfp_tr = row_costs(
        X_train, amount_median, rate_median,
        CENTRAL_LGD, CENTRAL_COST_OF_FUNDS, CENTRAL_DURATION
    )
    central_oof = optimal_threshold(ytr, oof_p, cfn_tr, cfp_tr)

    cfn_te, cfp_te = row_costs(
        X_test, amount_median, rate_median,
        CENTRAL_LGD, CENTRAL_COST_OF_FUNDS, CENTRAL_DURATION
    )
    central_test = cost_at_threshold(
        y_test.to_numpy(), test_p, cfn_te, cfp_te,
        central_oof["threshold"]
    )

    print("\n" + "=" * 78)
    print("C) SCENARIO CENTRAL")
    print("=" * 78)
    print(f"LGD               : {CENTRAL_LGD:.2%}")
    print(f"Cost of funds     : {CENTRAL_COST_OF_FUNDS:.2%}")
    print(f"Duration          : {CENTRAL_DURATION:.1f} years")
    print(f"Optimal threshold : {central_oof['threshold']:.2f} (selected on OOF TRAIN)")
    print("\nTEST — single final evaluation:")
    for k in ["total_cost", "fn_cost", "fp_cost", "n_fn", "n_fp", "precision", "recall", "f1"]:
        print(f"  {k:<12}: {central_test[k]:,.4f}" if isinstance(central_test[k], float) else f"  {k:<12}: {central_test[k]}")

    # ------------------------------------------------------------------
    # D) Stability summary.
    # ------------------------------------------------------------------
    thresholds = pd.concat([
        lgd_df["threshold"], grid_df["threshold"]
    ], ignore_index=True)
    print("\n" + "=" * 78)
    print("D) STABILITE DU SEUIL")
    print("=" * 78)
    print(f"Min threshold : {thresholds.min():.2f}")
    print(f"Max threshold : {thresholds.max():.2f}")
    print(f"Range         : {thresholds.max() - thresholds.min():.2f}")
    print(f"Std           : {thresholds.std(ddof=0):.4f}")

    # ------------------------------------------------------------------
    # Save data.
    # ------------------------------------------------------------------
    lgd_csv = REPORTS_DIR / "step8c_lgd_sensitivity.csv"
    grid_csv = REPORTS_DIR / "step8c_funding_duration_sensitivity.csv"
    lgd_df.to_csv(lgd_csv, index=False)
    grid_df.to_csv(grid_csv, index=False)

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(lgd_df["lgd"] * 100, lgd_df["threshold"], marker="o")
    ax.set_xlabel("LGD (%)")
    ax.set_ylabel("Optimal threshold (OOF TRAIN)")
    ax.set_title("FinPredict — Sensibilité du seuil à la LGD")
    fig.tight_layout()
    lgd_png = REPORTS_DIR / "step8c_threshold_sensitivity_lgd.png"
    fig.savefig(lgd_png, dpi=150)
    plt.close(fig)

    pivot = grid_df.pivot(
        index="duration_years",
        columns="cost_of_funds",
        values="threshold"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, label="Optimal threshold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1%}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0f}y" for v in pivot.index])
    ax.set_xlabel("Cost of funds")
    ax.set_ylabel("Duration")
    ax.set_title("FinPredict — Sensibilité du seuil au financement et à la durée")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    grid_png = REPORTS_DIR / "step8c_threshold_sensitivity_funding_duration.png"
    fig.savefig(grid_png, dpi=150)
    plt.close(fig)

    report = {
        "methodology": {
            "probabilities": "sigmoid calibrated OOF probabilities, 5-fold",
            "threshold_selection": "OOF TRAIN only",
            "test_usage": "single final evaluation of pre-declared central scenario",
        },
        "lgd_scenarios": [
            {"name": name, "lgd": value} for name, value in LGD_SCENARIOS
        ],
        "cost_of_funds_scenarios": COST_OF_FUNDS_SCENARIOS,
        "duration_scenarios_years": DURATION_SCENARIOS,
        "central_scenario": {
            "lgd": CENTRAL_LGD,
            "cost_of_funds": CENTRAL_COST_OF_FUNDS,
            "duration_years": CENTRAL_DURATION,
        },
        "central_oof": central_oof,
        "central_test": central_test,
        "threshold_stability": {
            "min": float(thresholds.min()),
            "max": float(thresholds.max()),
            "range": float(thresholds.max() - thresholds.min()),
            "std": float(thresholds.std(ddof=0)),
        },
        "artifacts": {
            "lgd_csv": str(lgd_csv),
            "funding_duration_csv": str(grid_csv),
            "lgd_plot": str(lgd_png),
            "funding_duration_plot": str(grid_png),
        },
    }

    report_path = REPORTS_DIR / "step8c_economic_sensitivity_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("ARTEFACTS SAUVEGARDES")
    print("=" * 78)
    print(f"LGD CSV          : {lgd_csv}")
    print(f"Funding/Duration : {grid_csv}")
    print(f"LGD graph        : {lgd_png}")
    print(f"Funding graph    : {grid_png}")
    print(f"JSON report      : {report_path}")
    print("\nSTEP 8c TERMINEE")
    print("=" * 78)


if __name__ == "__main__":
    main()
