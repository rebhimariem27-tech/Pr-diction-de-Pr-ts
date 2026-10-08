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


# =============================================================================
# FINPREDICT AI PRO — STEP 8d
# ROBUST THRESHOLD SELECTION + STEP 8b/8c RECONCILIATION
# =============================================================================

DATASET_PATH = Path("data/processed/donnees_prets_features.csv")
REPORTS_DIR = Path("data/reports/ml")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "loan_status"

NUMERIC_FEATURES = [
    "person_age", "person_income", "loan_amnt", "loan_int_rate",
    "person_emp_length", "loan_percent_income",
    "cb_person_cred_hist_length", "interest_income_ratio",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Deterministic settings make the Step 8b/8c discrepancy easier to diagnose.
LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

CALIBRATION_CV = 5
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

LGD_SCENARIOS = [
    ("LGD_30pct", 0.30),
    ("LGD_36pct", 0.36),
    ("LGD_40pct_FinPredict", 0.40),
    ("LGD_42_53pct", 0.4253),
    ("LGD_54_8pct", 0.548),
]
COST_OF_FUNDS_SCENARIOS = [0.025, 0.030, 0.035, 0.040]
DURATION_SCENARIOS = [2.0, 3.0, 4.0]

CENTRAL_LGD = 0.40
CENTRAL_COF = 0.03
CENTRAL_DURATION = 3.0


# -----------------------------------------------------------------------------
# Model pipeline
# -----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", num, NUMERIC_FEATURES),
        ("categorical", cat, CATEGORICAL_FEATURES),
    ])
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_pipeline() -> CalibratedClassifierCV:
    base = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", lgb.LGBMClassifier(**LGB_PARAMS)),
    ])
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=CALIBRATION_CV,
    )


# -----------------------------------------------------------------------------
# Monetary cost
# -----------------------------------------------------------------------------
def row_costs(X, amount_median, rate_median, lgd, cof, duration):
    amount = X["loan_amnt"].fillna(amount_median).to_numpy(dtype=float)
    rate = X["loan_int_rate"].fillna(rate_median).to_numpy(dtype=float)

    cost_fn = lgd * amount
    margin_rate = np.maximum(rate / 100.0 - cof, 0.0)
    cost_fp = margin_rate * amount * duration
    return cost_fn, cost_fp


def cost_at_threshold(y, p, cost_fn, cost_fp, threshold):
    pred = (p >= threshold).astype(int)
    is_fn = (pred == 0) & (y == 1)
    is_fp = (pred == 1) & (y == 0)

    fn_cost = float(cost_fn[is_fn].sum())
    fp_cost = float(cost_fp[is_fp].sum())

    return {
        "threshold": float(threshold),
        "n_fn": int(is_fn.sum()),
        "n_fp": int(is_fp.sum()),
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "total_cost": fn_cost + fp_cost,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def scenario_curve(name, lgd, cof, duration, y, p, cfn, cfp):
    rows = []
    for t in THRESHOLDS:
        rows.append({
            "scenario": name,
            "lgd": lgd,
            "cost_of_funds": cof,
            "duration_years": duration,
            **cost_at_threshold(y, p, cfn, cfp, t),
        })

    out = pd.DataFrame(rows)
    best_cost = float(out["total_cost"].min())
    out["normalized_cost"] = out["total_cost"] / best_cost
    out["regret"] = out["normalized_cost"] - 1.0
    return out


def build_scenarios():
    scenarios = []
    for lgd_name, lgd in LGD_SCENARIOS:
        for cof in COST_OF_FUNDS_SCENARIOS:
            for duration in DURATION_SCENARIOS:
                scenarios.append({
                    "name": f"{lgd_name}_CoF{cof:.3f}_Dur{duration:.0f}y",
                    "lgd": lgd,
                    "cof": cof,
                    "duration": duration,
                })
    return scenarios


# -----------------------------------------------------------------------------
# Robust threshold summary
# -----------------------------------------------------------------------------
def summarize_thresholds(curves):
    summary = (
        curves.groupby("threshold", as_index=False)
        .agg(
            mean_normalized_cost=("normalized_cost", "mean"),
            worst_case_normalized_cost=("normalized_cost", "max"),
            mean_regret=("regret", "mean"),
            max_regret=("regret", "max"),
            median_regret=("regret", "median"),
            scenarios_optimal=("regret", lambda s: int(np.isclose(s, 0.0).sum())),
            mean_dollar_cost=("total_cost", "mean"),
            worst_dollar_cost=("total_cost", "max"),
        )
    )
    summary["mean_regret_pct"] = 100 * summary["mean_regret"]
    summary["max_regret_pct"] = 100 * summary["max_regret"]

    # Primary criterion: minimize worst-case normalized regret.
    # Tie-breakers: mean normalized cost, mean regret, distance from 0.50.
    ordered = summary.sort_values(
        ["max_regret", "mean_normalized_cost", "mean_regret"],
        ascending=[True, True, True],
    ).copy()
    min_max = ordered.iloc[0]["max_regret"]
    tied = ordered[np.isclose(ordered["max_regret"], min_max, atol=1e-12)].copy()
    min_mean = tied["mean_normalized_cost"].min()
    tied = tied[np.isclose(tied["mean_normalized_cost"], min_mean, atol=1e-12)].copy()
    min_regret = tied["mean_regret"].min()
    tied = tied[np.isclose(tied["mean_regret"], min_regret, atol=1e-12)].copy()
    robust_threshold = float(tied.iloc[np.argmin(np.abs(tied["threshold"].to_numpy() - 0.50))]["threshold"])

    mean_cost_threshold = float(summary.loc[summary["mean_normalized_cost"].idxmin(), "threshold"])

    summary["robust_rank"] = np.where(
        np.isclose(summary["threshold"], robust_threshold), 1, np.nan
    )
    summary["mean_cost_rank"] = np.where(
        np.isclose(summary["threshold"], mean_cost_threshold), 1, np.nan
    )
    return summary, robust_threshold, mean_cost_threshold


# -----------------------------------------------------------------------------
# Reconcile prior results
# -----------------------------------------------------------------------------
def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def reconcile(previous_step8b, previous_step8c, central_curve, current_best):
    result = {
        "previous_step8b_best_threshold": None,
        "previous_step8c_central_threshold": None,
        "current_deterministic_central_threshold": float(current_best["threshold"]),
        "current_cost_at_048": None,
        "current_cost_at_068": None,
        "current_cost_048_minus_068": None,
    }

    if previous_step8b:
        result["previous_step8b_best_threshold"] = previous_step8b.get("best_threshold_monetary")
    if previous_step8c:
        result["previous_step8c_central_threshold"] = previous_step8c.get("central_oof", {}).get("threshold")

    at48 = central_curve[np.isclose(central_curve["threshold"], 0.48)]
    at68 = central_curve[np.isclose(central_curve["threshold"], 0.68)]
    if not at48.empty:
        result["current_cost_at_048"] = float(at48.iloc[0]["total_cost"])
    if not at68.empty:
        result["current_cost_at_068"] = float(at68.iloc[0]["total_cost"])
    if result["current_cost_at_048"] is not None and result["current_cost_at_068"] is not None:
        result["current_cost_048_minus_068"] = result["current_cost_at_048"] - result["current_cost_at_068"]
    return result


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_threshold_analysis(summary, robust_threshold, mean_cost_threshold, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary["threshold"], summary["mean_normalized_cost"], marker="o", label="Mean normalized cost")
    ax.plot(summary["threshold"], summary["worst_case_normalized_cost"], marker="o", label="Worst-case normalized cost")
    ax.axvline(robust_threshold, linestyle="--", label=f"Robust threshold = {robust_threshold:.2f}")
    ax.axvline(mean_cost_threshold, linestyle=":", label=f"Mean-cost threshold = {mean_cost_threshold:.2f}")
    ax.axvline(0.48, linestyle="-.", label="Step 8b = 0.48")
    ax.axvline(0.68, linestyle="-.", label="Step 8c = 0.68")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Normalized monetary cost")
    ax.set_title("FinPredict — Robust threshold analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_regret_heatmap(central_scenarios, robust_threshold, out):
    subset = central_scenarios[np.isclose(central_scenarios["threshold"], robust_threshold)].copy()
    pivot = subset.pivot(index="duration_years", columns="cost_of_funds", values="robust_regret_pct")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, label="Regret at robust threshold (%)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1%}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0f}y" for v in pivot.index])
    ax.set_xlabel("Cost of funds")
    ax.set_ylabel("Duration")
    ax.set_title(f"FinPredict — Regret at robust threshold {robust_threshold:.2f}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}%", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("FINPREDICT — STEP 8d: ROBUST THRESHOLD SELECTION")
    print("=" * 78)

    df = pd.read_csv(DATASET_PATH)
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTRAIN : {len(X_train)}")
    print(f"TEST  : {len(X_test)}")
    print("TEST = holdout indépendant.")
    print("Aucun seuil n'est sélectionné sur TEST.")
    print("\nReproductibilité LightGBM : n_jobs=1, deterministic=True")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nCalcul des probabilités OOF calibrées (sigmoid, 5 folds)...")
    oof_p = cross_val_predict(
        build_pipeline(), X_train, y_train, cv=cv,
        method="predict_proba", n_jobs=1
    )[:, 1]

    print("Entraînement du modèle final calibré sur TRAIN complet...")
    final_model = build_pipeline()
    final_model.fit(X_train, y_train)
    test_p = final_model.predict_proba(X_test)[:, 1]

    amount_median = X_train["loan_amnt"].median()
    rate_median = X_train["loan_int_rate"].median()
    ytr = y_train.to_numpy()
    yte = y_test.to_numpy()

    scenarios = build_scenarios()
    print(f"\nScénarios testés : {len(scenarios)} = 5 LGD × 4 CoF × 3 durées")

    curves = []
    for s in scenarios:
        cfn, cfp = row_costs(
            X_train, amount_median, rate_median,
            s["lgd"], s["cof"], s["duration"]
        )
        curves.append(
            scenario_curve(
                s["name"], s["lgd"], s["cof"], s["duration"],
                ytr, oof_p, cfn, cfp
            )
        )
    curves_df = pd.concat(curves, ignore_index=True)

    summary, robust_threshold, mean_cost_threshold = summarize_thresholds(curves_df)
    robust_row = summary[np.isclose(summary["threshold"], robust_threshold)].iloc[0]

    print("\n" + "=" * 78)
    print("A) CANDIDATS DE SEUIL")
    print("=" * 78)
    candidate_values = [0.18, 0.44, 0.48, 0.50, 0.58, 0.62, 0.68, 0.70, 0.74]
    candidates = summary[summary["threshold"].isin(candidate_values)][[
        "threshold", "mean_normalized_cost", "worst_case_normalized_cost",
        "mean_regret_pct", "max_regret_pct", "scenarios_optimal"
    ]]
    print(candidates.to_string(index=False))

    print("\n" + "=" * 78)
    print("B) SELECTION ROBUSTE")
    print("=" * 78)
    print(f"Robust threshold (minimax max-regret) : {robust_threshold:.2f}")
    print(f"Mean-cost threshold                   : {mean_cost_threshold:.2f}")
    print(f"Mean regret @ robust                  : {robust_row['mean_regret_pct']:.3f}%")
    print(f"Max regret @ robust                   : {robust_row['max_regret_pct']:.3f}%")
    print(f"Scenarios optimal                     : {int(robust_row['scenarios_optimal'])}/{len(scenarios)}")

    # Central scenario curve and reconciliation.
    central_name = "LGD_40pct_FinPredict_CoF0.030_Dur3y"
    central_curve = curves_df[curves_df["scenario"] == central_name].copy().sort_values("threshold")
    central_best = central_curve.loc[central_curve["total_cost"].idxmin()]

    prev8b = read_json(REPORTS_DIR / "step8b_monetary_cost_results.json")
    prev8c = read_json(REPORTS_DIR / "step8c_economic_sensitivity_results.json")
    reconciliation = reconcile(prev8b, prev8c, central_curve, central_best)

    print("\n" + "=" * 78)
    print("C) RECONCILIATION STEP 8b vs STEP 8c")
    print("=" * 78)
    print(f"Current deterministic central optimum : {central_best['threshold']:.2f}")
    print(f"Current OOF cost @ 0.48               : ${reconciliation['current_cost_at_048']:,.2f}")
    print(f"Current OOF cost @ 0.68               : ${reconciliation['current_cost_at_068']:,.2f}")
    print(f"Current cost(0.48) - cost(0.68)      : ${reconciliation['current_cost_048_minus_068']:,.2f}")
    print(f"Previous Step 8b optimum              : {reconciliation['previous_step8b_best_threshold']}")
    print(f"Previous Step 8c central optimum      : {reconciliation['previous_step8c_central_threshold']}")

    # Apply robust threshold once to TEST under pre-declared central scenario.
    cfn_test, cfp_test = row_costs(
        X_test, amount_median, rate_median,
        CENTRAL_LGD, CENTRAL_COF, CENTRAL_DURATION
    )
    robust_test = cost_at_threshold(yte, test_p, cfn_test, cfp_test, robust_threshold)
    test_050 = cost_at_threshold(yte, test_p, cfn_test, cfp_test, 0.50)
    test_048 = cost_at_threshold(yte, test_p, cfn_test, cfp_test, 0.48)
    test_068 = cost_at_threshold(yte, test_p, cfn_test, cfp_test, 0.68)

    print("\n" + "=" * 78)
    print("D) TEST — CENTRAL SCENARIO, UNE SEULE FOIS")
    print("=" * 78)
    print(f"Central LGD        : {CENTRAL_LGD:.2%}")
    print(f"Central CoF        : {CENTRAL_COF:.2%}")
    print(f"Central duration   : {CENTRAL_DURATION:.1f} years")
    print(f"Robust threshold   : {robust_threshold:.2f}")
    print(f"Total cost         : ${robust_test['total_cost']:,.2f}")
    print(f"FN cost            : ${robust_test['fn_cost']:,.2f}")
    print(f"FP cost            : ${robust_test['fp_cost']:,.2f}")
    print(f"n_FN               : {robust_test['n_fn']}")
    print(f"n_FP               : {robust_test['n_fp']}")
    print(f"Precision          : {robust_test['precision']:.4f}")
    print(f"Recall             : {robust_test['recall']:.4f}")
    print(f"F1                 : {robust_test['f1']:.4f}")

    # Scenario-level regret at robust threshold.
    best_by_scenario = curves_df.groupby("scenario", as_index=False).agg(best_cost=("total_cost", "min"))
    robust_scenario = curves_df[np.isclose(curves_df["threshold"], robust_threshold)].merge(
        best_by_scenario, on="scenario", how="left"
    )
    robust_scenario["robust_regret_pct"] = 100 * (
        robust_scenario["total_cost"] / robust_scenario["best_cost"] - 1
    )

    # Save artefacts.
    curves_path = REPORTS_DIR / "step8d_all_scenario_threshold_curves.csv"
    summary_path = REPORTS_DIR / "step8d_threshold_robustness_summary.csv"
    scenario_path = REPORTS_DIR / "step8d_robust_threshold_by_scenario.csv"
    reconciliation_path = REPORTS_DIR / "step8d_step8b_step8c_reconciliation.json"
    json_path = REPORTS_DIR / "step8d_robust_threshold_results.json"
    plot_path = REPORTS_DIR / "step8d_robust_threshold_analysis.png"
    heatmap_path = REPORTS_DIR / "step8d_robust_threshold_regret_heatmap.png"

    curves_df.to_csv(curves_path, index=False)
    summary.to_csv(summary_path, index=False)
    robust_scenario.to_csv(scenario_path, index=False)
    reconciliation_path.write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")

    plot_threshold_analysis(summary, robust_threshold, mean_cost_threshold, plot_path)
    central_robust = robust_scenario[np.isclose(robust_scenario["lgd"], CENTRAL_LGD)]
    plot_regret_heatmap(central_robust, robust_threshold, heatmap_path)

    report = {
        "methodology": {
            "model": "LightGBM + sigmoid calibration",
            "oof_cv": 5,
            "deterministic": True,
            "threshold_selection": "OOF TRAIN only",
            "test_usage": "single final evaluation after threshold selection",
            "primary_criterion": "minimum maximum normalized regret",
            "normalization": "scenario cost / scenario minimum cost",
        },
        "scenario_count": len(scenarios),
        "robust_threshold": robust_threshold,
        "mean_cost_threshold": mean_cost_threshold,
        "robust_metrics": {
            "mean_regret_pct": float(robust_row["mean_regret_pct"]),
            "max_regret_pct": float(robust_row["max_regret_pct"]),
            "scenarios_optimal": int(robust_row["scenarios_optimal"]),
        },
        "central_oof": {
            "threshold": float(central_best["threshold"]),
            "total_cost": float(central_best["total_cost"]),
            "cost_at_048": reconciliation["current_cost_at_048"],
            "cost_at_068": reconciliation["current_cost_at_068"],
        },
        "central_test": {
            "robust": robust_test,
            "at_050": test_050,
            "at_048": test_048,
            "at_068": test_068,
        },
        "reconciliation": reconciliation,
        "artifacts": {
            "all_curves": str(curves_path),
            "summary": str(summary_path),
            "scenario_level": str(scenario_path),
            "reconciliation": str(reconciliation_path),
            "report": str(json_path),
            "plot": str(plot_path),
            "heatmap": str(heatmap_path),
        },
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("ARTEFACTS SAUVEGARDES")
    print("=" * 78)
    print(f"All curves         : {curves_path}")
    print(f"Threshold summary  : {summary_path}")
    print(f"Scenario-level     : {scenario_path}")
    print(f"Reconciliation     : {reconciliation_path}")
    print(f"Main JSON          : {json_path}")
    print(f"Threshold plot     : {plot_path}")
    print(f"Regret heatmap     : {heatmap_path}")
    print("\nSTEP 8d TERMINEE")
    print("=" * 78)


if __name__ == "__main__":
    main()
