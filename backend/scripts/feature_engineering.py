from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_NUMERIC_FEATURES = [
    "person_age", "person_income", "loan_amnt", "loan_int_rate",
    "person_emp_length", "loan_percent_income",
    "cb_person_cred_hist_length",
]

BASE_CATEGORICAL_FEATURES = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]

TARGET_COLUMN = "loan_status"

# Feature retenue pour le futur modèle.
ENGINEERED_FEATURES = ["interest_income_ratio"]

# Features conservées uniquement pour le diagnostic.
DIAGNOSTIC_FEATURES = [
    "loan_to_income",
    "income_per_employment_year",
    "loan_percent_income_diff",
    "loan_to_monthly_income",
    "employment_age_ratio",
]


def load_clean_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Le dataset est vide.")

    required = (
        BASE_NUMERIC_FEATURES
        + BASE_CATEGORICAL_FEATURES
        + [TARGET_COLUMN]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes nécessaires manquantes : " + ", ".join(missing)
        )
    return df


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Division sûre : zéro au dénominateur devient NaN."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features retenues et les features de diagnostic.

    Aucune imputation et aucun encodage ne sont effectués ici.
    """
    result = df.copy()

    monthly_income = result["person_income"] / 12

    # Feature retenue pour le futur modèle.
    monthly_interest = (
        result["loan_amnt"] * result["loan_int_rate"] / 100 / 12
    )
    result["interest_income_ratio"] = safe_divide(
        monthly_interest,
        monthly_income,
    )

    # Diagnostics : ces variables restent dans le CSV pour analyse,
    # mais ne feront pas partie du premier modèle.
    result["loan_to_income"] = safe_divide(
        result["loan_amnt"],
        result["person_income"],
    )

    result["income_per_employment_year"] = safe_divide(
        result["person_income"],
        result["person_emp_length"],
    )

    result["loan_percent_income_diff"] = (
        result["loan_percent_income"]
        - result["loan_to_income"]
    ).abs()

    result["loan_to_monthly_income"] = safe_divide(
        result["loan_amnt"],
        monthly_income,
    )

    result["employment_age_ratio"] = safe_divide(
        result["person_emp_length"],
        result["person_age"],
    )

    for column in ENGINEERED_FEATURES + DIAGNOSTIC_FEATURES:
        result[column] = result[column].replace(
            [np.inf, -np.inf], np.nan
        )

    return result


def get_model_features() -> list[str]:
    """Liste officielle des features utilisées avant encodage."""
    return (
        BASE_NUMERIC_FEATURES
        + BASE_CATEGORICAL_FEATURES
        + ENGINEERED_FEATURES
    )


def get_feature_catalog() -> pd.DataFrame:
    rows = [
        ("person_age", "base", "model", "Âge du demandeur"),
        ("person_income", "base", "model", "Revenu annuel"),
        ("loan_amnt", "base", "model", "Montant demandé"),
        ("loan_int_rate", "base", "model", "Taux d'intérêt"),
        (
            "person_emp_length",
            "base",
            "model",
            "Ancienneté professionnelle",
        ),
        (
            "loan_percent_income",
            "base",
            "model",
            "Part du prêt / revenu fournie par le dataset",
        ),
        (
            "cb_person_cred_hist_length",
            "base",
            "model",
            "Ancienneté de l'historique de crédit",
        ),
        (
            "person_home_ownership",
            "base",
            "model",
            "Statut immobilier",
        ),
        ("loan_intent", "base", "model", "Objet du prêt"),
        ("loan_grade", "base", "model", "Grade de crédit"),
        (
            "cb_person_default_on_file",
            "base",
            "model",
            "Défaut historique déclaré",
        ),
        (
            "interest_income_ratio",
            "engineered",
            "model",
            "Charge mensuelle approximative des intérêts / revenu mensuel",
        ),
        (
            "loan_to_income",
            "engineered",
            "diagnostic",
            "Montant du prêt / revenu annuel",
        ),
        (
            "income_per_employment_year",
            "engineered",
            "diagnostic",
            "Revenu / ancienneté professionnelle",
        ),
        (
            "loan_percent_income_diff",
            "diagnostic",
            "diagnostic",
            "Écart entre ratio fourni et ratio recalculé",
        ),
        (
            "loan_to_monthly_income",
            "engineered",
            "diagnostic",
            "Montant du prêt / revenu mensuel",
        ),
        (
            "employment_age_ratio",
            "engineered",
            "diagnostic",
            "Ancienneté / âge",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["feature", "type", "role", "description"],
    )


def build_report(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> dict:
    diff = feature_df["loan_percent_income_diff"]
    model_features = get_model_features()

    return {
        "generated_at": datetime.now().isoformat(),
        "raw_rows": int(len(raw_df)),
        "rows": int(len(feature_df)),
        "target": TARGET_COLUMN,
        "model_features_before_encoding": model_features,
        "model_feature_count_before_encoding": len(model_features),
        "diagnostic_features": DIAGNOSTIC_FEATURES,
        "diagnostic_feature_count": len(DIAGNOSTIC_FEATURES),
        "model_feature_missing_values": {
            column: int(feature_df[column].isna().sum())
            for column in model_features
        },
        "diagnostic_feature_missing_values": {
            column: int(feature_df[column].isna().sum())
            for column in DIAGNOSTIC_FEATURES
        },
        "loan_percent_income_diagnostic": {
            "rows_compared": int(diff.notna().sum()),
            "rows_with_diff_gt_0_05": int((diff > 0.05).sum()),
            "rows_with_diff_gt_0_10": int((diff > 0.10).sum()),
            "max_difference": (
                float(diff.max()) if diff.notna().any() else None
            ),
        },
        "processing": {
            "imputation": False,
            "encoding": False,
            "scaling": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering V2 centralisé pour FinPredict"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="data/processed/donnees_prets_features.csv",
    )
    parser.add_argument(
        "--report",
        default="data/reports/feature_engineering_report.json",
    )
    parser.add_argument(
        "--catalog",
        default="data/reports/feature_catalog.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    catalog_path = Path(args.catalog)

    raw_df = load_clean_data(input_path)
    feature_df = build_features(raw_df)
    report = build_report(raw_df, feature_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(output_path, index=False)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    get_feature_catalog().to_csv(catalog_path, index=False)

    print("=" * 72)
    print("FINPREDICT - FEATURE ENGINEERING V2")
    print("=" * 72)
    print(f"Lignes                         : {len(feature_df)}")
    print(
        "Features modèle avant encodage : "
        f"{len(get_model_features())}"
    )
    print(
        f"Features diagnostic            : {len(DIAGNOSTIC_FEATURES)}"
    )
    print("\nFEATURES MODÈLE :")
    for feature in get_model_features():
        print(f"  - {feature}")

    print("\nFEATURES DIAGNOSTIC :")
    for feature in DIAGNOSTIC_FEATURES:
        print(f"  - {feature}")

    print(f"\nDataset   : {output_path}")
    print(f"Rapport   : {report_path}")
    print(f"Catalogue : {catalog_path}")
    print("\nAucune imputation, aucun encodage et aucun scaling.")


if __name__ == "__main__":
    main()
