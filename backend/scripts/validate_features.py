"""
FinPredict AI Pro - Final Feature Validation V2
================================================

Valide le dataset après feature engineering, avant le pipeline ML.

Important :
- Les features de diagnostic sont présentes dans le CSV pour analyse,
  mais elles ne sont PAS considérées comme des features du modèle.
- Les corrélations et contrôles ML portent donc uniquement sur les
  features officiellement retenues.
- Ce script ne modifie jamais le dataset.

Usage:
    python validate_features.py --input data\processed\donnees_prets_features.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. DÉFINITION OFFICIELLE DES FEATURES
# ============================================================

TARGET = "loan_status"


# Features numériques réellement utilisées par le modèle.
MODEL_NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "person_emp_length",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "interest_income_ratio",
]


# Features catégorielles réellement utilisées par le modèle.
MODEL_CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


# Ensemble complet des features utilisées par le modèle.
MODEL_FEATURES = (
    MODEL_NUMERIC_FEATURES
    + MODEL_CATEGORICAL_FEATURES
)


# Ces variables sont présentes dans le dataset pour diagnostic,
# mais ne seront PAS utilisées par le modèle ML.
DIAGNOSTIC_FEATURES = [
    "loan_to_income",
    "income_per_employment_year",
    "loan_percent_income_diff",
    "loan_to_monthly_income",
    "employment_age_ratio",
]


# Toutes les colonnes attendues.
REQUIRED = MODEL_FEATURES + [TARGET]


# ============================================================
# 2. CHARGEMENT
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    """Charge le dataset enrichi."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "Le dataset est vide."
        )

    return df


# ============================================================
# 3. VALIDATION DU SCHÉMA
# ============================================================

def check_schema(df: pd.DataFrame) -> dict:
    """
    Vérifie que toutes les features nécessaires existent.
    """

    missing = [
        column
        for column in REQUIRED
        if column not in df.columns
    ]

    unexpected = [
        column
        for column in df.columns
        if column not in MODEL_FEATURES
        and column not in DIAGNOSTIC_FEATURES
        and column != TARGET
    ]

    return {
        "valid": len(missing) == 0,
        "missing_model_columns": missing,
        "unexpected_columns": unexpected,
        "diagnostic_columns_present": [
            column
            for column in DIAGNOSTIC_FEATURES
            if column in df.columns
        ],
    }


# ============================================================
# 4. VALEURS MANQUANTES
# ============================================================

def check_missing_model_features(
    df: pd.DataFrame,
) -> dict:
    """
    Vérifie les valeurs manquantes uniquement sur les
    features utilisées par le modèle.
    """

    counts = df[MODEL_FEATURES].isna().sum()

    result = {}

    for column in MODEL_FEATURES:
        count = int(counts[column])

        if count > 0:
            result[column] = {
                "count": count,
                "percentage": round(
                    float(count / len(df) * 100),
                    3,
                ),
            }

    return result


# ============================================================
# 5. VALEURS INFINIES
# ============================================================

def check_infinite_model_features(
    df: pd.DataFrame,
) -> dict:
    """
    Vérifie +inf et -inf uniquement sur les features
    numériques du modèle.
    """

    numeric = df[MODEL_NUMERIC_FEATURES]

    result = {}

    for column in numeric.columns:

        count = int(
            np.isinf(numeric[column]).sum()
        )

        if count > 0:
            result[column] = count

    return result


# ============================================================
# 6. TARGET
# ============================================================

def check_target(
    df: pd.DataFrame,
) -> dict:
    """
    Analyse la distribution de loan_status.
    """

    counts = df[TARGET].value_counts(
        dropna=False
    )

    result = {}

    for value, count in counts.items():

        if pd.isna(value):
            key = "MISSING"
        else:
            key = str(int(value))

        result[key] = {
            "count": int(count),
            "percentage": round(
                float(count / len(df) * 100),
                3,
            ),
        }

    return result


# ============================================================
# 7. STATISTIQUES NUMÉRIQUES
# ============================================================

def check_numeric_stats(
    df: pd.DataFrame,
) -> dict:
    """
    Statistiques descriptives des features numériques ML.
    """

    stats = df[
        MODEL_NUMERIC_FEATURES
    ].describe().T

    result = {}

    for column, row in stats.iterrows():

        result[column] = {
            key: round(
                float(row[key]),
                6,
            )
            for key in [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
            if pd.notna(row[key])
        }

    return result


# ============================================================
# 8. CONTRÔLE DES FEATURES INGÉNIERÉES
# ============================================================

def check_engineered_ranges(
    df: pd.DataFrame,
) -> dict:
    """
    Contrôle la feature dérivée réellement utilisée par le modèle.
    """

    columns = [
        "interest_income_ratio"
    ]

    result = {}

    for column in columns:

        series = df[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        result[column] = {
            "negative_count": int(
                (series < 0).sum()
            ),
            "zero_count": int(
                (series == 0).sum()
            ),
            "missing_count": int(
                series.isna().sum()
            ),
            "min": (
                round(
                    float(series.min()),
                    8,
                )
                if series.notna().any()
                else None
            ),
            "max": (
                round(
                    float(series.max()),
                    8,
                )
                if series.notna().any()
                else None
            ),
        }

    return result


# ============================================================
# 9. CORRÉLATIONS
# ============================================================

def check_correlations(
    df: pd.DataFrame,
    threshold: float = 0.90,
) -> list[dict]:
    """
    Recherche uniquement les corrélations fortes entre les
    features numériques réellement utilisées par le modèle.

    Les features de diagnostic sont volontairement exclues.
    """

    numeric = df[
        MODEL_NUMERIC_FEATURES
    ]

    corr = numeric.corr().abs()

    pairs = []

    columns = list(
        corr.columns
    )

    for i, feature_1 in enumerate(columns):

        for feature_2 in columns[i + 1:]:

            value = corr.loc[
                feature_1,
                feature_2,
            ]

            if (
                pd.notna(value)
                and value >= threshold
            ):
                pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "absolute_correlation": round(
                        float(value),
                        4,
                    ),
                })

    return sorted(
        pairs,
        key=lambda x:
            x["absolute_correlation"],
        reverse=True,
    )


# ============================================================
# 10. VARIABLES CATÉGORIELLES
# ============================================================

def check_categorical_cardinality(
    df: pd.DataFrame,
) -> dict:
    """
    Vérifie les modalités présentes dans les variables
    catégorielles du modèle.
    """

    result = {}

    for column in MODEL_CATEGORICAL_FEATURES:

        counts = df[column].value_counts(
            dropna=False
        )

        result[column] = {
            str(value): int(count)
            for value, count in counts.items()
        }

    return result


# ============================================================
# 11. FEATURES DE DIAGNOSTIC
# ============================================================

def check_diagnostic_features(
    df: pd.DataFrame,
) -> dict:
    """
    Vérifie seulement la présence des colonnes de diagnostic.

    Elles ne participent PAS aux contrôles ML.
    """

    result = {}

    for column in DIAGNOSTIC_FEATURES:

        if column not in df.columns:

            result[column] = {
                "present": False,
            }

            continue

        result[column] = {
            "present": True,
            "missing_count": int(
                df[column].isna().sum()
            ),
        }

    return result


# ============================================================
# 12. RAPPORT GLOBAL
# ============================================================

def build_report(
    df: pd.DataFrame,
) -> dict:

    schema = check_schema(df)

    return {
        "metadata": {
            "generated_at":
                datetime.now().isoformat(),
        },

        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },

        "schema": schema,

        "model_features": {
            "count": len(MODEL_FEATURES),
            "numeric": MODEL_NUMERIC_FEATURES,
            "categorical":
                MODEL_CATEGORICAL_FEATURES,
        },

        "diagnostic_features":
            DIAGNOSTIC_FEATURES,

        "missing_values_model_features":
            check_missing_model_features(df),

        "infinite_values_model_features":
            check_infinite_model_features(df),

        "target_distribution":
            check_target(df),

        "numeric_statistics_model_features":
            check_numeric_stats(df),

        "engineered_model_feature_ranges":
            check_engineered_ranges(df),

        "strong_correlations_model_features_abs_ge_0_90":
            check_correlations(df),

        "categorical_cardinality":
            check_categorical_cardinality(df),

        "diagnostic_feature_checks":
            check_diagnostic_features(df),
    }


# ============================================================
# 13. MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validation finale des features "
            "FinPredict V2"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Dataset généré par "
            "feature_engineering.py"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "data/reports/"
            "final_feature_validation.json"
        ),
        help="Rapport JSON",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    report_path = Path(
        args.report
    )

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df = load_data(input_path)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    report = build_report(df)

    # --------------------------------------------------------
    # Sauvegarde rapport
    # --------------------------------------------------------

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print("=" * 72)
    print(
        "FINPREDICT — "
        "FINAL FEATURE VALIDATION V2"
    )
    print("=" * 72)

    print(
        f"Lignes       : {len(df)}"
    )

    print(
        f"Colonnes     : {len(df.columns)}"
    )

    print(
        f"Features ML  : "
        f"{len(MODEL_FEATURES)}"
    )

    print(
        f"Diagnostics  : "
        f"{len(DIAGNOSTIC_FEATURES)}"
    )

    print()

    print(
        "Schéma       : "
        + (
            "OK"
            if report["schema"]["valid"]
            else "ERREUR"
        )
    )

    # --------------------------------------------------------
    # Inf
    # --------------------------------------------------------

    print(
        "Inf/-Inf ML  : "
        + (
            "Aucune"
            if not report[
                "infinite_values_model_features"
            ]
            else str(
                report[
                    "infinite_values_model_features"
                ]
            )
        )
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    print(
        "NaN features : "
        + (
            "Aucun"
            if not report[
                "missing_values_model_features"
            ]
            else str(
                {
                    key: value["count"]
                    for key, value
                    in report[
                        "missing_values_model_features"
                    ].items()
                }
            )
        )
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    print()
    print(
        "Target loan_status :"
    )

    for value, info in report[
        "target_distribution"
    ].items():

        print(
            f"  - {value}: "
            f"{info['count']} "
            f"({info['percentage']}%)"
        )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print()

    print(
        "Corrélations fortes entre "
        "FEATURES ML (|r| >= 0.90) :"
    )

    pairs = report[
        "strong_correlations_model_features_abs_ge_0_90"
    ]

    if not pairs:

        print(
            "  Aucune"
        )

    else:

        for pair in pairs:

            print(
                f"  - "
                f"{pair['feature_1']} <-> "
                f"{pair['feature_2']} : "
                f"{pair['absolute_correlation']}"
            )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()

    print(
        "Features de diagnostic "
        "non utilisées par le modèle :"
    )

    for feature in DIAGNOSTIC_FEATURES:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------

    print()

    print(
        f"Rapport : {report_path}"
    )

    print(
        "Le dataset n'a pas été modifié."
    )


if __name__ == "__main__":
    main()