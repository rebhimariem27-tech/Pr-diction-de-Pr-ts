"""
FinPredict AI Pro - ML Pipeline | Step 1
========================================

Étape 1 :
    - charger le dataset enrichi
    - définir X (features) et y (target)
    - effectuer le split TRAIN / TEST
    - vérifier les dimensions et la distribution de la target

Cette version NE fait PAS encore :
    - imputation
    - encodage
    - scaling
    - entraînement LightGBM

Usage:
    python model_ml_step1.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/processed/donnees_prets_features.csv"
)

TARGET = "loan_status"


# Features officielles retenues pour le modèle.
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

MODEL_CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

MODEL_FEATURES = (
    MODEL_NUMERIC_FEATURES
    + MODEL_CATEGORICAL_FEATURES
)


# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================

def load_dataset(
    path: Path = DATASET_PATH,
) -> pd.DataFrame:
    """Charge le dataset feature-engineered."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "Le dataset est vide."
        )

    return df


# ============================================================
# 3. VALIDATION DES COLONNES
# ============================================================

def validate_columns(
    df: pd.DataFrame,
) -> None:
    """Vérifie que les features et la target existent."""

    required = MODEL_FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing)
        )


# ============================================================
# 4. CONSTRUCTION DE X ET y
# ============================================================

def build_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    X = variables utilisées pour prédire
    y = variable que le modèle doit apprendre à prédire
    """

    X = df[MODEL_FEATURES].copy()

    y = df[TARGET].copy()

    return X, y


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Sépare les données en :

        80 % TRAIN
        20 % TEST

    stratify=y conserve approximativement la même
    proportion des classes dans TRAIN et TEST.
    """

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


# ============================================================
# 6. DISTRIBUTION DES CLASSES
# ============================================================

def print_distribution(
    name: str,
    y: pd.Series,
) -> None:
    """Affiche la distribution de la target."""

    print(f"\n{name}")

    counts = (
        y.value_counts()
        .sort_index()
    )

    percentages = (
        y.value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    for class_value in counts.index:
        print(
            f"  Classe {class_value}: "
            f"{counts[class_value]} "
            f"({percentages[class_value]:.2f}%)"
        )


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:

    print("=" * 72)
    print(
        "FINPREDICT — ML PIPELINE "
        "STEP 1 : X / y + TRAIN / TEST SPLIT"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nDataset chargé : {DATASET_PATH}"
    )

    print(
        f"Observations : {len(df)}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_columns(df)

    print(
        "Validation des colonnes : OK"
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X, y = build_xy(df)

    print(
        f"\nX : {X.shape[0]} lignes × "
        f"{X.shape[1]} features"
    )

    print(
        f"y : {len(y)} observations"
    )

    print(
        "\nFeatures utilisées :"
    )

    for feature in MODEL_FEATURES:
        print(f"  - {feature}")

    print(
        f"\nTarget : {TARGET}"
    )

    # --------------------------------------------------------
    # Distribution globale
    # --------------------------------------------------------

    print_distribution(
        "Distribution globale",
        y,
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(X, y)

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    print("\nTRAIN / TEST SPLIT")

    print(
        f"  X_train : "
        f"{X_train.shape[0]} lignes"
    )

    print(
        f"  X_test  : "
        f"{X_test.shape[0]} lignes"
    )

    print(
        f"  Train   : "
        f"{len(X_train) / len(X) * 100:.2f}%"
    )

    print(
        f"  Test    : "
        f"{len(X_test) / len(X) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Distribution après split
    # --------------------------------------------------------

    print_distribution(
        "Distribution TRAIN",
        y_train,
    )

    print_distribution(
        "Distribution TEST",
        y_test,
    )

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    assert len(X_train) + len(X_test) == len(X)

    assert len(y_train) + len(y_test) == len(y)

    assert set(y_train.unique()) == set(y.unique())

    assert set(y_test.unique()) == set(y.unique())

    print(
        "\n✅ Vérification du split : OK"
    )

    print(
        "\nAucun preprocessing n'a encore été effectué."
    )

    print(
        "Prochaine étape : "
        "imputation + encodage sans data leakage."
    )


if __name__ == "__main__":
    main()