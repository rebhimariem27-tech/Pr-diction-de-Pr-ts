from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = Path(
    "data/processed/donnees_prets_features.csv"
)

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

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def load_dataset() -> pd.DataFrame:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError(
            "Le dataset est vide."
        )

    return df


def build_xy(
    df: pd.DataFrame,
):
    X = df[MODEL_FEATURES].copy()

    y = df[TARGET].copy()

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def main():

    print("=" * 72)
    print(
        "FINPREDICT — ML PIPELINE "
        "STEP 2 : PREPROCESSING"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    df = load_dataset()

    X, y = build_xy(df)

    print(
        f"\nDataset : {len(df)} lignes"
    )

    print(
        f"Features : {len(MODEL_FEATURES)}"
    )

    # --------------------------------------------------------
    # 2. Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(X, y)

    print(
        f"\nX_train : {X_train.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    # --------------------------------------------------------
    # 3. Preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    # IMPORTANT :
    # fit uniquement sur TRAIN
    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # TEST :
    # transform uniquement
    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # 4. Vérifications
    # --------------------------------------------------------

    print(
        "\nAprès preprocessing :"
    )

    print(
        f"X_train_processed : "
        f"{X_train_processed.shape}"
    )

    print(
        f"X_test_processed  : "
        f"{X_test_processed.shape}"
    )

    print(
        "\nNaN train :",
        pd.DataFrame(
            X_train_processed
        ).isna().sum().sum(),
    )

    print(
        "NaN test  :",
        pd.DataFrame(
            X_test_processed
        ).isna().sum().sum(),
    )

    print(
        "\n✅ Preprocessing terminé."
    )

    print(
        "✅ Imputation apprise uniquement sur TRAIN."
    )

    print(
        "✅ Encoder apprise uniquement sur TRAIN."
    )

    print(
        "✅ TEST uniquement transformé."
    )


if __name__ == "__main__":
    main()