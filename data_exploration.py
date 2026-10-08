from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("data/raw/donnees_prets_reels.csv")
REPORT_PATH = Path("data/reports/data_exploration_report.json")
PLOTS_DIR = Path("data/reports/eda_plots")


NUMERIC_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET = "loan_status"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {INPUT_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("DATA EXPLORATION — FINPREDICT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Dimensions
    # ---------------------------------------------------------
    print("\n1. Dimensions")
    print(f"Lignes  : {df.shape[0]}")
    print(f"Colonnes: {df.shape[1]}")

    # ---------------------------------------------------------
    # 2. Types
    # ---------------------------------------------------------
    print("\n2. Types de données")
    print(df.dtypes)

    # ---------------------------------------------------------
    # 3. Aperçu
    # ---------------------------------------------------------
    print("\n3. Aperçu du dataset")
    print(df.head())

    # ---------------------------------------------------------
    # 4. Valeurs manquantes
    # ---------------------------------------------------------
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(3)

    print("\n4. Valeurs manquantes")
    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_pct
    })

    print(missing_table[missing_table["missing_count"] > 0])

    # ---------------------------------------------------------
    # 5. Doublons
    # ---------------------------------------------------------
    duplicate_count = int(df.duplicated().sum())

    print("\n5. Doublons")
    print(f"Doublons exacts : {duplicate_count}")

    # ---------------------------------------------------------
    # 6. Distribution de la cible
    # ---------------------------------------------------------
    print("\n6. Distribution de loan_status")

    target_counts = df[TARGET].value_counts().sort_index()
    target_pct = (target_counts / len(df) * 100).round(2)

    target_table = pd.DataFrame({
        "count": target_counts,
        "percent": target_pct
    })

    print(target_table)

    # Graphique cible
    plt.figure(figsize=(7, 5))
    target_counts.plot(kind="bar")
    plt.title("Distribution de loan_status")
    plt.xlabel("loan_status")
    plt.ylabel("Nombre de prêts")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 7. Statistiques numériques
    # ---------------------------------------------------------
    print("\n7. Statistiques descriptives")
    numeric_stats = df[NUMERIC_COLUMNS].describe().round(3)
    print(numeric_stats)

    # ---------------------------------------------------------
    # 8. Variables catégorielles
    # ---------------------------------------------------------
    print("\n8. Variables catégorielles")

    categorical_summary = {}

    for col in CATEGORICAL_COLUMNS:
        counts = df[col].value_counts(dropna=False)
        categorical_summary[col] = {
            str(k): int(v)
            for k, v in counts.items()
        }

        print(f"\n{col}")
        print(counts)

    # ---------------------------------------------------------
    # 9. Corrélations numériques
    # ---------------------------------------------------------
    print("\n9. Matrice de corrélation")

    corr = df[NUMERIC_COLUMNS + [TARGET]].corr(numeric_only=True)

    print(corr.round(3))

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 10. Distributions numériques
    # ---------------------------------------------------------
    for col in NUMERIC_COLUMNS:
        plt.figure(figsize=(7, 5))
        df[col].hist(bins=40)
        plt.title(f"Distribution de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / f"distribution_{col}.png",
            dpi=150
        )
        plt.close()

    # ---------------------------------------------------------
    # 11. Relations avec la cible
    # ---------------------------------------------------------
    target_means = {}

    for col in NUMERIC_COLUMNS:
        grouped = df.groupby(TARGET)[col].mean()
        target_means[col] = {
            str(k): round(float(v), 4)
            for k, v in grouped.items()
        }

    # Exemple : taux de défaut par grade
    default_rate_by_grade = (
        df.groupby("loan_grade")[TARGET]
        .mean()
        .round(4)
        .to_dict()
    )

    # ---------------------------------------------------------
    # 12. Rapport JSON
    # ---------------------------------------------------------
    report = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist()
        },
        "missing_values": {
            col: {
                "count": int(missing[col]),
                "percent": float(missing_pct[col])
            }
            for col in df.columns
            if missing[col] > 0
        },
        "duplicates": duplicate_count,
        "target_distribution": {
            str(k): {
                "count": int(target_counts[k]),
                "percent": float(target_pct[k])
            }
            for k in target_counts.index
        },
        "categorical_summary": categorical_summary,
        "target_group_means": target_means,
        "default_rate_by_loan_grade": {
            str(k): float(v)
            for k, v in default_rate_by_grade.items()
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("EDA TERMINÉE")
    print("=" * 70)
    print(f"Rapport : {REPORT_PATH}")
    print(f"Graphiques : {PLOTS_DIR}")


if __name__ == "__main__":
    main()