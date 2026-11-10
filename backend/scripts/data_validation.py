from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
    "loan_int_rate", "loan_status", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
]

NUMERIC_COLUMNS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_status", "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLUMNS = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]


def to_builtin(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    actual = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in actual]
    extra = [c for c in actual if c not in EXPECTED_COLUMNS]
    return {
        "valid": not missing,
        "expected_columns": EXPECTED_COLUMNS,
        "actual_columns": actual,
        "missing_columns": missing,
        "extra_columns": extra,
    }


def analyze_missing(df: pd.DataFrame) -> dict[str, Any]:
    counts = df.isna().sum()
    return {
        c: {"count": int(counts[c]), "percentage": round(float(counts[c] / len(df) * 100), 3)}
        for c in df.columns if counts[c] > 0
    }


def analyze_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    mask = df.duplicated(keep=False)
    return {
        "duplicate_rows_after_first_occurrence": int(df.duplicated().sum()),
        "rows_in_duplicate_groups": int(mask.sum()),
        "duplicate_groups": int(df[mask].drop_duplicates().shape[0]) if mask.any() else 0,
    }


def analyze_target(df: pd.DataFrame) -> dict[str, Any]:
    result = {"column": "loan_status", "exists": "loan_status" in df.columns}
    if "loan_status" not in df.columns:
        return result
    counts = df["loan_status"].value_counts(dropna=False)
    result["missing"] = int(df["loan_status"].isna().sum())
    result["unique_values"] = [to_builtin(v) for v in df["loan_status"].drop_duplicates().tolist()]
    result["distribution"] = {
        str(to_builtin(v)): {"count": int(n), "percentage": round(float(n / len(df) * 100), 3)}
        for v, n in counts.items()
    }
    return result


def analyze_ranges(df: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    def add(name: str, mask: pd.Series, description: str) -> None:
        invalid = int(mask.fillna(False).sum())
        checks[name] = {"description": description, "invalid_rows": invalid, "valid": invalid == 0}

    if "person_age" in df:
        add("age", (df.person_age < 18) | (df.person_age > 100), "18 <= person_age <= 100")
    if "person_income" in df:
        add("income", df.person_income < 0, "person_income >= 0")
    if "loan_amnt" in df:
        add("loan_amount", df.loan_amnt <= 0, "loan_amnt > 0")
    if "loan_int_rate" in df:
        add("interest_rate", (df.loan_int_rate < 0) | (df.loan_int_rate > 100), "0 <= loan_int_rate <= 100")
    if "loan_percent_income" in df:
        add("loan_percent_income", (df.loan_percent_income < 0) | (df.loan_percent_income > 1), "0 <= loan_percent_income <= 1")
    if "person_emp_length" in df:
        add("employment_length", (df.person_emp_length < 0) | (df.person_age.notna() & (df.person_emp_length > df.person_age)), "0 <= employment length <= age")
    if "cb_person_cred_hist_length" in df:
        add("credit_history_length", (df.cb_person_cred_hist_length < 0) | (df.person_age.notna() & (df.cb_person_cred_hist_length > df.person_age)), "0 <= credit history length <= age")
    return checks


def analyze_consistency(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    required = {"loan_amnt", "person_income", "loan_percent_income"}
    if required.issubset(df.columns):
        comparable = df.person_income.gt(0) & df.loan_percent_income.notna()
        expected = df.loc[comparable, "loan_amnt"] / df.loc[comparable, "person_income"]
        diff = (df.loc[comparable, "loan_percent_income"] - expected).abs()
        out["loan_percent_income_consistency"] = {
            "rows_compared": int(comparable.sum()),
            "rows_difference_gt_0_05": int((diff > 0.05).sum()),
            "max_absolute_difference": round(float(diff.max()), 6) if len(diff) else None,
            "note": "Diagnostic uniquement; la définition finale de la feature sera validée avant entraînement.",
        }
    return out


def analyze_categorical(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for c in CATEGORICAL_COLUMNS:
        if c in df.columns:
            counts = df[c].dropna().astype(str).value_counts()
            out[c] = {str(v): int(n) for v, n in counts.items()}
    return out


def analyze_numeric_stats(df: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if not cols:
        return {}
    stats = df[cols].describe().T
    fields = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    return {
        c: {k: round(float(row[k]), 4) for k in fields if pd.notna(row[k])}
        for c, row in stats.iterrows()
    }


def build_report(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    return {
        "metadata": {"generated_at": datetime.now().isoformat(), "input_file": str(path)},
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 3),
        },
        "schema": validate_schema(df),
        "missing_values": analyze_missing(df),
        "duplicates": analyze_duplicates(df),
        "target": analyze_target(df),
        "numeric_statistics": analyze_numeric_stats(df),
        "categorical_values": analyze_categorical(df),
        "range_checks": analyze_ranges(df),
        "consistency_checks": analyze_consistency(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit non destructif du dataset FinPredict")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", default="data_quality_report.json")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    df = pd.read_csv(path)
    report = build_report(df, path)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70)
    print("FINPREDICT — DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Lignes       : {report['dataset']['rows']}")
    print(f"Colonnes     : {report['dataset']['columns']}")
    print(f"Doublons     : {report['duplicates']['duplicate_rows_after_first_occurrence']}")
    print(f"Rapport JSON : {args.report}")
    print("\nValeurs manquantes :")
    for c, x in report["missing_values"].items():
        print(f"  - {c}: {x['count']} ({x['percentage']}%)")
    print("\nContrôles de plage :")
    for name, x in report["range_checks"].items():
        print(f"  - {name}: {'OK' if x['valid'] else 'ALERTE'} ({x['invalid_rows']} lignes)")
    print("\nDistribution target :")
    for value, x in report["target"].get("distribution", {}).items():
        print(f"  - {value}: {x['count']} ({x['percentage']}%)")
    print("\nLe CSV source n'a pas été modifié.")


if __name__ == "__main__":
    main()
