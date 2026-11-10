from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

EXPECTED_COLUMNS = [
    'person_age','person_income','person_home_ownership','person_emp_length',
    'loan_intent','loan_grade','loan_amnt','loan_int_rate','loan_status',
    'loan_percent_income','cb_person_default_on_file','cb_person_cred_hist_length'
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Dataset introuvable : {path}')
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError('Le dataset est vide.')
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError('Colonnes obligatoires manquantes : ' + ', '.join(missing))


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric = [
        'person_age','person_income','person_emp_length','loan_amnt',
        'loan_int_rate','loan_status','loan_percent_income','cb_person_cred_hist_length'
    ]
    categorical = [
        'person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file'
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in categorical:
        df[c] = df[c].astype('string').str.strip().str.upper()
    return df


def remove_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    masks: dict[str, pd.Series] = {}
    masks['age'] = df['person_age'].notna() & ((df['person_age'] < 18) | (df['person_age'] > 100))
    masks['income'] = df['person_income'].notna() & (df['person_income'] < 0)
    masks['loan_amount'] = df['loan_amnt'].notna() & (df['loan_amnt'] <= 0)
    masks['interest_rate'] = df['loan_int_rate'].notna() & ((df['loan_int_rate'] < 0) | (df['loan_int_rate'] > 100))
    masks['loan_percent_income'] = df['loan_percent_income'].notna() & ((df['loan_percent_income'] < 0) | (df['loan_percent_income'] > 1))
    masks['employment_length'] = df['person_emp_length'].notna() & df['person_age'].notna() & (
        (df['person_emp_length'] < 0) | (df['person_emp_length'] > df['person_age'])
    )
    masks['credit_history_length'] = df['cb_person_cred_hist_length'].notna() & df['person_age'].notna() & (
        (df['cb_person_cred_hist_length'] < 0) | (df['cb_person_cred_hist_length'] > df['person_age'])
    )
    masks['target'] = df['loan_status'].notna() & ~df['loan_status'].isin([0, 1])
    invalid = pd.Series(False, index=df.index)
    for m in masks.values():
        invalid |= m.fillna(False)
    report = {k: int(v.fillna(False).sum()) for k, v in masks.items()}
    report['total_invalid_rows'] = int(invalid.sum())
    return df.loc[~invalid].copy(), report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n = int(df.duplicated().sum())
    return df.drop_duplicates().copy(), n


def categorical_domains(df: pd.DataFrame) -> dict[str, Any]:
    domains = {
        'person_home_ownership': {'RENT','OWN','MORTGAGE','OTHER'},
        'loan_intent': {'PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION'},
        'loan_grade': {'A','B','C','D','E','F','G'},
        'cb_person_default_on_file': {'Y','N'},
    }
    out = {}
    for col, expected in domains.items():
        observed = set(df[col].dropna().astype(str).unique())
        out[col] = {'observed_values': sorted(observed), 'unexpected_values': sorted(observed - expected), 'valid': not bool(observed - expected)}
    return out


def missing_report(df: pd.DataFrame) -> dict[str, Any]:
    counts = df.isna().sum()
    return {c: {'count': int(n), 'percentage': round(float(n/len(df)*100), 3)} for c, n in counts.items() if n > 0}


def target_report(df: pd.DataFrame) -> dict[str, Any]:
    counts = df['loan_status'].value_counts(dropna=False)
    dist = {}
    for v, n in counts.items():
        key = 'MISSING' if pd.isna(v) else str(int(v))
        dist[key] = {'count': int(n), 'percentage': round(float(n/len(df)*100), 3)}
    return {'column': 'loan_status', 'distribution': dist}


def feature_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    req = {'loan_amnt','person_income','loan_percent_income'}
    if not req.issubset(df.columns):
        return {}
    comparable = df['person_income'].gt(0) & df['loan_percent_income'].notna()
    expected = df.loc[comparable, 'loan_amnt'] / df.loc[comparable, 'person_income']
    actual = df.loc[comparable, 'loan_percent_income']
    diff = (actual - expected).abs()
    return {
        'loan_percent_income': {
            'rows_compared': int(comparable.sum()),
            'difference_gt_0_05': int((diff > 0.05).sum()),
            'difference_gt_0_10': int((diff > 0.10).sum()),
            'max_absolute_difference': round(float(diff.max()), 6) if len(diff) else None,
            'note': 'Diagnostic uniquement; la définition métier de cette variable sera confirmée avant modification.'
        }
    }


def preprocess(input_path: Path, output_path: Path, report_path: Path) -> pd.DataFrame:
    raw = load_data(input_path)
    validate_schema(raw)
    rows_raw = len(raw)
    df = normalize_types(raw)
    missing_target = int(df['loan_status'].isna().sum())
    df = df.loc[df['loan_status'].notna()].copy()
    df, invalid_report = remove_invalid_rows(df)
    df, duplicates_removed = remove_duplicates(df)
    report = {
        'metadata': {'generated_at': datetime.now().isoformat(), 'input_file': str(input_path), 'output_file': str(output_path)},
        'dataset': {'rows_raw': rows_raw, 'rows_clean': int(len(df)), 'rows_removed_total': int(rows_raw-len(df))},
        'removed_rows': {'missing_target': missing_target, 'duplicates_exact': duplicates_removed, **invalid_report},
        'missing_values_after_cleaning': missing_report(df),
        'target_after_cleaning': target_report(df),
        'categorical_domains': categorical_domains(df),
        'feature_diagnostics': feature_diagnostics(df),
        'imputation': {'performed': False, 'reason': 'L’imputation sera apprise uniquement sur TRAIN dans la pipeline ML afin d’éviter le data leakage.'}
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return df


def main() -> None:
    p = argparse.ArgumentParser(description='Nettoyage contrôlé du dataset FinPredict')
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='data/processed/donnees_prets_clean.csv')
    p.add_argument('--report', default='data/reports/preprocessing_report.json')
    a = p.parse_args()
    df = preprocess(Path(a.input), Path(a.output), Path(a.report))
    print('='*72)
    print('FINPREDICT — PREPROCESSING')
    print('='*72)
    print(f'Lignes après nettoyage : {len(df)}')
    print(f'Dataset clean          : {a.output}')
    print(f'Rapport                : {a.report}')
    print('Aucune imputation globale : elle sera faite plus tard uniquement sur TRAIN.')

if __name__ == '__main__':
    main()
