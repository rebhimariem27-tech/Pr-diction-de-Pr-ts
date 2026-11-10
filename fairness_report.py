"""Rapport d'équité du modèle de production, par groupe (âge, logement).

    python fairness_report.py [--csv data/processed/donnees_prets_features.csv] [--out data/reports/fairness_report.json]

Utilise le MÊME split TEST (random_state=42, stratify) que le pipeline, donc
le modèle n'a jamais vu ces lignes. Pour chaque groupe :
  - taux de défaut observé, taux de refus du modèle
  - recall (défauts correctement refusés) et taux de faux refus (bons clients refusés)
  - ratio d'acceptation vs groupe de référence ("règle des 4/5" : < 0.80 = signal à investiguer)

Ce rapport est un OUTIL DE DÉTECTION, pas un avis juridique : un écart n'est pas
automatiquement une discrimination, ni l'absence d'écart une conformité. Faites
valider l'usage de person_age par votre service juridique / conformité.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ARTIFACT = Path("data/production/finpredict_production_model.joblib")
AGE_BINS = [17, 24, 34, 44, 54, 120]
AGE_LABELS = ["<=24", "25-34", "35-44", "45-54", "55+"]


def group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group_col, observed=True):
        pos, neg = g[g.y == 1], g[g.y == 0]
        rows.append({
            group_col: str(name), "n": len(g),
            "default_rate": g.y.mean(),
            "refusal_rate": g.refused.mean(),
            "acceptance_rate": 1 - g.refused.mean(),
            "recall_defaults_refused": pos.refused.mean() if len(pos) else np.nan,
            "false_refusal_rate": neg.refused.mean() if len(neg) else np.nan,
        })
    out = pd.DataFrame(rows)
    ref = out.loc[out.acceptance_rate.idxmax(), "acceptance_rate"]
    out["acceptance_ratio_vs_best"] = out.acceptance_rate / ref
    out["flag_4_5_rule"] = out.acceptance_ratio_vs_best < 0.80
    return out.round(4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/processed/donnees_prets_features.csv"))
    p.add_argument("--out", type=Path, default=Path("data/reports/fairness_report.json"))
    a = p.parse_args()

    artifact = joblib.load(ARTIFACT)
    feats, thr = artifact["model_features"], artifact["production_threshold"]
    df = pd.read_csv(a.csv)
    X, y = df[feats], df[artifact["target"]].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    proba = artifact["pipeline"].predict_proba(X_test)[:, 1]
    t = X_test.copy()
    t["y"], t["refused"] = y_test.values, (proba >= thr).astype(int)
    t["age_band"] = pd.cut(t.person_age, AGE_BINS, labels=AGE_LABELS)

    report = {"threshold": thr, "n_test": len(t)}
    for col in ["age_band", "person_home_ownership"]:
        table = group_table(t, col)
        print(f"\n=== {col} ===\n{table.to_string(index=False)}")
        report[col] = table.to_dict(orient="records")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nRapport écrit : {a.out}")
    print("Pistes si des écarts apparaissent : (1) réentraîner SANS person_age et comparer ROC-AUC et coût ;")
    print("(2) seuils par groupe NON recommandés sans avis juridique ; (3) documenter la décision.")


if __name__ == "__main__":
    main()
