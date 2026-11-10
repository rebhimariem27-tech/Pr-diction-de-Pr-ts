"""Analyse les lignes où loan_percent_income du CSV s'écarte de features.py.

    python inspect_loan_percent_mismatch.py

Répond à 3 questions :
  1. De combien s'écartent-elles ? (petits écarts d'arrondi ou vraies incohérences ?)
  2. Une autre règle d'arrondi (floor, ceil, demi vers le haut) les expliquerait-elle ?
  3. Est-ce que ça change les DÉCISIONS du modèle ? (mêmes lignes scorées avec la
     valeur du CSV, puis avec la valeur recalculée par l'API : combien de décisions basculent ?)

Écrit aussi data/reports/loan_percent_income_mismatch.csv (les lignes concernées)
pour examen métier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.features import derive_features

ARTIFACT = Path("data/production/finpredict_production_model.joblib")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/processed/donnees_prets_features.csv"))
    p.add_argument("--out", type=Path, default=Path("data/reports/loan_percent_income_mismatch.csv"))
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    ok = df[["loan_amnt", "person_income"]].notna().all(axis=1) & (df["person_income"] > 0)
    d = df.loc[ok].copy()
    raw = d["loan_amnt"] / d["person_income"]
    d["recomputed"] = np.round(raw, 2)
    d["diff"] = (d["loan_percent_income"] - d["recomputed"]).abs()
    bad = d[d["diff"] >= 1e-6].copy()
    print(f"Lignes comparées : {len(d)} | écartées : {len(bad)} ({len(bad) / len(d):.2%})")

    # 1. Taille des écarts
    print("\n--- 1. Taille des écarts ---")
    bins = pd.cut(bad["diff"], [0, 0.0101, 0.0201, 0.05, 1], labels=["<=0.01", "0.01-0.02", "0.02-0.05", ">0.05"])
    print(bins.value_counts().sort_index().to_string())
    print(f"écart moyen : {bad['diff'].mean():.4f} | médian : {bad['diff'].median():.4f}")

    # 2. Autres règles d'arrondi
    print("\n--- 2. Une autre règle d'arrondi explique-t-elle ces lignes ? ---")
    rules = {
        "floor 2 déc.": np.floor(raw[bad.index] * 100) / 100,
        "ceil 2 déc.": np.ceil(raw[bad.index] * 100) / 100,
        "demi vers le haut": np.floor(raw[bad.index] * 100 + 0.5) / 100,
        "arrondi 3 déc.": np.round(raw[bad.index], 3),
        "pas d'arrondi": raw[bad.index],
    }
    for name, values in rules.items():
        share = float(((values - bad["loan_percent_income"]).abs() < 1e-6).mean())
        print(f"  {name:<20} explique {share:6.1%} des lignes écartées")

    # Un écart lié au revenu ou au montant ?
    print("\n--- Profil des lignes écartées vs les autres ---")
    for col in ["person_income", "loan_amnt", "loan_int_rate"]:
        if col in d:
            print(f"  {col:<15} écartées médiane={bad[col].median():>10.1f} | autres médiane={d.loc[d['diff'] < 1e-6, col].median():>10.1f}")
    if "loan_status" in d:
        print(f"  taux de défaut  écartées={bad['loan_status'].mean():.3f} | autres={d.loc[d['diff'] < 1e-6, 'loan_status'].mean():.3f}")

    # 3. Impact sur les décisions
    print("\n--- 3. Impact sur les décisions du modèle (seuil de production) ---")
    if not ARTIFACT.exists():
        print(f"  Artefact introuvable ({ARTIFACT}) : étape ignorée.")
    else:
        art = joblib.load(ARTIFACT)
        feats, thr = art["model_features"], art["production_threshold"]
        as_in_csv = bad[feats].copy()
        as_served = bad[feats].copy()
        as_served["loan_percent_income"] = bad["recomputed"]
        p_csv = art["pipeline"].predict_proba(as_in_csv)[:, 1]
        p_srv = art["pipeline"].predict_proba(as_served)[:, 1]
        flips = int(((p_csv >= thr) != (p_srv >= thr)).sum())
        print(f"  Écart moyen de probabilité : {np.abs(p_csv - p_srv).mean():.4f} | max : {np.abs(p_csv - p_srv).max():.4f}")
        print(f"  Décisions qui basculent    : {flips} sur {len(bad)} ({flips / len(bad):.2%}) "
              f"soit {flips / len(d):.3%} du jeu complet")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    bad.sort_values("diff", ascending=False).to_csv(a.out, index=False)
    print(f"\nLignes concernées écrites dans : {a.out}")


if __name__ == "__main__":
    main()
