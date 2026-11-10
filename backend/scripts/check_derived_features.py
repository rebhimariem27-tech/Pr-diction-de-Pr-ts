"""Vérifie quelle formule reproduit les variables dérivées du dataset d'entraînement.

    python check_derived_features.py [--csv data/processed/donnees_prets_features.csv]

Sortie : part de lignes reproduites exactement par chaque formule candidate. La formule
retenue doit en reproduire au moins 95 % ; sinon training et serving divergent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.app.features import ACTIVE_FORMULAS, FORMULAS

# Part minimale de lignes que la formule retenue doit reproduire exactement.
MIN_MATCH_RATE = 0.95


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/processed/donnees_prets_features.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    ok = df[["loan_amnt", "person_income", "loan_int_rate"]].notna().all(axis=1) & (df["person_income"] > 0)
    d = df.loc[ok]
    status = 0

    for feature, candidates in FORMULAS.items():
        print(f"\n=== {feature} ({len(d)} lignes comparées) ===")
        results = []
        for name, fn in candidates.items():
            diff = (fn(d["loan_amnt"], d["person_income"], d["loan_int_rate"]) - d[feature]).abs()
            match = float((diff < 1e-6).mean())
            results.append((-match, diff.mean(), diff.max(), name))
        results.sort()
        for neg_match, mean, mx, name in results:
            marker = "  <-- ACTIVE" if name == ACTIVE_FORMULAS[feature] else ""
            print(f"  {name:<36} lignes identiques={-neg_match:7.2%} mean={mean:.6f} max={mx:.6f}{marker}")
        best_match, best_name = -results[0][0], results[0][3]
        if best_name != ACTIVE_FORMULAS[feature]:
            print(f"  ATTENTION : la meilleure formule est '{best_name}', mettez à jour ACTIVE_FORMULAS.")
            status = 1
        elif best_match < MIN_MATCH_RATE:
            print(f"  ATTENTION : seulement {best_match:.2%} des lignes sont reproduites (minimum {MIN_MATCH_RATE:.0%}).")
            status = 1
        elif best_match < 1:
            n_off = int(round((1 - best_match) * len(d)))
            print(f"  Note : {n_off} lignes ({1 - best_match:.2%}) s'écartent de la formule : incohérence des données sources.")
    print("\nOK : formules cohérentes avec le dataset." if status == 0 else "\nÀ CORRIGER avant production.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
