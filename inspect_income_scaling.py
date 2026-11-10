"""Vérifie si certaines lignes du CSV ont un revenu modifié APRÈS le calcul de loan_percent_income.

    python inspect_income_scaling.py

Hypothèse testée : le revenu de ces lignes a été multiplié par un facteur (0.85) alors que
loan_percent_income est resté calculé sur l'ancien revenu. Le script mesure :
  A. les lignes qui ne correspondent TOUJOURS pas à la formule retenue (après l'arrondi corrigé) ;
  B. parmi elles, combien sont expliquées par « revenu / 0.85 » ;
  C. leur profil (objet du prêt, défaut) comparé au reste du jeu ;
  D. des doublons quasi identiques (risque de fuite entre TRAIN et TEST).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features import FORMULAS, ACTIVE_FORMULAS

SCALE = 0.85


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/processed/donnees_prets_features.csv"))
    p.add_argument("--out", type=Path, default=Path("data/reports/income_scaling_suspects.csv"))
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    d = df[df[["loan_amnt", "person_income"]].notna().all(axis=1) & (df["person_income"] > 0)].copy()
    fn = FORMULAS["loan_percent_income"][ACTIVE_FORMULAS["loan_percent_income"]]
    d["recomputed"] = fn(d["loan_amnt"], d["person_income"], d["loan_int_rate"])
    d["mismatch"] = (d["loan_percent_income"] - d["recomputed"]).abs() >= 1e-6
    bad, good = d[d.mismatch].copy(), d[~d.mismatch]
    print(f"A. Lignes comparées : {len(d)} | encore écartées : {len(bad)} ({len(bad) / len(d):.2%})")
    if bad.empty:
        print("   Aucune ligne écartée : rien à analyser."); return

    # B. hypothèse revenu x 0.85
    alt = fn(bad["loan_amnt"], bad["person_income"] / SCALE, bad["loan_int_rate"])
    bad["explained_by_scale"] = (alt - bad["loan_percent_income"]).abs() < 1e-6
    print(f"\nB. Expliquées par « revenu / {SCALE} » : {bad.explained_by_scale.sum()} sur {len(bad)} "
          f"({bad.explained_by_scale.mean():.1%})")
    print(f"   Revenus multiples de 850 (= 0.85 x un nombre rond) : "
          f"{(bad.person_income % 850 == 0).mean():.1%} des lignes écartées vs "
          f"{(good.person_income % 850 == 0).mean():.1%} des autres")

    # C. profil
    print("\nC. Profil des lignes écartées vs les autres")
    print(f"   taux de défaut : écartées={bad.loan_status.mean():.1%} | autres={good.loan_status.mean():.1%}")
    print("   objet du prêt (écartées) :")
    print((bad.loan_intent.value_counts(normalize=True).head(3) * 100).round(1).to_string().replace("\n", "\n     "))
    print("   objet du prêt (autres) :")
    print((good.loan_intent.value_counts(normalize=True).head(3) * 100).round(1).to_string().replace("\n", "\n     "))
    print(f"   cb_person_cred_hist_length : écartées médiane={bad.cb_person_cred_hist_length.median():.0f} | "
          f"autres={good.cb_person_cred_hist_length.median():.0f}")

    # D. quasi-doublons (mêmes caractéristiques du prêt, historique de crédit différent)
    key = ["person_income", "loan_amnt", "loan_int_rate", "loan_grade", "loan_intent", "person_home_ownership"]
    dup_all = d.duplicated(subset=key, keep=False)
    print("\nD. Quasi-doublons (même revenu, montant, taux, note, objet, logement)")
    print(f"   parmi les lignes écartées : {dup_all[bad.index].mean():.1%} | parmi les autres : {dup_all[good.index].mean():.1%}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    bad.sort_values("loan_percent_income").to_csv(a.out, index=False)
    print(f"\nLignes suspectes écrites dans : {a.out}")


if __name__ == "__main__":
    main()
