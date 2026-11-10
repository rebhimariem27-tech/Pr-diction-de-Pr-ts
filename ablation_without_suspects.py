"""Mesure l'effet des lignes suspectes (revenu x 0.85, tous MEDICAL, tous défauts) sur le modèle.

    python ablation_without_suspects.py

Utilise le MÊME split que la production (random_state=42, stratify) et la MÊME définition du
pipeline (importée de model_ml_final_pipeline.py). Trois mesures sur le jeu TEST :

  1. Modèle A (entraîné AVEC les suspects), évalué sur le TEST complet vs sur le TEST propre
     -> si les métriques baissent sur le TEST propre, elles étaient gonflées par les suspects.
  2. Modèle B (entraîné SANS les suspects), évalué sur le même TEST propre
     -> le modèle B est-il aussi bon, meilleur ou moins bon que A sur de vraies données ?
  3. Mémorisation : part des suspects du TEST que le modèle A refuse, comparée à la part des
     autres défauts MEDICAL qu'il refuse.

Ce script n'écrit rien dans data/production : il ne modifie pas votre modèle en service.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from features import ACTIVE_FORMULAS, FORMULAS
from model_ml_final_pipeline import (MODEL_FEATURES, PRODUCTION_THRESHOLD, TARGET,
                                     build_calibrated_pipeline)

SCALE = 0.85


def flag_suspects(df: pd.DataFrame) -> pd.Series:
    """Ligne dont loan_percent_income ne colle pas à la formule MAIS colle avec revenu/0.85."""
    fn = FORMULAS["loan_percent_income"][ACTIVE_FORMULAS["loan_percent_income"]]
    ok = df[["loan_amnt", "person_income"]].notna().all(axis=1) & (df["person_income"] > 0)
    flags = pd.Series(False, index=df.index)
    d = df[ok]
    mismatch = (d["loan_percent_income"] - fn(d["loan_amnt"], d["person_income"], d["loan_int_rate"])).abs() >= 1e-6
    alt = fn(d["loan_amnt"], d["person_income"] / SCALE, d["loan_int_rate"])
    explained = (alt - d["loan_percent_income"]).abs() < 1e-6
    flags.loc[d.index] = (mismatch & explained).to_numpy()
    return flags


def metrics(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {"n": len(y), "taux_defaut": float(np.mean(y)),
            "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "brier": brier_score_loss(y, proba),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0)}


def show(title: str, rows: dict) -> None:
    print(f"\n{title}")
    print(pd.DataFrame(rows).T.round(4).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/processed/donnees_prets_features.csv"))
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    suspects = flag_suspects(df)
    X, y = df[MODEL_FEATURES], df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    s_tr, s_te = suspects.loc[X_tr.index], suspects.loc[X_te.index]
    print(f"Suspects : {int(suspects.sum())} | dans TRAIN : {int(s_tr.sum())} | dans TEST : {int(s_te.sum())}")
    print(f"Seuil de production : {PRODUCTION_THRESHOLD}")

    print("\nEntraînement du modèle A (avec suspects)...")
    model_a = build_calibrated_pipeline().fit(X_tr, y_tr)
    print("Entraînement du modèle B (sans suspects)...")
    model_b = build_calibrated_pipeline().fit(X_tr[~s_tr], y_tr[~s_tr])

    clean = ~s_te
    pa_all, pa_clean = model_a.predict_proba(X_te)[:, 1], model_a.predict_proba(X_te[clean])[:, 1]
    pb_clean = model_b.predict_proba(X_te[clean])[:, 1]

    show("1 & 2. Métriques sur le TEST (seuil de production)", {
        "A sur TEST complet (suspects inclus)": metrics(y_te, pa_all, PRODUCTION_THRESHOLD),
        "A sur TEST propre": metrics(y_te[clean], pa_clean, PRODUCTION_THRESHOLD),
        "B sur TEST propre": metrics(y_te[clean], pb_clean, PRODUCTION_THRESHOLD),
    })

    # 3. mémorisation
    print("\n3. Mémorisation par le modèle A")
    med = (X_te["loan_intent"] == "MEDICAL")
    if s_te.any():
        refus_susp = float((pa_all[s_te.to_numpy()] >= PRODUCTION_THRESHOLD).mean())
        print(f"   Suspects du TEST refusés par A              : {refus_susp:.1%}  (n={int(s_te.sum())})")
    real_med_def = med & ~s_te & (y_te == 1)
    if real_med_def.any():
        refus_real = float((pa_all[real_med_def.to_numpy()] >= PRODUCTION_THRESHOLD).mean())
        print(f"   Autres défauts MEDICAL du TEST refusés par A : {refus_real:.1%}  (n={int(real_med_def.sum())})")
    clean_med = med & ~s_te
    print(f"   Probabilité moyenne prédite sur les prêts MEDICAL propres : "
          f"A={model_a.predict_proba(X_te[clean_med])[:, 1].mean():.3f} | "
          f"B={model_b.predict_proba(X_te[clean_med])[:, 1].mean():.3f} | "
          f"défaut observé={y_te[clean_med].mean():.3f}")
    print("\nLecture : si B égale A sur le TEST propre, les suspects n'apportent rien de réel au modèle.")
    print("Si A refuse presque tous les suspects mais bien moins les autres défauts MEDICAL, il a mémorisé l'artefact.")


if __name__ == "__main__":
    main()
