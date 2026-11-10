"""Explicabilité par prédiction (codes de raison), sans dépendance SHAP.

LightGBM calcule nativement des contributions de type SHAP (pred_contrib=True).
Le modèle de production est un CalibratedClassifierCV(cv=5) : on moyenne les
contributions des 5 pipelines de base, puis on regroupe les colonnes one-hot
sur leur variable d'origine.

Unité : contribution en log-odds au score brut LightGBM (AVANT calibration
sigmoïde). Le SIGNE et le CLASSEMENT sont fiables pour justifier une décision ;
la valeur absolue n'est pas une probabilité.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ContributionExplainer:
    def __init__(self, calibrated_model, feature_columns: list[str], categorical_features: list[str]):
        self.feature_columns = list(feature_columns)
        self.categorical = list(categorical_features)

        self.pipelines = []
        for cc in calibrated_model.calibrated_classifiers_:
            base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if base is None:
                raise TypeError("Impossible de retrouver le pipeline de base du calibrateur.")
            self.pipelines.append(base)

        pre = self.pipelines[0].named_steps["preprocessor"]
        self.origin = [self._origin_of(n) for n in pre.get_feature_names_out()]

    def _origin_of(self, transformed_name: str) -> str:
        name = transformed_name.split("__", 1)[-1]
        for cat in self.categorical:
            if name.startswith(cat + "_"):
                return cat
        if name not in self.feature_columns:
            raise ValueError(f"Colonne transformée inconnue : {transformed_name}")
        return name

    def explain(self, X: pd.DataFrame, top_n: int = 4) -> dict[str, Any]:
        """X : une seule ligne, colonnes = feature_columns."""
        per_fold = []
        for pipe in self.pipelines:
            Xt = pipe.named_steps["preprocessor"].transform(X)
            contrib = pipe.named_steps["model"].predict(Xt, pred_contrib=True)
            per_fold.append(np.asarray(contrib)[0, :-1])  # dernière colonne = biais
        mean_contrib = np.mean(per_fold, axis=0)

        grouped: dict[str, float] = {}
        for origin, value in zip(self.origin, mean_contrib):
            grouped[origin] = grouped.get(origin, 0.0) + float(value)

        row = X.iloc[0]
        factors = []
        for feature, value in grouped.items():
            raw = row[feature]
            raw = None if pd.isna(raw) else (raw.item() if hasattr(raw, "item") else raw)
            factors.append({
                "feature": feature,
                "value": raw,
                "contribution": round(value, 4),
                "effect": "increases_risk" if value > 0 else "decreases_risk",
            })
        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

        return {
            "method": "lightgbm_pred_contrib_mean_of_5_folds",
            "unit": "log-odds, raw LightGBM score before sigmoid calibration",
            "top_risk_factors": [f for f in factors if f["contribution"] > 0][:top_n],
            "top_protective_factors": [f for f in factors if f["contribution"] < 0][:top_n],
        }
