"""Source UNIQUE de vérité pour les variables dérivées et les règles métier.

Utilisé par l'API (inférence), le script de vérification et, idéalement, par
le pipeline d'entraînement / preprocessing. Ainsi training et serving ne
peuvent plus diverger.

!! AVANT LA MISE EN PRODUCTION !!
Lancez `python check_derived_features.py` : il compare chaque formule candidate
aux colonnes réelles de data/processed/donnees_prets_features.csv et indique
laquelle reproduit exactement les données d'entraînement. Reportez le nom
gagnant dans ACTIVE_FORMULAS ci-dessous.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable

import numpy as np


def _half_up_2(value):
    """Arrondi à 2 décimales, demi vers le haut, sur la valeur binaire EXACTE du flottant.
    C'est la règle qui a produit le CSV (ex. 0.125 -> 0.13, mais 0.075 -> 0.07 car 0.075 est
    stocké comme 0.07499999...). np.round / round() arrondissent 0.125 en 0.12 : ils divergent."""
    if hasattr(value, "map"):  # pandas.Series
        return value.map(_half_up_2)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return value
    return float(Decimal(float(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# Chaque formule fonctionne sur des scalaires ET sur des pandas.Series.
FORMULAS: dict[str, dict[str, Callable]] = {
    "loan_percent_income": {
        "exact": lambda amnt, income, rate: amnt / income,
        "rounded_2": lambda amnt, income, rate: np.round(amnt / income, 2),
        "half_up_2": lambda amnt, income, rate: _half_up_2(amnt / income),
    },
    "interest_income_ratio": {
        "amnt_x_rate_pct_over_income": lambda amnt, income, rate: amnt * rate / 100.0 / income,
        "amnt_x_rate_pct_over_income_half": lambda amnt, income, rate: amnt * rate / 100.0 / income / 2.0,
        "amnt_x_rate_over_income": lambda amnt, income, rate: amnt * rate / income,
    },
}

# Confirmé avec check_derived_features.py sur le CSV d'entraînement :
#  - interest_income_ratio : correspondance EXACTE.
#  - loan_percent_income   : montant/revenu arrondi à 2 décimales, DEMI VERS LE HAUT sur la valeur
#    binaire exacte ("half_up_2"). np.round (demi pair) donnait 0.12 là où le CSV a 0.13.
#    Les lignes restantes qui ne correspondent pas viennent des données sources (cf.
#    inspect_income_scaling.py), pas d'une formule.
ACTIVE_FORMULAS = {
    "loan_percent_income": "half_up_2",
    "interest_income_ratio": "amnt_x_rate_pct_over_income",
}

DERIVED_FEATURES = tuple(ACTIVE_FORMULAS)


def _is_missing(*values) -> bool:
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)


def derive_features(loan_amnt, person_income, loan_int_rate) -> dict[str, float | None]:
    """Calcule les variables dérivées. Renvoie None si une entrée manque
    (l'imputer du pipeline traitera alors la valeur manquante)."""
    if _is_missing(loan_amnt, person_income) or person_income <= 0:
        return {name: None for name in DERIVED_FEATURES}

    out: dict[str, float | None] = {}
    for name, formula_name in ACTIVE_FORMULAS.items():
        if name == "interest_income_ratio" and _is_missing(loan_int_rate):
            out[name] = None
            continue
        value = FORMULAS[name][formula_name](loan_amnt, person_income, loan_int_rate)
        out[name] = float(value)
    return out


def business_rule_violations(
    *, person_age, person_emp_length, cb_person_cred_hist_length,
    loan_amnt, person_income, loan_int_rate,
) -> list[str]:
    """Mêmes règles que preprocessing.remove_invalid_rows : une demande que
    l'entraînement aurait rejetée comme invalide ne doit pas être scorée."""
    problems: list[str] = []
    if person_age is not None:
        if person_emp_length is not None and person_emp_length > person_age:
            problems.append("person_emp_length ne peut pas dépasser person_age.")
        if cb_person_cred_hist_length is not None and cb_person_cred_hist_length > person_age:
            problems.append("cb_person_cred_hist_length ne peut pas dépasser person_age.")
    if loan_int_rate is not None and loan_int_rate > 100:
        problems.append("loan_int_rate doit être <= 100 (en pourcentage).")
    if not _is_missing(loan_amnt, person_income) and person_income > 0:
        if loan_amnt / person_income > 1:
            problems.append("loan_amnt / person_income dépasse 1 (loan_percent_income > 1 rejeté à l'entraînement).")
    return problems
