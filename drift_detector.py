import pandas as pd
import numpy as np
from logger_decisions import charger_log
from extraire_depuis_qdrant import extraire_dataset

def calculer_statistiques(df: pd.DataFrame, colonne: str) -> dict:
    """Calcule moyenne et écart-type d'une colonne."""
    return {
        "moyenne": df[colonne].mean(),
        "std":     df[colonne].std(),
        "min":     df[colonne].min(),
        "max":     df[colonne].max(),
    }

def detecter_drift(seuil_alerte: float = 0.25) -> dict:
    """
    Compare les données récentes (log) avec les données
    d'entraînement (Qdrant).

    seuil_alerte : si la moyenne dévie de plus de 25%,
                   on considère qu'il y a drift.

    Retourne un dict avec le statut de chaque feature.
    """
    log      = charger_log()
    reference = extraire_dataset()

    if len(log) < 5:
        return {"statut": "INSUFFISANT",
                "message": f"Seulement {len(log)} décisions loggées. "
                            "Minimum 5 requis pour analyser le drift.",
                "alertes": []}

    features_a_surveiller = {
        "montant": ("montant", "montant"),   # (col dans log, col dans référence)
        "taux":    ("taux",    "taux"),
        "salaire": ("salaire", "revenu"),
        "age":     ("age",     "age"),
    }

    alertes  = []
    rapports = {}

    for nom, (col_log, col_ref) in features_a_surveiller.items():
        if col_log not in log.columns or col_ref not in reference.columns:
            continue

        stats_ref = calculer_statistiques(reference, col_ref)
        stats_rec = calculer_statistiques(log,       col_log)

        moyenne_ref = stats_ref["moyenne"]
        moyenne_rec = stats_rec["moyenne"]

        # Deviation relative
        if moyenne_ref != 0:
            deviation = abs(moyenne_rec - moyenne_ref) / abs(moyenne_ref)
        else:
            deviation = 0

        en_drift = deviation > seuil_alerte

        rapports[nom] = {
            "moyenne_reference": round(moyenne_ref, 2),
            "moyenne_recente":   round(moyenne_rec, 2),
            "deviation_pct":     round(deviation * 100, 1),
            "en_drift":          en_drift,
        }

        if en_drift:
            alertes.append(
                f"⚠️ DRIFT sur '{nom}' : "
                f"référence={moyenne_ref:.1f}, "
                f"récent={moyenne_rec:.1f} "
                f"(+{deviation*100:.1f}%)"
            )

    statut = "ALERTE" if alertes else "OK"

    return {
        "statut":  statut,
        "alertes": alertes,
        "rapport": rapports,
        "nb_decisions_recentes": len(log),
    }

def verifier_distribution_decisions(log: pd.DataFrame) -> dict:
    """
    Surveille si le taux d'approbation change anormalement.
    Un modèle qui approuve soudainement 95% des dossiers est suspect.
    """
    if len(log) < 10:
        return {"statut": "INSUFFISANT"}

    counts = log['decision'].value_counts(normalize=True) * 100
    taux_approbation = counts.get('APPROUVÉ', 0)

    alerte = None
    if taux_approbation > 85:
        alerte = f"⚠️ Taux d'approbation anormalement élevé : {taux_approbation:.1f}%"
    elif taux_approbation < 15:
        alerte = f"⚠️ Taux d'approbation anormalement bas : {taux_approbation:.1f}%"

    return {
        "taux_approbation":  round(taux_approbation, 1),
        "distribution":      counts.to_dict(),
        "alerte":            alerte,
    }