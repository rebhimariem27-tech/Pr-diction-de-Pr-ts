import csv
import os
from datetime import datetime

LOG_FILE = "decisions_log.csv"

COLONNES = [
    "timestamp", "salaire", "montant", "taux", "grade",
    "intention", "age", "anciennete", "defaut_passe",
    "score_ml", "score_qdrant", "score_final", "decision"
]

def init_log():
    """Crée le fichier CSV avec les en-têtes s'il n'existe pas."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLONNES)
            writer.writeheader()

def logger_decision(
    salaire, montant, taux, grade, intention,
    age, anciennete, defaut_passe,
    score_ml, score_qdrant, score_final, decision
):
    """Enregistre une décision dans le log."""
    init_log()
    ligne = {
        "timestamp":   datetime.now().isoformat(),
        "salaire":     salaire,
        "montant":     montant,
        "taux":        taux,
        "grade":       grade,
        "intention":   intention,
        "age":         age,
        "anciennete":  anciennete,
        "defaut_passe": defaut_passe,
        "score_ml":    round(score_ml, 2),
        "score_qdrant": round(score_qdrant, 2),
        "score_final": round(score_final, 2),
        "decision":    decision
    }
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES)
        writer.writerow(ligne)

def charger_log():
    """Retourne le log complet sous forme de DataFrame."""
    import pandas as pd
    init_log()
    df = pd.read_csv(LOG_FILE)
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df