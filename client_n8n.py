import requests
import os
from datetime import datetime

# Mettez cette URL dans votre .env après création du webhook n8n
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "http://localhost:5678/webhook/decision-pret"
)

def envoyer_decision_n8n(
    salaire, montant, grade, taux, intention, age, anciennete, proprietaire, defaut_passe, score_ml, score_final, decision, explication, email_client) -> bool:
    """
    Envoie la décision au webhook n8n pour déclencher
    les automatisations (email, Google Sheets, etc.)
    Retourne True si succès, False sinon.
    """
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "salaire": salaire,
        "montant": montant,
        "grade": grade,
        "taux": taux,
        "intention": intention,
        "age": age,
        "anciennete": anciennete,
        "proprietaire": proprietaire,
        "defaut_passe": defaut_passe,
        "score_ml": score_ml,
        "score_final": score_final,
        "decision": decision,
        "explication": explication,
        "email_client": email_client
    }

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=5,
            headers={"Content-Type": "application/json"}
        )
        # Affiche le code de retour dans le terminal pour débugger
        print(f"DEBUG N8N: Status {response.status_code}") 
        return response.status_code == 200
    except Exception as e:
        # Affiche l'erreur réelle dans le terminal
        print(f"ERREUR CONNEXION N8N: {e}") 
        return False