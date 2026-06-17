import jwt
import hashlib
import hmac
import os
import streamlit as st
from datetime import datetime, timedelta

SECRET_KEY    = os.getenv("JWT_SECRET", "finpredict_secret_changez_moi")
SESSION_DUREE = 30  # minutes

def generer_token(manager_id: str) -> str:
    """Génère un JWT signé valable 30 minutes."""
    payload = {
        "manager_id": manager_id,
        "exp":        datetime.utcnow() + timedelta(minutes=SESSION_DUREE),
        "iat":        datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verifier_token(token: str) -> dict | None:
    """
    Vérifie le JWT. Retourne le payload si valide,
    None si expiré ou invalide.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def verifier_mot_de_passe(mot_de_passe: str, hash_stocke: str) -> bool:
    """
    Compare le mot de passe saisi avec le hash stocké.
    Utilise HMAC-SHA512 — cohérent avec votre projet.
    """
    hash_saisi = hmac.new(
        SECRET_KEY.encode(),
        mot_de_passe.encode(),
        hashlib.sha512
    ).hexdigest()
    return hmac.compare_digest(hash_saisi, hash_stocke)

def est_authentifie() -> bool:
    """Vérifie si le manager a une session active dans Streamlit."""
    token = st.session_state.get("auth_token")
    if not token:
        return False
    payload = verifier_token(token)
    return payload is not None

def connecter(manager_id: str):
    """Ouvre une session Streamlit après authentification réussie."""
    token = generer_token(manager_id)
    st.session_state["auth_token"]  = token
    st.session_state["manager_id"]  = manager_id
    st.session_state["auth_time"]   = datetime.now().isoformat()

def deconnecter():
    """Ferme la session."""
    for key in ["auth_token", "manager_id", "auth_time"]:
        st.session_state.pop(key, None)