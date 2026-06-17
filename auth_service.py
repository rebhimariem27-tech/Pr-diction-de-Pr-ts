import os
import hashlib
import hmac
import streamlit as st
from datetime import datetime

# ── Import de VOS modules existants ──────────────────────────────
from security_utils import generate_and_upload_qr, MASTER_KEY_512
from face_service import (
    load_dataset_encodings,
    authenticate_sequence
)

# ── Configuration ─────────────────────────────────────────────────
MON_BUCKET   = os.getenv("AWS_BUCKET_NAME", "pfe-face-auth-storage-votre-nom")
SECRET_KEY   = os.getenv("JWT_SECRET", "finpredict_secret_changez_moi")

# Hash du mot de passe manager — générez-le une fois :
# python -c "import hmac,hashlib; print(hmac.new(b'finpredict_secret', b'votre_mdp', hashlib.sha512).hexdigest())"
MANAGER_PASSWORD_HASH = os.getenv("MANAGER_PASSWORD_HASH", "REMPLACEZ_PAR_LE_HASH")

def verifier_mot_de_passe(mdp_saisi: str) -> bool:
    """Étape 1 — HMAC-512, cohérent avec security_utils."""
    hash_saisi = hmac.new(
        SECRET_KEY.encode(),
        mdp_saisi.encode(),
        hashlib.sha512
    ).hexdigest()
    return hmac.compare_digest(hash_saisi, MANAGER_PASSWORD_HASH)

def generer_qr_code(manager_id: str) -> tuple[str, bytes]:
    """
    Étape 2 — Appelle directement generate_and_upload_qr()
    de votre security_utils.py.
    Retourne (trust_code, qr_bytes)
    """
    import uuid
    task_id = f"FINPREDICT-{manager_id}-{int(__import__('time').time())}"

    # Stocke le task_id dans la session pour l'étape 3
    st.session_state["mfa_task_id"] = task_id

    trust_code, qr_bytes = generate_and_upload_qr(
        user_name=manager_id,
        bucket_name=MON_BUCKET,
        task_id=task_id
    )
    return trust_code, qr_bytes

def lancer_verification_faciale(manager_id: str) -> dict:
    """
    Étape 3 — Lance authenticate_sequence() de face_service.py
    en mode headless (pas de PyQt6, pas de fenêtre OpenCV).
    Les callbacks mettent à jour st.session_state.
    """
    encodings, names, poses = load_dataset_encodings()

    if len(encodings) == 0:
        return {
            "succes":  False,
            "message": "Dataset vide — enregistrez d'abord le manager."
        }

    if manager_id not in names:
        return {
            "succes":  False,
            "message": f"Manager '{manager_id}' non enregistré dans le dataset."
        }

    resultat = {"succes": False, "message": "", "confiance": 0}

    def on_status(txt: str):
        st.session_state["face_status"] = txt

    def on_progress(val: int):
        st.session_state["face_progress"] = val

    # authenticate_sequence tourne en bloquant jusqu'à succès ou échec
    # total_challenges=2 comme dans votre config TOTAL_CHALLENGES
    try:
        result = authenticate_sequence(
            known_encodings=encodings,
            known_names=names,
            known_poses=poses,
            total_challenges=2,
            challenge_timeout=12,
            status_callback=on_status,
            progress_callback=on_progress,
        )

        if result.get("result") == "success":
            resultat["succes"]    = True
            resultat["confiance"] = 0.99
            resultat["message"]   = f"Authentifié : {result.get('user')}"
        else:
            resultat["message"] = "Authentification échouée ou annulée."

    except Exception as e:
        resultat["message"] = f"Erreur : {str(e)}"

    return resultat