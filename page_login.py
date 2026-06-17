import streamlit as st
import time
from auth_service import (
    verifier_mot_de_passe,
    generer_qr_code,
    lancer_verification_faciale
)
from session_manager import connecter

def afficher_login():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0F172A; }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation état
    if "login_etape" not in st.session_state:
        st.session_state.login_etape     = 1
        st.session_state.manager_id_tmp  = ""
        st.session_state.code_confiance  = None
        st.session_state.face_status     = ""
        st.session_state.face_progress   = 0

    st.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <h1 style='color:#06B6D4;'>🏦 FinPredict AI Pro</h1>
        <p style='color:#94A3B8;'>Accès sécurisé — Manager</p>
    </div>
    """, unsafe_allow_html=True)

    etape = st.session_state.login_etape
    st.progress((etape - 1) / 3, text=f"Étape {etape} / 3")
    st.write("")

    # ── ÉTAPE 1 : MOT DE PASSE ──────────────────────────────────
    if etape == 1:
        st.markdown("### 🔑 Étape 1 — Identifiants")

        manager_id = st.text_input("Identifiant manager", placeholder="manager_01")
        mdp        = st.text_input("Mot de passe", type="password")

        if st.button("Valider →", use_container_width=True):
            if not manager_id or not mdp:
                st.error("Remplissez tous les champs.")
            elif verifier_mot_de_passe(mdp):
                st.session_state.manager_id_tmp = manager_id
                st.session_state.login_etape    = 2
                st.rerun()
            else:
                st.error("❌ Identifiants incorrects.")
                _logger_echec(manager_id, "ECHEC_MDP")

    # ── ÉTAPE 2 : QR CODE + TRUST CODE ─────────────────────────
    elif etape == 2:
        st.markdown("### 📱 Étape 2 — Code de confiance")

        manager_id = st.session_state.manager_id_tmp

        # Génération unique du QR Code (votre generate_and_upload_qr)
        if "qr_bytes" not in st.session_state:
            with st.spinner("Génération du QR Code..."):
                code, qr_bytes = generer_qr_code(manager_id)
                st.session_state.code_confiance = code
                st.session_state.qr_bytes       = qr_bytes

        # Affichage QR Code
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(st.session_state.qr_bytes, width=220)

        st.info(
            "Scannez ce QR Code avec votre mobile. "
            "L'URL pointe vers votre Lambda AWS `VerifyMFATask`. "
            "Entrez ensuite le code affiché ici."
        )

        # Affichage du code pour saisie manuelle
        code_saisi = st.text_input(
            "Code de confiance (format TR-XXX-XXX)",
            placeholder="TR-ABC-123",
            max_chars=10
        ).upper().strip()

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Retour"):
                for k in ["qr_bytes", "code_confiance", "mfa_task_id"]:
                    st.session_state.pop(k, None)
                st.session_state.login_etape = 1
                st.rerun()
        with col_next:
            if st.button("Valider →", use_container_width=True):
                if code_saisi == st.session_state.code_confiance:
                    st.session_state.pop("qr_bytes", None)
                    st.session_state.login_etape = 3
                    st.rerun()
                else:
                    st.error(
                        f"❌ Code incorrect. "
                        f"Attendu : {st.session_state.code_confiance}"
                    )

    # ── ÉTAPE 3 : ROTATION VISAGE ───────────────────────────────
    elif etape == 3:
        st.markdown("### 👁️ Étape 3 — Vérification faciale")

        st.info("""
        **Instructions :**
        1. Placez-vous face à la caméra
        2. Attendez d'être identifié
        3. Suivez les instructions de rotation (gauche, droite, haut, bas)
        4. Maintenez chaque pose **3 secondes**

        Le système effectue **2 défis** aléatoires.
        """)

        # Affichage du statut en temps réel
        status_placeholder  = st.empty()
        progress_placeholder = st.empty()

        col_back, col_auth = st.columns(2)
        with col_back:
            if st.button("← Retour"):
                st.session_state.login_etape = 2
                st.rerun()

        with col_auth:
            if st.button("🎥 Lancer la vérification faciale",
                          use_container_width=True):

                manager_id = st.session_state.manager_id_tmp

                with st.spinner(
                    "Vérification en cours — "
                    "suivez les instructions devant la caméra..."
                ):
                    # Lance authenticate_sequence() headless
                    # La caméra s'ouvre en arrière-plan
                    resultat = lancer_verification_faciale(manager_id)

                if resultat["succes"]:
                    # Ouvrir la session JWT
                    connecter(manager_id)
                    st.success(
                        f"✅ Authentification réussie — "
                        f"Bienvenue {manager_id}"
                    )
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {resultat['message']}")
                    _logger_echec(manager_id, "ECHEC_VISAGE")

        # Mise à jour statut temps réel
        if st.session_state.get("face_status"):
            status_placeholder.info(
                f"🎥 {st.session_state.face_status}"
            )
        if st.session_state.get("face_progress", 0) > 0:
            progress_placeholder.progress(
                st.session_state.face_progress / 100
            )

    return False

def _logger_echec(manager_id: str, raison: str):
    """Log des tentatives échouées pour audit."""
    import csv
    from datetime import datetime
    with open("tentatives_auth.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(), manager_id, raison
        ])