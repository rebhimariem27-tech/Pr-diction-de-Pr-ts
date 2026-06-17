import datetime
from datetime import datetime
import streamlit as st
import plotly.express as px
import pandas as pd
from serveur_mcp import chercher_prets_similaires
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import joblib
from explicabilite import expliquer_decision
from client_n8n import envoyer_decision_n8n
from logger_decisions import logger_decision
load_dotenv()
@st.cache_resource
def charger_modele():
    try:
        return joblib.load('artefacts_modele.pkl')
    except FileNotFoundError:
        return None

def predire(artefacts, salaire, montant, taux, grade, intention, age, anciennete, proprietaire, defaut_passe):
    model    = artefacts['model']
    encoders = artefacts['encoders']
    features = artefacts['features']

    data = {
        'montant':              montant,
        'taux':                 taux,
        'revenu':               salaire,
        'age':                  age,
        'anciennete':           anciennete,
        'ratio_revenu':         montant / (salaire + 1),
        'historique_credit':    7,
        'ratio_endettement':    montant / (salaire + 1),
        'effort_mensuel':       (montant * taux / 100 / 12) / (salaire / 12 + 1),
        'revenu_par_anciennete': salaire / (anciennete + 1),
        'grade_enc':            encoders['grade'].transform([grade])[0],
        'intention_enc':        encoders['intention'].transform([intention])[0],
        'proprietaire_enc':     encoders['proprietaire'].transform([proprietaire])[0],
        'defaut_passe_enc':     encoders['defaut_passe'].transform([defaut_passe])[0],
    }

    X = pd.DataFrame([data])[features]
    return round(model.predict_proba(X)[0][1] * 100, 1)
# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="FinPredict AI Pro", layout="wide", initial_sidebar_state="collapsed",page_icon="🏦")

# --- DESIGN CSS PERSONNALISÉ (PALETTE PRO SLATE & BLUE) ---
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: #0F172A;
        color: #F8FAFC;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    .stNumberInput div div input, .stSlider div {{
        background-color: #1E293B !important;
        color: white !important;
    }}
    .stButton>button {{
        background-color: #2563EB !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        height: 3.5rem !important;
        width: 100% !important;
    }}
    .metric-card {{
        background-color: #020617;
        border: 1px solid #1E293B;
        border-left: 5px solid #06B6D4;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }}
    .metric-card p {{ color: #94A3B8 !important; margin:0; font-size: 0.9rem; }}
    .metric-card h2 {{ color: #F8FAFC !important; margin:0; font-size: 1.8rem; }}
    h1, h2, h3 {{ color: #F8FAFC !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- EN-TÊTE ---
st.markdown("<h1 style='margin-bottom:0;'>FinPredict <span style='color:#06B6D4;'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8 !important;'>Analyse Hybride : Mathématique + Vectorielle (Qdrant)</p>", unsafe_allow_html=True)
st.write("") 

# --- MISE EN PAGE PRINCIPALE ---
col_form, col_spacer, col_res = st.columns([1, 0.1, 1.8])

with col_form:
    st.markdown("### 📝 Nouveau Dossier")

    # --- INFORMATIONS FINANCIÈRES ---
    st.markdown("**💰 Informations financières**")
    salaire = st.number_input("Revenu annuel (€)", min_value=0, value=45000, step=1000)
    montant = st.number_input("Montant du prêt (€)", min_value=0, value=10000, step=500)
    taux = st.number_input("Taux d'intérêt (%)", min_value=0.0, max_value=35.0, value=12.0, step=0.5)

    st.markdown("**📊 Profil de crédit**")
    grade= st.selectbox("Grade de crédit",
                        ["A", "B", "C", "D", "E", "F", "G"],
                          index=1,
                          help="A = meilleur profil, G = profil risqué")
    intention = st.selectbox("Intention du prêt", [
                             "PERSONAL", "EDUCATION", "MEDICAL",
                             "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                             help="Objet de la demande de prêt")
    defaut_passe = st.radio("Défaut de paiement passé ?",
                             ["N", "Y"],
                             horizontal=True,
                             help="Y = a déjà été en défaut")

    st.markdown("**👤 Profil personnel**")
    age        = st.slider("Âge", 18, 75, 35)
    anciennete = st.slider("Ancienneté emploi (années)", 0, 40, 5)
    proprietaire = st.selectbox("Statut immobilier",
                                 ["RENT", "OWN", "MORTGAGE", "OTHER"])

    st.markdown("**📧 Contact**")
    email_client = st.text_input("Email du client", value="client@exemple.com")

    st.write("")
    analyser = st.button("LANCER L'ANALYSE")
# --- LOGIQUE DE PRÉDICTION ---
if analyser:
    artefacts = charger_modele()

    if artefacts is None:
        st.error("Lancez d'abord : python model_ml.py")
    else:
        # Score ML (70% du poids)
        score_ml = predire(
            artefacts,
            salaire=salaire, montant=montant, taux=taux,
            grade=grade, intention=intention,
            age=age, anciennete=anciennete,
            proprietaire=proprietaire, defaut_passe=defaut_passe
        )

        # Score Qdrant (30% du poids — contexte historique)
        resultats_qdrant = chercher_prets_similaires(
            salaire=salaire,
            montant=montant,
            score=score_ml
        )

        nb_ok = resultats_qdrant.count("Rentable")
        nb_total = nb_ok + resultats_qdrant.count("Non Rentable")

        score_qdrant = (
            nb_ok / nb_total * 100
        ) if nb_total > 0 else 50

        score_final = round(
            0.7 * score_ml + 0.3 * score_qdrant,
            1
        )

        if score_final > 70:
            decision, couleur = "APPROUVÉ", "#06B6D4"
        elif score_final > 45:
            decision, couleur = "À RÉVISER", "#FBBF24"
        else:
            decision, couleur = "REJETÉ", "#F87171"

        # --- EXPLICATION SHAP ---
        st.markdown("### 🔍 Pourquoi cette décision ?")

        features_client = {
            'montant':               montant,
            'taux':                  taux,
            'revenu':                salaire,
            'age':                   age,
            'anciennete':            anciennete,
            'ratio_revenu':          montant / (salaire + 1),
            'historique_credit':     7,
            'ratio_endettement':     montant / (salaire + 1),
            'effort_mensuel':        (montant * taux / 100 / 12) / (salaire / 12 + 1),
            'revenu_par_anciennete': salaire / (anciennete + 1),
            'grade_enc':             artefacts['encoders']['grade'].transform([grade])[0],
            'intention_enc':         artefacts['encoders']['intention'].transform([intention])[0],
            'proprietaire_enc':      artefacts['encoders']['proprietaire'].transform([proprietaire])[0],
            'defaut_passe_enc':      artefacts['encoders']['defaut_passe'].transform([defaut_passe])[0],
        }

        explication = expliquer_decision(features_client)

        col_fav, col_def = st.columns(2)

        with col_fav:
            st.markdown("**✅ Facteurs favorables**")

            for f in explication['favorables']:
                noms = {
    'taux':                  'Taux d\'intérêt',
    'montant':               'Montant du prêt',
    'revenu':                'Revenu annuel',
    'ratio_endettement':     'Ratio d\'endettement',
    'effort_mensuel':        'Effort mensuel',
    'grade_enc':             'Grade de crédit',
    'defaut_passe_enc':      'Défaut passé',
    'ratio_revenu':          'Part prêt/revenu',
    'historique_credit':     'Historique crédit',
    'age':                   'Âge',
    'anciennete':            'Ancienneté emploi',
    'proprietaire_enc':      'Statut immobilier',
    'intention_enc':         'Intention du prêt',
    'revenu_par_anciennete': 'Revenu / ancienneté',
            }
                

                nom = noms.get(f['feature'], f['feature'])
                barre = min(int(abs(f['impact']) * 200), 100)

                st.markdown(
                    f"""
                    <div style="margin-bottom:8px;">
                        <span style="color:#94A3B8;">{nom}</span>
                        <div style="background:#1E293B; border-radius:4px; height:8px; margin-top:4px;">
                            <div style="background:#06B6D4; width:{barre}%; height:8px; border-radius:4px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_def:
            st.markdown("**⚠️ Facteurs limitants**")

            for f in explication['defavorables']:
                noms = {
                    'taux':  'Taux d\'intérêt',
    'montant':               'Montant du prêt',
    'revenu':                'Revenu annuel',
    'ratio_endettement':     'Ratio d\'endettement',
    'effort_mensuel':        'Effort mensuel',
    'grade_enc':             'Grade de crédit',
    'defaut_passe_enc':      'Défaut passé',
    'ratio_revenu':          'Part prêt/revenu',
    'historique_credit':     'Historique crédit',
    'age':                   'Âge',
    'anciennete':            'Ancienneté emploi',
    'proprietaire_enc':      'Statut immobilier',
    'intention_enc':         'Intention du prêt',
    'revenu_par_anciennete': 'Revenu / ancienneté',
                }

                nom = noms.get(f['feature'], f['feature'])
                barre = min(int(abs(f['impact']) * 200), 100)

                st.markdown(
                    f"""
                    <div style="margin-bottom:8px;">
                        <span style="color:#94A3B8;">{nom}</span>
                        <div style="background:#1E293B; border-radius:4px; height:8px; margin-top:4px;">
                            <div style="background:#F87171; width:{barre}%; height:8px; border-radius:4px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with st.chat_message("assistant"):
            st.write(
                f"**Analyse experte :** "
                f"{explication['explication']}"
            )

        # Calculs pour le rapport
        nb_ok = resultats_qdrant.count("Rentable")
        nb_total = nb_ok + resultats_qdrant.count("Non Rentable")

        ratio_rent = (
            nb_ok / nb_total * 100
        ) if nb_total > 0 else 50

        from generateur_rapport import generer_rapport_pdf

        pdf_bytes = generer_rapport_pdf(
            salaire=salaire,
            montant=montant,
            grade=grade,
            intention=intention,
            score_ml=score_ml,
            score_qdrant=score_qdrant,
            score_final=score_final,
            decision=decision,
            favorables=explication['favorables'],
            defavorables=explication['defavorables'],
            explication=explication['explication'],
            nb_similaires=nb_total,
            ratio_rentable=ratio_rent,
        )

        st.download_button(
            label="📄 Télécharger le rapport PDF",
            data=pdf_bytes,
            file_name=f"rapport_pret_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
        succes_n8n = envoyer_decision_n8n(
        salaire=salaire,
        montant=montant,
        grade=grade,
        taux=taux,
        intention=intention,
        age=age,
        anciennete=anciennete,
        proprietaire=proprietaire,
        defaut_passe=defaut_passe,
        score_ml=score_ml,
        score_final=score_final,
        decision=decision,
        explication=explication['explication'],
        email_client=email_client
    )

    if succes_n8n:
        st.success("✅ Workflow n8n déclenché — email et log envoyés")
    else:
        st.info("ℹ️  n8n non disponible — rapport PDF disponible en téléchargement")
    # Logger la décision automatiquement
    logger_decision(
        salaire=salaire, montant=montant, taux=taux,
        grade=grade, intention=intention,
        age=age, anciennete=anciennete,
        defaut_passe=defaut_passe,
        score_ml=score_ml, score_qdrant=score_qdrant,
        score_final=score_final, decision=decision
    )
# --- CONFIGURATION ---
client_ai = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1"
) 

# Initialisation du client Qdrant
qd_db = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
collection_name = "prets_reels"

st.title("🤖 Assistant Analyse de Prêts")

# --- FONCTION DE RECHERCHE ULTRA-ROBUSTE ---
def chercher_contexte(question):
    vector = embed_model.encode(question).tolist()
    
    try:
        # METHODE 1 : La plus récente
        if hasattr(qd_db, 'query_points'):
            results = qd_db.query_points(collection_name=collection_name, query=vector, limit=10).points
        # METHODE 2 : La classique
        elif hasattr(qd_db, 'search'):
            results = qd_db.search(collection_name=collection_name, query_vector=vector, limit=3)
        # METHODE 3 : Sécurité
        else:
            # Si aucune n'est trouvée, on affiche les méthodes disponibles pour comprendre
            metodes = [m for m in dir(qd_db) if not m.startswith('_')]
            return f"Erreur : Méthodes introuvables. Disponibles : {metodes[:5]}"

        contexte = ""
        for res in results:
            p = res.payload
            m = p.get('montant', 'N/A')
            g = p.get('grade', 'N/A')
            pr = p.get('profit', 'N/A')
            t = p.get('taux', 'N/A')
            d = p.get('description', 'N/A') # Le revenu est caché là-dedans !
            
            contexte += f"- Prêt: {m}€, Grade: {g}, Rentable: {pr}, Taux: {t}%. Infos: {d}\n"
        
        return contexte

    except Exception as e:
        return f"Erreur technique lors de la recherche : {str(e)}"

# --- INTERFACE DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez une question sur les dossiers de prêt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyse des dossiers en cours..."):
            contexte_extraite = chercher_contexte(prompt)
            
            # On donne le contexte à l'IA
            instruction = f"""
            Tu es un assistant expert en analyse de risques bancaires. 
            Voici des données extraites de notre base historique (les dossiers les plus pertinents pour la question) :
            {contexte_extraite}
            
            Informations importantes pour ton analyse :
            - Le champ 'Resultat' (ou Profit) : 1 signifie que le prêt a été rentable/accepté, 0 signifie qu'il a échoué ou a été refusé.
            - Le champ 'Score' : c'est le score de crédit (plus il est haut, plus le client est fiable).
            
            Instructions :
            1. Si l'utilisateur te dit juste 'Bonjour', réponds poliment et propose ton aide pour analyser les prêts.
            2. Si l'utilisateur pose une question, utilise les données fournies pour justifier ta réponse.
            3. Si aucune donnée n'est pertinente, réponds avec tes connaissances générales en banque.
            """

            response = client_ai.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            reponse_finale = response.choices[0].message.content
            st.markdown(reponse_finale)
        
    st.session_state.messages.append({"role": "assistant", "content": reponse_finale})