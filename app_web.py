import streamlit as st
import plotly.express as px
import pandas as pd
from serveur_mcp import chercher_prets_similaires
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

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
    salaire = st.number_input("Salaire Annuel (€)", min_value=0, value=45000)
    montant = st.number_input("Montant du prêt (€)", min_value=0, value=10000)
    score = st.slider("Score de Crédit", 0, 1000, 700)
    
    st.write("")
    analyser = st.button("LANCER L'ANALYSE")

# --- LOGIQUE DE PRÉDICTION ---
if analyser:
    # 1. AGENT MATHÉMATIQUE (Vérification des limites logiques)
    # Calcul du ratio d'endettement (Prêt / Salaire)
    ratio = montant / (salaire if salaire > 0 else 1)
    est_absurde = False
    message_alerte = ""

    if ratio > 15: # Si le prêt dépasse 15x le salaire annuel
        est_absurde = True
        message_alerte = "Montant du prêt disproportionné par rapport au salaire."
    if score < 150:
        est_absurde = True
        message_alerte = "Score de crédit critique."

    # 2. AGENT VECTORIEL (Recherche Qdrant)
    try:
        resultats_qdrant = chercher_prets_similaires(salaire, montant, score)
        nb_ok = resultats_qdrant.count("Rentable")
        nb_nok = resultats_qdrant.count("Non Rentable")
        total_sim = nb_ok + nb_nok
        
        # Probabilité brute de Qdrant
        probabilite_brute = (nb_ok / total_sim * 100) if total_sim > 0 else 50
        
        # 3. SYNTHÈSE (A2A Hybrid Logic)
        if est_absurde:
            # On écrase le résultat de l'IA par une punition mathématique
            score_final = probabilite_brute * 0.05 # On réduit à presque rien
            decision = "REJET AUTOMATIQUE"
            couleur_decision = "#F87171" # Rouge
            explication = f"ALERTE : {message_alerte} Malgré des dossiers historiquement proches parfois positifs, les chiffres actuels présentent un risque mathématique majeur."
        else:
            score_final = probabilite_brute
            decision = "APPROUVÉ" if score_final > 65 else "À RÉVISER"
            couleur_decision = "#06B6D4" if score_final > 65 else "#FBBF24"
            explication = f"Analyse basée sur {total_sim} dossiers similaires trouvés dans la base Qdrant. Le profil semble cohérent avec les politiques de prêt habituelles."

        # --- AFFICHAGE DES RÉSULTATS ---
        with col_res:
            st.markdown("### 📈 Diagnostic Final")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <p>Indice de Confiance</p>
                    <h2>{score_final:.1f}%</h2>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card" style="border-left-color: {couleur_decision};">
                    <p>Décision Système</p>
                    <h2 style="color:{couleur_decision} !important;">{decision}</h2>
                </div>""", unsafe_allow_html=True)

            # Graphique Donut
            df_plot = pd.DataFrame({"Cat": ["Profit", "Risque"], "Val": [score_final, 100 - score_final]})
            fig = px.pie(df_plot, values='Val', names='Cat', hole=0.75, color_discrete_sequence=[couleur_decision, '#1E293B'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=0,b=0,l=0,r=0),
                              annotations=[dict(text=f"{int(score_final)}%", x=0.5, y=0.5, font_size=25, showarrow=False, font_color="white")])
            st.plotly_chart(fig, use_container_width=True)

            # Chatbot info
            with st.chat_message("assistant"):
                st.write(f"**Analyse de l'expert :** {explication}")

            with st.expander("Détails techniques (MCP / Qdrant)"):
                st.write(resultats_qdrant)
                
    except Exception as e:
        st.error(f"Erreur de connexion avec Qdrant : {e}")
        st.info("Assurez-vous que Docker tourne et que vous avez lancé l'ingestion.")
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