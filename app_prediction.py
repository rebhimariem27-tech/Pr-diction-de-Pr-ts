import streamlit as st
import plotly.express as px
import pandas as pd
from serveur_mcp import chercher_prets_similaires

# --- CONFIGURATION PRO ---
st.set_page_config(page_title="FinPredict AI Pro", layout="wide")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0F172A; color: #F8FAFC; }
    .metric-card { background-color: #020617; border-left: 5px solid #06B6D4; border-radius: 12px; padding: 20px; margin-bottom: 15px; }
    .stButton>button { background-color: #2563EB !important; color: white !important; border-radius: 8px !important; width: 100%; height: 3em;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("# 🏦 FinPredict <span style='color:#06B6D4;'>AI Pro</span>", unsafe_allow_html=True)
st.write("Analyse basée sur le dataset réel **HMEQ**")

col_form, col_res = st.columns([1, 1.8])

with col_form:
    st.subheader("📝 Dossier de demande")
    montant = st.number_input("Mont_Pret (LOAN)", value=20000)
    job = st.selectbox("Métier (JOB)", ["Office", "Sales", "Mgr", "ProfExe", "Self", "Other"])
    incidents = st.slider("Incidents de crédit (DEROG)", 0, 10, 0)
    valeur_bien = st.number_input("Valeur du bien immobilier (VALUE)", value=50000)
    
    # Simulation de l'ancien score de salaire par un ratio
    salaire_estime = valeur_bien / 2 # Hypothèse simple pour le calcul de ratio
    
    analyser = st.button("LANCER L'ANALYSE")

if analyser:
    # On adapte l'appel à notre outil pour inclure les nouveaux détails
    # On réutilise notre fonction en lui passant les nouvelles infos dans le texte
    resultats = chercher_prets_similaires(int(salaire_estime), int(montant), int(100 - incidents*10))
    
    nb_ok = resultats.count("Rentable")
    nb_nok = resultats.count("Non Rentable")
    total = nb_ok + nb_nok
    profit_pct = (nb_ok / total * 100) if total > 0 else 0

    with col_res:
        # LOGIQUE DE SÉCURITÉ : Si trop d'incidents, on bloque même si Qdrant est optimiste
        if incidents > 2:
            profit_pct = profit_pct * 0.3
            decision = "REJET (INCIDENTS TROP ÉLEVÉS)"
            color = "#F87171"
        elif profit_pct > 60:
            decision = "APPROUVÉ"
            color = "#06B6D4"
        else:
            decision = "RISQUÉ"
            color = "#FBBF24"

        st.markdown(f"""<div class="metric-card" style="border-left-color: {color};">
            <p style='color:gray;'>Score de Profitabilité</p>
            <h2 style='color:{color} !important;'>{profit_pct:.1f}% - {decision}</h2>
        </div>""", unsafe_allow_html=True)

        fig = px.pie(values=[profit_pct, 100-profit_pct], names=['Profit', 'Risque'], hole=0.7,
                     color_discrete_sequence=[color, '#1E293B'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=False, font_color="white")
        st.plotly_chart(fig)

        with st.chat_message("assistant"):
            st.write(f"Analyse terminée pour un profil **{job}**. "
                     f"La base Qdrant montre que les profils avec **{incidents} incidents** sont historiquement délicats.")