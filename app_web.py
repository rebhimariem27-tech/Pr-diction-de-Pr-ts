import streamlit as st
import matplotlib.pyplot as plt
from serveur_mcp import chercher_prets_similaires

# Configuration de la page
st.set_page_config(page_title="Banque IA - Analyse de Prêt", layout="wide")

st.title("🏦 Système Expert d'Analyse de Prêts")
st.write("Cet outil utilise **Qdrant** et **MCP** pour comparer le client avec des dossiers historiques.")

# Création de deux colonnes
col_form, col_graph = st.columns([1, 2])

with col_form:
    st.subheader("📝 Dossier Client")
    salaire = st.number_input("Salaire Annuel (€)", min_value=0, value=40000, step=1000)
    montant = st.number_input("Montant du Prêt (€)", min_value=0, value=10000, step=500)
    score = st.slider("Score de Crédit", 0, 1000, 650)
    
    analyser = st.button("Lancer l'Analyse", use_container_width=True)

if analyser:
    # 1. Appel de ton système
    resultats = chercher_prets_similaires(salaire, montant, score)
    
    # 2. Analyse des résultats
    nb_ok = resultats.count("Rentable")
    nb_nok = resultats.count("Non Rentable")
    total = nb_ok + nb_nok
    
    if total > 0:
        profit_pct = (nb_ok / total) * 100
    else:
        profit_pct = 0

    with col_graph:
        st.subheader("📊 Prédiction de Profitabilité")
        
        # Affichage du score
        st.metric("Score de Confiance", f"{profit_pct:.1f}%", 
                  delta="Favorable" if profit_pct > 50 else "Risqué",
                  delta_color="normal" if profit_pct > 50 else "inverse")

        # Graphique
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = ['#2ecc71', '#e74c3c'] # Vert et Rouge
        ax.pie([profit_pct, 100-profit_pct], 
               labels=['Profit Potentiel', 'Risque de Perte'], 
               autopct='%1.1f%%', 
               startangle=140, 
               colors=colors,
               wedgeprops={'edgecolor': 'white'})
        st.pyplot(fig)

        # Affichage des justifications
        st.info(f"💡 Analyse : Sur les dossiers les plus proches, nous avons trouvé {nb_ok} succès et {nb_nok} échecs.")
        with st.expander("Voir les dossiers historiques trouvés dans Qdrant"):
            st.text(resultats)