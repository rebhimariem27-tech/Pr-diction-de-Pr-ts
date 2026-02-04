import matplotlib.pyplot as plt
from serveur_mcp import chercher_prets_similaires # On importe l'outil créé avant

def agent_analyste(salaire, montant, score):
    print(f"🔍 Agent Analyste : Recherche de dossiers pour un prêt de {montant}€...")
    # Il utilise l'outil MCP pour interroger Qdrant
    donnees = chercher_prets_similaires(salaire, montant, score)
    return donnees

def agent_decideur(donnees_historiques):
    print("🧠 Agent Décideur : Analyse approfondie...")
    
    # On compte précisément les mots clés dans la réponse de l'analyste
    nb_non_rentable = donnees_historiques.count("Non Rentable")
    nb_rentable = donnees_historiques.count("Rentable")
    
    total = nb_rentable + nb_non_rentable
    
    if total == 0:
        return 50, "Aucune donnée trouvée."
    
    # Calcul du pourcentage
    pourcentage_profit = (nb_rentable / total) * 100
    
    justification = f"Sur {total} cas proches : {nb_rentable} succès et {nb_non_rentable} échecs."
    return pourcentage_profit, justification

def afficher_resultat(pourcentage, justification):
    # Création du Pie Chart (Graphique en camembert)
    labels = ['Profit Potentiel', 'Risque de Perte']
    tailles = [pourcentage, 100 - pourcentage]
    colors = ['#66b3ff', '#ff9999']

    plt.figure(figsize=(7, 7))
    plt.pie(tailles, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f"Prédiction de Profitabilité\n{justification}")
    plt.show()

# --- SIMULATION D'UNE UTILISATION ---
if __name__ == "__main__":
    # 1. On entre les données d'un NOUVEAU client
    nouveau_client = {"salaire": 200, "montant": 12000, "score": 650}
    
    # 2. L'Agent Analyste travaille
    resultats_bruts = agent_analyste(**nouveau_client)
    
    # 3. L'Agent Décideur analyse les résultats de l'analyste
    score_final, texte_justification = agent_decideur(resultats_bruts)
    
    # 4. On affiche le résultat final
    print(f"✅ Résultat : {score_final}% de profitabilité.")
    afficher_resultat(score_final, texte_justification)