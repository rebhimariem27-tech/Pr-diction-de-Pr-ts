import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# 1. Connexion à Qdrant (qui tourne sur Docker)
client = QdrantClient(host="localhost", port=6333)

# 2. Charger le modèle d'IA qui transforme le texte en nombres (Vecteurs)
# Ce modèle est petit et rapide pour débuter
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Créer une "Collection" dans Qdrant (C'est comme une table dans une base de données)
collection_name = "prets_historiques"

# On supprime la collection si elle existe déjà pour recommencer à zéro
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 4. Lire ton fichier CSV
df = pd.read_csv('donnees_prets.csv')

# 5. Préparer les données pour Qdrant
points = []
for index, row in df.iterrows():
    # On crée une description textuelle du profil client
    texte_complet = f"Salaire: {row['Salaire_Annuel']}, Prêt: {row['Montant_Pret']}, Score: {row['Score_Credit']}. Justification: {row['Justification_Historique']}"
    
    # On transforme ce texte en vecteur (une liste de 384 nombres)
    vector = model.encode(texte_complet).tolist()
    
    # On prépare le "point" à insérer
    points.append({
        "id": index,
        "vector": vector,
        "payload": { # Les données réelles qu'on veut garder
            "salaire": row['Salaire_Annuel'],
            "montant": row['Montant_Pret'],
            "score": row['Score_Credit'],
            "profit": row['Resultat'],
            "justification": row['Justification_Historique']
        }
    })

# 6. Envoyer tout ça dans Qdrant !
client.upsert(collection_name=collection_name, points=points)

print("Succès ! Les données sont dans Qdrant.")