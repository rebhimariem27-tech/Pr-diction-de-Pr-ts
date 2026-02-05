import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# On charge le nouveau dataset (limité à 1000 pour éviter les lenteurs)
df = pd.read_csv('donnees_prets_reels.csv').head(1000)
# On remplit les cases vides (NaN) par 0 pour ne pas faire planter l'IA
df = df.fillna(0)

collection_name = "prets_reels"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Encode en batch pour plus de rapidité
descriptions = []
for index, row in df.iterrows():
    texte_description = (
        f"Prêt de {row['loan_amnt']}€. Grade: {row['loan_grade']}. "
        f"Intention: {row['loan_intent']}. Taux: {row['loan_int_rate']}%. "
        f"Revenu: {row['person_income']}€. Âge: {row['person_age']} ans."
    )
    descriptions.append(texte_description)

print("Encodage des descriptions...")
vectors = model.encode(descriptions, batch_size=32, show_progress_bar=True)

points = []
for index, row in df.iterrows():
    profit_reel = 1 if row['loan_status'] == 0 else 0
    
    points.append(PointStruct(
        id=index,
        vector=vectors[index].tolist(),
        payload={
            "montant": row['loan_amnt'],
            "grade": row['loan_grade'],
            "profit": profit_reel,
            "taux": row['loan_int_rate'],
            "description": descriptions[index]
        }
    ))

client.upsert(collection_name=collection_name, points=points)
print(f"✅ {len(points)} dossiers réels importés dans Qdrant !")