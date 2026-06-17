import pandas as pd
from qdrant_client import QdrantClient

def extraire_dataset() -> pd.DataFrame:
    client = QdrantClient(host="localhost", port=6333)
    tous_les_points = []
    offset = None

    while True:
        resultats, offset_suivant = client.scroll(
            collection_name="prets_reels",
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        if not resultats:
            break

        for point in resultats:
            p = point.payload
            tous_les_points.append({
                "montant":           p.get("montant", 0),
                "grade":             p.get("grade", "C"),
                "taux":              p.get("taux", 10.0),
                "revenu":            p.get("revenu", 0),
                "age":               p.get("age", 30),
                "anciennete":        p.get("anciennete", 0),
                "intention":         p.get("intention", "PERSONAL"),
                "proprietaire":      p.get("proprietaire", "RENT"),
                "ratio_revenu":      p.get("ratio_revenu", 0.1),
                "historique_credit": p.get("historique_credit", 5),
                "defaut_passe":      p.get("defaut_passe", "N"),
                "profit":            p.get("profit", 0),  # cible
            })

        if offset_suivant is None:
            break
        offset = offset_suivant

    df = pd.DataFrame(tous_les_points).fillna(0)
    print(f"{len(df)} dossiers extraits de Qdrant")
    return df