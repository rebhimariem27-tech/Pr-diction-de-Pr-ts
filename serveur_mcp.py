from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer

mcp = FastMCP("Calculateur de Prêt")
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

@mcp.tool()
def chercher_prets_similaires(salaire: int, montant: int, score: int) -> str:

    requete = f"Prêt {montant}€, revenu {salaire}€, score crédit {score}"
    vector = model.encode(requete).tolist()

    resultats = []

    # Tentative 1 : filtre strict sur montant ET revenu
    if salaire > 0 and montant > 0:
        filtre_strict = Filter(must=[
            FieldCondition(key="montant", range=Range(
                gte=montant * 0.6, lte=montant * 1.4
            )),
            FieldCondition(key="revenu", range=Range(
                gte=salaire * 0.5, lte=salaire * 1.5
            )),
        ])
        reponse = client.query_points(
            collection_name="prets_reels",
            query=vector,
            query_filter=filtre_strict,
            limit=20
        )
        resultats = reponse.points

    # Tentative 2 : filtre sur montant seulement si pas assez de résultats
    if len(resultats) < 5 and montant > 0:
        filtre_montant = Filter(must=[
            FieldCondition(key="montant", range=Range(
                gte=montant * 0.6, lte=montant * 1.4
            )),
        ])
        reponse = client.query_points(
            collection_name="prets_reels",
            query=vector,
            query_filter=filtre_montant,
            limit=20
        )
        resultats = reponse.points

    # Tentative 3 : aucun filtre (fallback total)
    if len(resultats) < 5:
        reponse = client.query_points(
            collection_name="prets_reels",
            query=vector,
            limit=20
        )
        resultats = reponse.points

    print(f"\n--- DEBUG : Qdrant a trouvé {len(resultats)} dossiers ---")
    reponse_str = "Dossiers trouvés :\n"
    for res in resultats:
        p = res.payload
        print(f"Montant={p['montant']}€, Grade={p['grade']}, "
              f"Revenu={p.get('revenu','?')}€, Profit={p['profit']}")
        statut = "Rentable" if p['profit'] == 1 else "Non Rentable"
        reponse_str += (f"- Prêt {p['montant']}€ "
                        f"(Grade: {p['grade']}, Taux: {p['taux']}%) "
                        f"-> {statut}\n")

    return reponse_str
if __name__ == "__main__":
    mcp.run()