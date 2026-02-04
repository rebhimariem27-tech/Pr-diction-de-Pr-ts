from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

mcp = FastMCP("Calculateur de Prêt")
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

@mcp.tool()
def chercher_prets_similaires(salaire: int, montant: int, score: int) -> str:
    requete = f"Salaire: {salaire}, Prêt: {montant}, Score: {score}"
    vector = model.encode(requete).tolist()
    
    reponse_qdrant = client.query_points(
        collection_name="prets_historiques",
        query=vector,
        limit=5 # On en cherche 5 pour être plus précis
    )
    
    resultats = reponse_qdrant.points
    
    print(f"\n--- DEBUG : Qdrant a trouvé {len(resultats)} dossiers ---")
    reponse = "Dossiers trouvés :\n"
    for res in resultats:
        p = res.payload
        # On affiche dans la console pour comprendre
        print(f"Trouvé: Salaire={p['salaire']}, Profit={p['profit']}")
        
        statut = "Rentable" if p['profit'] == 1 else "Non Rentable"
        reponse += f"- Prêt {p['montant']}€ (Salaire: {p['salaire']}€) -> {statut}\n"
    
    return reponse

if __name__ == "__main__":
    mcp.run()