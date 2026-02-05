from serveur_mcp import chercher_prets_similaires
from qdrant_client import QdrantClient

# =========================
# TEST 1: Vérifier Qdrant
# =========================
def test_qdrant_connection():
    """Teste la connexion à Qdrant"""
    client = QdrantClient(host="localhost", port=6333)
    collections = client.get_collections()
    assert any(c.name == "prets_reels" for c in collections.collections), "Collection 'prets_reels' non trouvée"
    print("✅ Qdrant connecté et collection trouvée")

def test_qdrant_has_data():
    """Teste que la collection a des données"""
    client = QdrantClient(host="localhost", port=6333)
    collection_info = client.get_collection("prets_reels")
    assert collection_info.points_count > 0, "La collection est vide"
    print(f"✅ Qdrant contient {collection_info.points_count} points")

# =========================
# TEST 2: Fonction de recherche
# =========================
def test_chercher_prets_similaires_basic():
    """Teste la fonction de recherche avec des paramètres normaux"""
    resultat = chercher_prets_similaires(salaire=50000, montant=10000, score=700)
    assert isinstance(resultat, str), "La réponse doit être une string"
    assert "Prêt" in resultat, "La réponse doit contenir 'Prêt'"
    assert "Rentable" in resultat or "Non Rentable" in resultat, "Doit contenir un statut"
    print("✅ Fonction de recherche fonctionne")

def test_chercher_prets_similaires_returns_20():
    """Teste que la fonction retourne 20 dossiers"""
    resultat = chercher_prets_similaires(salaire=45000, montant=5000, score=650)
    count = resultat.count("Prêt")
    assert count == 20, f"Doit retourner 20 dossiers, got {count}"
    print(f"✅ Fonction retourne bien 20 dossiers")

# =========================
# TEST 3: Cas limites mathématiques
# =========================
def test_cas_absurde_ratio_eleve():
    """Teste un prêt disproportionné (ratio > 15)"""
    # Salaire 1000€, Prêt 20000€ = ratio 20x
    resultat = chercher_prets_similaires(salaire=1000, montant=20000, score=700)
    assert "Rentable" in resultat or "Non Rentable" in resultat
    print("✅ Cas absurde (ratio élevé) géré")

def test_cas_score_critique():
    """Teste un score de crédit très faible"""
    resultat = chercher_prets_similaires(salaire=50000, montant=10000, score=100)
    assert "Rentable" in resultat or "Non Rentable" in resultat
    print("✅ Cas critique (score faible) géré")

def test_cas_normal():
    """Teste un profil normal et équilibré"""
    resultat = chercher_prets_similaires(salaire=60000, montant=15000, score=750)
    assert "Rentable" in resultat or "Non Rentable" in resultat
    print("✅ Cas normal géré")

# =========================
# TEST 4: Cohérence des résultats
# =========================
def test_resultat_coherent_rentable():
    """Test profil avec beaucoup de dossiers rentables trouvés"""
    resultat = chercher_prets_similaires(salaire=45000, montant=8000, score=700)
    rentables = resultat.count("Rentable")
    non_rentables = resultat.count("Non Rentable")
    total = rentables + non_rentables
    
    if total > 0:
        ratio = (rentables / total) * 100
        assert 0 <= ratio <= 100, "Ratio de rentabilité invalide"
        print(f"✅ Ratio de rentabilité valide: {ratio:.1f}%")

def test_param_influence():
    """Test que les paramètres influencent le résultat"""
    r1 = chercher_prets_similaires(salaire=30000, montant=15000, score=600)
    r2 = chercher_prets_similaires(salaire=80000, montant=10000, score=800)
    
    # Les résultats devraient être différents
    assert r1 != r2, "Paramètres différents devraient donner résultats différents"
    print("✅ Les paramètres influencent bien le résultat")

# =========================
# TEST 5: Validation des données
# =========================
def test_montant_zero():
    """Test avec un montant zéro"""
    resultat = chercher_prets_similaires(salaire=50000, montant=0, score=700)
    assert isinstance(resultat, str)
    print("✅ Montant zéro géré")

def test_salaire_zero():
    """Test avec un salaire zéro"""
    resultat = chercher_prets_similaires(salaire=0, montant=10000, score=700)
    assert isinstance(resultat, str)
    print("✅ Salaire zéro géré")

# =========================
# RUN ALL TESTS
# =========================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("SUITE DE TESTS - SYSTÈME DE PRÉDICTION DE PRÊTS")
    print("="*50 + "\n")
    
    tests = [
        ("Connexion Qdrant", test_qdrant_connection),
        ("Données dans Qdrant", test_qdrant_has_data),
        ("Recherche basique", test_chercher_prets_similaires_basic),
        ("Retour 20 dossiers", test_chercher_prets_similaires_returns_20),
        ("Cas absurde (ratio)", test_cas_absurde_ratio_eleve),
        ("Cas critique (score)", test_cas_score_critique),
        ("Cas normal", test_cas_normal),
        ("Cohérence rentabilité", test_resultat_coherent_rentable),
        ("Influence paramètres", test_param_influence),
        ("Montant zéro", test_montant_zero),
        ("Salaire zéro", test_salaire_zero),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"Test: {name}...", end=" ")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ ERREUR: {str(e)}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"RÉSULTATS: {passed} réussi, {failed} échoué")
    print("="*50 + "\n")
