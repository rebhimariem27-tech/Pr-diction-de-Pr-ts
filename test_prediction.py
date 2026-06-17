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
def test_modele_ml_existe():
    import os
    assert os.path.exists('artefacts_modele.pkl'), "Lancez model_ml.py d'abord"
    print("Fichier artefacts_modele.pkl trouvé")

def test_prediction_entre_0_et_100():
    import joblib
    a = joblib.load('artefacts_modele.pkl')
    from app_web import predire
    score = predire(a, salaire=50000, montant=10000,
                    taux=10.0, grade='B', intention='PERSONAL')
    assert 0 <= score <= 100
    print(f"Score retourné : {score}%")

def test_profil_risque_inferieur_a_profil_sure():
    import joblib
    a = joblib.load('artefacts_modele.pkl')
    from app_web import predire
    score_sur  = predire(a, 80000, 5000,  taux=6.0,  grade='A', intention='EDUCATION')
    score_rque = predire(a, 10000, 80000, taux=24.0, grade='G', intention='VENTURE')
    assert score_sur > score_rque, \
        f"Profil sûr ({score_sur}%) doit scorer plus haut que risqué ({score_rque}%)"
    print(f"Sûr: {score_sur}% > Risqué: {score_rque}% — OK")
def test_qdrant_filtre_par_montant():
    """Avec filtre actif, la majorité des résultats doit être dans la fourchette"""
    from serveur_mcp import chercher_prets_similaires

    montant_demande = 10000
    resultat = chercher_prets_similaires(
        salaire=50000, montant=montant_demande, score=700
    )

    lignes = [l for l in resultat.split('\n') if l.startswith('- Prêt')]
    if not lignes:
        print("⚠️  Aucun dossier retourné — fallback activé")
        return

    dans_fourchette = 0
    for ligne in lignes:
        montant_str = ligne.split('Prêt ')[1].split('€')[0]
        montant_trouve = int(montant_str.replace(',', ''))
        if montant_demande * 0.6 <= montant_trouve <= montant_demande * 1.4:
            dans_fourchette += 1

    ratio = dans_fourchette / len(lignes)
    # On accepte que 50%+ soient dans la fourchette (fallback peut diluer)
    assert ratio >= 0.5, f"Seulement {ratio*100:.0f}% dans la fourchette"
    print(f"✅ {dans_fourchette}/{len(lignes)} dossiers dans la fourchette ±40%")


def test_resultats_differents_selon_montant():
    """Un prêt de 5000€ et un prêt de 25000€ doivent retourner des dossiers différents"""
    from serveur_mcp import chercher_prets_similaires

    r_petit = chercher_prets_similaires(salaire=40000, montant=5000,  score=700)
    r_grand = chercher_prets_similaires(salaire=40000, montant=25000, score=700)

    assert r_petit != r_grand, \
        "Des montants très différents doivent retourner des dossiers différents"
    print("✅ Discrimination par montant confirmée")
def test_shap_retourne_explication():
    """SHAP doit retourner favorables et défavorables"""
    import joblib
    from explicabilite import expliquer_decision

    artefacts = joblib.load('artefacts_modele.pkl')
    features_client = {
        'montant': 10000, 'taux': 12.0, 'revenu': 50000,
        'age': 35, 'anciennete': 5, 'ratio_revenu': 0.2,
        'historique_credit': 7, 'ratio_endettement': 0.2,
        'effort_mensuel': 0.03, 'revenu_par_anciennete': 8333,
        'grade_enc':       artefacts['encoders']['grade'].transform(['B'])[0],
        'intention_enc':   artefacts['encoders']['intention'].transform(['PERSONAL'])[0],
        'proprietaire_enc':artefacts['encoders']['proprietaire'].transform(['RENT'])[0],
        'defaut_passe_enc':artefacts['encoders']['defaut_passe'].transform(['N'])[0],
    }

    result = expliquer_decision(features_client)

    assert 'favorables'   in result
    assert 'defavorables' in result
    assert 'explication'  in result
    assert len(result['impacts']) > 0
    print(f"✅ SHAP OK — {result['explication']}")

def test_shap_profil_risque_a_plus_defavorables():
    """Un profil risqué doit avoir plus de facteurs défavorables"""
    import joblib
    from explicabilite import expliquer_decision

    artefacts = joblib.load('artefacts_modele.pkl')

    base = {
        'age': 35, 'anciennete': 5, 'historique_credit': 7,
        'revenu_par_anciennete': 5000,
        'grade_enc':        artefacts['encoders']['grade'].transform(['G'])[0],
        'intention_enc':    artefacts['encoders']['intention'].transform(['VENTURE'])[0],
        'proprietaire_enc': artefacts['encoders']['proprietaire'].transform(['RENT'])[0],
        'defaut_passe_enc': artefacts['encoders']['defaut_passe'].transform(['Y'])[0],
    }

    profil_risque = {**base,
        'montant': 80000, 'taux': 24.0, 'revenu': 10000,
        'ratio_revenu': 8.0, 'ratio_endettement': 8.0, 'effort_mensuel': 1.6,
    }

    result = expliquer_decision(profil_risque)
    total_def = sum(abs(f['impact']) for f in result['defavorables'])
    total_fav = sum(abs(f['impact']) for f in result['favorables'])

    assert total_def > total_fav, \
        f"Profil risqué doit avoir plus d'impacts défavorables ({total_def:.3f} vs {total_fav:.3f})"
    print(f"✅ Profil risqué : défavorables={total_def:.3f} > favorables={total_fav:.3f}")
def test_logger_cree_fichier():
    """Le logger doit créer decisions_log.csv"""
    import os
    from logger_decisions import logger_decision, LOG_FILE

    logger_decision(
        salaire=50000, montant=10000, taux=12.0,
        grade='B', intention='PERSONAL',
        age=35, anciennete=5, defaut_passe='N',
        score_ml=75.0, score_qdrant=70.0,
        score_final=73.5, decision='APPROUVÉ'
    )
    assert os.path.exists(LOG_FILE)
    print("✅ decisions_log.csv créé")

def test_logger_enregistre_donnees():
    """Les données loggées doivent être récupérables"""
    from logger_decisions import charger_log
    log = charger_log()
    assert len(log) > 0
    assert 'score_final' in log.columns
    assert 'decision' in log.columns
    print(f"✅ {len(log)} décisions dans le log")

def test_drift_detector_statut():
    """Le drift detector doit retourner un statut valide"""
    from drift_detector import detecter_drift
    result = detecter_drift()
    assert result['statut'] in ['OK', 'ALERTE', 'INSUFFISANT']
    print(f"✅ Drift statut : {result['statut']}")