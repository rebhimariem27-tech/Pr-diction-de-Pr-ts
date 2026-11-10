import shap
import joblib
import pandas as pd
import numpy as np

def charger_explainer():
    """
    Crée l'explainer SHAP une seule fois et le réutilise.
    TreeExplainer est optimisé pour LightGBM — rapide et exact.
    """
    artefacts = joblib.load('artefacts_modele.pkl')
    model     = artefacts['model']
    explainer = shap.TreeExplainer(model)
    return explainer, artefacts

def expliquer_decision(features_client: dict) -> dict:
    """
    Prend un dictionnaire de features et retourne :
    - les 3 facteurs favorables
    - les 3 facteurs défavorables
    - une phrase d'explication lisible
    """
    explainer, artefacts = charger_explainer()
    features  = artefacts['features']
    encoders  = artefacts['encoders']

    X = pd.DataFrame([features_client])[features]
    
    # Calcul des valeurs SHAP
    # shap_values[i] = contribution de la feature i à la décision finale
    # Positif  → pousse vers "Rentable"
    # Négatif  → pousse vers "Non Rentable"
    shap_values = explainer.shap_values(X)
    
    # LightGBM binaire retourne une liste [classe_0, classe_1]
    # On prend classe_1 = "Rentable"
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Construire le tableau feature → impact
    impacts = []
    for feat, val in zip(features, sv):
        impacts.append({
            'feature':    feat,
            'impact':     round(float(val), 4),
            'valeur':     round(float(features_client.get(feat, 0)), 2),
            'direction':  'favorable' if val > 0 else 'défavorable'
        })

    # Trier par impact absolu
    impacts.sort(key=lambda x: abs(x['impact']), reverse=True)

    favorables    = [i for i in impacts if i['direction'] == 'favorable'][:3]
    defavorables  = [i for i in impacts if i['direction'] == 'défavorable'][:3]

    # Noms lisibles pour l'affichage
    noms_lisibles = {
        'taux':               'Taux d\'intérêt',
        'montant':            'Montant du prêt',
        'revenu':             'Revenu annuel',
        'ratio_endettement':  'Ratio d\'endettement',
        'effort_mensuel':     'Effort mensuel',
        'grade_enc':          'Grade de crédit',
        'defaut_passe_enc':   'Défaut passé',
        'ratio_revenu':       'Part prêt/revenu',
        'historique_credit':  'Historique crédit',
        'age':                'Âge',
        'anciennete':         'Ancienneté emploi',
    }

    def nom(feat):
        return noms_lisibles.get(feat, feat)

    # Phrase d'explication
    if favorables:
        raisons_ok  = ', '.join([nom(f['feature']) for f in favorables[:2]])
    else:
        raisons_ok  = 'aucun facteur dominant'

    if defavorables:
        raisons_nok = ', '.join([nom(f['feature']) for f in defavorables[:2]])
    else:
        raisons_nok = 'aucun facteur limitant'

    explication = (
        f"Facteurs favorables : {raisons_ok}. "
        f"Facteurs limitants : {raisons_nok}."
    )

    return {
        'impacts':        impacts,
        'favorables':     favorables,
        'defavorables':   defavorables,
        'explication':    explication,
    }