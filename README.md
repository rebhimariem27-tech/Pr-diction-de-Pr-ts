<div align="center">
  <img src="finpredict-frontend/public/finpredict-mark.svg" alt="FinPredict" width="88" />
  <h1>FinPredict AI Pro</h1>
  <p><strong>Plateforme d'analyse du risque de crédit explicable, sécurisée et traçable.</strong></p>
  <p>
    Transformez un dossier de prêt en une recommandation lisible, justifiable et persistée,
    sans retirer la décision finale du contrôle humain.
  </p>

  <p>
    <a href="#fonctionnalites">Fonctionnalités</a> ·
    <a href="#demarrage-rapide">Démarrage rapide</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#api">API</a> ·
    <a href="#securite-et-gouvernance">Sécurité</a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+" />
    <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Next.js-15-000000?logo=next.js&logoColor=white" alt="Next.js 15" />
    <img src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
    <img src="https://img.shields.io/badge/Azure%20SQL-tracking-0078D4?logo=microsoftazure&logoColor=white" alt="Azure SQL" />
  </p>
</div>

> **Statut :** projet démonstrateur prêt à être présenté. Les prédictions sont des recommandations de modèle et ne constituent pas une décision automatique d'octroi ou de refus de crédit.

## Pourquoi FinPredict ?

FinPredict est une application full-stack de décision augmentée par l'IA. Elle réunit :

- un pipeline de classification binaire basé sur LightGBM ;
- une calibration probabiliste sigmoid pour rendre le score plus interprétable ;
- une interface métier en français, pensée pour l'analyse rapide d'un portefeuille ;
- une explication des facteurs qui font évoluer la probabilité de défaut ;
- une persistance transactionnelle des demandes et prédictions dans Azure SQL ;
- une authentification par clé API, du rate limiting et une séparation stricte des secrets.

L'objectif n'est pas de remplacer l'analyste : c'est de lui donner un signal de risque cohérent, contextualisé et auditable.

## Fonctionnalités

### Analyse individuelle

- formulaire guidé pour les caractéristiques financières et de crédit ;
- validation stricte des types, bornes et catégories côté API ;
- calcul serveur des variables dérivées pour éviter le *training-serving skew* ;
- probabilité de défaut, seuil opérationnel et recommandation `ACCEPT` / `REFUSE` ;
- identifiants de demande et de prédiction pour assurer la traçabilité.

### Explicabilité

- décomposition de la prédiction par groupes de facteurs ;
- lecture en points de probabilité par rapport à une probabilité de base ;
- facteurs positifs et négatifs affichés dans un langage métier ;
- rappel explicite qu'une explication de modèle n'est pas une relation causale.

### Pilotage et portefeuille

- tableau de bord avec indicateurs de synthèse ;
- consultation de l'historique des prédictions persistées ;
- état de santé de l'API et de la connexion Azure SQL ;
- exports de rapports de risque, monitoring et gouvernance ;
- écrans de surveillance et de paramétrage pour préparer l'industrialisation.

> Les KPI, tendances et graphiques de démonstration de l'interface sont signalés comme illustratifs. Les résultats d'une analyse et l'historique renvoyé par `/portfolio` proviennent, eux, de l'API et du stockage configurés.

## Performance du modèle

Les résultats ci-dessous proviennent du jeu de test associé à l'artefact de production `1.0.0`, au seuil opérationnel `0,48`. Ils ne constituent pas une garantie de performance sur de nouvelles populations.

| Indicateur | Résultat |
| --- | ---: |
| ROC-AUC | **0,9533** |
| PR-AUC | **0,9125** |
| Précision | **96,31 %** |
| Rappel | **75,39 %** |
| F1-score | **0,8457** |
| Brier score | **0,0497** |

Le modèle utilise 12 variables d'entrée, dont 8 numériques et 4 catégorielles. Les ratios `loan_percent_income` et `interest_income_ratio` sont recalculés côté serveur à partir des valeurs sources.

## Architecture

```text
┌──────────────────────┐       ┌────────────────────────┐
│ Next.js / TypeScript │──────▶│ Routes API serveur      │
│ Interface métier     │       │ /api/predict            │
│ Dashboard + analyse  │       │ /api/portfolio          │
└──────────┬───────────┘       └────────────┬───────────┘
           │                                │ X-API-Key
           │                                ▼
           │                    ┌────────────────────────┐
           └───────────────────▶│ FastAPI                 │
                                │ Validation + scoring    │
                                │ Explication + health    │
                                └────────────┬───────────┘
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
                Artefact LightGBM                         Azure SQL
                calibré et versionné                       tracking
```

### Technologies

| Couche | Choix |
| --- | --- |
| Frontend | Next.js 15, React 19, TypeScript, Recharts, Lucide |
| Backend | FastAPI, Pydantic, Uvicorn |
| Machine learning | LightGBM, scikit-learn, joblib, pandas, NumPy |
| Stockage | Azure SQL pour les demandes et les prédictions |
| Déploiement | Azure App Service pour l'API |
| Qualité | pytest, tests unitaires, évaluation ML et tests d'intégration |

## Démarrage rapide

### Prérequis

- Python 3.12 ;
- Node.js 20.9 ou plus récent ;
- une base Azure SQL si la persistance et les analyses réelles sont activées ;
- un artefact de modèle présent dans `data/production/`.

### 1. Configurer l'environnement backend

Depuis la racine du dépôt :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Renseignez ensuite les variables Azure SQL et la clé API dans `.env`. Ne versionnez jamais ce fichier.

Pour générer une clé API sans la stocker en clair :

```powershell
python -m backend.app.security gen-key frontend-local
```

En développement uniquement, l'authentification peut être désactivée avec `FINPREDICT_AUTH_DISABLED=true`. Ne jamais utiliser cette option en production.

### 2. Démarrer l'API

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Puis ouvrir la documentation interactive : <http://127.0.0.1:8000/docs>.

### 3. Démarrer le frontend

Dans un second terminal :

```powershell
cd finpredict-frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Ouvrir <http://localhost:3000>.

Les variables `NEXT_PUBLIC_*` sont intégrées au navigateur : elles ne doivent contenir aucun secret. La clé API utilisée par le proxy Next.js doit rester dans une variable serveur, sans préfixe `NEXT_PUBLIC_`.

### Problème connu sous Windows et OneDrive

Le pilote natif `mssql-python` peut échouer lorsque l'environnement virtuel se trouve dans OneDrive ou dans un chemin contenant des caractères accentués. Si cela arrive, créez le venv dans un chemin ASCII :

```powershell
py -3.12 -m venv C:\venvs\finpredict
C:\venvs\finpredict\Scripts\python.exe -m pip install -r requirements.txt
C:\venvs\finpredict\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

## API

| Méthode | Route | Authentification | Rôle |
| --- | --- | --- | --- |
| `GET` | `/` | publique | Vérifier que l'application répond |
| `GET` | `/health` | publique | Vérifier le modèle et Azure SQL |
| `GET` | `/model-info` | clé API | Lire la version, le seuil et les variables |
| `POST` | `/predict` | clé API | Valider, scorer, expliquer et persister une demande |
| `GET` | `/portfolio` | clé API | Lire l'historique paginé des prédictions |

Exemple minimal :

```powershell
$headers = @{ "X-API-Key" = $env:FINPREDICT_API_KEY }
$body = @{
  application_ref = "demo-001"
  person_age = 32
  person_income = 48000
  loan_amnt = 10000
  loan_int_rate = 11.14
  person_emp_length = 5
  cb_person_cred_hist_length = 8
  person_home_ownership = "RENT"
  loan_intent = "PERSONAL"
  loan_grade = "B"
  cb_person_default_on_file = "N"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

## Tests et qualité

Tests unitaires et évaluation du modèle :

```powershell
pytest tests/unit tests/ml_evaluation
```

Tests d'intégration, uniquement lorsque les services et variables nécessaires sont disponibles :

```powershell
pytest tests/integration
```

Scripts utiles :

```powershell
python -m backend.scripts.preprocessing --input data/raw/donnees_prets_reels.csv
python -m backend.scripts.setup_api_key
```

## Sécurité et gouvernance

- les clés API sont comparées via leur hash SHA-256 ; elles ne sont pas stockées en clair ;
- le rate limiting protège les accès valides et les tentatives invalides ;
- les secrets Azure SQL restent côté backend ;
- les entrées inattendues sont refusées par Pydantic (`extra="forbid"`) ;
- les variables dérivées sont recalculées côté serveur ;
- les journaux d'accès excluent le corps des requêtes ;
- la recommandation est présentée comme une aide à la décision, avec supervision humaine ;
- CORS doit être limité aux origines frontend réellement autorisées.

## Limites connues et feuille de route

Le projet est prêt pour une démonstration technique et une base d'industrialisation. Les prochaines étapes naturelles sont :

1. connecter les décisions humaines et les labels observés à maturité ;
2. automatiser le suivi du drift, de la qualité des données et de la fairness ;
3. ajouter une fiche de changement et une politique de conservation des modèles ;
4. remplacer les données illustratives du dashboard par des agrégations API ;
5. externaliser le rate limiting vers Redis ou Azure API Management en multi-instance ;
6. formaliser les contrôles de validation avant chaque mise en production.

## Structure du dépôt

```text
backend/
  app/                 API FastAPI, validation, sécurité et explicabilité
  scripts/             Préprocessing, entraînement, diagnostics et migration
data/
  production/          Métadonnées et artefacts du modèle de production
  reports/              Rapports d'évaluation générés
finpredict-frontend/
  app/                 Interface Next.js et routes proxy serveur
tests/
  unit/                Tests du cœur applicatif
  ml_evaluation/       Contrôles de qualité du modèle
  integration/         Tests avec API et services externes
```

## Licence

Ajoutez la licence de votre choix avant une publication open source officielle. En l'absence de fichier `LICENSE`, tous les droits restent réservés par défaut.

---

<div align="center">
  <sub>FinPredict AI Pro · Analyse du risque de crédit avec explicabilité et supervision humaine</sub>
</div>
