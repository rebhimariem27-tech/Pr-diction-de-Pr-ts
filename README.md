# FinPredict

Application de prédiction du risque de prêt avec un backend FastAPI et un
frontend Next.js.

## Organisation

```text
backend/
  app/                 Code runtime de l'API FastAPI et logique métier
  scripts/             Préprocessing, entraînement, diagnostics et migrations
tests/
  test_core.py         Tests unitaires
  ml_evaluation/       Tests de qualité du modèle
  integration/         Tests nécessitant une API ou une base externe
finpredict-frontend/   Application Next.js
data/                  Données, artefacts et rapports générés
```

## Lancer le backend

Depuis la racine du dépôt :

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Pour Azure App Service, la commande de production est disponible dans
[`azure_startup_command.txt`](./azure_startup_command.txt).

### Erreur du pilote Azure SQL sous Windows

`mssql-python` charge des bibliothèques natives. Le venv ne doit pas être
installé dans un chemin OneDrive ou dans un chemin contenant des caractères
accentués : le chargeur natif peut alors signaler à tort que
`mssql-python-odbc` est incomplet.

Créez le venv dans un chemin ASCII, installez les dépendances, puis utilisez
ce venv pour lancer l'API :

```powershell
py -3.12 -m venv C:\venvs\finpredict
C:\venvs\finpredict\Scripts\python.exe -m pip install -r requirements.txt
C:\venvs\finpredict\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Fermez d'abord Uvicorn avant de réinstaller `mssql-python` ou
`mssql-python-odbc`, car les fichiers `.pyd` peuvent rester verrouillés :

```powershell
C:\venvs\finpredict\Scripts\python.exe -m pip install --force-reinstall mssql-python==1.15.0 mssql-python-odbc
```

Si le pilote est chargé correctement mais qu'Azure renvoie ensuite
`Database ... is not currently available`, le problème ne concerne plus le
pilote local : vérifiez que la base Azure SQL est démarrée et disponible dans
le portail Azure, puis réessayez.

## Lancer le frontend

```powershell
cd finpredict-frontend
npm run dev
```

Le proxy Next.js utilise `FINPREDICT_API_URL` et `FINPREDICT_API_KEY` côté
serveur. Ne jamais exposer ces variables avec le préfixe `NEXT_PUBLIC_`.

## Exécuter les tests

Depuis la racine :

```powershell
pytest tests/unit tests/ml_evaluation
```

Les tests d'intégration sont séparés et peuvent nécessiter des services ou
des variables d'environnement supplémentaires :

```powershell
pytest tests/integration
```

Les scripts backend se lancent comme modules, par exemple :

```powershell
python -m backend.scripts.preprocessing --input data/raw/donnees_prets_reels.csv
python -m backend.scripts.setup_api_key
```
