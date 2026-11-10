# FinPredict AI Pro — Frontend

Interface Next.js / TypeScript inspirée de la maquette approuvée : navigation bleu nuit, cartes claires, accents turquoise et bleus, graphique de portefeuille, analyse de crédit, portefeuille, surveillance et paramètres.

## Prérequis

- Node.js 20.9 ou plus récent conseillé pour Next.js 15.
- Le backend FastAPI existant (`api_azure.py`) pour lancer une vraie prédiction.
- L'API doit être démarrée séparément.

## Installation locale

1. Décompresser `finpredict-frontend.zip` dans le dossier de votre choix (il s'agit d'une application frontend autonome, distincte du backend FinPredict).
2. Créer `.env.local` à partir de `.env.example` :

   ```dotenv
   NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
   ```

3. Installer et démarrer :

   ```powershell
   npm install
   npm run dev
   ```

4. Ouvrir http://localhost:3000.

## Connexion à FastAPI

Le formulaire poste les 12 caractéristiques vers `POST ${NEXT_PUBLIC_API_BASE_URL}/predict`. Il utilise la réponse API réelle pour afficher la probabilité de défaut, le seuil, la recommandation et les identifiants de tracking Azure SQL. Si l'API ne répond pas, l'interface affiche une erreur et ne fabrique pas un résultat.

Le backend FastAPI doit autoriser l'origine de cette interface via CORS. Pour un frontend local, ajoutez `http://localhost:3000` aux origines autorisées de FastAPI. Pour un déploiement public, ajoutez également l'origine HTTPS exacte du frontend. N'utilisez jamais `*` pour une API contenant des données sensibles en production.

Pour viser une API hébergée, créez `.env.local` avant le build avec son URL publique, par exemple :

```dotenv
NEXT_PUBLIC_API_BASE_URL=https://VOTRE-DOMAINE-API
```

Les variables `NEXT_PUBLIC_*` sont intégrées au bundle navigateur et ne doivent jamais contenir de secrets. Les identifiants Azure SQL restent côté backend.

## Ce qui est réel et ce qui reste en démonstration

- **Réel si l'API est joignable :** indicateur de santé obtenu depuis `/health`, version annoncée depuis `/health`, prédiction depuis `/predict`, référence et identifiants retournés par Azure SQL.
- **Démonstration :** KPI du dashboard, tendance mensuelle, répartition des décisions, lignes du portefeuille et courbes de surveillance. Ils sont explicitement marqués comme données illustratives. Ils ne sont pas présentés comme des données Azure.
- Les écrans rapports et paramètres constituent la base UI ; l'export et les agrégations historiques nécessitent des endpoints FastAPI supplémentaires.

## Design reference

L'image d'inspiration est conservée dans `docs/design-reference.png`.
