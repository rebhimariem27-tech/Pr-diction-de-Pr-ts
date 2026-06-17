import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
from extraire_depuis_qdrant import extraire_dataset

def encoder_categoriques(df):
    encoders = {}
    for col in ['grade', 'intention', 'proprietaire', 'defaut_passe']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def construire_features(df):
    # Features ingéniérées : les ratios sont plus prédictifs que les valeurs brutes
    df['ratio_endettement']    = df['montant'] / (df['revenu'] + 1)
    df['effort_mensuel']       = (df['montant'] * df['taux'] / 100 / 12) / (df['revenu'] / 12 + 1)
    df['revenu_par_anciennete'] = df['revenu'] / (df['anciennete'] + 1)

    features = [
        'montant', 'taux', 'revenu', 'age', 'anciennete',
        'ratio_revenu', 'historique_credit',
        'ratio_endettement', 'effort_mensuel', 'revenu_par_anciennete',
        'grade_enc', 'intention_enc', 'proprietaire_enc', 'defaut_passe_enc',
    ]
    return features

def entrainer():
    df = extraire_dataset()
    df, encoders = encoder_categoriques(df)
    features = construire_features(df)

    X = df[features]
    y = df['profit']  # 1 = rentable, 0 = non rentable

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation pour détecter l'overfitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"AUC cross-validation : {scores.mean():.3f} +/- {scores.std():.3f}")

    model.fit(X_train, y_train)
    import pandas as pd
    importance_df = pd.DataFrame({
        'feature':    features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nImportance des features :")
    print(importance_df.to_string(index=False))
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"AUC test final       : {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=['Non rentable', 'Rentable']))

    joblib.dump({
        'model':    model,
        'encoders': encoders,
        'features': features
    }, 'artefacts_modele.pkl')
    print("Modèle sauvegardé : artefacts_modele.pkl")

if __name__ == "__main__":
    entrainer()