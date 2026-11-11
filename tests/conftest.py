"""Fixtures : mini-modèle de même structure que la production + fausse base SQL.
Aucun accès Azure, aucun fichier de données réel requis."""
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NUMERIC = ["person_age", "person_income", "loan_amnt", "loan_int_rate", "person_emp_length",
           "loan_percent_income", "cb_person_cred_hist_length", "interest_income_ratio"]
CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
FEATURES = NUMERIC + CATEGORICAL
THRESHOLD = 0.48

VALID_PAYLOAD = {
    "person_age": 32, "person_income": 48000, "loan_amnt": 10000, "loan_int_rate": 11.14,
    "person_emp_length": 5, "cb_person_cred_hist_length": 8,
    "person_home_ownership": "rent", "loan_intent": "PERSONAL", "loan_grade": "b",
    "cb_person_default_on_file": "N",
}


def make_dataset(n=600, seed=0):
    from backend.app.features import derive_features
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "person_age": rng.integers(20, 60, n),
        "person_income": rng.integers(20000, 120000, n).astype(float),
        "loan_amnt": rng.integers(1000, 20000, n).astype(float),
        "loan_int_rate": rng.uniform(5, 20, n),
        "person_emp_length": rng.integers(0, 15, n).astype(float),
        "cb_person_cred_hist_length": rng.integers(2, 20, n),
        "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
        "loan_intent": rng.choice(["PERSONAL", "EDUCATION", "MEDICAL"], n),
        "loan_grade": rng.choice(list("ABCDEFG"), n),
        "cb_person_default_on_file": rng.choice(["Y", "N"], n),
    })
    d = [derive_features(a, i, r) for a, i, r in zip(df.loan_amnt, df.person_income, df.loan_int_rate)]
    df["loan_percent_income"] = [x["loan_percent_income"] for x in d]
    df["interest_income_ratio"] = [x["interest_income_ratio"] for x in d]
    logit = 4 * df.loan_percent_income + 0.2 * (df.loan_int_rate - 10) - 1.5
    df["loan_status"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="session")
def tiny_model():
    df = make_dataset()
    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL),
    ])
    base = Pipeline([("preprocessor", pre),
                     ("model", lgb.LGBMClassifier(n_estimators=40, num_leaves=7, random_state=0, verbosity=-1))])
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    model.fit(df[FEATURES], df["loan_status"])
    return model


@pytest.fixture(scope="session")
def api_module(tiny_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump({"pipeline": tiny_model, "model_version": "test-1", "production_threshold": THRESHOLD,
                 "model_features": FEATURES}, path)
    os.environ["FINPREDICT_MODEL_PATH"] = str(path)
    os.environ["FINPREDICT_ENV_FILE"] = str(path.parent / "none.env")
    from backend.app.security import hash_key
    os.environ["FINPREDICT_API_KEY_HASHES"] = f"tester:{hash_key('secret-test-key')}"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"
    from backend.app import main
    return main


class FakeCursor:
    def __init__(self, db):
        self.db, self._row, self._rows = db, None, []

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        if "FROM dbo.model_versions" in s:
            self._row = (self.db.registered_threshold, self.db.status)
        elif "SELECT application_id FROM dbo.loan_applications" in s:
            self._row = (1,) if params[0] in self.db.refs else None
        elif "INSERT INTO dbo.loan_applications" in s:
            if self.db.race_error:
                raise RuntimeError("Violation of UNIQUE KEY: UX_loan_applications_application_ref (2601)")
            self.db.refs.add(params[0]); self.db.apps.append(params); self._row = (len(self.db.apps),)
        elif "INSERT INTO dbo.prediction_logs" in s:
            self.db.preds.append(params); self._row = (len(self.db.preds),)
        elif "MAX(model_version)" in s:
            self._row = (2, 0.21, 1, 1, 2, 0, 0, 0, 15000.0, "2026-11-11T12:00:00+00:00", "test-1")
        elif "GROUP BY LEFT(submitted_at_utc, 7)" in s:
            self._rows = [("2026-11", 2, 0.21)]
        elif "SELECT TOP 1 la.application_id" in s:
            self._row = (1, 1, "demo-1", "2026-11-11T12:00:00+00:00",
                         '{"loan_amnt":10000}', 0.21, "ACCEPT", "PENDING", "test-1")
        elif "UPDATE dbo.loan_applications" in s:
            self._row = None
        else:
            self._row = (1,)

    def fetchone(self): return self._row
    def fetchall(self): return self._rows
    def close(self): pass


class FakeDB:
    def __init__(self):
        self.refs, self.apps, self.preds = set(), [], []
        self.registered_threshold, self.status, self.race_error = THRESHOLD, "ACTIVE", False
        self.committed = 0

    def cursor(self): return FakeCursor(self)
    def commit(self): self.committed += 1
    def rollback(self): pass
    def close(self): pass


@pytest.fixture
def db(api_module, monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(api_module, "get_db_connection", lambda: fake)
    return fake


@pytest.fixture
def client(api_module):
    from fastapi.testclient import TestClient
    return TestClient(api_module.app)


@pytest.fixture
def headers():
    return {"X-API-Key": "secret-test-key"}
