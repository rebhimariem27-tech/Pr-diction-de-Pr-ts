import json

import pandas as pd
import pytest

from tests.conftest import CATEGORICAL, FEATURES, THRESHOLD, VALID_PAYLOAD
from backend.app.features import business_rule_violations, derive_features
from backend.app.security import ApiAuth, RateLimiter, find_key_name, hash_key, load_key_hashes


# ---------- features ----------
def test_derive_features_values():
    d = derive_features(10000, 48000, 11.14)
    assert d["loan_percent_income"] == pytest.approx(0.21)
    assert d["interest_income_ratio"] is not None

def test_derive_features_missing_inputs():
    assert derive_features(None, 48000, 10) == {"loan_percent_income": None, "interest_income_ratio": None}
    assert derive_features(1000, 0, 10)["loan_percent_income"] is None
    assert derive_features(1000, 5000, None)["interest_income_ratio"] is None

def test_business_rules():
    base = dict(person_age=30, person_emp_length=5, cb_person_cred_hist_length=5,
                loan_amnt=1000, person_income=5000, loan_int_rate=10)
    assert business_rule_violations(**base) == []
    assert business_rule_violations(**{**base, "person_emp_length": 31})
    assert business_rule_violations(**{**base, "cb_person_cred_hist_length": 31})
    assert business_rule_violations(**{**base, "loan_amnt": 6000})
    assert business_rule_violations(**{**base, "loan_int_rate": 150})


# ---------- sécurité ----------
def test_key_hashing_and_lookup():
    hashes = load_key_hashes(f"a:{hash_key('k1')},b:{hash_key('k2')}")
    assert find_key_name("k2", hashes) == "b"
    assert find_key_name("bad", hashes) is None

def test_bad_hash_format_rejected():
    with pytest.raises(ValueError):
        load_key_hashes("a:tooshort")

def test_rate_limiter_blocks_then_reports_retry():
    rl = RateLimiter(window_seconds=60)
    assert rl.check("x", 2)[0] and rl.check("x", 2)[0]
    allowed, retry = rl.check("x", 2)
    assert not allowed and retry >= 1
    assert rl.check("y", 2)[0]

def test_auth_fails_closed(monkeypatch):
    monkeypatch.delenv("FINPREDICT_API_KEY_HASHES", raising=False)
    monkeypatch.delenv("FINPREDICT_AUTH_DISABLED", raising=False)
    with pytest.raises(RuntimeError):
        ApiAuth()


# ---------- explicabilité ----------
def test_explainer_structure(tiny_model):
    from backend.app.explain import ContributionExplainer
    ex = ContributionExplainer(tiny_model, FEATURES, CATEGORICAL)
    row = pd.DataFrame([{**VALID_PAYLOAD, "person_home_ownership": "RENT", "loan_grade": "B",
                         **derive_features(10000, 48000, 11.14)}], columns=FEATURES)
    out = ex.explain(row)
    names = {f["feature"] for f in out["top_risk_factors"] + out["top_protective_factors"]}
    assert names <= set(FEATURES)                       # one-hot regroupé sur la variable d'origine
    assert all(f["contribution"] > 0 for f in out["top_risk_factors"])
    assert all(f["contribution"] < 0 for f in out["top_protective_factors"])


# ---------- API ----------
def test_requires_api_key(client, db):
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 401
    assert client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "nope"}).status_code == 401
    assert db.apps == []

def test_predict_ok_server_computes_derived(client, db, headers):
    r = client.post("/predict", json=VALID_PAYLOAD, headers=headers)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["derived_features"]["loan_percent_income"] == pytest.approx(0.21)
    assert body["recommendation"] == ("REFUSE" if body["default_probability"] >= THRESHOLD else "ACCEPT")
    assert "top_risk_factors" in body["explanation"]
    stored_snapshot = json.loads(db.apps[0][2])
    assert stored_snapshot["loan_grade"] == "B" and stored_snapshot["person_home_ownership"] == "RENT"
    assert json.loads(db.preds[0][7])["method"]         # explanation_json persisté
    assert db.committed == 1

def test_client_derived_values_are_ignored(client, db, headers):
    r = client.post("/predict", headers=headers,
                    json={**VALID_PAYLOAD, "loan_percent_income": 0.9, "interest_income_ratio": 9.0})
    assert r.status_code == 201
    assert r.json()["derived_features"]["loan_percent_income"] == pytest.approx(0.21)

@pytest.mark.parametrize("override", [
    {"person_emp_length": 40},                 # > âge
    {"cb_person_cred_hist_length": 40},        # > âge
    {"loan_amnt": 60000},                      # > revenu
    {"loan_grade": "Z"},                       # catégorie inconnue
])
def test_invalid_requests_rejected_before_db(client, db, headers, override):
    r = client.post("/predict", json={**VALID_PAYLOAD, **override}, headers=headers)
    assert r.status_code == 422
    assert db.apps == []

def test_duplicate_ref_409(client, db, headers):
    payload = {**VALID_PAYLOAD, "application_ref": "dup-1"}
    assert client.post("/predict", json=payload, headers=headers).status_code == 201
    assert client.post("/predict", json=payload, headers=headers).status_code == 409

def test_race_on_unique_index_is_409_not_503(client, db, headers):
    db.race_error = True
    r = client.post("/predict", json={**VALID_PAYLOAD, "application_ref": "race-1"}, headers=headers)
    assert r.status_code == 409

def test_threshold_mismatch_and_inactive_model_return_503(client, db, headers):
    db.registered_threshold = 0.5
    assert client.post("/predict", json=VALID_PAYLOAD, headers=headers).status_code == 503
    db.registered_threshold, db.status = THRESHOLD, "RETIRED"
    assert client.post("/predict", json=VALID_PAYLOAD, headers=headers).status_code == 503


@pytest.mark.parametrize("amnt,income,expected", [
    (5000, 40000, 0.13),    # 0.125 exact -> demi vers le haut
    (9000, 40000, 0.23),    # 0.225
    (3000, 40000, 0.07),    # 0.075 est stocké 0.0749999... -> 0.07
    (7000, 40000, 0.17),    # 0.175 -> 0.17
    (1000, 40000, 0.03),    # 0.025 -> 0.03
    (3000, 200000, 0.01),   # 0.015 -> 0.01
    (10000, 48000, 0.21),
])
def test_loan_percent_income_matches_csv_rounding(amnt, income, expected):
    assert derive_features(amnt, income, 10.0)["loan_percent_income"] == expected
