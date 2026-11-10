from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from sql_compat import connect

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "person_emp_length",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "interest_income_ratio",
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def quote_connection_value(value: str) -> str:
    if any(char in value for char in ";={} "):
        return "{" + value.replace("}", "}}") + "}"
    return value


def make_connection_string() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    values = {
        "SQL_SERVER": os.getenv("SQL_SERVER"),
        "SQL_DATABASE": os.getenv("SQL_DATABASE"),
        "SQL_USERNAME": os.getenv("SQL_USERNAME"),
        "SQL_PASSWORD": os.getenv("SQL_PASSWORD"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Variables manquantes dans .env : " + ", ".join(missing)
        )
    return (
        f"Server={quote_connection_value(values['SQL_SERVER'])};"
        f"Database={quote_connection_value(values['SQL_DATABASE'])};"
        f"UID={quote_connection_value(values['SQL_USERNAME'])};"
        f"PWD={quote_connection_value(values['SQL_PASSWORD'])};"
        "Encrypt=yes;TrustServerCertificate=no;MultiSubnetFailover=yes;"
    )


def api_request(
    base_url: str, method: str, path: str, payload: dict | None = None
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        url=f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        response = urlopen(request, timeout=30)
        with response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw}
        return exc.code, parsed
    except URLError as exc:
        raise RuntimeError(
            f"Impossible de joindre l'API à {base_url}. "
            "Démarre-la avec `python -m uvicorn api_azure:app --reload`. "
            f"Détail réseau : {exc.reason}"
        ) from exc


def db_counts_for_reference(reference: str) -> tuple[int, int, list[tuple]]:
    conn = connect(make_connection_string(), timeout=60)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT application_id, feature_snapshot_sha256 "
            "FROM dbo.loan_applications WHERE application_ref = ?",
            (reference,),
        )
        applications = cursor.fetchall()
        predictions: list[tuple] = []
        if applications:
            app_id = int(applications[0][0])
            cursor.execute(
                "SELECT prediction_id, model_version, default_probability, "
                "threshold, recommended_decision, feature_snapshot_sha256 "
                "FROM dbo.prediction_logs WHERE application_id = ?",
                (app_id,),
            )
            predictions = cursor.fetchall()
        return len(applications), len(predictions), [tuple(row) for row in (applications + predictions)]
    finally:
        cursor.close()
        conn.close()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "FinPredict API end-to-end checks: health, prediction, Azure SQL persistence, "
            "duplicate reference and request validation. Writes one clearly marked test record."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "application_test.json",
        help="JSON d'une ligne de caractéristiques préparées.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERREUR : fichier de caractéristiques introuvable : {args.input}")
        print("Crée application_test.json à partir d'une ligne de donnees_prets_features.csv.")
        return 2

    try:
        with args.input.open("r", encoding="utf-8") as stream:
            base_payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERREUR : lecture JSON impossible : {exc}")
        return 2

    if not isinstance(base_payload, dict):
        print("ERREUR : le fichier d'entrée doit contenir un objet JSON.")
        return 2

    missing_columns = [name for name in FEATURE_COLUMNS if name not in base_payload]
    if missing_columns:
        print("ERREUR : caractéristiques manquantes : " + ", ".join(missing_columns))
        return 2

    try:
        print("=" * 68)
        print("FINPREDICT — TESTS DE BOUT EN BOUT")
        print("=" * 68)

        # 1. Health endpoint and Azure connectivity.
        status, health = api_request(args.base_url, "GET", "/health")
        assert_true(status == 200, f"GET /health : HTTP {status}, réponse={health}")
        assert_true(health.get("azure_sql") == "connected", "/health n'indique pas Azure SQL connecté.")
        print("[OK] /health : API disponible et Azure SQL connecté")

        # 2. Model metadata.
        status, model_info = api_request(args.base_url, "GET", "/model-info")
        assert_true(status == 200, f"GET /model-info : HTTP {status}, réponse={model_info}")
        model_version = str(model_info.get("model_version", ""))
        threshold = float(model_info.get("threshold", -1))
        assert_true(bool(model_version), "Version du modèle absente de /model-info.")
        assert_true(0 <= threshold <= 1, "Seuil invalide dans /model-info.")
        print(f"[OK] /model-info : modèle {model_version}, seuil {threshold:.2f}")

        # 3. Create one uniquely referenced test application and make prediction.
        test_ref = f"e2e-test-{uuid.uuid4().hex}"
        payload = dict(base_payload)
        payload["application_ref"] = test_ref
        status, result = api_request(args.base_url, "POST", "/predict", payload)
        assert_true(status == 201, f"POST /predict : HTTP {status}, réponse={result}")
        assert_true(result.get("application_ref") == test_ref, "La référence retournée diffère de celle envoyée.")
        assert_true(result.get("model_version") == model_version, "Version modèle inattendue dans la réponse.")
        assert_true(abs(float(result.get("threshold", -1)) - threshold) < 1e-9, "Seuil de réponse incohérent.")
        probability = float(result.get("default_probability", -1))
        recommendation = result.get("recommendation")
        assert_true(math.isfinite(probability) and 0 <= probability <= 1, "Probabilité hors intervalle [0,1].")
        expected_recommendation = "REFUSE" if probability >= threshold else "ACCEPT"
        assert_true(recommendation == expected_recommendation, "Recommandation incohérente avec la probabilité et le seuil.")
        assert_true(result.get("tracking_storage") == "Azure SQL", "Réponse n'indiquant pas Azure SQL.")
        app_id = int(result["application_id"])
        pred_id = int(result["prediction_id"])
        print(
            f"[OK] POST /predict : HTTP 201, application_id={app_id}, "
            f"prediction_id={pred_id}, P(default)={probability:.4f}, décision={recommendation}"
        )

        # 4. Confirm rows exist in Azure SQL, not just in the HTTP response.
        conn = connect(make_connection_string(), timeout=60)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT application_id, feature_snapshot_sha256 "
                "FROM dbo.loan_applications WHERE application_ref = ?",
                (test_ref,),
            )
            app_rows = cursor.fetchall()
            assert_true(len(app_rows) == 1, f"Azure SQL contient {len(app_rows)} demandes pour la référence de test.")
            db_app_id = int(app_rows[0][0])
            snapshot_hash = str(app_rows[0][1])
            assert_true(db_app_id == app_id, "L'ID de demande dans Azure diffère de la réponse API.")
            assert_true(bool(snapshot_hash), "Empreinte du snapshot absente en base.")

            cursor.execute(
                "SELECT prediction_id, model_version, default_probability, threshold, "
                "recommended_decision, feature_snapshot_sha256 "
                "FROM dbo.prediction_logs WHERE application_id = ?",
                (db_app_id,),
            )
            prediction_rows = cursor.fetchall()
            assert_true(len(prediction_rows) == 1, f"Azure SQL contient {len(prediction_rows)} prédictions pour la demande de test.")
            db_prediction = prediction_rows[0]
            assert_true(int(db_prediction[0]) == pred_id, "L'ID de prédiction Azure diffère de la réponse API.")
            assert_true(str(db_prediction[1]) == model_version, "Version modèle incohérente en base.")
            assert_true(abs(float(db_prediction[2]) - probability) < 1e-6, "Probabilité incohérente en base.")
            assert_true(abs(float(db_prediction[3]) - threshold) < 1e-9, "Seuil incohérent en base.")
            assert_true(str(db_prediction[4]) == recommendation, "Recommandation incohérente en base.")
            assert_true(str(db_prediction[5]) == snapshot_hash, "L'empreinte du snapshot diffère entre les tables.")
        finally:
            cursor.close()
            conn.close()
        print("[OK] Azure SQL : demande et prédiction réellement persistées, valeurs concordantes")

        # 5. Retry with same reference must not duplicate the application.
        status, duplicate_body = api_request(args.base_url, "POST", "/predict", payload)
        assert_true(status == 409, f"Référence dupliquée : HTTP {status} au lieu de 409, réponse={duplicate_body}")
        conn = connect(make_connection_string(), timeout=60)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM dbo.loan_applications WHERE application_ref = ?",
                (test_ref,),
            )
            assert_true(int(cursor.fetchone()[0]) == 1, "La demande dupliquée a créé plusieurs lignes.")
            cursor.execute(
                "SELECT COUNT(*) FROM dbo.prediction_logs WHERE application_id = ?",
                (app_id,),
            )
            assert_true(int(cursor.fetchone()[0]) == 1, "La tentative dupliquée a créé plusieurs prédictions.")
        finally:
            cursor.close()
            conn.close()
        print("[OK] Idempotence : la même référence est refusée (HTTP 409), sans doublon")

        # 6. Invalid category must be rejected before DB write.
        invalid_ref = f"e2e-invalid-{uuid.uuid4().hex}"
        invalid_payload = dict(base_payload)
        invalid_payload["application_ref"] = invalid_ref
        invalid_payload["loan_grade"] = "Z"
        status, invalid_body = api_request(args.base_url, "POST", "/predict", invalid_payload)
        assert_true(status == 422, f"Catégorie invalide : HTTP {status} au lieu de 422, réponse={invalid_body}")
        count_apps, _, _ = db_counts_for_reference(invalid_ref)
        assert_true(count_apps == 0, "Une requête invalide a été enregistrée dans Azure SQL.")
        print("[OK] Validation : catégorie invalide refusée (HTTP 422), aucune demande écrite")

        # 7. Required feature omission must be rejected before DB write.
        missing_ref = f"e2e-missing-{uuid.uuid4().hex}"
        missing_payload = dict(base_payload)
        missing_payload["application_ref"] = missing_ref
        missing_payload.pop("person_income", None)
        status, missing_body = api_request(args.base_url, "POST", "/predict", missing_payload)
        assert_true(status == 422, f"Feature manquante : HTTP {status} au lieu de 422, réponse={missing_body}")
        count_apps, _, _ = db_counts_for_reference(missing_ref)
        assert_true(count_apps == 0, "Une requête incomplète a été enregistrée dans Azure SQL.")
        print("[OK] Validation : feature obligatoire absente refusée (HTTP 422), aucune demande écrite")

        print("\nRÉSULTAT : tous les tests de bout en bout ont réussi.")
        print(f"Référence de test ajoutée dans Azure SQL : {test_ref}")
        print("Cette demande est une donnée technique de test, pas une décision de crédit réelle.")
        return 0

    except Exception as exc:
        print(f"\nÉCHEC DU TEST : {type(exc).__name__}: {exc}")
        print("Consulte également le terminal Uvicorn pour le traceback serveur si nécessaire.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
