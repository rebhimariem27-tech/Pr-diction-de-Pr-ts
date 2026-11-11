from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .explain import ContributionExplainer
from .features import business_rule_violations, derive_features
from .security import ApiAuth
from .sql_compat import connect

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv("FINPREDICT_MODEL_PATH")
    or PROJECT_ROOT / "data" / "production" / "finpredict_production_model.joblib"
)
# Mettez le .env HORS du dossier OneDrive : FINPREDICT_ENV_FILE=C:\secrets\finpredict.env
ENV_PATH = Path(os.getenv("FINPREDICT_ENV_FILE") or PROJECT_ROOT / ".env")
load_dotenv(ENV_PATH)

EXPECTED_FEATURES = [
    "person_age", "person_income", "loan_amnt", "loan_int_rate",
    "person_emp_length", "loan_percent_income", "cb_person_cred_hist_length",
    "interest_income_ratio", "person_home_ownership", "loan_intent",
    "loan_grade", "cb_person_default_on_file",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]
ALLOWED_CATEGORIES = {
    "person_home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_intent": {
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION",
    },
    "loan_grade": {"A", "B", "C", "D", "E", "F", "G"},
    "cb_person_default_on_file": {"Y", "N"},
}
# Écart toléré entre une variable dérivée envoyée par un client et la valeur serveur.
DERIVED_MISMATCH_TOLERANCE = 0.01

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("finpredict.api")
access_logger = logging.getLogger("finpredict.access")


# =====================================================================
# 2. ARTEFACT DE PRODUCTION
# =====================================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Production model artifact not found: {MODEL_PATH}. "
        "Run the production training/export script first."
    )

ARTIFACT = joblib.load(MODEL_PATH)
if not isinstance(ARTIFACT, dict):
    raise TypeError("The production artifact must be a dictionary.")

MODEL = ARTIFACT.get("pipeline")
MODEL_VERSION = str(ARTIFACT.get("model_version", "1.0.0"))
PRODUCTION_THRESHOLD = float(ARTIFACT.get("production_threshold", 0.48))
FEATURE_COLUMNS = list(ARTIFACT.get("model_features", EXPECTED_FEATURES))

if MODEL is None or not hasattr(MODEL, "predict_proba"):
    raise TypeError("The artifact does not contain a valid pipeline with predict_proba().")
if FEATURE_COLUMNS != EXPECTED_FEATURES:
    raise RuntimeError(
        "Feature order/schema in the model artifact does not match this API. "
        f"Artifact features: {FEATURE_COLUMNS}"
    )
if not 0.0 <= PRODUCTION_THRESHOLD <= 1.0:
    raise ValueError("The production threshold in the artifact is outside [0, 1].")

try:
    EXPLAINER: ContributionExplainer | None = ContributionExplainer(
        MODEL, FEATURE_COLUMNS, CATEGORICAL_FEATURES
    )
except Exception:
    logger.exception("Explainer unavailable: predictions will be returned without explanation.")
    EXPLAINER = None

AUTH = ApiAuth()  # échoue au démarrage si aucune clé n'est configurée


# =====================================================================
# 3. AZURE SQL
# =====================================================================

def quote_connection_value(value: str) -> str:
    if any(char in value for char in ";={} "):
        return "{" + value.replace("}", "}}") + "}"
    return value


def make_connection_string() -> str:
    values = {
        "SQL_SERVER": os.getenv("SQL_SERVER"),
        "SQL_DATABASE": os.getenv("SQL_DATABASE"),
        "SQL_APP_USERNAME": os.getenv("SQL_APP_USERNAME"),
        "SQL_APP_PASSWORD": os.getenv("SQL_APP_PASSWORD"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError("Missing Azure SQL settings in .env: " + ", ".join(missing))
    return (
        f"Server={quote_connection_value(values['SQL_SERVER'])};"
        f"Database={quote_connection_value(values['SQL_DATABASE'])};"
        f"UID={quote_connection_value(values['SQL_APP_USERNAME'])};"
        f"PWD={quote_connection_value(values['SQL_APP_PASSWORD'])};"
        "Encrypt=yes;TrustServerCertificate=no;MultiSubnetFailover=yes;"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_db_connection():
    return connect(make_connection_string(), timeout=60)


def is_duplicate_key_error(exc: Exception) -> bool:
    """Violation de l'index unique sur application_ref (course entre deux requêtes).
    SQL Server : erreurs 2601 (index unique) / 2627 (contrainte unique)."""
    text = str(exc)
    return (
        "UX_loan_applications_application_ref" in text
        or "2601" in text
        or "2627" in text
        or type(exc).__name__ == "IntegrityError"
    )


# =====================================================================
# 4. SCHEMAS
# =====================================================================

class LoanApplication(BaseModel):
    """Une demande de prêt ; n'incluez ni nom, ni pièce d'identité, ni mot de passe.

    Les variables dérivées (loan_percent_income, interest_income_ratio) sont
    calculées par le SERVEUR. Si un ancien client les envoie encore, elles sont
    ignorées (un écart notable est journalisé)."""

    model_config = ConfigDict(extra="forbid")

    application_ref: str | None = Field(default=None, min_length=1, max_length=255)

    person_age: int | None = Field(..., ge=18, le=100)
    person_income: float | None = Field(..., gt=0, allow_inf_nan=False)
    loan_amnt: float | None = Field(..., gt=0, allow_inf_nan=False)
    loan_int_rate: float | None = Field(..., ge=0, allow_inf_nan=False)
    person_emp_length: float | None = Field(..., ge=0, allow_inf_nan=False)
    cb_person_cred_hist_length: int | None = Field(..., ge=0)

    person_home_ownership: str | None = Field(..., max_length=40)
    loan_intent: str | None = Field(..., max_length=60)
    loan_grade: str | None = Field(..., max_length=5)
    cb_person_default_on_file: str | None = Field(..., max_length=5)

    # Dépréciés : acceptés pour compatibilité, jamais utilisés pour le scoring.
    loan_percent_income: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    interest_income_ratio: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_business_rules(self) -> "LoanApplication":
        problems = business_rule_violations(
            person_age=self.person_age,
            person_emp_length=self.person_emp_length,
            cb_person_cred_hist_length=self.cb_person_cred_hist_length,
            loan_amnt=self.loan_amnt,
            person_income=self.person_income,
            loan_int_rate=self.loan_int_rate,
        )
        if problems:
            raise ValueError(" ".join(problems))
        return self


class PredictionResponse(BaseModel):
    application_id: int
    application_ref: str
    prediction_id: int
    model_version: str
    default_probability: float
    threshold: float
    recommendation: Literal["ACCEPT", "REFUSE"]
    derived_features: dict[str, float | None]
    explanation: dict[str, Any]
    tracking_storage: Literal["Azure SQL"] = "Azure SQL"


class PortfolioRow(BaseModel):
    application_id: int
    prediction_id: int
    application_ref: str
    submitted_at_utc: str
    loan_amnt: float | None = None
    default_probability: float
    recommendation: Literal["ACCEPT", "REFUSE"]
    actual_decision: str
    model_version: str


class DashboardSummary(BaseModel):
    period_days: int
    data_available: bool
    analyzed_count: int
    average_default_probability: float | None = None
    recommendation_counts: dict[str, int]
    actual_decision_counts: dict[str, int]
    total_loan_amount: float
    monthly: list[dict[str, str | int | float]]
    latest_submitted_at_utc: str | None = None
    model_version: str | None = None


class MonitoringSummary(BaseModel):
    generated_at_utc: str
    overall_status: Literal["NORMAL", "WARNING", "ALERT", "UNKNOWN"]
    demo_mode: bool
    reference_rows: int
    current_rows: int
    high_drift_features: int
    moderate_drift_features: int
    high_missingness_features: int
    prediction_psi: float | None = None
    prediction_ks: float | None = None
    threshold: float
    reference_mean_default_probability: float | None = None
    current_mean_default_probability: float | None = None
    refusal_rate_delta_pp: float | None = None
    model_performance: dict[str, Any] | None = None
    drift_features: list[dict[str, Any]] = Field(default_factory=list)
    fairness: dict[str, Any] = Field(default_factory=dict)


class HumanDecisionUpdate(BaseModel):
    actual_decision: Literal["PENDING", "APPROVED", "REJECTED", "MANUAL_REVIEW"]


# =====================================================================
# 5. FASTAPI
# =====================================================================

app = FastAPI(
    title="FinPredict AI Pro API",
    version="1.1.0",
    description=(
        "Credit-default risk inference using the saved FinPredict production "
        "pipeline. Each accepted prediction is transactionally recorded in Azure SQL. "
        "The result is a model recommendation, not an automatic lending decision."
    ),
)

_origins = [o.strip() for o in os.getenv(
    "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-API-Key"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Journal d'accès : qui, quoi, résultat. Jamais le corps de la requête."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        access_logger.info(
            "request_id=%s method=%s path=%s status=%s caller=%s ip=%s duration_ms=%.1f",
            request_id, request.method, request.url.path, status,
            getattr(request.state, "caller", "-"),
            request.client.host if request.client else "-",
            (time.perf_counter() - started) * 1000.0,
        )


@app.get("/", tags=["General"])
def root():
    return {"application": "FinPredict AI Pro API", "status": "running", "docs": "/docs"}


@app.get("/health", tags=["Monitoring"])
def health():
    """Health endpoint (public) : vérifie aussi que Azure SQL est joignable."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return {"status": "healthy", "model_loaded": True,
                "model_version": MODEL_VERSION, "azure_sql": "connected"}
    except Exception:
        logger.exception("Health check could not reach Azure SQL.")
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": True,
                     "model_version": MODEL_VERSION, "azure_sql": "unavailable"},
        )
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


@app.get("/model-info", tags=["Model"], dependencies=[Depends(AUTH)])
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "threshold": PRODUCTION_THRESHOLD,
        "target": "loan_status",
        "positive_class": 1,
        "positive_class_meaning": "default",
        "features": FEATURE_COLUMNS,
        "server_computed_features": ["loan_percent_income", "interest_income_ratio"],
        "explanation_available": EXPLAINER is not None,
    }


@app.get("/dashboard-summary", response_model=DashboardSummary, tags=["Monitoring"])
def dashboard_summary(period_days: int = 30, caller: str = Depends(AUTH)) -> DashboardSummary:
    """Return real persisted aggregates for the dashboard."""
    if period_days < 1 or period_days > 3650:
        raise HTTPException(422, "period_days must be between 1 and 3650.")

    latest_cte = """
        WITH latest AS (
            SELECT la.application_id, la.submitted_at_utc, la.actual_decision,
                   la.feature_snapshot_json, pl.default_probability,
                   pl.recommended_decision, pl.model_version
            FROM dbo.loan_applications AS la
            INNER JOIN dbo.prediction_logs AS pl
                ON pl.application_id = la.application_id
            WHERE pl.prediction_id = (
                SELECT TOP 1 newest.prediction_id
                FROM dbo.prediction_logs AS newest
                WHERE newest.application_id = la.application_id
                ORDER BY newest.predicted_at_utc DESC, newest.prediction_id DESC
            )
              AND la.submitted_at_utc >= CONVERT(
                    NVARCHAR(50), DATEADD(day, ?, SYSUTCDATETIME()), 127
                  )
        )
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            latest_cte
            + """
            SELECT COUNT_BIG(*) AS analyzed_count,
                   AVG(default_probability) AS average_default_probability,
                   SUM(CASE WHEN recommended_decision = N'ACCEPT' THEN 1 ELSE 0 END) AS accepted_count,
                   SUM(CASE WHEN recommended_decision = N'REFUSE' THEN 1 ELSE 0 END) AS refused_count,
                   SUM(CASE WHEN actual_decision = N'PENDING' THEN 1 ELSE 0 END) AS pending_count,
                   SUM(CASE WHEN actual_decision = N'APPROVED' THEN 1 ELSE 0 END) AS approved_count,
                   SUM(CASE WHEN actual_decision = N'REJECTED' THEN 1 ELSE 0 END) AS rejected_count,
                   SUM(CASE WHEN actual_decision = N'MANUAL_REVIEW' THEN 1 ELSE 0 END) AS manual_review_count,
                   SUM(COALESCE(TRY_CONVERT(float, JSON_VALUE(feature_snapshot_json, '$.loan_amnt')), 0)) AS total_loan_amount,
                   MAX(submitted_at_utc) AS latest_submitted_at_utc,
                   MAX(model_version) AS model_version
            FROM latest
            """,
            (-period_days,),
        )
        summary = cursor.fetchone()
        if summary is None:
            raise RuntimeError("Azure SQL returned no dashboard summary.")

        cursor.execute(
            latest_cte
            + """
            SELECT LEFT(submitted_at_utc, 7) AS month,
                   COUNT_BIG(*) AS analyzed_count,
                   AVG(default_probability) AS average_default_probability
            FROM latest
            GROUP BY LEFT(submitted_at_utc, 7)
            ORDER BY month
            """,
            (-period_days,),
        )
        monthly_rows = cursor.fetchall()

        def count(value: Any) -> int:
            return int(value or 0)

        monthly = [
            {
                "month": str(row[0]),
                "analyzed_count": count(row[1]),
                "average_default_probability": float(row[2] or 0.0),
            }
            for row in monthly_rows
        ]
        analyzed_count = count(summary[0])
        return DashboardSummary(
            period_days=period_days,
            data_available=analyzed_count > 0,
            analyzed_count=analyzed_count,
            average_default_probability=(
                float(summary[1]) if summary[1] is not None else None
            ),
            recommendation_counts={
                "ACCEPT": count(summary[2]),
                "REFUSE": count(summary[3]),
            },
            actual_decision_counts={
                "PENDING": count(summary[4]),
                "APPROVED": count(summary[5]),
                "REJECTED": count(summary[6]),
                "MANUAL_REVIEW": count(summary[7]),
            },
            total_loan_amount=float(summary[8] or 0.0),
            monthly=monthly,
            latest_submitted_at_utc=(
                str(summary[9]) if summary[9] is not None else None
            ),
            model_version=str(summary[10]) if summary[10] is not None else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Azure SQL dashboard aggregation failed.")
        raise HTTPException(503, "Dashboard data is unavailable.") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


@app.patch(
    "/applications/{application_id}/decision",
    response_model=PortfolioRow,
    tags=["Human review"],
)
def update_human_decision(
    application_id: int,
    update: HumanDecisionUpdate,
    caller: str = Depends(AUTH),
) -> PortfolioRow:
    """Persist the human decision separately from the model recommendation."""
    if application_id < 1:
        raise HTTPException(422, "application_id must be positive.")

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT TOP 1 la.application_id, pl.prediction_id, la.application_ref,
                   la.submitted_at_utc, la.feature_snapshot_json,
                   pl.default_probability, pl.recommended_decision,
                   la.actual_decision, pl.model_version
            FROM dbo.loan_applications AS la
            INNER JOIN dbo.prediction_logs AS pl
                ON pl.application_id = la.application_id
            WHERE la.application_id = ?
            ORDER BY pl.predicted_at_utc DESC, pl.prediction_id DESC
            """,
            (application_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(404, "Application not found.")

        now = utc_now()
        cursor.execute(
            """
            UPDATE dbo.loan_applications
            SET actual_decision = ?, decision_at_utc = ?, updated_at_utc = ?
            WHERE application_id = ?
            """,
            (update.actual_decision, now, now, application_id),
        )
        conn.commit()
        logger.info(
            "Human decision updated application_id=%s decision=%s caller=%s",
            application_id, update.actual_decision, caller,
        )

        snapshot = json.loads(row[4])
        return PortfolioRow(
            application_id=int(row[0]),
            prediction_id=int(row[1]),
            application_ref=str(row[2]),
            submitted_at_utc=str(row[3]),
            loan_amnt=float(snapshot["loan_amnt"]) if snapshot.get("loan_amnt") is not None else None,
            default_probability=float(row[5]),
            recommendation=str(row[6]),
            actual_decision=update.actual_decision,
            model_version=str(row[8]),
        )
    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        logger.exception("Human decision update failed.")
        raise HTTPException(503, "Human decision could not be saved.") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


@app.post("/predict", response_model=PredictionResponse, status_code=201, tags=["Predictions"])
def predict(application: LoanApplication, caller: str = Depends(AUTH)) -> PredictionResponse:
    # 1. Séparer la référence de suivi et les features.
    payload = application.model_dump()
    application_ref = payload.pop("application_ref", None) or str(uuid.uuid4())
    client_derived = {k: payload.pop(k) for k in ("loan_percent_income", "interest_income_ratio")}

    # Variables dérivées : toujours recalculées côté serveur (anti training-serving skew).
    derived = derive_features(payload["loan_amnt"], payload["person_income"], payload["loan_int_rate"])
    for name, server_value in derived.items():
        sent = client_derived[name]
        if sent is not None and server_value is not None and abs(sent - server_value) > DERIVED_MISMATCH_TOLERANCE:
            logger.warning(
                "Client-supplied %s=%s differs from server value %.6f (caller=%s); using server value.",
                name, sent, server_value, caller,
            )
    payload.update(derived)

    for column in CATEGORICAL_FEATURES:
        value = payload[column]
        if value is None:
            continue
        normalized = value.strip().upper()
        if normalized not in ALLOWED_CATEGORIES[column]:
            raise HTTPException(
                status_code=422,
                detail={"message": f"Invalid category for {column}.",
                        "allowed_values": sorted(ALLOWED_CATEGORIES[column])},
            )
        payload[column] = normalized

    snapshot = {column: payload[column] for column in FEATURE_COLUMNS}
    snapshot_json = canonical_json(snapshot)
    snapshot_hash = sha256_text(snapshot_json)

    # 2. Score avec le pipeline de production exact.
    input_df = pd.DataFrame([snapshot], columns=FEATURE_COLUMNS)
    input_df = input_df.astype(object).where(pd.notna(input_df), np.nan)

    inference_started = time.perf_counter()
    try:
        probability = float(MODEL.predict_proba(input_df)[0, 1])
        latency_ms = (time.perf_counter() - inference_started) * 1000.0
    except Exception as exc:
        logger.exception("Production model inference failed.")
        raise HTTPException(500, "The model could not calculate a prediction for this request.") from exc

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        logger.error("The model returned an invalid default probability.")
        raise HTTPException(500, "The model returned an invalid prediction.")

    recommendation = "REFUSE" if probability >= PRODUCTION_THRESHOLD else "ACCEPT"

    # 2b. Explication (codes de raison). Non bloquante si elle échoue.
    explanation: dict[str, Any] = {}
    if EXPLAINER is not None:
        try:
            explanation = EXPLAINER.explain(input_df, expected_probability=probability)
            explanation["threshold"] = PRODUCTION_THRESHOLD
        except Exception:
            logger.exception("Explanation failed; continuing without it.")
    explanation_json = canonical_json(explanation)

    now = utc_now()

    # 3. Écriture transactionnelle dans Azure SQL.
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT threshold, status FROM dbo.model_versions WHERE model_version = ?",
            (MODEL_VERSION,),
        )
        registered_model = cursor.fetchone()
        if registered_model is None or str(registered_model[1]).upper() != "ACTIVE":
            conn.rollback()
            raise HTTPException(503, "The production model is not registered as ACTIVE in Azure SQL.")

        if abs(float(registered_model[0]) - PRODUCTION_THRESHOLD) > 1e-9:
            conn.rollback()
            raise HTTPException(503, "The model threshold differs between the artifact and Azure SQL.")

        # Contrôle rapide d'idempotence (réponse 409 propre dans le cas courant).
        cursor.execute(
            "SELECT application_id FROM dbo.loan_applications WHERE application_ref = ?",
            (application_ref,),
        )
        if cursor.fetchone() is not None:
            conn.rollback()
            raise HTTPException(409, "This application_ref has already been recorded. "
                                     "Use a new reference for a new application.")

        cursor.execute(
            """
            INSERT INTO dbo.loan_applications (
                application_ref, submitted_at_utc, feature_snapshot_json,
                feature_snapshot_sha256, actual_decision,
                created_at_utc, updated_at_utc
            )
            OUTPUT INSERTED.application_id
            VALUES (?, ?, ?, ?, N'PENDING', ?, ?)
            """,
            (application_ref, now, snapshot_json, snapshot_hash, now, now),
        )
        inserted_application = cursor.fetchone()
        if inserted_application is None or inserted_application[0] is None:
            raise RuntimeError("Azure SQL did not return the new application_id.")
        application_id = int(inserted_application[0])

        cursor.execute(
            """
            INSERT INTO dbo.prediction_logs (
                application_id, model_version, predicted_at_utc,
                default_probability, threshold, recommended_decision,
                feature_snapshot_sha256, explanation_json, latency_ms,
                created_at_utc
            )
            OUTPUT INSERTED.prediction_id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (application_id, MODEL_VERSION, now, probability, PRODUCTION_THRESHOLD,
             recommendation, snapshot_hash, explanation_json, latency_ms, now),
        )
        inserted_prediction = cursor.fetchone()
        if inserted_prediction is None or inserted_prediction[0] is None:
            raise RuntimeError("Azure SQL did not return the new prediction_id.")
        prediction_id = int(inserted_prediction[0])
        conn.commit()

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        if is_duplicate_key_error(exc):
            # Deux requêtes simultanées avec la même référence : l'index unique a tranché.
            logger.info("Concurrent duplicate application_ref rejected by unique index.")
            raise HTTPException(409, "This application_ref has already been recorded. "
                                     "Use a new reference for a new application.") from exc
        logger.exception("Azure SQL tracking write failed; prediction was not returned.")
        raise HTTPException(
            503, "Prediction tracking is unavailable. No result was returned; retry after service recovery."
        ) from exc
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    return PredictionResponse(
        application_id=application_id,
        application_ref=application_ref,
        prediction_id=prediction_id,
        model_version=MODEL_VERSION,
        default_probability=round(probability, 6),
        threshold=PRODUCTION_THRESHOLD,
        recommendation=recommendation,
        derived_features=derived,
        explanation=explanation,
    )


@app.get("/monitoring-summary", response_model=MonitoringSummary, tags=["Monitoring"])
def monitoring_summary(caller: str = Depends(AUTH)) -> MonitoringSummary:
    """Expose the latest generated drift, data-quality and fairness report."""
    report_path = PROJECT_ROOT / "data" / "reports" / "monitoring" / "step10_monitoring_report.json"
    if not report_path.is_file():
        logger.error("Monitoring report is missing: %s", report_path)
        raise HTTPException(503, "Monitoring report unavailable. Run the monitoring pipeline first.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        prediction = report.get("prediction_monitoring", {})
        drift_path = PROJECT_ROOT / "data" / "reports" / "monitoring" / "step10_feature_drift.csv"
        drift_features: list[dict[str, Any]] = []
        if drift_path.is_file():
            import csv
            with drift_path.open(newline="", encoding="utf-8") as file:
                drift_features = list(csv.DictReader(file))
        fairness_path = PROJECT_ROOT / "data" / "reports" / "fairness_report.json"
        fairness = json.loads(fairness_path.read_text(encoding="utf-8")) if fairness_path.is_file() else {}
        return MonitoringSummary(
            generated_at_utc=str(report.get("generated_at_utc", "")),
            overall_status=str(summary.get("overall_status", "UNKNOWN")),
            demo_mode=bool(summary.get("demo_mode", False)),
            reference_rows=int(summary.get("reference_rows", 0)),
            current_rows=int(summary.get("current_rows", 0)),
            high_drift_features=int(summary.get("high_drift_features", 0)),
            moderate_drift_features=int(summary.get("moderate_drift_features", 0)),
            high_missingness_features=int(summary.get("high_missingness_features", 0)),
            prediction_psi=summary.get("prediction_psi"),
            prediction_ks=summary.get("prediction_ks"),
            threshold=float(summary.get("threshold", PRODUCTION_THRESHOLD)),
            reference_mean_default_probability=prediction.get("reference_mean_default_probability"),
            current_mean_default_probability=prediction.get("current_mean_default_probability"),
            refusal_rate_delta_pp=prediction.get("refusal_rate_delta_pp"),
            model_performance=report.get("model_performance"),
            drift_features=drift_features,
            fairness=fairness,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        logger.exception("Monitoring report could not be loaded.")
        raise HTTPException(503, "Monitoring report is invalid or unavailable.")


@app.get("/portfolio", response_model=list[PortfolioRow], tags=["Predictions"])
def portfolio(limit: int = 50, offset: int = 0, caller: str = Depends(AUTH)) -> list[PortfolioRow]:
    """Return the persisted predictions shown by the portfolio screen."""
    if limit < 1 or limit > 100:
        raise HTTPException(422, "limit must be between 1 and 100.")
    if offset < 0:
        raise HTTPException(422, "offset must be non-negative.")

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT la.application_id, pl.prediction_id, la.application_ref,
                   la.submitted_at_utc, la.feature_snapshot_json,
                   pl.default_probability, pl.recommended_decision,
                   la.actual_decision, pl.model_version
            FROM dbo.loan_applications AS la
            INNER JOIN dbo.prediction_logs AS pl
                ON pl.application_id = la.application_id
            WHERE pl.prediction_id = (
                SELECT TOP 1 latest.prediction_id
                FROM dbo.prediction_logs AS latest
                WHERE latest.application_id = la.application_id
                ORDER BY latest.predicted_at_utc DESC, latest.prediction_id DESC
            )
            ORDER BY pl.predicted_at_utc DESC, pl.prediction_id DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """,
            (offset, limit),
        )
        rows = cursor.fetchall()
        result: list[PortfolioRow] = []
        for row in rows:
            snapshot = json.loads(row[4])
            result.append(PortfolioRow(
                application_id=int(row[0]),
                prediction_id=int(row[1]),
                application_ref=str(row[2]),
                submitted_at_utc=str(row[3]),
                loan_amnt=float(snapshot["loan_amnt"]) if snapshot.get("loan_amnt") is not None else None,
                default_probability=float(row[5]),
                recommendation=str(row[6]),
                actual_decision=str(row[7]),
                model_version=str(row[8]),
            ))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Azure SQL portfolio read failed.")
        raise HTTPException(503, "Portfolio history is unavailable.") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
