from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sql_compat import connect
from pydantic import BaseModel, ConfigDict, Field
from fastapi.middleware.cors import CORSMiddleware
# =====================================================================
# 1. CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "data" / "production" / "finpredict_production_model.joblib"
ENV_PATH = PROJECT_ROOT / ".env"

EXPECTED_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
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

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("finpredict.api")


# =====================================================================
# 2. LOAD THE EXISTING PRODUCTION ARTIFACT
#    The .joblib file is a DICTIONARY containing "pipeline", not the
#    pipeline object itself.
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
    raise TypeError(
        "The artifact does not contain a valid pipeline with predict_proba()."
    )
if FEATURE_COLUMNS != EXPECTED_FEATURES:
    raise RuntimeError(
        "Feature order/schema in the model artifact does not match this API. "
        f"Artifact features: {FEATURE_COLUMNS}"
    )
if not 0.0 <= PRODUCTION_THRESHOLD <= 1.0:
    raise ValueError("The production threshold in the artifact is outside [0, 1].")


# =====================================================================
# 3. AZURE SQL CONNECTION
# =====================================================================

def quote_connection_value(value: str) -> str:
    if any(char in value for char in ";={} "):
        return "{" + value.replace("}", "}}") + "}"
    return value


def make_connection_string() -> str:
    load_dotenv(ENV_PATH)
    # The API uses a dedicated least-privilege database user.
    # Keep SQL_USERNAME / SQL_PASSWORD for administrative CLI tasks only.
    values = {
        "SQL_SERVER": os.getenv("SQL_SERVER"),
        "SQL_DATABASE": os.getenv("SQL_DATABASE"),
        "SQL_APP_USERNAME": os.getenv("SQL_APP_USERNAME"),
        "SQL_APP_PASSWORD": os.getenv("SQL_APP_PASSWORD"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing Azure SQL settings in .env: " + ", ".join(missing)
        )

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
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_db_connection():
    return connect(make_connection_string(), timeout=60)


# =====================================================================
# 4. REQUEST/RESPONSE SCHEMAS
# =====================================================================

class LoanApplication(BaseModel):
    """One loan application; do not include names, ID numbers, or passwords."""

    model_config = ConfigDict(extra="forbid")

    # Optional, pseudonymous idempotency/reference key. A UUID is generated
    # if the caller does not provide one.
    application_ref: str | None = Field(default=None, min_length=1, max_length=255)

    person_age: int | None = Field(..., ge=18, le=100)
    person_income: float | None = Field(..., gt=0, allow_inf_nan=False)
    loan_amnt: float | None = Field(..., gt=0, allow_inf_nan=False)
    loan_int_rate: float | None = Field(..., ge=0, allow_inf_nan=False)
    person_emp_length: float | None = Field(..., ge=0, allow_inf_nan=False)
    loan_percent_income: float | None = Field(..., ge=0, le=1, allow_inf_nan=False)
    cb_person_cred_hist_length: int | None = Field(..., ge=0)

    person_home_ownership: str | None = Field(..., max_length=40)
    loan_intent: str | None = Field(..., max_length=60)
    loan_grade: str | None = Field(..., max_length=5)
    cb_person_default_on_file: str | None = Field(..., max_length=5)
    interest_income_ratio: float | None = Field(..., ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    application_id: int
    application_ref: str
    prediction_id: int
    model_version: str
    default_probability: float
    threshold: float
    recommendation: Literal["ACCEPT", "REFUSE"]
    tracking_storage: Literal["Azure SQL"] = "Azure SQL"


# =====================================================================
# 5. FASTAPI
# =====================================================================

app = FastAPI(
    title="FinPredict AI Pro API",
    version="1.0.0",
    description=(
        "Credit-default risk inference using the saved FinPredict production "
        "pipeline. Each accepted prediction is transactionally recorded in Azure SQL. "
        "The result is a model recommendation, not an automatic lending decision."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

@app.get("/", tags=["General"])
def root():
    return {
        "application": "FinPredict AI Pro API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Monitoring"])
def health():
    """Health endpoint also verifies that Azure SQL can be reached."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": MODEL_VERSION,
            "azure_sql": "connected",
        }
    except Exception:
        logger.exception("Health check could not reach Azure SQL.")
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "model_loaded": True,
                "model_version": MODEL_VERSION,
                "azure_sql": "unavailable",
            },
        )
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "threshold": PRODUCTION_THRESHOLD,
        "target": "loan_status",
        "positive_class": 1,
        "positive_class_meaning": "default",
        "features": FEATURE_COLUMNS,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=201,
    tags=["Predictions"],
)
def predict(application: LoanApplication) -> PredictionResponse:
    # 1. Extract model features separately from the tracking reference.
    payload = application.model_dump()
    application_ref = payload.pop("application_ref", None) or str(uuid.uuid4())

    # Normalize category casing and validate the known categories from the
    # training data. Null values are passed to the fitted imputer as NaN.
    for column in CATEGORICAL_FEATURES:
        value = payload[column]
        if value is None:
            continue
        normalized = value.strip().upper()
        if normalized not in ALLOWED_CATEGORIES[column]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Invalid category for {column}.",
                    "allowed_values": sorted(ALLOWED_CATEGORIES[column]),
                },
            )
        payload[column] = normalized

    snapshot = {column: payload[column] for column in FEATURE_COLUMNS}
    snapshot_json = canonical_json(snapshot)
    snapshot_hash = sha256_text(snapshot_json)

    # 2. Calculate risk using the exact saved production pipeline.
    # Convert None to np.nan so the fitted SimpleImputer can process missing
    # numeric and categorical values in the same way as the training pipeline.
    input_df = pd.DataFrame([snapshot], columns=FEATURE_COLUMNS)
    input_df = input_df.astype(object).where(pd.notna(input_df), np.nan)

    inference_started = time.perf_counter()
    try:
        probability = float(MODEL.predict_proba(input_df)[0, 1])
        latency_ms = (time.perf_counter() - inference_started) * 1000.0
    except Exception as exc:
        logger.exception("Production model inference failed.")
        raise HTTPException(
            status_code=500,
            detail="The model could not calculate a prediction for this request.",
        ) from exc

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        logger.error("The model returned an invalid default probability.")
        raise HTTPException(
            status_code=500,
            detail="The model returned an invalid prediction.",
        )

    recommendation = "REFUSE" if probability >= PRODUCTION_THRESHOLD else "ACCEPT"
    now = utc_now()

    # 3. Insert the application and prediction together in one Azure SQL
    # transaction. If cloud tracking fails, do not return an untracked result.
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # The API only logs predictions for the registered ACTIVE model, and
        # refuses to write if DB and artifact thresholds are inconsistent.
        cursor.execute(
            "SELECT threshold, status FROM dbo.model_versions WHERE model_version = ?",
            (MODEL_VERSION,),
        )
        registered_model = cursor.fetchone()
        if registered_model is None or str(registered_model[1]).upper() != "ACTIVE":
            conn.rollback()
            raise HTTPException(
                status_code=503,
                detail="The production model is not registered as ACTIVE in Azure SQL.",
            )

        registered_threshold = float(registered_model[0])
        if abs(registered_threshold - PRODUCTION_THRESHOLD) > 1e-9:
            conn.rollback()
            raise HTTPException(
                status_code=503,
                detail="The model threshold differs between the artifact and Azure SQL.",
            )

        # Idempotency safety: a caller-supplied reference may be used only once.
        cursor.execute(
            "SELECT application_id FROM dbo.loan_applications WHERE application_ref = ?",
            (application_ref,),
        )
        existing_application = cursor.fetchone()
        if existing_application is not None:
            conn.rollback()
            raise HTTPException(
                status_code=409,
                detail="This application_ref has already been recorded. Use a new reference for a new application.",
            )

        # Return the identity from the same INSERT statement. Calling
        # SCOPE_IDENTITY() in a separate cursor.execute() runs in a different
        # SQL batch/scope and can return NULL with Azure SQL.
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
            VALUES (?, ?, ?, ?, ?, ?, ?, N'{}', ?, ?)
            """,
            (
                application_id,
                MODEL_VERSION,
                now,
                probability,
                PRODUCTION_THRESHOLD,
                recommendation,
                snapshot_hash,
                latency_ms,
                now,
            ),
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
            conn.rollback()
        logger.exception("Azure SQL tracking write failed; prediction was not returned.")
        raise HTTPException(
            status_code=503,
            detail="Prediction tracking is unavailable. No result was returned; retry after service recovery.",
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
    )
