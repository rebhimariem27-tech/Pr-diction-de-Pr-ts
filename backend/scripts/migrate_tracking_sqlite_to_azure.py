"""Migrate FinPredict Step 11 tracking data from SQLite to Azure SQL.

Expected project layout:
  .env
  data/tracking/finpredict_tracking.db

Required .env variables:
  SQL_SERVER, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD

This script creates the six tracking tables if missing, preserves primary-key IDs,
checks existing rows before skipping them, and verifies the rows after migration.
It does not delete or alter the source SQLite database.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from backend.app.sql_compat import connect

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DB = PROJECT_ROOT / "data" / "tracking" / "finpredict_tracking.db"

# Create parent tables before tables with foreign keys.
DDL_STATEMENTS = [
    """
    IF OBJECT_ID(N'dbo.model_versions', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.model_versions (
            model_version NVARCHAR(100) NOT NULL PRIMARY KEY,
            registered_at_utc NVARCHAR(50) NOT NULL,
            artifact_path NVARCHAR(1024) NOT NULL,
            artifact_sha256 NVARCHAR(64) NULL,
            dataset_version NVARCHAR(100) NULL,
            threshold FLOAT NOT NULL,
            metrics_json NVARCHAR(MAX) NOT NULL CONSTRAINT DF_model_versions_metrics DEFAULT N'{}',
            parameters_json NVARCHAR(MAX) NOT NULL CONSTRAINT DF_model_versions_parameters DEFAULT N'{}',
            status NVARCHAR(20) NOT NULL CONSTRAINT DF_model_versions_status DEFAULT N'CANDIDATE',
            notes NVARCHAR(MAX) NULL,
            CONSTRAINT CK_model_versions_threshold CHECK (threshold >= 0 AND threshold <= 1),
            CONSTRAINT CK_model_versions_status CHECK (status IN (N'CANDIDATE', N'ACTIVE', N'RETIRED'))
        );
    END;
    """,
    """
    IF OBJECT_ID(N'dbo.loan_applications', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.loan_applications (
            application_id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            application_ref NVARCHAR(255) NULL,
            submitted_at_utc NVARCHAR(50) NOT NULL,
            feature_snapshot_json NVARCHAR(MAX) NOT NULL,
            feature_snapshot_sha256 NVARCHAR(64) NOT NULL,
            actual_decision NVARCHAR(20) NOT NULL CONSTRAINT DF_loan_applications_decision DEFAULT N'PENDING',
            decision_at_utc NVARCHAR(50) NULL,
            funded_at_utc NVARCHAR(50) NULL,
            created_at_utc NVARCHAR(50) NOT NULL,
            updated_at_utc NVARCHAR(50) NOT NULL,
            CONSTRAINT CK_loan_applications_decision CHECK (
                actual_decision IN (N'PENDING', N'APPROVED', N'REJECTED', N'MANUAL_REVIEW')
            )
        );
    END;
    """,
    """
    IF NOT EXISTS (
        SELECT 1 FROM sys.indexes
        WHERE name = N'UX_loan_applications_application_ref'
          AND object_id = OBJECT_ID(N'dbo.loan_applications')
    )
    BEGIN
        CREATE UNIQUE INDEX UX_loan_applications_application_ref
            ON dbo.loan_applications(application_ref)
            WHERE application_ref IS NOT NULL;
    END;
    """,
    """
    IF OBJECT_ID(N'dbo.prediction_logs', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.prediction_logs (
            prediction_id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            application_id BIGINT NOT NULL,
            model_version NVARCHAR(100) NOT NULL,
            predicted_at_utc NVARCHAR(50) NOT NULL,
            default_probability FLOAT NOT NULL,
            threshold FLOAT NOT NULL,
            recommended_decision NVARCHAR(10) NOT NULL,
            feature_snapshot_sha256 NVARCHAR(64) NOT NULL,
            explanation_json NVARCHAR(MAX) NOT NULL CONSTRAINT DF_prediction_logs_explanation DEFAULT N'{}',
            latency_ms FLOAT NULL,
            created_at_utc NVARCHAR(50) NOT NULL,
            CONSTRAINT FK_prediction_logs_application FOREIGN KEY (application_id)
                REFERENCES dbo.loan_applications(application_id),
            CONSTRAINT FK_prediction_logs_model FOREIGN KEY (model_version)
                REFERENCES dbo.model_versions(model_version),
            CONSTRAINT CK_prediction_logs_probability CHECK (default_probability >= 0 AND default_probability <= 1),
            CONSTRAINT CK_prediction_logs_threshold CHECK (threshold >= 0 AND threshold <= 1),
            CONSTRAINT CK_prediction_logs_decision CHECK (recommended_decision IN (N'ACCEPT', N'REFUSE')),
            CONSTRAINT CK_prediction_logs_latency CHECK (latency_ms IS NULL OR latency_ms >= 0)
        );
    END;
    """,
    """
    IF OBJECT_ID(N'dbo.loan_outcome_events', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.loan_outcome_events (
            event_id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            application_id BIGINT NOT NULL,
            event_timestamp_utc NVARCHAR(50) NOT NULL,
            event_type NVARCHAR(30) NOT NULL,
            amount FLOAT NULL,
            days_past_due INT NULL,
            source NVARCHAR(255) NULL,
            notes NVARCHAR(MAX) NULL,
            recorded_at_utc NVARCHAR(50) NOT NULL,
            CONSTRAINT FK_loan_outcome_events_application FOREIGN KEY (application_id)
                REFERENCES dbo.loan_applications(application_id),
            CONSTRAINT CK_loan_outcome_events_type CHECK (event_type IN (
                N'PAYMENT_DUE', N'PAYMENT_RECEIVED', N'DELINQUENCY', N'DEFAULT_CONFIRMED',
                N'RECOVERY', N'LOAN_CLOSED', N'CHARGED_OFF'
            )),
            CONSTRAINT CK_loan_outcome_events_amount CHECK (amount IS NULL OR amount >= 0),
            CONSTRAINT CK_loan_outcome_events_days CHECK (days_past_due IS NULL OR days_past_due >= 0)
        );
    END;
    """,
    """
    IF OBJECT_ID(N'dbo.outcome_labels', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.outcome_labels (
            application_id BIGINT NOT NULL PRIMARY KEY,
            label INT NOT NULL,
            label_definition NVARCHAR(MAX) NOT NULL,
            observation_window_days INT NOT NULL,
            label_matured BIT NOT NULL CONSTRAINT DF_outcome_labels_matured DEFAULT (0),
            matured_at_utc NVARCHAR(50) NULL,
            source NVARCHAR(255) NULL,
            created_at_utc NVARCHAR(50) NOT NULL,
            updated_at_utc NVARCHAR(50) NOT NULL,
            CONSTRAINT FK_outcome_labels_application FOREIGN KEY (application_id)
                REFERENCES dbo.loan_applications(application_id),
            CONSTRAINT CK_outcome_labels_label CHECK (label IN (0, 1)),
            CONSTRAINT CK_outcome_labels_window CHECK (observation_window_days > 0)
        );
    END;
    """,
    """
    IF OBJECT_ID(N'dbo.retraining_runs', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.retraining_runs (
            run_id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            candidate_model_version NVARCHAR(100) NULL,
            started_at_utc NVARCHAR(50) NOT NULL,
            finished_at_utc NVARCHAR(50) NULL,
            dataset_version NVARCHAR(100) NULL,
            dataset_rows BIGINT NULL,
            metrics_json NVARCHAR(MAX) NOT NULL CONSTRAINT DF_retraining_runs_metrics DEFAULT N'{}',
            validation_status NVARCHAR(20) NOT NULL CONSTRAINT DF_retraining_runs_validation DEFAULT N'PENDING',
            deployment_status NVARCHAR(20) NOT NULL CONSTRAINT DF_retraining_runs_deployment DEFAULT N'NOT_DEPLOYED',
            notes NVARCHAR(MAX) NULL,
            CONSTRAINT CK_retraining_runs_rows CHECK (dataset_rows IS NULL OR dataset_rows >= 0),
            CONSTRAINT CK_retraining_runs_validation CHECK (
                validation_status IN (N'PENDING', N'PASSED', N'FAILED', N'CANCELLED')
            ),
            CONSTRAINT CK_retraining_runs_deployment CHECK (
                deployment_status IN (N'NOT_DEPLOYED', N'DEPLOYED', N'ROLLED_BACK')
            )
        );
    END;
    """,
]

TABLE_CONFIG: list[dict[str, Any]] = [
    {
        "name": "model_versions",
        "pk": "model_version",
        "identity": False,
        "columns": [
            "model_version", "registered_at_utc", "artifact_path", "artifact_sha256",
            "dataset_version", "threshold", "metrics_json", "parameters_json", "status", "notes",
        ],
    },
    {
        "name": "loan_applications",
        "pk": "application_id",
        "identity": True,
        "columns": [
            "application_id", "application_ref", "submitted_at_utc", "feature_snapshot_json",
            "feature_snapshot_sha256", "actual_decision", "decision_at_utc", "funded_at_utc",
            "created_at_utc", "updated_at_utc",
        ],
    },
    {
        "name": "prediction_logs",
        "pk": "prediction_id",
        "identity": True,
        "columns": [
            "prediction_id", "application_id", "model_version", "predicted_at_utc",
            "default_probability", "threshold", "recommended_decision", "feature_snapshot_sha256",
            "explanation_json", "latency_ms", "created_at_utc",
        ],
    },
    {
        "name": "loan_outcome_events",
        "pk": "event_id",
        "identity": True,
        "columns": [
            "event_id", "application_id", "event_timestamp_utc", "event_type", "amount",
            "days_past_due", "source", "notes", "recorded_at_utc",
        ],
    },
    {
        "name": "outcome_labels",
        "pk": "application_id",
        "identity": False,
        "columns": [
            "application_id", "label", "label_definition", "observation_window_days",
            "label_matured", "matured_at_utc", "source", "created_at_utc", "updated_at_utc",
        ],
    },
    {
        "name": "retraining_runs",
        "pk": "run_id",
        "identity": True,
        "columns": [
            "run_id", "candidate_model_version", "started_at_utc", "finished_at_utc",
            "dataset_version", "dataset_rows", "metrics_json", "validation_status",
            "deployment_status", "notes",
        ],
    },
]


def quote_connection_value(value: str) -> str:
    """Escape values placed in a SQL-style connection string."""
    if any(char in value for char in ";={} "):
        return "{" + value.replace("}", "}}") + "}"
    return value


def make_connection_string() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    import os

    keys = {
        "SQL_SERVER": os.getenv("SQL_SERVER"),
        "SQL_DATABASE": os.getenv("SQL_DATABASE"),
        "SQL_USERNAME": os.getenv("SQL_USERNAME"),
        "SQL_PASSWORD": os.getenv("SQL_PASSWORD"),
    }
    missing = [key for key, value in keys.items() if not value]
    if missing:
        raise RuntimeError("Variables manquantes dans .env : " + ", ".join(missing))

    return (
        f"Server={quote_connection_value(keys['SQL_SERVER'])};"
        f"Database={quote_connection_value(keys['SQL_DATABASE'])};"
        f"UID={quote_connection_value(keys['SQL_USERNAME'])};"
        f"PWD={quote_connection_value(keys['SQL_PASSWORD'])};"
        "Encrypt=yes;TrustServerCertificate=no;MultiSubnetFailover=yes;"
    )


def normalize_for_compare(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, Decimal)):
        return float(value)
    return value


def values_equal(source_value: Any, target_value: Any) -> bool:
    a = normalize_for_compare(source_value)
    b = normalize_for_compare(target_value)
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def ensure_schema(cursor) -> None:
    for statement in DDL_STATEMENTS:
        cursor.execute(statement)


def fetch_source_rows(sqlite_conn: sqlite3.Connection, table: str, columns: list[str]):
    expected = set(columns)
    actual = {
        row[1]
        for row in sqlite_conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    }
    if not expected.issubset(actual):
        missing = sorted(expected - actual)
        raise RuntimeError(f"Colonnes manquantes dans SQLite pour {table}: {missing}")
    col_sql = ", ".join(f'"{col}"' for col in columns)
    return sqlite_conn.execute(f'SELECT {col_sql} FROM "{table}"').fetchall()


def migrate_table(sqlite_conn, azure_conn, config: dict[str, Any]) -> tuple[int, int]:
    table = config["name"]
    pk = config["pk"]
    columns = config["columns"]
    rows = fetch_source_rows(sqlite_conn, table, columns)
    cursor = azure_conn.cursor()

    source_count = len(rows)
    inserted_count = 0
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(f"[{column}]" for column in columns)
    insert_sql = f"INSERT INTO dbo.[{table}] ({column_sql}) VALUES ({placeholders})"
    select_sql = f"SELECT {column_sql} FROM dbo.[{table}] WHERE [{pk}] = ?"

    to_insert = []
    for row in rows:
        source_dict = dict(zip(columns, tuple(row)))
        cursor.execute(select_sql, (source_dict[pk],))
        existing = cursor.fetchone()
        if existing is None:
            to_insert.append(tuple(row))
            continue

        if not all(values_equal(source_dict[col], existing[i]) for i, col in enumerate(columns)):
            raise RuntimeError(
                f"Conflit de données pour {table}.{pk}={source_dict[pk]!r}: "
                "la ligne existe déjà dans Azure SQL mais ses valeurs diffèrent. "
                "Aucune donnée existante n'a été écrasée."
            )

    identity_on = False
    try:
        if config["identity"] and to_insert:
            cursor.execute(f"SET IDENTITY_INSERT dbo.[{table}] ON")
            identity_on = True

        for row in to_insert:
            cursor.execute(insert_sql, row)
            inserted_count += 1

        if identity_on:
            cursor.execute(f"SET IDENTITY_INSERT dbo.[{table}] OFF")
            identity_on = False

        azure_conn.commit()
    except Exception:
        azure_conn.rollback()
        if identity_on:
            try:
                cursor.execute(f"SET IDENTITY_INSERT dbo.[{table}] OFF")
                azure_conn.commit()
            except Exception:
                azure_conn.rollback()
        raise
    finally:
        cursor.close()

    # Verify that each source primary key now exists and all values match.
    cursor = azure_conn.cursor()
    try:
        for row in rows:
            source_dict = dict(zip(columns, tuple(row)))
            cursor.execute(select_sql, (source_dict[pk],))
            actual = cursor.fetchone()
            if actual is None or not all(
                values_equal(source_dict[col], actual[i]) for i, col in enumerate(columns)
            ):
                raise RuntimeError(f"Vérification après migration échouée pour {table}.{pk}={source_dict[pk]!r}")
    finally:
        cursor.close()

    return source_count, inserted_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrer les données de tracking FinPredict depuis SQLite vers Azure SQL."
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=DEFAULT_SOURCE_DB,
        help=f"Chemin de la base SQLite (défaut : {DEFAULT_SOURCE_DB})",
    )
    args = parser.parse_args()

    source_db = args.source_db.resolve()
    if not source_db.is_file():
        print(f"Base SQLite introuvable : {source_db}", file=sys.stderr)
        return 2

    sqlite_conn = sqlite3.connect(str(source_db))
    sqlite_conn.row_factory = sqlite3.Row
    azure_conn = None

    try:
        print(f"Source SQLite : {source_db}")
        connection_string = make_connection_string()
        print("Connexion à Azure SQL...")
        azure_conn = connect(connection_string, timeout=60)
        cursor = azure_conn.cursor()
        try:
            ensure_schema(cursor)
            azure_conn.commit()
        finally:
            cursor.close()
        print("Schéma Azure SQL vérifié/créé.")

        total_source = 0
        total_inserted = 0
        for config in TABLE_CONFIG:
            source_count, inserted_count = migrate_table(sqlite_conn, azure_conn, config)
            total_source += source_count
            total_inserted += inserted_count
            print(
                f"{config['name']:<22} source={source_count:<5} "
                f"nouveaux_insérés={inserted_count:<5} vérification=OK"
            )

        print("\nMigration et vérifications terminées.")
        print(f"Lignes présentes dans la source SQLite : {total_source}")
        print(f"Nouvelles lignes insérées dans Azure SQL : {total_inserted}")
        print("La base SQLite source n'a pas été modifiée ni supprimée.")
        return 0
    except Exception as exc:
        if azure_conn is not None:
            try:
                azure_conn.rollback()
            except Exception:
                pass
        print(f"\nERREUR : {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Aucune donnée de la base SQLite n'a été modifiée. "
            "Si la migration s'est arrêtée à mi-chemin, relancez le script après correction ; "
            "les lignes identiques déjà insérées seront vérifiées et ignorées.",
            file=sys.stderr,
        )
        return 1
    finally:
        sqlite_conn.close()
        if azure_conn is not None:
            try:
                azure_conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
