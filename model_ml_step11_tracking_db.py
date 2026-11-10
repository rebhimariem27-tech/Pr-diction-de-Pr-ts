from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

# FinPredict Step 11 — prediction & credit-outcome tracking
# SQLite MVP. Do not store names, national IDs, addresses, passwords or card data.

DEFAULT_DB = Path("data/tracking/finpredict_tracking.db")
MODEL_FEATURES = [
    "person_age", "person_income", "loan_amnt", "loan_int_rate",
    "person_emp_length", "loan_percent_income", "cb_person_cred_hist_length",
    "interest_income_ratio", "person_home_ownership", "loan_intent",
    "loan_grade", "cb_person_default_on_file",
]
NUMERIC_FEATURES = {
    "person_age", "person_income", "loan_amnt", "loan_int_rate",
    "person_emp_length", "loan_percent_income", "cb_person_cred_hist_length",
    "interest_income_ratio",
}
CAT_FEATURES = set(MODEL_FEATURES) - NUMERIC_FEATURES
DECISIONS = {"PENDING", "APPROVED", "REJECTED", "MANUAL_REVIEW"}
EVENT_TYPES = {
    "PAYMENT_DUE", "PAYMENT_RECEIVED", "DELINQUENCY", "DEFAULT_CONFIRMED",
    "RECOVERY", "LOAN_CLOSED", "CHARGED_OFF",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Le JSON doit être un objet : {path}")
    return payload


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def db_session(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path) -> None:
    with db_session(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_versions (
            model_version TEXT PRIMARY KEY,
            registered_at_utc TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_sha256 TEXT,
            dataset_version TEXT,
            threshold REAL NOT NULL CHECK(threshold BETWEEN 0 AND 1),
            metrics_json TEXT NOT NULL DEFAULT '{}',
            parameters_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'CANDIDATE'
                CHECK(status IN ('CANDIDATE','ACTIVE','RETIRED')),
            notes TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_one_active_model
            ON model_versions(status) WHERE status='ACTIVE';

        CREATE TABLE IF NOT EXISTS loan_applications (
            application_id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_ref TEXT UNIQUE,
            submitted_at_utc TEXT NOT NULL,
            feature_snapshot_json TEXT NOT NULL,
            feature_snapshot_sha256 TEXT NOT NULL,
            actual_decision TEXT NOT NULL DEFAULT 'PENDING'
                CHECK(actual_decision IN ('PENDING','APPROVED','REJECTED','MANUAL_REVIEW')),
            decision_at_utc TEXT,
            funded_at_utc TEXT,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_logs (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER NOT NULL REFERENCES loan_applications(application_id),
            model_version TEXT NOT NULL REFERENCES model_versions(model_version),
            predicted_at_utc TEXT NOT NULL,
            default_probability REAL NOT NULL CHECK(default_probability BETWEEN 0 AND 1),
            threshold REAL NOT NULL CHECK(threshold BETWEEN 0 AND 1),
            recommended_decision TEXT NOT NULL CHECK(recommended_decision IN ('ACCEPT','REFUSE')),
            feature_snapshot_sha256 TEXT NOT NULL,
            explanation_json TEXT NOT NULL DEFAULT '{}',
            latency_ms REAL CHECK(latency_ms IS NULL OR latency_ms >= 0),
            created_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_predictions_application_time
            ON prediction_logs(application_id, predicted_at_utc);
        CREATE INDEX IF NOT EXISTS ix_predictions_model_time
            ON prediction_logs(model_version, predicted_at_utc);

        CREATE TABLE IF NOT EXISTS loan_outcome_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER NOT NULL REFERENCES loan_applications(application_id),
            event_timestamp_utc TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN (
                'PAYMENT_DUE','PAYMENT_RECEIVED','DELINQUENCY','DEFAULT_CONFIRMED',
                'RECOVERY','LOAN_CLOSED','CHARGED_OFF')),
            amount REAL CHECK(amount IS NULL OR amount >= 0),
            days_past_due INTEGER CHECK(days_past_due IS NULL OR days_past_due >= 0),
            source TEXT,
            notes TEXT,
            recorded_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_outcome_events_application_time
            ON loan_outcome_events(application_id, event_timestamp_utc);

        CREATE TABLE IF NOT EXISTS outcome_labels (
            application_id INTEGER PRIMARY KEY REFERENCES loan_applications(application_id),
            label INTEGER NOT NULL CHECK(label IN (0,1)),
            label_definition TEXT NOT NULL,
            observation_window_days INTEGER NOT NULL CHECK(observation_window_days > 0),
            label_matured INTEGER NOT NULL DEFAULT 0 CHECK(label_matured IN (0,1)),
            matured_at_utc TEXT,
            source TEXT,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retraining_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_model_version TEXT,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            dataset_version TEXT,
            dataset_rows INTEGER CHECK(dataset_rows IS NULL OR dataset_rows >= 0),
            metrics_json TEXT NOT NULL DEFAULT '{}',
            validation_status TEXT NOT NULL DEFAULT 'PENDING'
                CHECK(validation_status IN ('PENDING','PASSED','FAILED','CANCELLED')),
            deployment_status TEXT NOT NULL DEFAULT 'NOT_DEPLOYED'
                CHECK(deployment_status IN ('NOT_DEPLOYED','DEPLOYED','ROLLED_BACK')),
            notes TEXT
        );
        """)
    print(f"Base initialisée : {path}")


def validate_snapshot(features: dict) -> dict:
    missing = [c for c in MODEL_FEATURES if c not in features]
    if missing:
        raise ValueError("Features manquantes : " + ", ".join(missing))
    snapshot = {k: features[k] for k in MODEL_FEATURES}
    for key in NUMERIC_FEATURES:
        value = snapshot[key]
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} doit être numérique ou null.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{key} doit être un nombre fini ou null (pas NaN/inf).")
        snapshot[key] = numeric
    for key in CAT_FEATURES:
        if snapshot[key] is not None:
            snapshot[key] = str(snapshot[key])
    return snapshot


def get_application(conn: sqlite3.Connection, application_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM loan_applications WHERE application_id=?", (application_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Application introuvable : {application_id}")
    return row


def cmd_register_model(args) -> None:
    status = args.status.upper()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold doit être compris entre 0 et 1.")
    if status == "ACTIVE" and not args.artifact_path.exists():
        raise FileNotFoundError(f"Artefact actif introuvable : {args.artifact_path}")
    metrics = read_json(args.metrics_json)
    parameters = read_json(args.parameters_json)
    with db_session(args.db) as conn:
        if status == "ACTIVE":
            conn.execute("UPDATE model_versions SET status='RETIRED' WHERE status='ACTIVE'")
        conn.execute("""
            INSERT INTO model_versions (
                model_version,registered_at_utc,artifact_path,artifact_sha256,
                dataset_version,threshold,metrics_json,parameters_json,status,notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(model_version) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                dataset_version=excluded.dataset_version,
                threshold=excluded.threshold,
                metrics_json=excluded.metrics_json,
                parameters_json=excluded.parameters_json,
                status=excluded.status, notes=excluded.notes
        """, (
            args.version, utc_now(), str(args.artifact_path), file_sha256(args.artifact_path),
            args.dataset_version, args.threshold, json_dumps(metrics), json_dumps(parameters),
            status, args.notes,
        ))
    print(f"Modèle enregistré : {args.version} | status={status} | threshold={args.threshold:.2f}")


def cmd_record_application(args) -> None:
    raw = read_json(args.features_json)
    snapshot = validate_snapshot(raw)
    encoded = json_dumps(snapshot)
    now = utc_now()
    with db_session(args.db) as conn:
        cur = conn.execute("""
            INSERT INTO loan_applications (
                application_ref,submitted_at_utc,feature_snapshot_json,
                feature_snapshot_sha256,actual_decision,created_at_utc,updated_at_utc
            ) VALUES (?,?,?,?, 'PENDING',?,?)
        """, (args.application_ref, args.submitted_at or now, encoded, sha256_text(encoded), now, now))
        application_id = int(cur.lastrowid)
    print(f"Application créée : id={application_id}")
    print("Référence interne :", args.application_ref or "(non fournie)")


def cmd_log_prediction(args) -> None:
    if not 0 <= args.probability <= 1:
        raise ValueError("probability doit être compris entre 0 et 1.")
    if args.latency_ms is not None and args.latency_ms < 0:
        raise ValueError("latency-ms doit être positif ou nul.")
    explanation = read_json(args.explanation_json)
    with db_session(args.db) as conn:
        app = get_application(conn, args.application_id)
        model = conn.execute(
            "SELECT threshold FROM model_versions WHERE model_version=?", (args.model_version,)
        ).fetchone()
        if model is None:
            raise ValueError(f"Modèle inconnu : {args.model_version}. Enregistre-le d'abord.")
        threshold = model["threshold"] if args.threshold is None else args.threshold
        if not 0 <= threshold <= 1:
            raise ValueError("threshold doit être compris entre 0 et 1.")
        rec = "REFUSE" if args.probability >= threshold else "ACCEPT"
        now = utc_now()
        cur = conn.execute("""
            INSERT INTO prediction_logs (
                application_id,model_version,predicted_at_utc,default_probability,
                threshold,recommended_decision,feature_snapshot_sha256,explanation_json,
                latency_ms,created_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            args.application_id, args.model_version, now, args.probability, threshold,
            rec, app["feature_snapshot_sha256"], json_dumps(explanation), args.latency_ms, now,
        ))
        prediction_id = int(cur.lastrowid)
    print(f"Prédiction enregistrée : id={prediction_id}")
    print(f"P(default)={args.probability:.4f} | seuil={threshold:.2f} | recommandation={rec}")


def cmd_record_decision(args) -> None:
    decision = args.decision.upper()
    if decision not in DECISIONS - {"PENDING"}:
        raise ValueError(f"decision doit être parmi {sorted(DECISIONS - {'PENDING'})}")
    with db_session(args.db) as conn:
        get_application(conn, args.application_id)
        now = utc_now()
        conn.execute("""
            UPDATE loan_applications
            SET actual_decision=?,decision_at_utc=?,updated_at_utc=?
            WHERE application_id=?
        """, (decision, now, now, args.application_id))
    print(f"Décision réelle enregistrée : application={args.application_id}, décision={decision}")


def cmd_mark_funded(args) -> None:
    with db_session(args.db) as conn:
        app = get_application(conn, args.application_id)
        if app["actual_decision"] != "APPROVED":
            raise ValueError("Le prêt ne peut être marqué financé que si la décision réelle est APPROVED.")
        now = utc_now()
        conn.execute(
            "UPDATE loan_applications SET funded_at_utc=?,updated_at_utc=? WHERE application_id=?",
            (args.funded_at or now, now, args.application_id),
        )
    print(f"Prêt financé enregistré pour application {args.application_id}.")


def cmd_add_event(args) -> None:
    event_type = args.event_type.upper()
    if event_type not in EVENT_TYPES:
        raise ValueError(f"event-type doit être parmi {sorted(EVENT_TYPES)}")
    if args.amount is not None and args.amount < 0:
        raise ValueError("amount doit être positif ou nul.")
    if args.days_past_due is not None and args.days_past_due < 0:
        raise ValueError("days-past-due doit être positif ou nul.")
    with db_session(args.db) as conn:
        app = get_application(conn, args.application_id)
        if not app["funded_at_utc"]:
            raise ValueError("Marque d'abord le prêt comme financé avec mark-funded.")
        cur = conn.execute("""
            INSERT INTO loan_outcome_events (
                application_id,event_timestamp_utc,event_type,amount,days_past_due,
                source,notes,recorded_at_utc
            ) VALUES (?,?,?,?,?,?,?,?)
        """, (
            args.application_id, args.event_date or utc_now(), event_type,
            args.amount, args.days_past_due, args.source, args.notes, utc_now(),
        ))
        event_id = int(cur.lastrowid)
    print(f"Événement enregistré : id={event_id}, type={event_type}")


def cmd_label_outcome(args) -> None:
    if args.label not in (0, 1):
        raise ValueError("label doit être 0 (non-default) ou 1 (default).")
    if args.window_days <= 0:
        raise ValueError("window-days doit être strictement positif.")
    if not args.definition.strip():
        raise ValueError("definition est obligatoire.")
    if not args.matured:
        raise ValueError(
            "Label non enregistré : ajoute --matured seulement quand l'horizon "
            "d'observation est vraiment terminé et le résultat est confirmé."
        )
    now = utc_now()
    with db_session(args.db) as conn:
        app = get_application(conn, args.application_id)
        if not app["funded_at_utc"]:
            raise ValueError("On ne peut labelliser que les prêts effectivement financés.")
        conn.execute("""
            INSERT INTO outcome_labels (
                application_id,label,label_definition,observation_window_days,
                label_matured,matured_at_utc,source,created_at_utc,updated_at_utc
            ) VALUES (?,?,?,?,1,?,?,?,?)
            ON CONFLICT(application_id) DO UPDATE SET
                label=excluded.label,
                label_definition=excluded.label_definition,
                observation_window_days=excluded.observation_window_days,
                label_matured=1,matured_at_utc=excluded.matured_at_utc,
                source=excluded.source,updated_at_utc=excluded.updated_at_utc
        """, (
            args.application_id, args.label, args.definition, args.window_days,
            args.matured_at or now, args.source, now, now,
        ))
    meaning = "default" if args.label == 1 else "non-default"
    print(f"Label arrivé à maturité : application={args.application_id}, loan_status={args.label} ({meaning}).")


def cmd_export_labeled(args) -> None:
    rows = []
    with db_session(args.db) as conn:
        labels = conn.execute("""
            SELECT ol.application_id,ol.label,la.feature_snapshot_json,la.feature_snapshot_sha256
            FROM outcome_labels ol
            JOIN loan_applications la ON la.application_id=ol.application_id
            WHERE ol.label_matured=1 AND la.funded_at_utc IS NOT NULL
            ORDER BY ol.application_id
        """).fetchall()
        for row in labels:
            pred = conn.execute("""
                SELECT feature_snapshot_sha256 FROM prediction_logs
                WHERE application_id=? ORDER BY predicted_at_utc DESC,prediction_id DESC LIMIT 1
            """, (row["application_id"],)).fetchone()
            if pred is None or pred["feature_snapshot_sha256"] != row["feature_snapshot_sha256"]:
                continue
            features = json.loads(row["feature_snapshot_json"])
            item = {feature: features.get(feature) for feature in MODEL_FEATURES}
            item["loan_status"] = int(row["label"])
            rows.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MODEL_FEATURES + ["loan_status"]).to_csv(args.output, index=False)
    print(f"Labels mûrs exportés : {len(rows)} lignes")
    print(f"CSV : {args.output}")
    print("Seuls les prêts financés avec label mûr et prédiction enregistrée sont exportés.")


def cmd_log_retraining(args) -> None:
    metrics = read_json(args.metrics_json)
    now = utc_now()
    with db_session(args.db) as conn:
        cur = conn.execute("""
            INSERT INTO retraining_runs (
                candidate_model_version,started_at_utc,finished_at_utc,dataset_version,
                dataset_rows,metrics_json,validation_status,deployment_status,notes
            ) VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            args.candidate_version, now, now, args.dataset_version, args.dataset_rows,
            json_dumps(metrics), args.validation_status, args.deployment_status, args.notes,
        ))
        run_id = int(cur.lastrowid)
    print(f"Run de réentraînement enregistré : {run_id}")


def cmd_summary(args) -> None:
    with db_session(args.db) as conn:
        print("=" * 62)
        print("FINPREDICT — TRACKING DATABASE SUMMARY")
        print("=" * 62)
        queries = {
            "model_versions": "SELECT COUNT(*) FROM model_versions",
            "applications": "SELECT COUNT(*) FROM loan_applications",
            "predictions": "SELECT COUNT(*) FROM prediction_logs",
            "funded_loans": "SELECT COUNT(*) FROM loan_applications WHERE funded_at_utc IS NOT NULL",
            "outcome_events": "SELECT COUNT(*) FROM loan_outcome_events",
            "mature_labels": "SELECT COUNT(*) FROM outcome_labels WHERE label_matured=1",
            "retraining_runs": "SELECT COUNT(*) FROM retraining_runs",
        }
        for label, query in queries.items():
            print(f"{label:<20}: {conn.execute(query).fetchone()[0]}")
        active = conn.execute(
            "SELECT model_version,threshold,artifact_path FROM model_versions WHERE status='ACTIVE'"
        ).fetchone()
        if active:
            print("\nActive model:")
            print(f"  version   : {active['model_version']}")
            print(f"  threshold : {active['threshold']:.2f}")
            print(f"  artifact  : {active['artifact_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FinPredict Step 11 — prediction and outcome tracking database"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("init", help="Initialize database schema")

    p = subs.add_parser("register-model", help="Register a model version")
    p.add_argument("--version", required=True)
    p.add_argument("--artifact-path", type=Path, required=True)
    p.add_argument("--dataset-version", default=None)
    p.add_argument("--threshold", type=float, default=0.48)
    p.add_argument("--status", choices=["CANDIDATE", "ACTIVE", "RETIRED"], default="CANDIDATE")
    p.add_argument("--metrics-json", type=Path)
    p.add_argument("--parameters-json", type=Path)
    p.add_argument("--notes")

    p = subs.add_parser("record-application", help="Record application and feature snapshot JSON")
    p.add_argument("--features-json", type=Path, required=True)
    p.add_argument("--application-ref", help="Pseudonymous internal reference; do not use a name/ID number")
    p.add_argument("--submitted-at")

    p = subs.add_parser("log-prediction", help="Log model probability and recommendation")
    p.add_argument("--application-id", type=int, required=True)
    p.add_argument("--model-version", required=True)
    p.add_argument("--probability", type=float, required=True)
    p.add_argument("--threshold", type=float)
    p.add_argument("--explanation-json", type=Path)
    p.add_argument("--latency-ms", type=float)

    p = subs.add_parser("record-decision", help="Record the actual human/business decision")
    p.add_argument("--application-id", type=int, required=True)
    p.add_argument("--decision", choices=["APPROVED", "REJECTED", "MANUAL_REVIEW"], required=True)

    p = subs.add_parser("mark-funded", help="Record the date the approved loan was actually funded")
    p.add_argument("--application-id", type=int, required=True)
    p.add_argument("--funded-at")

    p = subs.add_parser("add-event", help="Add a repayment/outcome event for a funded loan")
    p.add_argument("--application-id", type=int, required=True)
    p.add_argument("--event-type", choices=sorted(EVENT_TYPES), required=True)
    p.add_argument("--event-date")
    p.add_argument("--amount", type=float)
    p.add_argument("--days-past-due", type=int)
    p.add_argument("--source")
    p.add_argument("--notes")

    p = subs.add_parser("label-outcome", help="Set a confirmed, mature outcome label")
    p.add_argument("--application-id", type=int, required=True)
    p.add_argument("--label", type=int, choices=[0, 1], required=True, help="0=non-default; 1=default")
    p.add_argument("--definition", required=True, help="E.g. default definition and observation horizon")
    p.add_argument("--window-days", type=int, required=True)
    p.add_argument("--matured", action="store_true", help="Confirm observation window has matured")
    p.add_argument("--matured-at")
    p.add_argument("--source")

    p = subs.add_parser("export-labeled", help="Export funded loans with mature labels to training CSV")
    p.add_argument("--output", type=Path, default=Path("data/processed/outcome_labeled_loans.csv"))

    p = subs.add_parser("log-retraining-run", help="Track a candidate training/validation run")
    p.add_argument("--candidate-version")
    p.add_argument("--dataset-version")
    p.add_argument("--dataset-rows", type=int)
    p.add_argument("--metrics-json", type=Path)
    p.add_argument("--validation-status", choices=["PENDING", "PASSED", "FAILED", "CANCELLED"], default="PENDING")
    p.add_argument("--deployment-status", choices=["NOT_DEPLOYED", "DEPLOYED", "ROLLED_BACK"], default="NOT_DEPLOYED")
    p.add_argument("--notes")

    subs.add_parser("summary", help="Show table counts and active model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init":
        init_db(args.db)
        print("Tables créées : model_versions, loan_applications, prediction_logs,")
        print("               loan_outcome_events, outcome_labels, retraining_runs")
        return

    init_db(args.db)
    dispatch = {
        "register-model": cmd_register_model,
        "record-application": cmd_record_application,
        "log-prediction": cmd_log_prediction,
        "record-decision": cmd_record_decision,
        "mark-funded": cmd_mark_funded,
        "add-event": cmd_add_event,
        "label-outcome": cmd_label_outcome,
        "export-labeled": cmd_export_labeled,
        "log-retraining-run": cmd_log_retraining,
        "summary": cmd_summary,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
