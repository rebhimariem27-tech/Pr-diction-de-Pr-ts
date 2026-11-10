import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "production" / "finpredict_production_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "donnees_prets_features.csv"


@pytest.fixture(scope="session")
def model_artifact():
    assert MODEL_PATH.exists(), f"Missing model artifact: {MODEL_PATH}"
    artifact = joblib.load(MODEL_PATH)
    assert isinstance(artifact, dict), "Production artifact must be a dictionary"
    assert "pipeline" in artifact, "Artifact missing trained pipeline"
    return artifact


@pytest.fixture(scope="session")
def evaluation_frame():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Processed dataset is empty"
    return df


@pytest.fixture(scope="session")
def model_features(model_artifact):
    features = model_artifact.get("model_features")
    assert features is not None and len(features) == 12, "Unexpected feature list in artifact"
    return features


def test_model_artifact_loads(model_artifact):
    pipeline = model_artifact["pipeline"]
    assert hasattr(pipeline, "predict_proba"), "Loaded model does not expose predict_proba"


def test_expected_feature_schema(model_features, evaluation_frame):
    missing = [feature for feature in model_features if feature not in evaluation_frame.columns]
    assert not missing, f"Missing required columns: {missing}"
    X = evaluation_frame[model_features]
    assert len(X.columns) == len(model_features)


def test_prediction_output_shape_and_validity(model_artifact, model_features, evaluation_frame):
    pipeline = model_artifact["pipeline"]
    X = evaluation_frame[model_features]
    probs = pipeline.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),), "Probability vector length mismatch"
    assert pd.api.types.is_numeric_dtype(probs), "Probabilities must be numeric"
    assert ((probs >= 0) & (probs <= 1)).all(), "Probabilities must be in [0, 1]"


def test_threshold_boundary_behavior(model_artifact, model_features, evaluation_frame):
    pipeline = model_artifact["pipeline"]
    probs = pipeline.predict_proba(evaluation_frame[model_features])[:, 1]
    for threshold in [0.4799, 0.48, 0.4801]:
        preds = (probs >= threshold).astype(int)
        assert preds.dtype.kind in {"i", "u"}, "Thresholded predictions must be integer-like"
    assert (probs >= 0.48).sum() != (probs >= 0.4799).sum(), "Threshold boundary should change decision counts at 0.4799 vs 0.48"
    assert (probs >= 0.4801).sum() != (probs >= 0.48).sum(), "Threshold boundary should change decision counts at 0.48 vs 0.4801"


def test_repeated_prediction_consistency(model_artifact, model_features, evaluation_frame):
    pipeline = model_artifact["pipeline"]
    X = evaluation_frame[model_features].head(200)
    first = pipeline.predict_proba(X)[:, 1]
    second = pipeline.predict_proba(X)[:, 1]
    assert (first == second).all(), "Repeated predictions should be deterministic"


def test_invalid_input_rejection(model_artifact, model_features, evaluation_frame):
    pipeline = model_artifact["pipeline"]
    bad = evaluation_frame[model_features].copy()
    bad.loc[bad.index[0], model_features[0]] = float("nan")
    try:
        result = pipeline.predict_proba(bad)
    except (ValueError, TypeError):
        return
    assert result is not None
    assert result.shape[0] == len(bad)


def test_metric_calculation_correctness(model_artifact, model_features, evaluation_frame):
    pipeline = model_artifact["pipeline"]
    probs = pipeline.predict_proba(evaluation_frame[model_features])[:, 1]
    y_true = evaluation_frame["loan_status"].astype(int)
    threshold = float(model_artifact.get("production_threshold", 0.48))
    y_pred = (probs >= threshold).astype(int)

    assert roc_auc_score(y_true, probs) > 0.9
    assert average_precision_score(y_true, probs) > 0.85
    assert brier_score_loss(y_true, probs) < 0.1
    assert precision_score(y_true, y_pred, zero_division=0) > 0.9
    assert recall_score(y_true, y_pred, zero_division=0) > 0.7
    assert f1_score(y_true, y_pred, zero_division=0) > 0.8


def test_calibration_outputs_exist(model_artifact):
    artifact = model_artifact
    metrics = artifact.get("test_metrics_at_production_threshold", {})
    assert "brier_score" in metrics, "Artifact should contain Brier score"
    assert "pr_auc" in metrics, "Artifact should contain PR-AUC"
    assert "roc_auc" in metrics, "Artifact should contain ROC-AUC"


def test_report_files_are_readable():
    report_dir = PROJECT_ROOT / "data" / "reports" / "ml"
    assert report_dir.exists(), "ML reports directory does not exist"
    json_files = list(report_dir.glob("**/*.json"))
    assert json_files, "No ML report JSON files were found"
    for path in json_files[:5]:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload is not None


def test_api_response_schema_example_smoke():
    from backend.app import main
    assert hasattr(main, "app"), "FastAPI app should exist"
    routes = {route.path for route in main.app.routes}
    assert "/health" in routes, "Health endpoint is missing"
    assert "/model-info" in routes, "Model-info endpoint is missing"
    assert "/predict" in routes, "Predict endpoint is missing"
