from pathlib import Path
import json
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from prometheus_client import (
    Counter,
    Histogram,
    make_asgi_app,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_disease_api")

# Count HTTP requests, grouped by method, route and status code.
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

# Measure the duration of each HTTP request.
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["path"],
)

# Count server-side HTTP errors (5xx).
SERVER_ERRORS = Counter(
    "api_server_errors_total",
    "Total HTTP 5xx responses",
    ["path"],
)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Educational demonstration only; not for medical use.",
)

# Prometheus can scrape metrics from /metrics.
app.mount("/metrics", make_asgi_app())

model = None
metadata = {}
feature_columns = []


@app.on_event("startup")
def load_artifacts():
    global model, metadata, feature_columns

    if not MODEL_PATH.exists():
        logger.error("Model artifact not found: %s", MODEL_PATH)
        return

    model = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())

    feature_columns = metadata.get(
        "feature_columns",
        list(model.feature_names_in_),
    )

    logger.info("Model loaded successfully")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    path = request.url.path

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        logger.exception("Unhandled request exception")
        raise
    finally:
        elapsed = time.perf_counter() - start

        REQUEST_COUNT.labels(
            request.method, path, str(status)
        ).inc()

        REQUEST_LATENCY.labels(path).observe(elapsed)

        if status >= 500:
            SERVER_ERRORS.labels(path).inc()

        logger.info(
            "method=%s path=%s status=%s duration_seconds=%.4f",
            request.method,
            path,
            status,
            elapsed,
        )


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Train the model first.",
        )

    supplied = request.features
    missing = set(feature_columns) - set(supplied)
    extra = set(supplied) - set(feature_columns)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(extra),
            },
        )

    try:
        row = pd.DataFrame([{
            column: supplied[column]
            for column in feature_columns
        }])

        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        positive_index = list(model.classes_).index(1)

        probability = float(probabilities[positive_index])

        logger.info(
            "Prediction completed; predicted_class=%s",
            prediction,
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "model_name": metadata.get(
                "selected_model", "saved_pipeline"
            ),
            "disclaimer": "Educational use only; not for medical decisions.",
        }

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc