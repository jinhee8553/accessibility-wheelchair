from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = [
    ROOT / "artifacts" / "accessibility_classifier_ordinal.joblib",
    ROOT / "artifacts" / "accessibility_classifier_ensemble.joblib",
    ROOT / "artifacts" / "accessibility_classifier.joblib",
]


class AccessibilityRequest(BaseModel):
    genre_group: str = Field(..., examples=["클래식"])
    venue_type: str = Field(..., examples=["대형"])
    is_weekend: int = Field(..., ge=0, le=1, examples=[1])
    duration_days: float = Field(..., ge=1.0, examples=[2.0])
    organizer_type: str = Field(..., examples=["대관"])
    organizer: str = Field(..., examples=["(재)서울시립교향악단"])


def load_model_bundle() -> dict[str, Any]:
    for model_file in MODEL_FILES:
        if model_file.exists():
            return joblib.load(model_file)
    raise RuntimeError("모델 파일이 없습니다. 먼저 학습 스크립트를 실행하세요.")


bundle = load_model_bundle()
app = FastAPI(title="SAC Accessibility Classifier", version="1.0.0")


def best_model_name() -> str:
    best_model = bundle.get("metrics", {}).get("best_model", "unknown")
    if isinstance(best_model, dict):
        return str(best_model.get("name", "unknown"))
    return str(best_model)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": best_model_name(),
        "target": bundle["target"],
        "features": bundle["features"],
    }


@app.post("/predict")
def predict(payload: AccessibilityRequest) -> dict[str, Any]:
    row = pd.DataFrame([payload.model_dump()])
    pipeline = bundle["pipeline"]
    label = pipeline.predict(row)[0]
    response: dict[str, Any] = {"prediction": label}
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        classes = getattr(pipeline, "classes_", bundle.get("label_order", []))
        response["probabilities"] = {
            class_label: float(probability)
            for class_label, probability in zip(classes, probabilities)
        }
    return response
