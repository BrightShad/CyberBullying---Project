from contextlib import asynccontextmanager

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import joblib
from cyberbullying.config import get_tfidf_vectorizer_path, get_model_path
from cyberbullying.inference import load_checkpoint, load_sklearn_baseline, predict_text, predict_text_sklearn

_device = torch.device("cpu")
_models = {
    "binary": {"bilstm": None, "logistic": None, "rf": None},
    "multiclass": {"bilstm": None, "logistic": None, "rf": None}
}
_vectorizers = {"binary": None, "multiclass": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _vectorizers

    for task in ["binary", "multiclass"]:
        # Vectorizer
        tfidf_path = get_tfidf_vectorizer_path(task)
        if tfidf_path.exists():
            _vectorizers[task] = joblib.load(tfidf_path)

        # BiLSTM
        bilstm_path = get_model_path("bilstm", task)
        if bilstm_path.exists():
            model, vocab, label_map = load_checkpoint(bilstm_path, _device)
            _models[task]["bilstm"] = {"model": model, "vocab": vocab, "label_map": label_map}
        
        # Sklearn models
        if _vectorizers[task] is not None:
            for m_type in ["logistic", "rf"]:
                m_path = get_model_path(m_type, task)
                if m_path.exists():
                    model, _, label_map = load_sklearn_baseline(m_path, tfidf_path, task)
                    _models[task][m_type] = {"model": model, "label_map": label_map}
    yield


app = FastAPI(
    title="Cyberbullying Classifier API",
    description="REST endpoint for tweet cyberbullying detection",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Tweet text to classify")
    model: str = Field("bilstm", description="Model to use: bilstm, logistic, or rf")
    task: str = Field("binary", description="Task to use: binary or multiclass")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    task: str


@app.get("/health")
def health():
    ready = any(model is not None for models in _models.values() for model in models.values())
    return {"status": "ok", "model_loaded": ready}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.task not in _models:
        raise HTTPException(status_code=400, detail="Invalid task. Choose 'binary' or 'multiclass'.")
    if request.model not in _models[request.task]:
        raise HTTPException(status_code=400, detail="Invalid model. Choose 'bilstm', 'logistic', or 'rf'.")
    
    model_data = _models[request.task][request.model]
    if model_data is None:
        raise HTTPException(status_code=503, detail=f"{request.model} model for {request.task} not loaded.")

    try:
        if request.model == "bilstm":
            result = predict_text(request.text, model_data["model"], model_data["vocab"], model_data["label_map"], _device)
        else:
            result = predict_text_sklearn(request.text, model_data["model"], _vectorizers[request.task], model_data["label_map"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(**result)
