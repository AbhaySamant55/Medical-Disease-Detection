"""FastAPI backend for the disease detection dashboard.

Serves the trained models over a small JSON API and hosts the static frontend.
The heavy objects — six tabular ensembles and up to three image ensembles — are
loaded once at startup and reused, because unpickling an ensemble takes long
enough to be felt on every request otherwise.

    python -m web.server
    # or
    uvicorn web.server:app --reload --port 8000

The imaging endpoints need TensorFlow, which cannot share an environment with
mediapipe. Run this from `.venv-tf`; the tabular half works anywhere.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.data_prep import load_disease

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Medical Disease Detection", version="2.0")

_tabular: dict = {}
_images: dict = {}
_overview: dict = {}


# --------------------------------------------------------------------- load
def _load_tabular() -> None:
    for key in config.DISEASES:
        path = config.bundle_path(key)
        if not path.exists():
            continue
        bundle = joblib.load(path)
        spec = config.DISEASES[key]
        ens = bundle["ensemble"]

        # the field list drives the form the browser renders
        fields = []
        for col in bundle["columns"]:
            meta = dict(bundle["feature_meta"].get(col, {}))
            meta["name"] = col
            meta["label"] = _prettify(col)
            fields.append(meta)

        _tabular[key] = {
            "bundle": bundle,
            "summary": {
                "key": key,
                "name": spec["name"],
                "icon": spec["icon"],
                "blurb": spec["blurb"],
                "source": spec["source"],
                "positive_label": spec["positive_label"],
                "negative_label": spec["negative_label"],
                "algorithms": len(ens.fitted_),
                "samples": bundle["info"]["n_samples"],
                "features": bundle["info"]["n_features"],
                "baseline": bundle["info"]["majority_rate"],
                "metrics": bundle["test_metrics"]["ensemble"],
                "confusion": bundle["confusion_matrix"],
            },
            "fields": fields,
            "weights": ens.weights_frame().to_dict("records"),
            "members": bundle["test_metrics"]["members"],
        }
    print(f"  tabular ensembles loaded: {list(_tabular)}")


def _load_images() -> None:
    try:
        from src.image_ensemble import ImageEnsemble, is_available
    except Exception as exc:
        print(f"  image ensembles unavailable: {exc}")
        return
    for key, task in config.IMAGE_TASKS.items():
        if not is_available(key):
            continue
        try:
            ens = ImageEnsemble.load(key)
        except Exception as exc:
            print(f"  {key}: could not load ({exc.__class__.__name__}: {exc})")
            continue
        _images[key] = {
            "ensemble": ens,
            "summary": {
                "key": key, "name": task["name"], "icon": task["icon"],
                "blurb": task["blurb"], "classes": ens.classes,
                "algorithms": len(ens.members), "backbone": ens.backbone_name,
            },
            "weights": ens.weights_frame().to_dict("records"),
        }
    print(f"  image ensembles loaded: {list(_images)}")


def _prettify(col: str) -> str:
    """Turn a raw column name into something a human reads comfortably."""
    special = {
        "bmi": "BMI", "dpf": "Diabetes Pedigree", "hba1c": "HbA1c",
        "trestbps": "Resting Blood Pressure", "chol": "Cholesterol",
        "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
        "thalach": "Max Heart Rate", "exang": "Exercise Angina",
        "oldpeak": "ST Depression", "ca": "Major Vessels", "thal": "Thalassemia",
        "cp": "Chest Pain Type", "sex": "Sex", "age": "Age",
    }
    if col.lower() in special:
        return special[col.lower()]
    out = col.replace("_", " ")
    out = "".join(f" {c}" if c.isupper() and i and not out[i - 1].isupper()
                  else c for i, c in enumerate(out))
    return out.strip().title()


@app.on_event("startup")
def _startup() -> None:
    t0 = time.perf_counter()
    print("Loading models ...")
    _load_tabular()
    _load_images()
    try:
        rows = []
        for key, entry in _tabular.items():
            s = entry["summary"]
            rows.append({"key": key, "name": s["name"],
                         "accuracy": s["metrics"]["accuracy"],
                         "roc_auc": s["metrics"]["roc_auc"]})
        _overview["diseases"] = rows
    except Exception:
        _overview["diseases"] = []
    print(f"ready in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------- api
@app.get("/api/catalog")
def catalog():
    """Everything the frontend needs to build its navigation."""
    return {
        "tabular": [e["summary"] for e in _tabular.values()],
        "imaging": [e["summary"] for e in _images.values()],
        "imaging_available": bool(_images),
    }


@app.get("/api/disease/{key}")
def disease(key: str):
    entry = _tabular.get(key)
    if entry is None:
        raise HTTPException(404, f"unknown disease '{key}'")
    return {"summary": entry["summary"], "fields": entry["fields"],
            "weights": entry["weights"], "members": entry["members"]}


@app.post("/api/predict/{key}")
def predict(key: str, payload: dict):
    entry = _tabular.get(key)
    if entry is None:
        raise HTTPException(404, f"unknown disease '{key}'")

    bundle = entry["bundle"]
    ens = bundle["ensemble"]
    spec = config.DISEASES[key]

    values = payload.get("values", {})
    row = {}
    for col in bundle["columns"]:
        meta = bundle["feature_meta"].get(col, {})
        v = values.get(col, meta.get("default"))
        if meta.get("kind") == "categorical":
            row[col] = str(v)
        else:
            try:
                row[col] = float(v)
            except (TypeError, ValueError):
                row[col] = float(meta.get("default") or 0.0)

    X = pd.DataFrame([row])[bundle["columns"]]
    proba = ens.predict_proba(X)[0]
    p_disease = float(proba[1])
    detail = ens.explain(X)

    models = []
    for _, r in detail.iterrows():
        models.append({
            "algorithm": r["Algorithm"],
            "family": r["Family"],
            "vote": spec["positive_label"] if int(r["Prediction"]) == 1
                    else spec["negative_label"],
            "votes_positive": bool(int(r["Prediction"]) == 1),
            "surety": float(r["Surety %"]),
            "p_disease": float(r["P(disease) %"]),
            "accuracy": float(r["Accuracy %"]),
            "weight": float(r["Weight %"]),
            "about": r["About"],
        })

    n_pos = sum(1 for m in models if m["votes_positive"])
    positive = p_disease >= 0.5
    return {
        "probability": p_disease,
        "positive": positive,
        "verdict": spec["positive_label"] if positive else spec["negative_label"],
        "confidence": max(p_disease, 1 - p_disease),
        "agreement": n_pos if positive else len(models) - n_pos,
        "n_models": len(models),
        "baseline": bundle["info"]["majority_rate"],
        "models": models,
    }


@app.get("/api/imaging/{key}")
def imaging(key: str):
    entry = _images.get(key)
    if entry is None:
        raise HTTPException(404, f"no trained models for '{key}'")
    return {"summary": entry["summary"], "weights": entry["weights"]}


@app.post("/api/imaging/{key}/predict")
async def imaging_predict(key: str, file: UploadFile = File(...)):
    entry = _images.get(key)
    if entry is None:
        raise HTTPException(404, f"no trained models for '{key}'")
    from PIL import Image

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read that image")

    ens = entry["ensemble"]
    probs, detail = ens.analyse(img)
    top = int(np.argmax(probs))
    verdict = ens.classes[top]
    healthy = any(w in verdict.lower()
                  for w in ("no_tumor", "no tumour", "normal", "healthy"))

    models = [{
        "algorithm": r["Algorithm"],
        "vote": r["Vote"],
        "agrees": bool(r["Vote"] == verdict),
        "surety": float(r["Surety %"]),
        "accuracy": float(r["Accuracy %"]),
        "weight": float(r["Weight %"]),
    } for _, r in detail.iterrows()]

    return {
        "verdict": verdict,
        "pretty": verdict.replace("_", " ").title(),
        "healthy": healthy,
        "probability": float(probs[top]),
        "classes": ens.classes,
        "probabilities": [float(p) for p in probs],
        "agreement": sum(1 for m in models if m["agrees"]),
        "n_models": len(models),
        "models": models,
    }


@app.get("/api/performance")
def performance():
    out = {"tabular": None, "imaging": None}
    if config.TABULAR_METRICS.exists():
        out["tabular"] = json.loads(config.TABULAR_METRICS.read_text(encoding="utf-8"))
    img = config.REPORTS_DIR / "image_ensemble_metrics.json"
    if img.exists():
        out["imaging"] = json.loads(img.read_text(encoding="utf-8"))
    return out


@app.get("/api/health")
def health():
    return {"ok": True, "tabular": list(_tabular), "imaging": list(_images)}


# ------------------------------------------------------------------ static
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
