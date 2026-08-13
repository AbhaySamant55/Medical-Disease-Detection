"""Inference for the image tasks: CNN embedding -> 10 classifiers -> weighted vote.

Mirrors the tabular pipeline exactly. The only difference is how a sample
becomes a feature vector: tabular rows arrive as numbers already, whereas a
scan is pushed through a fine-tuned EfficientNetB0 and reduced to a 1280-d
embedding first. From there the same ten algorithms and the same
accuracy-weighted vote apply.

Artefacts come from `notebooks/kaggle_image_10_models.ipynb`:

    models/image/<task>_backbone.keras   fine-tuned feature extractor
    models/image/<task>_heads.joblib     the 10 fitted classifiers + weights

TensorFlow is needed to run the backbone. It is imported lazily so the tabular
half of the project keeps working in environments where TensorFlow is broken or
absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def backbone_path(task: str):
    return config.IMAGE_MODEL_PATH / f"{task}_backbone.keras"


def heads_path(task: str):
    return config.IMAGE_MODEL_PATH / f"{task}_heads.joblib"


def is_available(task: str) -> bool:
    return backbone_path(task).exists() and heads_path(task).exists()


class ImageEnsemble:
    """Ten classifiers voting on a CNN embedding, weighted by CV accuracy."""

    def __init__(self, bundle: dict, extractor):
        self.task = bundle["task"]
        self.classes = bundle["classes"]
        self.img_size = bundle.get("img_size", 224)
        self.members = bundle["members"]
        self.scores = bundle["scores"]
        self.majority_rate = bundle.get("majority_rate", 0.0)
        self.backbone_name = bundle.get("backbone", "EfficientNetB0")
        self.extractor = extractor

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, task: str) -> "ImageEnsemble":
        import joblib

        if not is_available(task):
            raise FileNotFoundError(
                f"Missing {backbone_path(task).name} / {heads_path(task).name} "
                f"in models/image/. Train them with "
                f"notebooks/kaggle_image_10_models.ipynb."
            )
        try:
            import tensorflow as tf
        except Exception as exc:
            raise ImportError(
                f"TensorFlow is required to run the image models "
                f"({exc.__class__.__name__}: {exc}). Note that TensorFlow needs "
                f"protobuf>=5.28 while mediapipe pins protobuf<5, so install it "
                f"in its own environment."
            ) from exc

        extractor = tf.keras.models.load_model(backbone_path(task), compile=False)
        return cls(joblib.load(heads_path(task)), extractor)

    # --------------------------------------------------------------- predict
    def embed(self, image) -> np.ndarray:
        """PIL image (or array) -> the backbone's pooled embedding."""
        from PIL import Image

        if not isinstance(image, np.ndarray):
            image = np.asarray(image.convert("RGB").resize(
                (self.img_size, self.img_size), Image.BILINEAR))
        # the saved model carries its own preprocess_input layer, so the raw
        # 0-255 values are what it expects
        return self.extractor.predict(image[None, ...].astype("float32"), verbose=0)

    def predict_proba(self, embedding: np.ndarray) -> np.ndarray:
        weighted = None
        for name, model in self.members.items():
            w = self.scores[name]["weight"]
            if w <= 0:
                continue
            p = model.predict_proba(embedding) * w
            weighted = p if weighted is None else weighted + p
        if weighted is None:
            weighted = np.full((len(embedding), len(self.classes)),
                               1.0 / len(self.classes))
        return weighted

    def explain(self, embedding: np.ndarray) -> pd.DataFrame:
        """Per-algorithm breakdown for one scan — what the UI table shows."""
        rows = []
        for name, model in self.members.items():
            s = self.scores[name]
            p = model.predict_proba(embedding)[0]
            k = int(np.argmax(p))
            rows.append({
                "Algorithm": name,
                "Vote": self.classes[k],
                "Surety %": 100 * float(p[k]),
                "Accuracy %": 100 * s["cv_accuracy"],
                "Weight %": 100 * s["weight"],
            })
        return (pd.DataFrame(rows).sort_values("Weight %", ascending=False)
                .reset_index(drop=True))

    def analyse(self, image):
        """Full result for one image: (probabilities, per-model table)."""
        emb = self.embed(image)
        return self.predict_proba(emb)[0], self.explain(emb)

    def weights_frame(self) -> pd.DataFrame:
        return (pd.DataFrame([{
            "Algorithm": n,
            "CV accuracy %": 100 * s["cv_accuracy"],
            "Weight %": 100 * s["weight"],
        } for n, s in self.scores.items()])
            .sort_values("Weight %", ascending=False).reset_index(drop=True))
