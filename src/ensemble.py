"""Accuracy-weighted soft-voting ensemble over the ten algorithms.

Every disease is run through all ten models. The final verdict combines two
numbers per model:

  * **accuracy %** — how often that algorithm was right during cross-validation
    on the *training* split. This is a property of the model, fixed after
    training. It never touches the test set, so the held-out evaluation stays
    honest.
  * **surety %** — the probability that algorithm assigns to *this particular*
    patient. This varies from case to case.

        final_probability = Σ ( weightₘ × suretyₘ ) / Σ weightₘ

With `WEIGHT_SCHEME = "above_baseline"` the weight is how far the model beat
the "always guess the majority class" baseline:

        weightₘ = max( cv_accuracyₘ − majority_class_rate , 0 )

That correction matters on the skewed datasets here. Liver disease is 71%
positive, so a model scoring 71% accuracy has learned precisely nothing — under
raw-accuracy weighting it would still cast a large vote and drag the ensemble
toward the majority class. Subtracting the baseline gives it a weight of zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Soft voting where each member's vote is scaled by its CV accuracy."""

    def __init__(self, members: dict, cv_folds: int = config.CV_FOLDS,
                 weight_scheme: str = config.WEIGHT_SCHEME,
                 weight_metric: str = config.WEIGHT_METRIC,
                 random_state: int = config.RANDOM_STATE):
        self.members = members              # {name: {"pipeline", "family", "about"}}
        self.cv_folds = cv_folds
        self.weight_scheme = weight_scheme
        self.weight_metric = weight_metric
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, verbose: bool = True) -> "WeightedEnsemble":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts = np.bincount(y.astype(int))
        self.majority_rate_ = float(counts.max() / counts.sum())

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)

        self.fitted_: dict = {}
        self.member_scores_: dict = {}

        if verbose:
            print(f"    {'model':<30}{'CV acc':>9}{'ROC-AUC':>9}"
                  f"{'F1':>8}{'weight':>9}")
            print("    " + "-" * 65)

        for name, spec in self.members.items():
            pipe = spec["pipeline"]

            # out-of-fold probabilities -> honest, leak-free estimates
            proba = cross_val_predict(clone(pipe), X, y, cv=cv,
                                      method="predict_proba", n_jobs=1)
            pred = proba.argmax(axis=1)

            acc = accuracy_score(y, pred)
            try:
                auc = roc_auc_score(y, proba[:, 1])
            except (ValueError, IndexError):
                auc = float("nan")

            scores = {
                "cv_accuracy": float(acc),
                "cv_roc_auc": float(auc),
                "cv_f1": float(f1_score(y, pred, zero_division=0)),
                "cv_precision": float(precision_score(y, pred, zero_division=0)),
                "cv_recall": float(recall_score(y, pred, zero_division=0)),
                "family": spec.get("family", ""),
                "about": spec.get("about", ""),
            }

            if self.weight_metric == "roc_auc" and not np.isnan(auc):
                # 0.5 is the chance level for AUC, whatever the class balance
                score, floor = auc, 0.5
            else:
                score, floor = acc, self.majority_rate_

            if self.weight_scheme == "above_baseline":
                weight = max(score - floor, 0.0)
            else:
                weight = float(score)
            scores["weight_raw"] = float(weight)

            fitted = clone(pipe).fit(X, y)
            self.fitted_[name] = fitted
            self.member_scores_[name] = scores

            if verbose:
                print(f"    {name:<30}{acc:>9.4f}{auc:>9.4f}"
                      f"{scores['cv_f1']:>8.4f}{weight:>9.4f}")

        total = sum(s["weight_raw"] for s in self.member_scores_.values())
        if total <= 0:
            # every model was at or below the baseline — fall back to equal votes
            for s in self.member_scores_.values():
                s["weight"] = 1.0 / len(self.member_scores_)
        else:
            for s in self.member_scores_.values():
                s["weight"] = s["weight_raw"] / total

        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, X) -> np.ndarray:
        weighted = None
        for name, model in self.fitted_.items():
            w = self.member_scores_[name]["weight"]
            if w <= 0:
                continue
            p = model.predict_proba(X)
            weighted = p * w if weighted is None else weighted + p * w
        if weighted is None:                       # all weights zero
            weighted = np.column_stack([
                np.full(len(X), 1 - self.majority_rate_),
                np.full(len(X), self.majority_rate_)])
        return weighted

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # -------------------------------------------------------------- explain
    def explain(self, X) -> pd.DataFrame:
        """Per-model breakdown for a single sample — what the UI table shows."""
        rows = []
        for name, model in self.fitted_.items():
            s = self.member_scores_[name]
            p = model.predict_proba(X)[0]
            positive = float(p[1]) if len(p) > 1 else float(p[0])
            rows.append({
                "Algorithm": name,
                "Family": s["family"],
                "Prediction": int(np.argmax(p)),
                "Surety %": 100 * float(p.max()),
                "P(disease) %": 100 * positive,
                "Accuracy %": 100 * s["cv_accuracy"],
                "ROC-AUC": s["cv_roc_auc"],
                "Weight %": 100 * s["weight"],
                "Contribution": s["weight"] * positive,
                "About": s["about"],
            })
        df = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------- summary
    def weights_frame(self) -> pd.DataFrame:
        rows = [{
            "Algorithm": name,
            "Family": s["family"],
            "CV accuracy %": 100 * s["cv_accuracy"],
            "ROC-AUC": s["cv_roc_auc"],
            "F1": s["cv_f1"],
            "Precision": s["cv_precision"],
            "Recall": s["cv_recall"],
            "Weight %": 100 * s["weight"],
        } for name, s in self.member_scores_.items()]
        return (pd.DataFrame(rows)
                .sort_values("Weight %", ascending=False)
                .reset_index(drop=True))
