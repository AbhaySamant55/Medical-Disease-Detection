"""Train all ten algorithms for every tabular disease and build the ensemble.

    python -m src.train_tabular              # every disease
    python -m src.train_tabular diabetes     # just one

For each disease:
  1. stratified 80/20 hold-out
  2. all ten algorithms cross-validated on the *training* split — this produces
     the accuracy figure each model's vote is weighted by
  3. every model refitted on the full training split
  4. the weighted ensemble and all ten members scored on the untouched test split
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config
from .data_prep import feature_metadata, load_disease, split
from .ensemble import WeightedEnsemble
from .models import build_models, make_pipeline

RULE = "=" * 78


def evaluate(y_true, y_pred, y_proba) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except (ValueError, IndexError):
        out["roc_auc"] = float("nan")
    return out


def train_one(key: str) -> dict:
    spec = config.DISEASES[key]
    # the console here is cp1252 on Windows, so keep terminal output ASCII —
    # the emoji in config.DISEASES are for the Streamlit UI
    print(f"\n{RULE}")
    print(spec["name"].upper())
    print(RULE)

    X, y, info = load_disease(key)
    print(f"  {info['n_samples']} samples | {info['n_features']} features | "
          f"{100 * info['positive_rate']:.1f}% positive | "
          f"{100 * info['missing_pct']:.1f}% missing cells")
    print(f"  numeric: {len(info['numeric_cols'])}   "
          f"categorical: {len(info['categorical_cols'])}")
    print(f"  majority-class baseline accuracy: {info['majority_rate']:.4f}")

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"  train {len(X_train)} / test {len(X_test)}\n")

    members = {
        name: {
            "pipeline": make_pipeline(spec_m["estimator"],
                                      info["numeric_cols"],
                                      info["categorical_cols"]),
            "family": spec_m["family"],
            "about": spec_m["about"],
        }
        for name, spec_m in build_models().items()
    }
    print(f"  cross-validating {len(members)} algorithms "
          f"({config.CV_FOLDS}-fold on the training split):")

    t0 = time.perf_counter()
    ens = WeightedEnsemble(members).fit(X_train, y_train)
    print(f"    ({time.perf_counter() - t0:.1f}s)")

    # ---------------------------------------------- held-out test results
    print(f"\n  Held-out test set ({len(X_test)} samples):")
    print(f"    {'model':<30}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}{'AUC':>8}")
    print("    " + "-" * 70)

    member_test: dict[str, dict] = {}
    for name, model in ens.fitted_.items():
        proba = model.predict_proba(X_test)
        pred = proba.argmax(axis=1)
        m = evaluate(y_test, pred, proba[:, 1])
        member_test[name] = m
        print(f"    {name:<30}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}{m['f1']:>8.4f}{m['roc_auc']:>8.4f}")

    ens_proba = ens.predict_proba(X_test)
    ens_pred = ens_proba.argmax(axis=1)
    ens_metrics = evaluate(y_test, ens_pred, ens_proba[:, 1])
    print("    " + "-" * 70)
    print(f"    {'>> WEIGHTED ENSEMBLE':<30}{ens_metrics['accuracy']:>8.4f}"
          f"{ens_metrics['precision']:>8.4f}{ens_metrics['recall']:>8.4f}"
          f"{ens_metrics['f1']:>8.4f}{ens_metrics['roc_auc']:>8.4f}")

    # The fair comparison. In deployment you have to choose your single model
    # using cross-validation on the training split, because the test set is not
    # available yet — so that is what the ensemble has to beat.
    cv_pick = max(ens.member_scores_,
                  key=lambda n: ens.member_scores_[n]["cv_accuracy"])
    cv_pick_auc = member_test[cv_pick]["roc_auc"]
    delta = ens_metrics["roc_auc"] - cv_pick_auc
    verdict = "beats" if delta > 1e-9 else ("ties" if abs(delta) <= 1e-9 else "trails")
    print(f"\n    model picked by CV on train: {cv_pick}")
    print(f"      its test AUC {cv_pick_auc:.4f}  ->  ensemble {verdict} it "
          f"by {delta:+.4f}")

    # Reported for transparency only: the best model chosen *with hindsight* on
    # the test set. Nobody can pick this in advance, so it is an oracle, not a
    # baseline the ensemble is expected to beat.
    oracle_name = max(member_test, key=lambda n: member_test[n]["roc_auc"]
                      if not np.isnan(member_test[n]["roc_auc"]) else -1)
    oracle_auc = member_test[oracle_name]["roc_auc"]
    print(f"    (oracle best-on-test: {oracle_name} at {oracle_auc:.4f} — "
          f"selectable only in hindsight)")

    print(f"\n{classification_report(y_test, ens_pred, digits=4, target_names=[spec['negative_label'], spec['positive_label']])}")

    # ------------------------------------------------------------- save
    bundle = {
        "key": key,
        "ensemble": ens,
        "info": info,
        "feature_meta": feature_metadata(X, info["numeric_cols"],
                                         info["categorical_cols"]),
        "columns": X.columns.tolist(),
        "test_metrics": {"ensemble": ens_metrics, "members": member_test},
        "confusion_matrix": confusion_matrix(y_test, ens_pred).tolist(),
    }
    path = config.bundle_path(key)
    joblib.dump(bundle, path, compress=3)
    print(f"  saved -> {path.relative_to(config.ROOT)} "
          f"({path.stat().st_size / 1e6:.1f} MB)")

    return {
        "info": {k: v for k, v in info.items()
                 if k not in ("numeric_cols", "categorical_cols")},
        "majority_baseline": info["majority_rate"],
        "cv_weights": ens.weights_frame().to_dict("records"),
        "test": {"ensemble": ens_metrics, "members": member_test},
        "cv_selected_model": {"name": cv_pick, "test_roc_auc": cv_pick_auc},
        "oracle_best_on_test": {"name": oracle_name, "roc_auc": oracle_auc},
        "ensemble_gain_auc": float(delta),
        "confusion_matrix": bundle["confusion_matrix"],
    }


def main() -> None:
    keys = sys.argv[1:] or list(config.DISEASES)
    unknown = [k for k in keys if k not in config.DISEASES]
    if unknown:
        raise SystemExit(f"Unknown disease(s): {unknown}. "
                         f"Available: {list(config.DISEASES)}")

    all_results, t_start = {}, time.perf_counter()
    for key in keys:
        try:
            all_results[key] = train_one(key)
        except FileNotFoundError as exc:
            print(f"  ! skipping {key}: {exc}")

    # -------------------------------------------------------- summary
    print(f"\n{RULE}"); print("SUMMARY — weighted ensemble on held-out test sets")
    print(RULE)
    rows = []
    for key, r in all_results.items():
        e = r["test"]["ensemble"]
        rows.append({
            "Disease": config.DISEASES[key]["name"],
            "Samples": r["info"]["n_samples"],
            "Baseline": round(r["majority_baseline"], 4),
            "Accuracy": round(e["accuracy"], 4),
            "Precision": round(e["precision"], 4),
            "Recall": round(e["recall"], 4),
            "F1": round(e["f1"], 4),
            "ROC-AUC": round(e["roc_auc"], 4),
            "CV-picked model": r["cv_selected_model"]["name"],
            "its AUC": round(r["cv_selected_model"]["test_roc_auc"], 4),
            "Ensemble gain": round(r["ensemble_gain_auc"], 4),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    config.TABULAR_METRICS.write_text(
        json.dumps({"diseases": all_results,
                    "weight_scheme": config.WEIGHT_SCHEME,
                    "cv_folds": config.CV_FOLDS,
                    "summary": rows}, indent=2, default=str),
        encoding="utf-8")
    print(f"\nSaved -> {config.TABULAR_METRICS.relative_to(config.ROOT)}")
    print(f"Total time: {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
