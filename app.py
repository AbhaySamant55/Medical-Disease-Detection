"""Streamlit UI for the multi-disease detection system.

Every disease is scored by ten algorithms; the app shows the final weighted
verdict and the full per-model breakdown behind it.

    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Medical Disease Detection System",
                   page_icon="🩺", layout="wide")

DISCLAIMER = (
    "This is a student machine-learning project trained on small public research "
    "datasets. It is **not a medical device** and must not be used for diagnosis. "
    "Always consult a qualified clinician."
)


@st.cache_resource
def load_bundle(key: str):
    path = config.bundle_path(key)
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_metrics():
    if config.TABULAR_METRICS.exists():
        return json.loads(config.TABULAR_METRICS.read_text(encoding="utf-8"))
    return None


@st.cache_resource
def load_image_ensemble(task_key: str):
    """Returns (ensemble, error_message)."""
    from src.image_ensemble import ImageEnsemble, is_available
    if not is_available(task_key):
        return None, "missing"
    try:
        return ImageEnsemble.load(task_key), None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 Disease Detection")
mode = st.sidebar.radio("Mode", ["Clinical data (10-model ensemble)",
                                 "Medical imaging (10-model ensemble)",
                                 "Model performance"])
st.sidebar.divider()

available = [k for k in config.DISEASES if config.bundle_path(k).exists()]
if not available and mode == "Clinical data (10-model ensemble)":
    st.error("No trained models found. Run `python -m src.train_tabular` first.")
    st.stop()


# ================================================================== TABULAR
if mode == "Clinical data (10-model ensemble)":
    key = st.sidebar.selectbox(
        "Condition", available,
        format_func=lambda k: f"{config.DISEASES[k]['icon']} {config.DISEASES[k]['name']}")
    bundle = load_bundle(key)
    ens, info, meta = bundle["ensemble"], bundle["info"], bundle["feature_meta"]
    spec = config.DISEASES[key]

    st.title(f"{spec['icon']} {spec['name']}")
    st.caption(spec["blurb"])

    c1, c2, c3, c4 = st.columns(4)
    t = bundle["test_metrics"]["ensemble"]
    c1.metric("Test accuracy", f"{t['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    c3.metric("Recall (sensitivity)", f"{t['recall']:.1%}")
    c4.metric("Algorithms", len(ens.fitted_))

    st.divider()
    st.subheader("Patient measurements")
    st.caption("Defaults are the dataset median for each field.")

    columns = bundle["columns"]
    values = {}
    ncol = 3
    cols = st.columns(ncol)
    for n, feature in enumerate(columns):
        m = meta.get(feature, {})
        target = cols[n % ncol]
        with target:
            if m.get("kind") == "categorical":
                opts = m.get("options", [])
                default = m.get("default")
                idx = opts.index(default) if default in opts else 0
                values[feature] = st.selectbox(feature, opts, index=idx, key=f"{key}_{feature}")
            else:
                lo, hi = float(m.get("min", 0.0)), float(m.get("max", 1.0))
                default = float(np.clip(m.get("default", lo), lo, hi))
                if m.get("integer"):
                    values[feature] = st.number_input(
                        feature, min_value=int(lo), max_value=int(hi),
                        value=int(default), step=1, key=f"{key}_{feature}")
                else:
                    values[feature] = st.number_input(
                        feature, min_value=lo, max_value=hi, value=default,
                        step=float(m.get("step", 0.01)), format="%.4f",
                        key=f"{key}_{feature}")

    st.divider()
    if st.button("🔬 Run all 10 algorithms", type="primary", use_container_width=True):
        X = pd.DataFrame([values])[columns]
        proba = ens.predict_proba(X)[0]
        p_disease = float(proba[1])
        detail = ens.explain(X)

        verdict = spec["positive_label"] if p_disease >= 0.5 else spec["negative_label"]
        votes_positive = int((detail["Prediction"] == 1).sum())
        n_models = len(detail)

        st.subheader("Final weighted verdict")
        left, right = st.columns([2, 3])
        with left:
            if p_disease >= 0.5:
                st.error(f"### {verdict}")
            else:
                st.success(f"### {verdict}")
            st.metric("Weighted probability of disease", f"{p_disease:.1%}")
            st.progress(min(max(p_disease, 0.0), 1.0))
            st.metric("Model agreement",
                      f"{votes_positive}/{n_models} say {spec['positive_label'].lower()}")
            confidence = max(p_disease, 1 - p_disease)
            st.caption(
                f"Combined surety **{confidence:.1%}**. The verdict is the "
                "accuracy-weighted average of all ten probabilities, not a simple "
                "majority vote."
            )
        with right:
            chart = (detail.set_index("Algorithm")[["P(disease) %"]]
                     .sort_values("P(disease) %"))
            st.bar_chart(chart, height=330)

        st.subheader("How each algorithm voted")
        st.caption(
            "**Surety %** is that model's confidence in *this* case. "
            "**Accuracy %** is its cross-validated accuracy on the training split. "
            "**Weight %** is its share of the final answer — models that beat the "
            f"majority-class baseline ({info['majority_rate']:.1%}) by more, count for more."
        )
        show = detail.copy()
        show["Vote"] = np.where(show["Prediction"] == 1,
                                spec["positive_label"], spec["negative_label"])
        show = show[["Algorithm", "Family", "Vote", "Surety %", "P(disease) %",
                     "Accuracy %", "Weight %"]]
        st.dataframe(
            show.style
                .format({"Surety %": "{:.1f}", "P(disease) %": "{:.1f}",
                         "Accuracy %": "{:.1f}", "Weight %": "{:.1f}"})
                .background_gradient(subset=["Weight %"], cmap="Blues")
                .background_gradient(subset=["P(disease) %"], cmap="Reds"),
            use_container_width=True, hide_index=True)

        with st.expander("What each algorithm does"):
            for _, row in detail.iterrows():
                st.markdown(f"**{row['Algorithm']}** *({row['Family']})* — {row['About']}")

    st.divider()
    st.info(DISCLAIMER)


# ==================================================================== IMAGE
elif mode == "Medical imaging (10-model ensemble)":
    task_key = st.sidebar.selectbox(
        "Scan type", list(config.IMAGE_TASKS),
        format_func=lambda k: f"{config.IMAGE_TASKS[k]['icon']} {config.IMAGE_TASKS[k]['name']}")
    task = config.IMAGE_TASKS[task_key]

    st.title(f"{task['icon']} {task['name']}")
    st.caption(task["blurb"])
    st.caption(
        "A fine-tuned EfficientNetB0 turns the scan into a 1280-dimensional "
        "embedding, then the **same ten algorithms** used for the clinical data "
        "vote on it, weighted by their cross-validated accuracy."
    )

    ens, err = load_image_ensemble(task_key)
    if ens is None:
        if err == "missing":
            st.warning(
                f"No trained models found for **{task['name']}**.\n\n"
                f"Expected `models/image/{task_key}_backbone.keras` and "
                f"`models/image/{task_key}_heads.joblib`.\n\n"
                "Train them on a Kaggle GPU with "
                "`notebooks/kaggle_image_10_models.ipynb`, then download both "
                "files from the notebook's Output panel into `models/image/`."
            )
        else:
            st.error(f"Could not load the image models.\n\n`{err}`")
            st.info(
                "TensorFlow needs `protobuf>=5.28` while mediapipe pins "
                "`protobuf<5`. Install TensorFlow in its own virtual "
                "environment to run the imaging models."
            )
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithms", len(ens.members))
    c2.metric("Classes", len(ens.classes))
    c3.metric("Backbone", ens.backbone_name)

    up = st.file_uploader("Upload a scan", type=["jpg", "jpeg", "png"])
    if up is not None:
        from PIL import Image
        img = Image.open(up).convert("RGB")

        with st.spinner("Embedding the scan and polling 10 algorithms ..."):
            probs, detail = ens.analyse(img)

        top = int(np.argmax(probs))
        verdict = ens.classes[top]
        votes = detail["Vote"].value_counts()

        left, right = st.columns([2, 3])
        with left:
            st.image(img, caption="Uploaded scan", use_container_width=True)

        with right:
            st.subheader("Final weighted verdict")
            healthy = any(w in verdict.lower() for w in ("no_tumor", "no tumour",
                                                         "normal", "healthy"))
            (st.success if healthy else st.error)(f"### {verdict}")
            st.metric("Weighted probability", f"{probs[top]:.1%}")
            st.progress(float(np.clip(probs[top], 0, 1)))
            st.metric("Model agreement",
                      f"{int(votes.get(verdict, 0))}/{len(detail)} agree")
            st.bar_chart(
                pd.DataFrame({"class": ens.classes, "probability": probs})
                .set_index("class"), height=240)

        st.subheader("How each algorithm voted")
        st.caption(
            "**Surety %** is that model's confidence in this scan. "
            "**Accuracy %** is its cross-validated accuracy during training. "
            "**Weight %** is its share of the final answer."
        )
        st.dataframe(
            detail.style
                  .format({"Surety %": "{:.1f}", "Accuracy %": "{:.1f}",
                           "Weight %": "{:.1f}"})
                  .background_gradient(subset=["Weight %"], cmap="Blues"),
            use_container_width=True, hide_index=True)
    else:
        st.info("Upload a scan to run all ten algorithms on it.")
        with st.expander("Algorithm weights for this task"):
            st.dataframe(
                ens.weights_frame().style.format({
                    "CV accuracy %": "{:.2f}", "Weight %": "{:.2f}"}),
                use_container_width=True, hide_index=True)

    st.divider()
    st.info(DISCLAIMER)


# ============================================================== PERFORMANCE
else:
    st.title("📊 Model performance")
    metrics = load_metrics()
    if metrics is None:
        st.warning("No metrics yet. Run `python -m src.train_tabular`.")
        st.stop()

    st.caption(
        f"Weighting scheme: `{metrics['weight_scheme']}` · "
        f"{metrics['cv_folds']}-fold cross-validation on the training split. "
        "All figures below are on held-out test sets never seen during training "
        "or weighting."
    )

    summary = pd.DataFrame(metrics["summary"])
    st.subheader("Weighted ensemble across all conditions")
    st.dataframe(
        summary.style.format({
            "Baseline": "{:.3f}", "Accuracy": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}",
            "its AUC": "{:.3f}", "Ensemble gain": "{:+.4f}",
        }).background_gradient(subset=["ROC-AUC"], cmap="Greens"),
        use_container_width=True, hide_index=True)

    st.divider()
    key = st.selectbox("Per-algorithm detail", list(metrics["diseases"]),
                       format_func=lambda k: config.DISEASES[k]["name"])
    d = metrics["diseases"][key]

    st.write("**Cross-validated on the training split** — this is what sets each "
             "model's weight:")
    st.dataframe(
        pd.DataFrame(d["cv_weights"]).style.format({
            "CV accuracy %": "{:.2f}", "ROC-AUC": "{:.4f}", "F1": "{:.4f}",
            "Precision": "{:.4f}", "Recall": "{:.4f}", "Weight %": "{:.2f}",
        }).background_gradient(subset=["Weight %"], cmap="Blues"),
        use_container_width=True, hide_index=True)

    st.write("**On the held-out test set:**")
    members = pd.DataFrame(d["test"]["members"]).T.reset_index(names="Algorithm")
    ens_row = pd.DataFrame([{"Algorithm": ">> WEIGHTED ENSEMBLE",
                             **d["test"]["ensemble"]}])
    st.dataframe(
        pd.concat([members, ens_row], ignore_index=True).style.format({
            "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
            "f1": "{:.4f}", "roc_auc": "{:.4f}"}),
        use_container_width=True, hide_index=True)

    cm = np.array(d["confusion_matrix"])
    spec = config.DISEASES[key]
    st.write("**Ensemble confusion matrix:**")
    st.dataframe(pd.DataFrame(
        cm,
        index=[f"actual: {spec['negative_label']}", f"actual: {spec['positive_label']}"],
        columns=[f"predicted: {spec['negative_label']}",
                 f"predicted: {spec['positive_label']}"]),
        use_container_width=True)

    st.info(DISCLAIMER)
