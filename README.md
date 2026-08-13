# 🩺 Medical Disease Detection System

A multi-disease screening system covering **9 conditions — 6 from clinical data
and 3 from medical images**. Every condition is run through **10 different
machine-learning algorithms**, and the final verdict is a soft vote in which
each algorithm's opinion is weighted by how good that algorithm actually is.

Images go through the same ten algorithms: a fine-tuned CNN turns the scan into
an embedding, and the ten classifiers vote on that.

> ⚠️ **Not a medical device.** A student project on small public research
> datasets. It cannot diagnose anyone and must never be used to.

---

## Conditions covered

| # | Condition | Data | Samples | Approach |
|---|---|---|---|---|
| 1 | Diabetes | Pima Indians (UCI/NIDDK) | 768 | 10-model ensemble |
| 2 | Heart Disease | UCI Cleveland | 303 | 10-model ensemble |
| 3 | Breast Cancer | Wisconsin Diagnostic | 569 | 10-model ensemble |
| 4 | Parkinson's Disease | UCI (voice measures) | 195 | 10-model ensemble |
| 5 | Liver Disease | Indian Liver Patient (UCI) | 583 | 10-model ensemble |
| 6 | Chronic Kidney Disease | UCI CKD | 399 | 10-model ensemble |
| 7 | Brain Tumour | MRI, 4 classes | 3,264 images | CNN embedding + 10-model ensemble |
| 8 | Pneumonia | Chest X-ray | 5,863 images | CNN embedding + 10-model ensemble |
| 9 | COVID-19 | Chest X-ray | 2,284 images | CNN embedding + 10-model ensemble |

---

## Results — weighted ensemble on held-out test sets

Every figure is on a stratified 20% split that was used **only** for this
table: never for training, never for tuning, never for setting the ensemble
weights.

| Condition | Baseline* | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Chronic Kidney Disease | 0.627 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| Breast Cancer | 0.627 | **0.9561** | 1.0000 | 0.8810 | 0.9367 | **0.9983** |
| Parkinson's Disease | 0.754 | **0.9231** | 0.9333 | 0.9655 | 0.9492 | **0.9759** |
| Heart Disease | 0.541 | **0.8852** | 0.8387 | 0.9286 | 0.8814 | **0.9491** |
| Diabetes | 0.651 | **0.7338** | 0.6383 | 0.5556 | 0.5941 | **0.8159** |
| Liver Disease | 0.714 | **0.7179** | 0.7500 | 0.9036 | 0.8197 | **0.8037** |

\* accuracy you would get by always predicting the majority class.

Kidney reaching a perfect score is a property of the dataset, not a triumph of
modelling — UCI CKD is close to linearly separable, and several individual
models hit 1.000 as well. Diabetes and liver are the genuinely hard ones, and
the numbers are reported as they came out.

---

## How the 10-model ensemble works

Each condition trains all ten algorithms, then combines them using two numbers
per model:

- **Accuracy %** — how often that algorithm was right during 5-fold
  cross-validation *on the training split*. A fixed property of the model.
- **Surety %** — the probability that algorithm assigns to *this particular*
  patient. Varies case by case.

```
final_probability  =  Σ ( weightₘ × suretyₘ )  /  Σ weightₘ
```

### The weight is skill above baseline, not raw accuracy

```
weightₘ = max( cv_accuracyₘ − majority_class_rate , 0 )
```

This correction matters. The liver dataset is **71.4% positive**, so a model
scoring 71% accuracy has learned exactly nothing — it is just predicting
"diseased" every time. Under raw-accuracy weighting it would still cast a large
vote and drag the ensemble toward the majority class. Subtracting the baseline
gives it a weight of zero.

Worked example — heart disease, weights actually assigned:

| Algorithm | CV accuracy | Weight |
|---|---|---|
| Support Vector Machine | 83.06% | 11.29% |
| Logistic Regression | 82.64% | 11.13% |
| Gaussian Naive Bayes | 81.82% | 10.81% |
| Random Forest | 81.40% | 10.65% |
| Extra Trees | 81.40% | 10.65% |
| K-Nearest Neighbours | 79.75% | 10.00% |
| XGBoost | 78.51% | 9.52% |
| Gradient Boosting | 77.69% | 9.19% |
| Neural Network (MLP) | 77.69% | 9.19% |
| Decision Tree | 73.55% | 7.58% |

The weakest model still contributes, but at two-thirds the influence of the
strongest.

### Does the ensemble actually help?

Honest answer: **sometimes.**

| Condition | Ensemble AUC | Model picked by CV | Its AUC | Gain |
|---|---|---|---|---|
| Heart Disease | 0.9491 | SVM | 0.9437 | **+0.0054** |
| Breast Cancer | 0.9983 | SVM | 0.9947 | **+0.0036** |
| Diabetes | 0.8159 | Logistic Regression | 0.8130 | **+0.0030** |
| Chronic Kidney | 1.0000 | SVM | 1.0000 | 0.0000 |
| Parkinson's | 0.9759 | Extra Trees | 0.9828 | −0.0069 |
| Liver Disease | 0.8037 | Logistic Regression | 0.8306 | −0.0269 |

It wins on 3 of 6, ties on 1, loses on 2, averaging −0.0036 AUC. The ensemble's
real benefit is **robustness rather than peak accuracy** — it is never the
disaster that a single badly-chosen model can be, and you do not have to guess
which algorithm suits a dataset in advance.

**One methodological point worth stating plainly.** The comparison above is
against the model chosen by cross-validation on the *training* split, because
that is the only model you could actually pick in advance. Comparing against
the best model *on the test set* would be comparing against an oracle: the
maximum of ten noisy test scores beats a blend almost automatically. Reported
that way the ensemble would look worse on all six diseases, and the comparison
would be meaningless. Both numbers are in `reports/tabular_metrics.json`.

---

## The ten algorithms

Chosen to span different inductive biases — ten near-identical models would
just average to one of themselves.

| Algorithm | Family | What it brings |
|---|---|---|
| Logistic Regression | Linear | Well-calibrated probabilities |
| Gaussian Naive Bayes | Probabilistic | Strong when features are near-independent |
| K-Nearest Neighbours | Instance-based | No training; pure local similarity |
| Decision Tree | Tree | Readable if/else rules |
| Random Forest | Bagging | The reliable tabular all-rounder |
| Extra Trees | Bagging | Randomised splits, lower variance |
| Gradient Boosting | Boosting | Sequential error correction |
| XGBoost | Boosting | Regularised boosting, usually the strongest single model |
| Support Vector Machine | Kernel | Maximum-margin separation in kernel space |
| Neural Network (MLP) | Neural | 128→64 dense net, Adam + early stopping |

If XGBoost is not installed the roster falls back to Linear Discriminant
Analysis, keeping the count at ten.

---

## Data handling that changes the results

- **Zeros that are really missing values.** In the Pima diabetes data, `Glucose`,
  `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` contain 0 — nobody has a
  BMI of 0. Left alone, the models learn that 0 is a meaningful reading.
  They are converted to `NaN` and imputed instead.
- **Chronic kidney disease is 25% missing cells** and mixes numeric and
  categorical columns. A `ColumnTransformer` median-imputes and scales the
  numerics, mode-imputes and one-hot encodes the categoricals.
- **Label direction is made consistent.** scikit-learn ships breast cancer as
  0 = malignant; UCI ships liver as 1 = patient, 2 = healthy. Both are flipped so
  that across all six datasets `1` always means "has the disease".
- **Everything lives inside a `Pipeline`,** so imputation and scaling are fitted
  per cross-validation fold. Fitting a scaler on the whole dataset first is the
  most common way to leak information and inflate these scores.

---

## Quick start

```bash
pip install -r requirements.txt
python -m src.download_data      # rebuild all six CSVs from source (~200 KB)
python -m src.train_tabular      # 6 diseases x 10 algorithms (~2 min, CPU)
streamlit run app.py
```

Train just one condition:

```bash
python -m src.train_tabular diabetes
```

The web UI has three modes: enter patient measurements and see the weighted
verdict with the full 10-algorithm breakdown; upload a scan for the CNN models;
or browse the complete performance dashboard.

---

## Imaging: the same 10 algorithms, on scans

**Brain tumour MRI and chest X-ray pneumonia are detected by the same ten-model
weighted ensemble as the clinical data** — not by a single CNN.

The ten classifiers cannot read pixels, so a CNN converts each scan into a
vector first:

```
scan -> fine-tuned EfficientNetB0 -> 1280-d embedding -> PCA 128
     -> 10 classifiers -> accuracy-weighted vote -> verdict
```

The backbone is **fine-tuned on the medical images before extracting features**,
not left frozen. Frozen ImageNet features are tuned for cats and cars; a few
epochs of fine-tuning moves them toward tissue texture and hands the ten heads a
far more separable space.

That the CNN step is doing real work is measurable. Running the identical ten
algorithms on **raw 64×64 pixels instead of CNN embeddings** gives, on the brain
MRI test set:

| Features | Ensemble accuracy | Glioma recall |
|---|---|---|
| Raw pixels (no CNN) | 0.7234 | 0.2000 |
| CNN embedding | — run the notebook | — |

Raw pixels essentially cannot find gliomas. Whatever the embedding version
scores, that 0.7234 is the floor it has to clear, and the gap is the CNN's
contribution. The weighting behaves the same way it does on tabular data — even
on raw pixels the ensemble beat the CV-selected single model by **+0.0152**.

### A dataset defect worth knowing about

The first trained brain-tumour ensemble scored 72.9% overall, which looked like
a mediocre model. The per-class breakdown said otherwise:

| Class | Training folder | Testing folder |
|---|---|---|
| glioma | 20/20 | **1/20** |
| meningioma | 20/20 | 19/20 |
| no tumour | 20/20 | 20/20 |
| pituitary | 19/20 | 18/20 |

Everything except glioma was fine. Checking the images explained it: in
`Training/glioma_tumor` every image is 512×512, while `Testing/glioma_tumor`
is a mix of 236×236, 524×581, 554×554 and more — the two folders were populated
from different sources. **This is a known defect in
`sartajbhuvaji/brain-tumor-classification-mri`**, and it means that class's test
score measures the dataset rather than the model. On the three sound classes the
same ensemble scores **95%**.

Use **`masoudnickparvar/brain-tumor-mri-dataset`** instead — the cleaned,
re-labelled release (7,023 images) without the problem. The notebook now also
compares image dimensions across train/test at load time and prints a warning
when a class looks inconsistent, so this gets caught before training rather than
after.

### Training them

One notebook does both tasks. Attach either dataset or both:

| Notebook | Datasets to attach on Kaggle |
|---|---|
| **`notebooks/kaggle_image_10_models.ipynb`** | `masoudnickparvar/brain-tumor-mri-dataset` and/or `paultimothymooney/chest-xray-pneumonia` |

1. open on [kaggle.com/code](https://www.kaggle.com/code) → **File → Import Notebook**
2. **Add Input** → attach one or both datasets (paths are auto-discovered)
3. **Settings → Accelerator → GPU T4 x2** → **Run All** (~10 min per dataset)
4. download into `models/image/`:
   - `brain_tumor_backbone.keras` + `brain_tumor_heads.joblib`
   - `pneumonia_backbone.keras` + `pneumonia_heads.joblib`

The app picks them up automatically and shows the full 10-algorithm breakdown
for an uploaded scan, exactly as it does for the clinical data.

Two single-CNN notebooks (`kaggle_brain_tumor_cnn.ipynb`,
`kaggle_pneumonia_cnn.ipynb`) are also included as the simpler baseline; the
10-model notebook reports a plain fine-tuned CNN score alongside the ensemble so
the two are directly comparable.

The pneumonia notebook handles two traps in that dataset that are easy to miss:
its official validation split contains only **16 images** (so a proper one is
carved out of train instead), and the classes are **3:1 imbalanced** — meaning a
model that always answers "pneumonia" scores 74%. Class weights are applied and
a threshold sweep is reported instead of assuming 0.5 is correct.

To train any of the three locally instead — slow on CPU, and needs TensorFlow:

```bash
python -m src.download_data --images --covid
python -m src.train_image brain_tumor
python -m src.train_image covid_xray
```

> **Environment note.** TensorFlow requires `protobuf >= 5.28` while
> `mediapipe` pins `protobuf < 5`, so the two cannot share one environment. The
> tabular half of this project needs neither and runs anywhere.

---

## Project layout

```
02-medical-disease-detection/
├── app.py                        # Streamlit UI (3 modes)
├── requirements.txt
├── data/tabular/                 # the six cleaned CSVs (committed)
├── data/images/                  # git-ignored; rebuild with download_data
├── models/tabular/               # one bundle per disease
├── notebooks/                    # Kaggle GPU notebooks for the CNNs
├── reports/tabular_metrics.json  # every number in this README
└── src/
    ├── config.py                 # disease registry + weighting scheme
    ├── data_prep.py              # per-disease cleaning + UI field metadata
    ├── models.py                 # the 10 algorithms + preprocessing
    ├── ensemble.py               # accuracy-weighted soft voting
    ├── train_tabular.py          # trains and scores everything
    ├── train_image.py            # CNN training (local or Kaggle)
    └── download_data.py
```

## Tech stack

Python · scikit-learn · XGBoost · pandas · NumPy · Streamlit ·
TensorFlow/Keras (imaging) · Matplotlib · seaborn

## Datasets

All public research datasets: UCI Machine Learning Repository (Cleveland Heart
Disease, Parkinsons, Indian Liver Patient, Chronic Kidney Disease), the Pima
Indians Diabetes Database, the Wisconsin Diagnostic Breast Cancer set shipped
with scikit-learn, and public brain-MRI and chest X-ray collections.

## License

MIT — see [LICENSE](LICENSE).
