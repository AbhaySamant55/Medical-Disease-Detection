"""Disease registry, paths and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TABULAR_DIR = DATA_DIR / "tabular"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# How much a model's vote counts in the final ensemble.
#   "above_baseline" - weight = max(cv_accuracy - majority_class_rate, 0)
#                      A model that only matches the "always predict the
#                      majority class" baseline has learned nothing, so it
#                      contributes nothing. This is the default.
#   "raw"            - weight = cv_accuracy (every model votes, in proportion
#                      to its accuracy)
WEIGHT_SCHEME = "above_baseline"

# Which cross-validated metric the weight is derived from.
#   "accuracy" - the default, and what the "accuracy %" column in the UI shows.
#   "roc_auc"  - threshold-independent, and generally the better choice on the
#                imbalanced datasets here (liver is 71% positive, so accuracy
#                is a weak signal). Weight becomes max(auc - 0.5, 0).
# Switch this to "roc_auc" and re-run src.train_tabular to compare.
WEIGHT_METRIC = "accuracy"

# --- tabular diseases ----------------------------------------------------
DISEASES = {
    "diabetes": {
        "name": "Diabetes",
        "icon": "🩸",
        "file": "diabetes.csv",
        "target": "Outcome",
        "positive_label": "Diabetic",
        "negative_label": "Not diabetic",
        "source": "Pima Indians Diabetes Database (UCI / NIDDK)",
        "blurb": "Predicts diabetes in female patients of Pima Indian heritage "
                 "aged 21+, from routine diagnostic measurements.",
        # columns where 0 is physiologically impossible and really means "missing"
        "zero_is_missing": ["Glucose", "BloodPressure", "SkinThickness",
                            "Insulin", "BMI"],
    },
    "heart": {
        "name": "Heart Disease",
        "icon": "❤️",
        "file": "heart.csv",
        "target": "target",
        "positive_label": "Heart disease present",
        "negative_label": "No heart disease",
        "source": "UCI Cleveland Heart Disease",
        "blurb": "Detects the presence of angiographic coronary artery disease "
                 "from clinical and exercise-test measurements.",
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "icon": "🎗️",
        "file": "breast_cancer.csv",
        "target": "target",
        "positive_label": "Malignant",
        "negative_label": "Benign",
        "source": "Wisconsin Diagnostic Breast Cancer (WDBC)",
        "blurb": "Classifies a breast mass as malignant or benign from features "
                 "computed on a digitised fine-needle aspirate image.",
    },
    "parkinsons": {
        "name": "Parkinson's Disease",
        "icon": "🧠",
        "file": "parkinsons.csv",
        "target": "target",
        "positive_label": "Parkinson's detected",
        "negative_label": "Healthy",
        "source": "UCI Parkinsons (Oxford / Max Little)",
        "blurb": "Detects Parkinson's from acoustic measures of sustained "
                 "vowel phonation — jitter, shimmer and nonlinear voice measures.",
    },
    "liver": {
        "name": "Liver Disease",
        "icon": "🫀",
        "file": "liver.csv",
        "target": "target",
        "positive_label": "Liver disease",
        "negative_label": "No liver disease",
        "source": "Indian Liver Patient Dataset (UCI)",
        "blurb": "Predicts liver disease from a standard liver function blood panel.",
    },
    "kidney": {
        "name": "Chronic Kidney Disease",
        "icon": "🫘",
        "file": "kidney.csv",
        "target": "target",
        "positive_label": "Chronic kidney disease",
        "negative_label": "Healthy",
        "source": "UCI Chronic Kidney Disease",
        "blurb": "Detects chronic kidney disease from blood and urine tests. "
                 "This dataset is heavily missing-value laden, which makes the "
                 "imputation step matter as much as the classifier.",
    },
}

# --- image diseases ------------------------------------------------------
IMAGE_TASKS = {
    "brain_tumor": {
        "name": "Brain Tumour (MRI)",
        "icon": "🧠",
        "dir": IMAGES_DIR / "brain_tumor",
        "train_subdir": "Training",
        "test_subdir": "Testing",
        "classes": ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
        "pretty_classes": ["Glioma", "Meningioma", "No tumour", "Pituitary"],
        "img_size": 224,
        "source": "Brain Tumour Classification MRI dataset",
        "blurb": "Four-way classification of brain MRI slices.",
    },
    "pneumonia": {
        "name": "Pneumonia (Chest X-ray)",
        "icon": "🫁",
        "dir": IMAGES_DIR / "pneumonia",
        "train_subdir": "train",
        "test_subdir": "test",
        "classes": ["NORMAL", "PNEUMONIA"],
        "pretty_classes": ["Normal", "Pneumonia"],
        "img_size": 224,
        "source": "Kermany paediatric chest X-ray dataset",
        "blurb": "Binary classification of chest radiographs. The classes are "
                 "3:1 imbalanced, so always answering 'pneumonia' scores 74% — "
                 "the models are weighted against that baseline, not against 50%.",
    },
    "covid_xray": {
        "name": "COVID-19 (Chest X-ray)",
        "icon": "🦠",
        "dir": IMAGES_DIR / "covid_xray",
        "train_subdir": "train",
        "test_subdir": "test",
        "classes": ["COVID19", "NORMAL"],
        "pretty_classes": ["COVID-19", "Normal"],
        "img_size": 224,
        "source": "COVID-19 chest radiography collection",
        "blurb": "Binary classification of chest radiographs.",
    },
}

TABULAR_MODEL_PATH = MODELS_DIR / "tabular"
IMAGE_MODEL_PATH = MODELS_DIR / "image"
TABULAR_METRICS = REPORTS_DIR / "tabular_metrics.json"

for _d in (MODELS_DIR, REPORTS_DIR, TABULAR_MODEL_PATH, IMAGE_MODEL_PATH):
    _d.mkdir(parents=True, exist_ok=True)


def disease_path(key: str) -> Path:
    return TABULAR_DIR / DISEASES[key]["file"]


def bundle_path(key: str) -> Path:
    return TABULAR_MODEL_PATH / f"{key}.joblib"
