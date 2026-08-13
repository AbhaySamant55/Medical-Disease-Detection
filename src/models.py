"""The ten algorithms every disease is trained with.

They deliberately span very different inductive biases — linear, probabilistic,
instance-based, single tree, bagged trees, two boosting families, kernel and
neural. Models that make different *kinds* of mistake are what makes the
weighted ensemble in `ensemble.py` worth building; ten near-identical models
would just average to one of themselves.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Median-impute + scale numerics, mode-impute + one-hot the categoricals.

    Scaling is required by the distance- and gradient-based models (KNN, SVM,
    MLP, logistic regression) and harmless for the tree models, so one shared
    preprocessor keeps every pipeline comparable.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", numeric_pipe, numeric_cols)]

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def build_models(random_state: int = config.RANDOM_STATE) -> dict:
    """The ten classifiers, each with a short human-readable description."""
    rs = random_state
    models = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=3000, C=1.0, random_state=rs),
            "family": "Linear",
            "about": "Fits a linear decision boundary; well-calibrated probabilities.",
        },
        "Gaussian Naive Bayes": {
            "estimator": GaussianNB(),
            "family": "Probabilistic",
            "about": "Assumes features are independent given the class. Fast, "
                     "strong when that assumption roughly holds.",
        },
        "Linear Discriminant Analysis": {
            "estimator": LinearDiscriminantAnalysis(),
            "family": "Probabilistic",
            "about": "Models each class as a Gaussian with a shared covariance "
                     "matrix and separates them linearly.",
        },
        "K-Nearest Neighbours": {
            "estimator": KNeighborsClassifier(n_neighbors=9, weights="distance"),
            "family": "Instance-based",
            "about": "Classifies by looking at the 9 most similar patients in "
                     "the training set. No training phase at all.",
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=5, random_state=rs),
            "family": "Tree",
            "about": "A single readable tree of if/else rules. Depth-limited to "
                     "stop it memorising the training set.",
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=rs),
            "family": "Bagging",
            "about": "400 decorrelated trees voting together — the reliable "
                     "all-rounder on tabular clinical data.",
        },
        "Extra Trees": {
            "estimator": ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=rs),
            "family": "Bagging",
            "about": "Like a random forest but with randomised split points, "
                     "which lowers variance further.",
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=rs),
            "family": "Boosting",
            "about": "Builds shallow trees in sequence, each correcting the "
                     "errors left by the previous ones.",
        },
        "Support Vector Machine": {
            "estimator": SVC(C=1.0, kernel="rbf", gamma="scale",
                             probability=True, random_state=rs),
            "family": "Kernel",
            "about": "Finds the widest margin between classes in a high-"
                     "dimensional kernel space.",
        },
        "Neural Network (MLP)": {
            "estimator": MLPClassifier(
                hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=1500,
                early_stopping=True, n_iter_no_change=30, random_state=rs),
            "family": "Neural",
            "about": "A two-hidden-layer fully connected network trained with "
                     "Adam and early stopping.",
        },
    }

    if HAS_XGB:
        models["XGBoost"] = {
            "estimator": XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=rs, n_jobs=-1, tree_method="hist"),
            "family": "Boosting",
            "about": "Regularised gradient boosting — usually the strongest "
                     "single model on tabular data.",
        }
        # keep the roster at exactly ten by dropping the weakest overlapping model
        models.pop("Linear Discriminant Analysis")

    return models


def make_pipeline(estimator, numeric_cols, categorical_cols) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", estimator),
    ])
