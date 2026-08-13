"""Rebuild every dataset from its original public source.

    python -m src.download_data              # tabular only (a few hundred KB)
    python -m src.download_data --images     # also the brain MRI set (~91 MB)
    python -m src.download_data --covid      # also the COVID X-rays (large, see below)

The tabular CSVs are small enough to be committed, so this is mainly here for
reproducibility. The image folders are git-ignored and must be fetched.
"""
from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

UA = {"User-Agent": "Mozilla/5.0 (medical-disease-detection dataset fetcher)"}

PIMA_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "pima-indians-diabetes.data.csv")
CLEVELAND_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                 "heart-disease/processed.cleveland.data")
PARKINSONS_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                  "parkinsons/parkinsons.data")
ILPD_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00225/"
            "Indian%20Liver%20Patient%20Dataset%20(ILPD).csv")
CKD_URL = "https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip"
BRAIN_TUMOR_URL = ("https://codeload.github.com/SartajBhuvaji/"
                   "Brain-Tumor-Classification-DataSet/zip/refs/heads/master")
COVID_URL = "https://codeload.github.com/education454/datasets/zip/refs/heads/master"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _save(df: pd.DataFrame, name: str) -> None:
    config.TABULAR_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABULAR_DIR / name
    df.to_csv(path, index=False)
    print(f"  {name:<20} {df.shape[0]:>4} rows x {df.shape[1]:>2} cols")


def tabular() -> None:
    print("Tabular datasets")

    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(io.BytesIO(_fetch(PIMA_URL)), header=None, names=cols)
    _save(df, "diabetes.csv")

    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df = pd.read_csv(io.BytesIO(_fetch(CLEVELAND_URL)), header=None, names=cols,
                     na_values="?")
    df["target"] = (df.pop("num") > 0).astype(int)
    _save(df, "heart.csv")

    df = pd.read_csv(io.BytesIO(_fetch(PARKINSONS_URL)))
    df = df.drop(columns=["name"]).rename(columns={"status": "target"})
    _save(df, "parkinsons.csv")

    cols = ["Age", "Gender", "TotalBilirubin", "DirectBilirubin", "Alkphos",
            "SgptAlamine", "SgotAspartate", "TotalProteins", "Albumin",
            "AGRatio", "Selector"]
    df = pd.read_csv(io.BytesIO(_fetch(ILPD_URL)), header=None, names=cols)
    # UCI encodes 1 = liver patient, 2 = healthy -> flip to the usual 1 = disease
    df["target"] = (df.pop("Selector") == 1).astype(int)
    _save(df, "liver.csv")

    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer(as_frame=True)
    df = bc.frame.rename(columns=lambda c: c.replace(" ", "_"))
    # sklearn ships 0 = malignant, 1 = benign -> flip so 1 is always "has disease"
    df["target"] = 1 - df["target"]
    _save(df, "breast_cancer.csv")

    _kidney()


def _kidney() -> None:
    """UCI ships CKD as an ARFF inside a RAR inside a ZIP."""
    tmp = config.DATA_DIR / "_ckd_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(_fetch(CKD_URL))).extractall(tmp)

    rar = next(tmp.rglob("*.rar"), None)
    arff = next(tmp.rglob("*full.arff"), None)
    if arff is None and rar is not None:
        for exe in (r"C:\Program Files\WinRAR\UnRAR.exe",
                    r"C:\Program Files\7-Zip\7z.exe", "unrar", "7z"):
            if shutil.which(exe) or Path(exe).exists():
                import subprocess
                cmd = ([exe, "x", "-o+", str(rar), str(tmp)] if "UnRAR" in exe or exe == "unrar"
                       else [exe, "x", f"-o{tmp}", str(rar), "-y"])
                subprocess.run(cmd, capture_output=True)
                arff = next(tmp.rglob("*full.arff"), None)
                if arff:
                    break

    if arff is None:
        print("  kidney.csv           SKIPPED — could not unpack the RAR. "
              "Install WinRAR/7-Zip, or extract chronic_kidney_disease_full.arff "
              f"into {tmp} by hand and re-run.")
        return

    names, rows, in_data = [], [], False
    for line in arff.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            names.append(s.split()[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            parts = [p.strip().strip("'\"") for p in s.rstrip(",").split(",")]
            if len(parts) == len(names):
                rows.append(parts)

    df = pd.DataFrame(rows, columns=names).replace({"?": None, "": None})
    df["target"] = (df.pop(names[-1]).str.strip().str.lower() == "ckd").astype(int)
    for c in df.columns:
        if c == "target":
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        # treat a column as numeric only when nearly every present value parsed
        if conv.notna().sum() >= df[c].notna().sum() * 0.9:
            df[c] = conv
        else:
            df[c] = df[c].str.strip()
    _save(df, "kidney.csv")
    shutil.rmtree(tmp, ignore_errors=True)


def brain_tumor() -> None:
    dest = config.IMAGES_DIR / "brain_tumor"
    if (dest / "Training").exists():
        print(f"\nBrain MRI: already present at {dest}")
        return
    print("\nBrain MRI: downloading ~91 MB ...")
    z = zipfile.ZipFile(io.BytesIO(_fetch(BRAIN_TUMOR_URL)))
    tmp = config.IMAGES_DIR / "_bt_tmp"
    z.extractall(tmp)
    root = next(tmp.iterdir())
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("Training", "Testing"):
        shutil.move(str(root / split), str(dest / split))
    shutil.rmtree(tmp, ignore_errors=True)
    n = sum(1 for _ in dest.rglob("*.jpg"))
    print(f"  {n} images -> {dest}")


def covid_xray(max_side: int = 256) -> None:
    """Large download (>1 GB) — images are downscaled on the way in."""
    dest = config.IMAGES_DIR / "covid_xray"
    if (dest / "train").exists():
        print(f"\nCOVID X-rays: already present at {dest}")
        return
    print("\nCOVID X-rays: downloading (>1 GB, this takes a while) ...")
    from PIL import Image

    payload = _fetch(COVID_URL)
    print(f"  {len(payload) / 1e6:.0f} MB downloaded, resizing to <= {max_side}px ...")
    count = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) < 5 or parts[1] != "Data":
                continue
            split, cls, fname = parts[2], parts[3], parts[4]
            if split not in ("train", "test") or not fname.lower().endswith(
                    (".jpg", ".jpeg", ".png")):
                continue
            out_dir = dest / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                with z.open(name) as fh:
                    img = Image.open(io.BytesIO(fh.read())).convert("L")
                img.thumbnail((max_side, max_side), Image.LANCZOS)
                img.save(out_dir / (Path(fname).stem + ".jpg"), "JPEG", quality=88)
                count += 1
            except Exception as exc:
                print(f"    skipped {fname}: {exc}")
    print(f"  {count} images -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", action="store_true", help="also fetch brain MRI")
    ap.add_argument("--covid", action="store_true", help="also fetch COVID X-rays")
    args = ap.parse_args()

    tabular()
    if args.images:
        brain_tumor()
    if args.covid:
        covid_xray()

    print("\nNext:  python -m src.train_tabular")


if __name__ == "__main__":
    main()
