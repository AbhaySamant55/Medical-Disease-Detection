"""Check that the Kaggle-trained image models are in the right place and load.

    python -m src.check_image_models

Reports which of the expected files are present, and — if TensorFlow is
available — actually loads each backbone and head bundle and runs one dummy
scan through the full ensemble, so a corrupt or half-downloaded file is caught
here rather than in the middle of a demo.
"""
from __future__ import annotations

import sys

import numpy as np

from . import config
from .image_ensemble import backbone_path, heads_path, is_available

EXPECTED_METRICS = config.REPORTS_DIR / "image_ensemble_metrics.json"


def main() -> None:
    print(f"Looking in {config.IMAGE_MODEL_PATH}\n")

    any_present = False
    for task_key, task in config.IMAGE_TASKS.items():
        bp, hp = backbone_path(task_key), heads_path(task_key)
        mark = "OK " if is_available(task_key) else "-- "
        print(f"{mark}{task['name']}")
        for path in (bp, hp):
            if path.exists():
                any_present = True
                print(f"     found   {path.name:<34} "
                      f"{path.stat().st_size / 1e6:>7.1f} MB")
            else:
                print(f"     MISSING {path.name}")
        print()

    print(f"{'found  ' if EXPECTED_METRICS.exists() else 'MISSING'} "
          f"reports/{EXPECTED_METRICS.name}\n")

    if not any_present:
        print("Nothing downloaded yet. Run notebooks/kaggle_image_10_models.ipynb "
              "on Kaggle, then copy the outputs here.")
        return

    # ------------------------------------------------- try actually loading
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        print(f"TensorFlow unavailable ({exc.__class__.__name__}), so the files "
              f"cannot be load-tested from this interpreter.")
        print("Use the TensorFlow environment:")
        print(r"    .venv-tf\Scripts\python.exe -m src.check_image_models")
        return

    from .image_ensemble import ImageEnsemble

    ok = True
    for task_key, task in config.IMAGE_TASKS.items():
        if not is_available(task_key):
            continue
        print(f"Load test: {task['name']}")
        try:
            ens = ImageEnsemble.load(task_key)
            dummy = np.random.default_rng(0).integers(
                0, 255, (ens.img_size, ens.img_size, 3)).astype("float32")
            probs, detail = ens.analyse(dummy)
            print(f"   {len(ens.members)} algorithms, classes {ens.classes}")
            print(f"   embedding -> probabilities {np.round(probs, 3).tolist()}")
            print(f"   weights sum to {sum(s['weight'] for s in ens.scores.values()):.3f}")
            print("   OK\n")
        except Exception as exc:
            ok = False
            print(f"   FAILED: {exc.__class__.__name__}: {exc}\n")

    if ok:
        print("All present models load and run. Start the app with:")
        print(r"    .venv-tf\Scripts\python.exe -m streamlit run app.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
