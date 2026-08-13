"""Train a CNN for the imaging tasks (brain MRI, COVID chest X-ray).

    python -m src.train_image brain_tumor
    python -m src.train_image covid_xray --epochs 25

Transfer learning from ImageNet: a frozen MobileNetV2 backbone is trained for a
few epochs, then the top of the backbone is unfrozen and fine-tuned at a low
learning rate. On a few thousand medical images that reliably beats training a
CNN from scratch, because the early filters (edges, texture) transfer even
though ImageNet contains no MRIs.

Requires TensorFlow. If TensorFlow and mediapipe are installed side by side they
will fight over protobuf versions — use a separate environment, or just run the
Kaggle notebooks in notebooks/, which is what the GPU-trained models come from.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import config


def build_datasets(task: dict, batch_size: int, seed: int):
    import tensorflow as tf

    root = task["dir"]
    train_dir = root / task["train_subdir"]
    test_dir = root / task["test_subdir"]
    if not train_dir.exists():
        raise FileNotFoundError(
            f"{train_dir} not found. Run `python -m src.download_data --images`.")

    size = (task["img_size"], task["img_size"])
    common = dict(image_size=size, batch_size=batch_size, label_mode="int",
                  class_names=task["classes"])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=seed, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=seed, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, **common)

    autotune = tf.data.AUTOTUNE
    return (train_ds.cache().shuffle(1000).prefetch(autotune),
            val_ds.cache().prefetch(autotune),
            test_ds.cache().prefetch(autotune))


def build_model(num_classes: int, img_size: int):
    import tensorflow as tf

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augment")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=list(config.IMAGE_TASKS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(
            f"TensorFlow is unavailable ({exc.__class__.__name__}: {exc}).\n"
            "Use the Kaggle notebooks in notebooks/ instead, or install "
            "TensorFlow in a clean environment.")

    task = config.IMAGE_TASKS[args.task]
    tf.random.set_seed(config.RANDOM_STATE)
    print(f"{task['name']}  ({len(task['classes'])} classes)")
    print("GPU:", tf.config.list_physical_devices("GPU") or "none (CPU — slow)")

    train_ds, val_ds, test_ds = build_datasets(task, args.batch_size,
                                               config.RANDOM_STATE)
    model, base = build_model(len(task["classes"]), task["img_size"])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
    ]

    print("\nStage 1 — frozen backbone")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                   callbacks=callbacks)

    print("\nStage 2 — fine-tuning the top of the backbone")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                   callbacks=callbacks)

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}   loss: {loss:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    names = task["pretty_classes"]
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    out = config.IMAGE_MODEL_PATH / f"{args.task}.keras"
    model.save(out)
    print(f"saved -> {out.relative_to(config.ROOT)}")

    (config.REPORTS_DIR / f"{args.task}_metrics.json").write_text(json.dumps({
        "task": args.task,
        "classes": names,
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=names, output_dict=True),
        "history": {"frozen": {k: [float(v) for v in vals] for k, vals in h1.history.items()},
                    "finetune": {k: [float(v) for v in vals] for k, vals in h2.history.items()}},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
