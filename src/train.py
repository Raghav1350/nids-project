"""Train the intrusion-detection model and persist it to disk.

Usage:
    python -m src.train [--config config.yaml] [--synthetic]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score

from src.data_loader import generate_synthetic, load_datasets
from src.features import build_pipeline, feature_columns, top_features
from src.utils import ensure_dirs, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="Force synthetic data even if NSL-KDD is present.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.synthetic:
        seed = cfg["data"].get("random_state", 42)
        train_df = generate_synthetic(cfg["data"].get("synthetic_n_train", 20000),
                                      seed=seed)
        # train_df labels are "normal"/"attack" strings -> binarize like NSL-KDD
        from src.data_loader import binarize_labels
        train_df = binarize_labels(train_df)
        test_df = None
    else:
        train_df, test_df = load_datasets(cfg)

    label_col = cfg["features"]["label_col"]
    categorical, numeric = feature_columns(train_df, cfg)
    print(f"Features: {len(categorical)} categorical, {len(numeric)} numeric")

    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]

    pipeline = build_pipeline(cfg, categorical, numeric)
    print("Training RandomForest ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Trained in {time.time() - t0:.1f}s")

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    print(f"Train accuracy: {train_acc:.4f}")

    print("\nTop features:")
    print(top_features(pipeline).to_string(index=False))

    bundle = {
        "pipeline": pipeline,
        "categorical": categorical,
        "numeric": numeric,
        "label_col": label_col,
        "binary": cfg["features"].get("binary_labels", True),
        "config": cfg,
    }
    model_path = Path(cfg["paths"]["model_file"])
    joblib.dump(bundle, model_path)
    print(f"\nModel saved to {model_path}")

    if test_df is not None:
        from src.evaluate import evaluate_bundle
        evaluate_bundle(bundle, test_df, cfg)


if __name__ == "__main__":
    main()
