"""Train the autoencoder anomaly detector and persist it to disk.

An MLP autoencoder (bottleneck) trained to reconstruct NORMAL traffic only.
Anomaly score = per-flow reconstruction MSE; higher = more suspicious.
The bundle uses the same format as the IsolationForest model, so it is a
drop-in replacement in detect.py / app.py via --anomaly-model.

Usage:
    python -m src.train_autoencoder [--config config_autoencoder.yaml] [--synthetic]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.anomaly_models import AutoencoderAnomaly
from src.data_loader import (
    binarize_labels,
    generate_synthetic,
    load_datasets,
)
from src.features import build_preprocessor, feature_columns
from src.train_anomaly import evaluate_bundle
from src.utils import ensure_dirs, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_autoencoder.yaml")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.synthetic:
        seed = cfg["data"].get("random_state", 42)
        train_df = generate_synthetic(cfg["data"].get("synthetic_n_train", 20000),
                                      seed=seed)
        train_df = binarize_labels(train_df)
        train_df["label"] = train_df["label"].map({0: "normal", 1: "unknown"})
        test_df = None
    else:
        cfg["features"]["binary_labels"] = False
        train_df, test_df = load_datasets(cfg)

    label_col = cfg["features"]["label_col"]
    categorical, numeric = feature_columns(train_df, cfg)
    X_train = train_df.drop(columns=[label_col])
    X_normal = X_train[train_df[label_col].to_numpy() == "normal"]
    print(f"Train flows: {len(X_train):,} ({len(X_normal):,} normal)")

    mcfg = cfg["model"]
    sub = mcfg.get("train_subsample", 20000)
    if len(X_normal) > sub:
        X_fit = X_normal.sample(sub, random_state=mcfg.get("random_state", 42))
        print(f"Subsampling {sub:,} normal flows for MLP training")
    else:
        X_fit = X_normal

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(categorical, numeric)),
            ("detector", AutoencoderAnomaly(
                hidden_layer_sizes=mcfg.get("hidden_layer_sizes", [64, 32, 64]),
                activation=mcfg.get("activation", "relu"),
                solver=mcfg.get("solver", "adam"),
                max_iter=mcfg.get("max_iter", 150),
                early_stopping=mcfg.get("early_stopping", True),
                validation_fraction=mcfg.get("validation_fraction", 0.1),
                n_iter_no_change=mcfg.get("n_iter_no_change", 10),
                random_state=mcfg.get("random_state", 42),
            )),
        ]
    )

    print("Training MLP autoencoder on normal traffic ...")
    t0 = time.time()
    pipeline.fit(X_fit)
    print(f"Trained in {time.time() - t0:.1f}s "
          f"(iterations used: {pipeline.named_steps['detector'].mlp_.n_iter_})")

    # Calibrate like the IsolationForest: threshold at the calibration_fpr
    # quantile of normal-train reconstruction errors.
    train_scores = -pipeline.decision_function(X_normal)
    fpr_target = mcfg.get("calibration_fpr", 0.05)
    threshold = float(np.quantile(train_scores, 1.0 - fpr_target))
    print(f"Calibrated threshold: {threshold:.4f} "
          f"(~{fpr_target:.0%} FPR on train normal traffic)")

    bundle = {
        "pipeline": pipeline,
        "categorical": categorical,
        "numeric": numeric,
        "label_col": label_col,
        "anomaly": True,
        "anomaly_threshold": threshold,
        "config": cfg,
    }
    model_path = Path(cfg["paths"]["model_file"])
    joblib.dump(bundle, model_path)
    print(f"Model saved to {model_path}")

    if test_df is not None:
        evaluate_bundle(bundle, test_df, cfg, threshold)


if __name__ == "__main__":
    main()
