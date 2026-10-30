"""Train the anomaly-based intrusion detector and persist it to disk.

Trains an IsolationForest on NORMAL traffic only (one-class setup), then
calibrates the alert threshold from normal-train scores and evaluates on
the full test set -- with a per-attack-category breakdown so the R2L/U2R
gap left by the supervised models can be measured directly.

Usage:
    python -m src.train_anomaly [--config config_anomaly.yaml] [--synthetic]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.data_loader import (
    binarize_labels,
    generate_synthetic,
    load_datasets,
)
from src.features import build_preprocessor, feature_columns
from src.utils import ensure_dirs, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_anomaly.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="Force synthetic data even if NSL-KDD is present.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.synthetic:
        seed = cfg["data"].get("random_state", 42)
        train_df = generate_synthetic(cfg["data"].get("synthetic_n_train", 20000),
                                      seed=seed)
        train_df = binarize_labels(train_df).rename(
            columns={"label": "label"})  # 0=normal, 1=attack
        # Represent as category-style labels for uniform handling below.
        train_df["label"] = train_df["label"].map({0: "normal", 1: "unknown"})
        test_df = None
    else:
        # Multiclass labels (normal/dos/probe/r2l/u2r/unknown): "normal" rows
        # are identical to the binary setup; categories give the breakdown.
        cfg["features"]["binary_labels"] = False
        train_df, test_df = load_datasets(cfg)

    label_col = cfg["features"]["label_col"]
    categorical, numeric = feature_columns(train_df, cfg)
    print(f"Features: {len(categorical)} categorical, {len(numeric)} numeric")

    X_train = train_df.drop(columns=[label_col])
    X_normal = X_train[train_df[label_col].to_numpy() == "normal"]
    print(f"Train flows: {len(X_train):,} ({len(X_normal):,} normal used for fitting)")

    mcfg = cfg["model"]
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(categorical, numeric)),
            ("detector", IsolationForest(
                n_estimators=mcfg.get("n_estimators", 200),
                max_samples=mcfg.get("max_samples", 256),
                random_state=mcfg.get("random_state", 42),
                n_jobs=mcfg.get("n_jobs", -1),
            )),
        ]
    )

    print("Training IsolationForest on normal traffic ...")
    t0 = time.time()
    pipeline.fit(X_normal)
    print(f"Trained in {time.time() - t0:.1f}s")

    # Anomaly score: higher = more suspicious. (IF's decision_function is
    # higher for inliers, so negate it.)
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


def evaluate_bundle(bundle, test_df: pd.DataFrame, cfg: dict,
                    threshold: float) -> dict:
    """Score the test set; report overall metrics + per-category recall."""
    label_col = cfg["features"]["label_col"]
    cats = test_df[label_col].to_numpy()
    y_test = (cats != "normal").astype(int)

    X_test = test_df.drop(columns=[label_col])
    scores = -bundle["pipeline"].decision_function(X_test)
    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    metrics = {
        "model": "isolation_forest",
        "n_test": int(len(test_df)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "threshold": threshold,
        "calibration_fpr": cfg["model"].get("calibration_fpr", 0.05),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr_test": float((y_pred[y_test == 0] == 1).mean()),
        "tp": tp, "fp": fp, "fn": fn,
    }

    per_cat = {}
    for cat in ["normal", "dos", "probe", "r2l", "u2r", "unknown"]:
        mask = cats == cat
        if mask.sum() == 0:
            continue
        per_cat[cat] = {
            "n": int(mask.sum()),
            ("fpr" if cat == "normal" else "recall"):
                float((y_pred[mask] == 1).mean()),
        }
    metrics["per_category"] = per_cat

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Anomaly detector ({cfg['model'].get('type', 'model')}) "
          f"on NSL-KDD KDDTest+ ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | "
          f"precision: {metrics['precision']:.3f} | "
          f"recall: {metrics['recall']:.3f} | "
          f"test FPR: {metrics['fpr_test']:.3f}")
    print("\nPer-category:")
    for cat, m in per_cat.items():
        key = "FPR" if cat == "normal" else "recall"
        val = m["fpr" if cat == "normal" else "recall"]
        print(f"  {cat:>7}: n={m['n']:>6,}  {key}={val:.3f}")

    # Operating-point sweep: what does recall look like at several
    # false-alarm budgets? Thresholds are set on TEST-normal score
    # quantiles, i.e. the budget a deployer would actually choose.
    normal_scores = scores[y_test == 0]
    print("\nThreshold sweep (FPR budget on normal traffic):")
    sweep = []
    for budget in (0.01, 0.02, 0.05, 0.10, 0.20):
        thr = float(np.quantile(normal_scores, 1.0 - budget))
        yp = (scores >= thr).astype(int)
        row = {
            "fpr_budget": budget,
            "threshold": thr,
            "precision": float(yp[y_test == 1].sum() / max(yp.sum(), 1)),
            "recall": float(yp[y_test == 1].mean()),
            "r2l_recall": float(yp[cats == "r2l"].mean()) if (cats == "r2l").sum() else None,
            "u2r_recall": float(yp[cats == "u2r"].mean()) if (cats == "u2r").sum() else None,
            "probe_recall": float(yp[cats == "probe"].mean()),
            "dos_recall": float(yp[cats == "dos"].mean()),
        }
        sweep.append(row)
        print(f"  FPR<={budget:>4.0%}: thr={thr:+.4f} "
              f"P={row['precision']:.3f} R={row['recall']:.3f} | "
              f"dos={row['dos_recall']:.2f} probe={row['probe_recall']:.2f} "
              f"r2l={row['r2l_recall']:.2f} u2r={row['u2r_recall']:.2f}")
    metrics["threshold_sweep"] = sweep
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {reports_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
