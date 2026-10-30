"""Evaluate a trained model bundle on a test set and write a report.

Usage:
    python -m src.evaluate [--config config.yaml] [--synthetic]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.data_loader import binarize_labels, generate_synthetic, load_datasets
from src.utils import ensure_dirs, load_config


def evaluate_bundle(bundle: dict, test_df: pd.DataFrame, cfg: dict) -> dict:
    pipeline = bundle["pipeline"]
    label_col = bundle["label_col"]
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"
                                     if bundle["binary"] else "weighted",
                                     zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary"
                               if bundle["binary"] else "weighted",
                               zero_division=0),
        "f1": f1_score(y_test, y_pred, average="binary"
                       if bundle["binary"] else "weighted", zero_division=0),
    }
    if bundle["binary"]:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_test, pipeline.predict_proba(X_test)[:, 1])
        except Exception:
            pass

    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports_dir / "classification_report.txt").write_text(report)
    pd.DataFrame(cm).to_csv(reports_dir / "confusion_matrix.csv", index=False)

    # Optional confusion-matrix heatmap (skipped if matplotlib is missing).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        fig.savefig(reports_dir / "confusion_matrix.png")
        plt.close(fig)
    except ImportError:
        pass

    print("\n==== Evaluation ====")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print("\n" + report)
    print(f"Reports written to {reports_dir}/")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    bundle = joblib.load(cfg["paths"]["model_file"])

    if args.synthetic:
        seed = cfg["data"].get("random_state", 42)
        test_df = binarize_labels(
            generate_synthetic(cfg["data"].get("synthetic_n_test", 5000),
                               seed=seed + 1))
    else:
        _, test_df = load_datasets(cfg)
    evaluate_bundle(bundle, test_df, cfg)


if __name__ == "__main__":
    main()
