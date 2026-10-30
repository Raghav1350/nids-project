"""Per-alert explanations using SHAP (TreeExplainer on the supervised RF).

For each alerted flow, shows the top features pushing the model toward
"attack" (+) or toward "normal" (-), with the flow's actual values.

Usage:
    python -m src.explain --input alerts.csv [--top-n 5] [--output explanations.csv]
    python -m src.explain --mode ensemble --input alerts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.detect import AnomalyDetector, EnsembleDetector, NIDSDetector
from src.utils import load_config


def _clean(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def _raw_display(orig_name: str, row: pd.Series,
                 categorical: list[str]) -> str:
    """Human-readable raw value for a transformed feature name."""
    if orig_name.startswith("cat__"):
        rest = orig_name[len("cat__"):]
        for col in categorical:
            if rest.startswith(col + "_"):
                return f"{col}={row[col]}"
        return rest
    col = orig_name.replace("num__", "")
    try:
        return f"{float(row[col]):.4g}"
    except (KeyError, ValueError, TypeError):
        return "?"


def explain_supervised(detector: NIDSDetector, df: pd.DataFrame,
                       top_n: int) -> pd.DataFrame:
    import shap

    X = detector._align(df)
    pre = detector.pipeline.named_steps["preprocess"]
    clf = detector.pipeline.named_steps["classifier"]
    Xt = pre.transform(X)
    orig_names = list(pre.get_feature_names_out())
    feature_names = [_clean(n) for n in orig_names]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    # Binary RF: shap_values may be (n, f) or (n, f, 2); take class-1.
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    rows = []
    for i in range(len(df)):
        vals = sv[i]
        order = sorted(range(len(vals)), key=lambda j: -abs(vals[j]))[:top_n]
        entry = {"row": i}
        for rank, j in enumerate(order, 1):
            entry[f"feature_{rank}"] = feature_names[j]
            entry[f"shap_{rank}"] = round(float(vals[j]), 4)
            entry[f"value_{rank}"] = _raw_display(
                orig_names[j], X.iloc[i], detector.categorical)
            entry[f"direction_{rank}"] = (
                "toward ATTACK" if vals[j] > 0 else "toward normal")
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV of flows (or alerts).")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="supervised",
                    choices=["supervised", "anomaly", "ensemble"])
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--output", default="explanations.csv")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.input)
    # Only explain rows the detector actually alerted on, if flagged.
    work = df[df["alert"]] if "alert" in df.columns else df
    if work.empty:
        print("No alerts to explain.")
        return

    if args.mode != "supervised":
        print("SHAP explanations are currently supported for the supervised "
              "model only (TreeExplainer needs the RF).")
        return

    model_path = args.model or cfg["paths"]["model_file"]
    if not Path(model_path).exists():
        raise SystemExit(f"Model not found at {model_path}")
    detector = NIDSDetector(model_path,
                            args.threshold if hasattr(args, "threshold") else 0.5)

    print(f"Explaining {len(work)} alerts with SHAP (top-{args.top_n}) ...")
    exp = explain_supervised(detector, work, args.top_n)
    exp.to_csv(args.output, index=False)

    for _, r in exp.head(10).iterrows():
        print(f"\nAlert row {int(r['row'])}:")
        for k in range(1, args.top_n + 1):
            print(f"  {r[f'feature_{k}']:<28} shap={r[f'shap_{k}']:+.3f} "
                  f"({r[f'value_{k}']}) -> {r[f'direction_{k}']}")
    print(f"\nFull explanations written to {args.output}")


if __name__ == "__main__":
    main()
