"""Inference engine: score flow records and raise alerts.

Batch mode (CSV of flow records -> alerts CSV):
    python -m src.detect --input flows.csv --output alerts.csv

Ensemble mode (supervised RandomForest + IsolationForest anomaly):
    python -m src.detect --mode ensemble --input flows.csv --output alerts.csv

Single-flow JSON via stdin:
    echo '{"duration": 0, "protocol_type": "tcp", ...}' | python -m src.detect --stdin

Live demo capture (requires scapy + root, approximate features):
    sudo python -m src.detect --live --iface eth0 --seconds 60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import COLUMN_NAMES
from src.utils import load_config


class _BaseScorer:
    """Shared bundle loading + feature alignment for all detectors."""

    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.label_col = bundle["label_col"]
        # Every column the model was trained on (minus the label).
        self.expected_cols = [c for c in COLUMN_NAMES if c != "label"]
        # Categorical vs numeric split (saved at train time) so missing
        # columns can be filled with a type-appropriate placeholder.
        # Falls back to inspecting the fitted preprocessor for old bundles.
        self.categorical = bundle.get("categorical") or []
        self.numeric = bundle.get("numeric") or []
        if not self.categorical or not self.numeric:
            try:
                trans = {
                    name: list(cols)
                    for name, _, cols in
                    self.pipeline.named_steps["preprocess"].transformers_
                }
                self.categorical = self.categorical or trans.get("cat", [])
                self.numeric = self.numeric or trans.get("num", [])
            except Exception:
                pass
        if not self.numeric:
            self.numeric = [c for c in self.expected_cols
                            if c not in self.categorical]

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all expected feature columns exist.

        Missing categoricals get a "missing" placeholder (the encoder
        ignores unknown categories); missing numerics get 0.
        """
        df = df.copy()
        for col in self.categorical:
            if col not in df.columns:
                df[col] = "missing"
        for col in self.numeric:
            if col not in df.columns:
                df[col] = 0
        ordered = [c for c in self.categorical + self.numeric
                   if c in df.columns]
        return df[ordered]


class NIDSDetector(_BaseScorer):
    """Supervised detector: loads a trained classifier bundle, scores flows."""

    def __init__(self, model_path: str, threshold: float = 0.70):
        super().__init__(model_path)
        self.threshold = threshold

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the input rows annotated with prediction + attack probability."""
        X = self._align(df)
        proba = self.pipeline.predict_proba(X)
        # Positive class = attack (1) for binary; else max proba.
        if proba.shape[1] == 2:
            attack_proba = proba[:, 1]
        else:
            attack_proba = proba.max(axis=1)
        out = df.copy()
        out["prediction"] = self.pipeline.predict(X)
        out["attack_probability"] = attack_proba
        out["alert"] = attack_proba >= self.threshold
        return out

    def alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = self.score(df)
        return scored[scored["alert"]].sort_values(
            "attack_probability", ascending=False)


class AnomalyDetector(_BaseScorer):
    """One-class detector: IsolationForest trained on normal traffic only.

    Anomaly score is ``-decision_function`` so higher = more suspicious,
    matching the convention in train_anomaly.py.
    """

    def __init__(self, model_path: str, threshold: float | None = None):
        super().__init__(model_path)
        if threshold is None:
            threshold = self.bundle.get("anomaly_threshold")
        if threshold is None:
            raise ValueError(
                "No anomaly threshold given and none saved in the bundle.")
        self.threshold = float(threshold)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._align(df)
        scores = -self.pipeline.decision_function(X)
        out = df.copy()
        out["anomaly_score"] = scores
        out["anomaly_alert"] = scores >= self.threshold
        return out

    def alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = self.score(df)
        return scored[scored["anomaly_alert"]].sort_values(
            "anomaly_score", ascending=False)


class EnsembleDetector:
    """Supervised + anomaly ensemble.

    The supervised model catches known attack patterns with high precision;
    the anomaly detector acts as a second opinion for novel / stealthy
    activity (the R2L/U2R blind spot). ``rule`` controls how the two votes
    combine: "either" (default), "both", "supervised" or "anomaly".

    ``ensemble_score`` ranks the combined alert list: max(sup_proba,
    sigmoid(anomaly_score - threshold)). It is a documented ranking
    heuristic, not a probability.
    """

    RULES = ("either", "both", "supervised", "anomaly")

    def __init__(self,
                 supervised_path: str,
                 anomaly_path: str,
                 supervised_threshold: float = 0.50,
                 anomaly_threshold: float | None = None,
                 rule: str = "either"):
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}")
        self.supervised = NIDSDetector(supervised_path, supervised_threshold)
        self.anomaly = AnomalyDetector(anomaly_path, anomaly_threshold)
        self.rule = rule
        self.supervised_threshold = supervised_threshold
        self.anomaly_threshold = self.anomaly.threshold

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        sup = self.supervised.score(df)
        ano = self.anomaly.score(df)
        out = df.copy()
        out["attack_probability"] = sup["attack_probability"]
        out["prediction"] = sup["prediction"]
        out["anomaly_score"] = ano["anomaly_score"]
        out["supervised_alert"] = sup["alert"]
        out["anomaly_alert"] = ano["anomaly_alert"]

        if self.rule == "either":
            out["alert"] = out["supervised_alert"] | out["anomaly_alert"]
        elif self.rule == "both":
            out["alert"] = out["supervised_alert"] & out["anomaly_alert"]
        elif self.rule == "supervised":
            out["alert"] = out["supervised_alert"]
        else:  # anomaly
            out["alert"] = out["anomaly_alert"]

        out["alert_source"] = "none"
        out.loc[out["supervised_alert"] & ~out["anomaly_alert"],
                "alert_source"] = "supervised"
        out.loc[~out["supervised_alert"] & out["anomaly_alert"],
                "alert_source"] = "anomaly"
        out.loc[out["supervised_alert"] & out["anomaly_alert"],
                "alert_source"] = "both"

        # Ranking heuristic: sigmoid centers the anomaly score at its
        # calibrated threshold (-> 0.5), saturating toward 1 above it.
        ano_conf = 1.0 / (1.0 + np.exp(-(out["anomaly_score"].to_numpy()
                                         - self.anomaly_threshold)))
        out["ensemble_score"] = np.maximum(
            out["attack_probability"].to_numpy(), ano_conf)
        return out

    def alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = self.score(df)
        return scored[scored["alert"]].sort_values(
            "ensemble_score", ascending=False)


def live_capture(iface: str | None, seconds: int, packet_limit: int = 2000):
    """Rough live demo: aggregate packets into flows, score them.

    NOTE: only a handful of NSL-KDD features can be derived from raw
    packets (duration, protocol, bytes, counts...); the rest are zeroed.
    Treat this as a demo, not production-grade feature extraction.
    """
    try:
        from scapy.all import IP, TCP, UDP, sniff
    except ImportError:
        sys.exit("Live mode needs scapy: pip install scapy (and run as root).")

    flows: dict[tuple, dict] = defaultdict(
        lambda: {"packets": 0, "bytes": 0, "start": None, "end": None,
                 "proto": "other"})
    import time

    def handle(pkt):
        if not pkt.haslayer(IP):
            return
        ip = pkt[IP]
        proto = "tcp" if pkt.haslayer(TCP) else "udp" if pkt.haslayer(UDP) else "other"
        key = (ip.src, ip.dst, proto)
        f = flows[key]
        now = time.time()
        f["packets"] += 1
        f["bytes"] += len(pkt)
        f["proto"] = proto
        f["start"] = now if f["start"] is None else f["start"]
        f["end"] = now

    print(f"Capturing on {iface or 'default interface'} for {seconds}s ...")
    sniff(iface=iface, prn=handle, timeout=seconds,
          stop_filter=lambda _: sum(f["packets"] for f in flows.values()) >= packet_limit)

    rows = []
    for (src, dst, proto), f in flows.items():
        rows.append({
            "duration": (f["end"] or 0) - (f["start"] or 0),
            "protocol_type": proto if proto in ("tcp", "udp") else "tcp",
            "service": "other",
            "flag": "SF",
            "src_bytes": f["bytes"],
            "dst_bytes": 0,
            "count": f["packets"],
            "srv_count": f["packets"],
        })
    df = pd.DataFrame(rows)
    print(f"Aggregated {len(df)} flows.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None,
                    help="Supervised model bundle (default from config).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Supervised P(attack) alert threshold.")
    ap.add_argument("--mode", default="supervised",
                    choices=["supervised", "anomaly", "ensemble"],
                    help="Which detector to run.")
    ap.add_argument("--rule", default=None,
                    choices=["either", "both", "supervised", "anomaly"],
                    help="Ensemble combination rule (default from config).")
    ap.add_argument("--anomaly-model", default=None,
                    help="Anomaly model bundle (default from config).")
    ap.add_argument("--anomaly-threshold", type=float, default=None,
                    help="Override the calibrated anomaly threshold.")
    ap.add_argument("--input", help="CSV of flow records to score")
    ap.add_argument("--output", default="alerts.csv")
    ap.add_argument("--stdin", action="store_true",
                    help="Read one flow as JSON from stdin")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--iface", default=None)
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg.get("ensemble", {})

    if args.live:
        df = live_capture(args.iface, args.seconds)
    elif args.stdin:
        df = pd.DataFrame([json.loads(sys.stdin.read())])
    elif args.input:
        df = pd.read_csv(args.input)
    else:
        ap.error("Provide --input, --stdin, or --live.")

    if args.mode == "ensemble":
        sup_path = args.model or ecfg.get("supervised_model")
        ano_path = args.anomaly_model or ecfg.get("anomaly_model")
        for p, name in ((sup_path, "supervised"), (ano_path, "anomaly")):
            if not p or not Path(p).exists():
                sys.exit(f"{name} model not found at {p}. "
                         f"Train first: python -m src.train "
                         f"and python -m src.train_anomaly --config config_anomaly.yaml")
        detector = EnsembleDetector(
            sup_path, ano_path,
            supervised_threshold=(args.threshold
                                  if args.threshold is not None
                                  else ecfg.get("supervised_threshold", 0.50)),
            anomaly_threshold=(args.anomaly_threshold
                               if args.anomaly_threshold is not None
                               else ecfg.get("anomaly_threshold")),
            rule=args.rule or ecfg.get("rule", "either"),
        )
        scored = detector.score(df)
        alert_df = scored[scored["alert"]]
        n_alerts = len(alert_df)
        n_sup = int(alert_df["supervised_alert"].sum())
        n_ano = int(alert_df["anomaly_alert"].sum())
        print(f"Scored {len(scored)} flows -> {n_alerts} ensemble alerts "
              f"({n_sup} supervised, {n_ano} anomaly, rule={detector.rule}).")
        show_cols = ["ensemble_score", "alert_source", "attack_probability",
                     "anomaly_score"] + [c for c in
                     ("protocol_type", "service", "flag")
                     if c in alert_df.columns]
    elif args.mode == "anomaly":
        ano_path = args.anomaly_model or ecfg.get("anomaly_model")
        if not ano_path or not Path(ano_path).exists():
            sys.exit(f"Anomaly model not found at {ano_path}. Train first: "
                     f"python -m src.train_anomaly --config config_anomaly.yaml")
        detector = AnomalyDetector(
            ano_path,
            args.anomaly_threshold if args.anomaly_threshold is not None
            else ecfg.get("anomaly_threshold"))
        scored = detector.score(df)
        alert_df = scored[scored["anomaly_alert"]]
        n_alerts = len(alert_df)
        print(f"Scored {len(scored)} flows -> {n_alerts} anomaly alerts "
              f"(threshold={detector.threshold:.4f}).")
        show_cols = ["anomaly_score"] + [c for c in
                     ("protocol_type", "service", "flag", "src_bytes", "dst_bytes")
                     if c in alert_df.columns]
    else:
        model_path = args.model or cfg["paths"]["model_file"]
        threshold = args.threshold or cfg["detection"]["alert_threshold"]
        if not Path(model_path).exists():
            sys.exit(f"Model not found at {model_path}. Train first: python -m src.train")
        detector = NIDSDetector(model_path, threshold)
        scored = detector.score(df)
        alert_df = scored[scored["alert"]]
        n_alerts = len(alert_df)
        print(f"Scored {len(scored)} flows -> {n_alerts} alerts "
              f"(threshold={threshold}).")
        show_cols = ["attack_probability"] + [c for c in
                     ("protocol_type", "service", "flag", "src_bytes", "dst_bytes")
                     if c in alert_df.columns]

    if n_alerts:
        print(alert_df[show_cols].head(20).to_string(index=False))
        alert_df.to_csv(args.output, index=False)
        print(f"Alerts written to {args.output}")
    else:
        print("No alerts - all clear.")


if __name__ == "__main__":
    main()
