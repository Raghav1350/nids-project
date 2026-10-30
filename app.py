"""Streamlit dashboard for the AI-powered NIDS.

Run:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.detect import AnomalyDetector, EnsembleDetector, NIDSDetector
from src.utils import load_config

st.set_page_config(page_title="AI Network Intrusion Detection",
                   page_icon="🛡️", layout="wide")


@st.cache_resource
def get_supervised(model_path: str, threshold: float):
    if not Path(model_path).exists():
        return None
    return NIDSDetector(model_path, threshold)


@st.cache_resource
def get_anomaly(model_path: str):
    if not Path(model_path).exists():
        return None
    return AnomalyDetector(model_path)


@st.cache_resource
def get_ensemble(sup_path: str, ano_path: str,
                 sup_threshold: float, rule: str):
    if not (Path(sup_path).exists() and Path(ano_path).exists()):
        return None
    return EnsembleDetector(sup_path, ano_path, sup_threshold, None, rule)


def _load_input():
    uploaded = st.file_uploader("Upload flow records (CSV)", type=["csv"])
    use_demo = st.button("Or try demo data (synthetic)")
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if use_demo:
        from src.data_loader import generate_synthetic
        return generate_synthetic(2000, seed=7).drop(
            columns=["label", "difficulty"])
    return None


def _alert_table(alerts: pd.DataFrame, sort_col: str):
    if alerts.empty:
        st.success("No alerts - network looks clean. ✅")
    else:
        st.dataframe(alerts.sort_values(sort_col, ascending=False),
                     use_container_width=True)
        st.download_button("Download alerts CSV",
                           alerts.to_csv(index=False),
                           "alerts.csv", "text/csv")


def main() -> None:
    cfg = load_config("config.yaml")
    ecfg = cfg.get("ensemble", {})
    st.title("🛡️ AI-Powered Network Intrusion Detection")
    st.caption("Supervised RandomForest + IsolationForest anomaly ensemble "
               "over NSL-KDD-style flow features")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Detection mode",
                        ["Supervised", "Anomaly", "Ensemble"],
                        help="Supervised: known attack patterns (high precision). "
                             "Anomaly: flags whatever doesn't look like normal "
                             "traffic (catches novel/stealthy activity). "
                             "Ensemble: both, combined.")
        st.divider()

    if mode == "Supervised":
        with st.sidebar:
            model_path = st.text_input("Model file", "models/nids_rf.joblib")
            threshold = st.slider("Alert threshold P(attack)", 0.0, 1.0,
                                  float(cfg["detection"]["alert_threshold"]), 0.05)
        detector = get_supervised(model_path, threshold)
        if detector is None:
            st.warning(f"Model not found at `{model_path}`. Train it first:\n"
                       "```\npython -m src.train\n```")
            return
        df = _load_input()
        if df is None:
            st.info("Upload a CSV or generate demo data to begin.")
            return
        with st.spinner("Scoring flows..."):
            scored = detector.score(df)
        n_alerts = int(scored["alert"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Flows scored", f"{len(scored):,}")
        c2.metric("Alerts raised", f"{n_alerts:,}")
        c3.metric("Alert rate", f"{100 * n_alerts / max(len(scored), 1):.1f}%")
        tab1, tab2 = st.tabs(["🚨 Alerts", "📊 All flows"])
        with tab1:
            _alert_table(scored[scored["alert"]], "attack_probability")
        with tab2:
            st.dataframe(scored, use_container_width=True)
        st.subheader("Attack probability distribution")
        st.bar_chart(scored["attack_probability"].value_counts(bins=20, sort=False))

    elif mode == "Anomaly":
        with st.sidebar:
            model_path = st.text_input("Model file", "models/nids_iforest.joblib")
            st.caption("Threshold is the calibrated value saved in the bundle "
                       "(~5% false alarms on train-normal traffic).")
        detector = get_anomaly(model_path)
        if detector is None:
            st.warning(f"Model not found at `{model_path}`. Train it first:\n"
                       "```\npython -m src.train_anomaly --config config_anomaly.yaml\n```")
            return
        df = _load_input()
        if df is None:
            st.info("Upload a CSV or generate demo data to begin.")
            return
        with st.spinner("Scoring flows..."):
            scored = detector.score(df)
        n_alerts = int(scored["anomaly_alert"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Flows scored", f"{len(scored):,}")
        c2.metric("Anomalies flagged", f"{n_alerts:,}")
        c3.metric("Flag rate", f"{100 * n_alerts / max(len(scored), 1):.1f}%")
        st.caption(f"Calibrated anomaly threshold: {detector.threshold:.4f} "
                   "(higher score = more suspicious)")
        tab1, tab2 = st.tabs(["🚨 Anomalies", "📊 All flows"])
        with tab1:
            _alert_table(scored[scored["anomaly_alert"]], "anomaly_score")
        with tab2:
            st.dataframe(scored, use_container_width=True)
        st.subheader("Anomaly score distribution")
        st.bar_chart(scored["anomaly_score"].value_counts(bins=20, sort=False))

    else:  # Ensemble
        with st.sidebar:
            sup_path = st.text_input("Supervised model",
                                     ecfg.get("supervised_model",
                                              "models/nids_rf.joblib"))
            ano_path = st.text_input("Anomaly model",
                                     ecfg.get("anomaly_model",
                                              "models/nids_iforest.joblib"))
            sup_threshold = st.slider("Supervised threshold P(attack)",
                                      0.0, 1.0,
                                      float(ecfg.get("supervised_threshold", 0.5)),
                                      0.05)
            rule = st.selectbox("Combination rule",
                                ["either", "both", "supervised", "anomaly"],
                                help="either: alert if any model fires (recommended). "
                                     "both: alert only on agreement.")
        detector = get_ensemble(sup_path, ano_path, sup_threshold, rule)
        if detector is None:
            st.warning("One of the model files was not found. Train both:\n"
                       "```\npython -m src.train\n"
                       "python -m src.train_anomaly --config config_anomaly.yaml\n```")
            return
        df = _load_input()
        if df is None:
            st.info("Upload a CSV or generate demo data to begin.")
            return
        with st.spinner("Scoring flows with both models..."):
            scored = detector.score(df)
        alerts = scored[scored["alert"]]
        n_sup = int(alerts["supervised_alert"].sum())
        n_ano = int(alerts["anomaly_alert"].sum())
        n_both = int(((alerts["alert_source"] == "both")).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flows scored", f"{len(scored):,}")
        c2.metric("Ensemble alerts", f"{len(alerts):,}")
        c3.metric("Supervised hits", f"{n_sup:,}")
        c4.metric("Anomaly hits", f"{n_ano:,}")
        st.caption(f"{n_both} flows flagged by **both** models "
                   f"(rule: {rule}) · anomaly threshold "
                   f"{detector.anomaly_threshold:.4f}")
        tab1, tab2 = st.tabs(["🚨 Alerts", "📊 All flows"])
        with tab1:
            _alert_table(alerts, "ensemble_score")
        with tab2:
            st.dataframe(scored, use_container_width=True)
        st.subheader("Ensemble score distribution")
        st.bar_chart(scored["ensemble_score"].value_counts(bins=20, sort=False))


if __name__ == "__main__":
    main()
