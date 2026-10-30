# 🛡️ AI-Powered Network Intrusion Detection System

A ready-to-run machine learning pipeline that learns what normal network
traffic looks like and flags malicious flows — DoS floods, port scans,
brute-force logins, privilege escalation and more.

**How it works:** raw flow records (NSL-KDD schema, 41 features) →
preprocessing (one-hot + scaling) → RandomForest classifier → alerts with
attack probabilities.

## ✨ Features

- **End-to-end ML pipeline** — data loading, feature engineering, training,
  evaluation, all in one command
- **Real NSL-KDD dataset** support (auto-download), with a **synthetic
  fallback** so it works fully offline
- **Batch detection** — score thousands of flows from CSV, get ranked alerts
- **Live demo capture** — sniff packets and score flows in real time
  (scapy, demo-grade features)
- **Streamlit dashboard** — upload flows, visualize alerts, tune thresholds
- **Evaluation reports** — accuracy, precision/recall/F1, ROC-AUC,
  confusion matrix

## 🚀 Quickstart

```bash
cd ai-network-intrusion-detection

# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (downloads NSL-KDD automatically; use --synthetic to skip)
python -m src.train
# or fully offline:
python -m src.train --synthetic

# 3. Evaluate on the held-out test set
python -m src.evaluate

# 4. Score new flows and get alerts
python -m src.detect --input sample_flows.csv --output alerts.csv

# 5. Launch the dashboard
streamlit run app.py
```

Expected results on the real NSL-KDD benchmark (binary, KDDTest+):
**76% accuracy, 97% precision, 60% recall, 0.96 ROC-AUC.**
The test set deliberately contains attack types never seen in training,
so recall is the hard part — the model ranks attacks very well
(ROC-AUC 0.96), and lowering the alert threshold trades precision for
recall:

| threshold | precision | recall | F1   |
|-----------|-----------|--------|------|
| 0.3       | 0.966     | 0.713  | 0.82 |
| 0.5       | 0.967     | 0.602  | 0.74 |
| 0.7       | 0.967     | 0.560  | 0.71 |

On synthetic data expect ~99–100% (demo attacks are cleanly separated).

## 🎯 Multiclass (attack-type) classifier

Beyond binary normal/attack detection, the pipeline trains a 5-class model
that names the attack category — `normal` / `dos` / `probe` / `r2l` / `u2r`
(uses `config_multiclass.yaml`; reports land in `reports/multiclass/`):

```bash
python -m src.train --config config_multiclass.yaml
python -m src.detect --config config_multiclass.yaml --input flows.csv
```

Real NSL-KDD results (KDDTest+, weighted averages: **74% accuracy,
79% precision, 74% recall**):

| class  | precision | recall | support |
|--------|-----------|--------|---------|
| dos    | 0.96      | 0.76   | 7,458   |
| normal | 0.63      | 0.97   | 9,711   |
| probe  | 0.86      | 0.59   | 2,421   |
| r2l    | 0.81      | 0.00   | 2,887   |
| u2r    | 0.33      | 0.01   | 67      |

The honest read: volumetric attacks (DoS floods, probes) are caught well,
but **R2L/U2R are almost entirely missed** — they're semantically
normal-looking single connections (a stolen-password login looks like a
login), and the training set has only 995 R2L / 52 U2R examples. This is
the known hard part of the benchmark: catching stealthy, low-volume
attacks needs anomaly detection or content-level features, not just a
flow classifier.

## 🚨 Anomaly detector (for the R2L/U2R gap)

Supervised classifiers learn attack *fingerprints* — so stealthy attacks
that look like normal traffic (R2L, U2R) are nearly invisible to them
(recall ~0.00/0.01 above). The anomaly model flips the problem: an
**IsolationForest trained on normal traffic only** flags whatever
deviates from normal, no attack labels needed
(uses `config_anomaly.yaml`; reports in `reports/anomaly/`):

```bash
python -m src.train_anomaly --config config_anomaly.yaml
```

Real NSL-KDD results (KDDTest+, threshold calibrated to ~5% false alarms
on train-normal traffic): **0.94 ROC-AUC, 97% precision, 64% recall** —
and the per-category picture shows exactly where it earns its keep:

| class  | recall | n      | vs supervised |
|--------|--------|--------|---------------|
| dos    | 0.77   | 7,458  | ≈ same (0.76) |
| probe  | **0.93** | 2,421  | up from 0.59  |
| r2l    | 0.06   | 2,887  | up from 0.00, still weak |
| u2r    | 0.24   | 67     | up from 0.01  |

The threshold is again the lever — but the tradeoff is starker for
stealthy attacks. R2L/U2R scores overlap heavily with normal traffic, so
catching most of them requires accepting a much higher false-alarm
budget (the realistic use is a **ranked triage queue**, not binary
alerting):

| FPR budget | precision | recall | dos  | probe | r2l  | u2r  |
|------------|-----------|--------|------|-------|------|------|
| 1%         | 0.99      | 0.53   | 0.68 | 0.72  | 0.01 | 0.06 |
| 5%         | 0.95      | 0.68   | 0.82 | 0.98  | 0.09 | 0.28 |
| 10%        | 0.91      | 0.77   | 0.92 | 1.00  | 0.19 | 0.52 |
| 20%        | 0.86      | 0.90   | 0.94 | 1.00  | 0.69 | 0.88 |

### 🔬 Deeper option: MLP autoencoder

A bottleneck autoencoder (64→32→64) trained to *reconstruct* normal
traffic; anomaly score = reconstruction MSE. Same bundle format as the
IsolationForest, so it's a drop-in `--anomaly-model` with no code
changes (`config_autoencoder.yaml`, `src/train_autoencoder.py`,
reports in `reports/autoencoder/`):

```bash
.venv/bin/python -m src.train_autoencoder --config config_autoencoder.yaml
.venv/bin/python -m src.detect --mode ensemble \
    --anomaly-model models/nids_autoencoder.joblib --input flows.csv
```

It beats the IsolationForest on the hard categories
(KDDTest+, ~5% FPR calibration): **0.965 ROC-AUC, 92% precision,
84% recall** — dos 0.96, probe 0.98, **r2l 0.41** (vs 0.06),
**u2r 0.78** (vs 0.24). The tradeoff vs IsolationForest on a 3,000-flow
sample: ensemble recall 75%→87% at precision 97%→91%. Pick the forest
for precision, the autoencoder for recall.

One implementation note: the estimator lives in
`src/anomaly_models.py` (not in the train script) so pickled bundles
load from any process.

### 🔍 Per-alert explanations (SHAP)

`src/explain.py` uses SHAP's TreeExplainer on the supervised
RandomForest to show *why* each alert fired — the top features pushing
toward "attack", with the flow's actual values:

```bash
.venv/bin/python -m src.explain --input alerts.csv --top-n 5
# Alert row 0:
#   dst_bytes          shap=+0.135 (0) -> toward ATTACK
#   service_ecr_i      shap=+0.100 (service=ecr_i) -> toward ATTACK
#   protocol_type_icmp shap=+0.091 (protocol_type=icmp) -> toward ATTACK
```

Smurf alerts are explained by `service=ecr_i` + `protocol=icmp` + zero
return bytes; neptune by zero-byte flows with high error rates — the
model's reasoning matches the known attack mechanics.

Recommended deployment: **ensemble** — the supervised RandomForest
(97% precision on known attack patterns) plus the anomaly detector as a
second opinion / ranked review queue for novel and stealthy activity.
It is now wired in (`src/detect.py::EnsembleDetector`, `config.yaml`
`ensemble:` section, dashboard mode selector):

```bash
# CLI: supervised (default) | anomaly | ensemble
python -m src.detect --mode ensemble --input flows.csv --output alerts.csv
python -m src.detect --mode ensemble --rule both --input flows.csv  # agree-only
```

Verified on 3,000 KDDTest+ flows — ensemble vs supervised alone
(threshold 0.5): **recall 62% → 75% at 97% precision**.
The 228 anomaly-only alerts were 96% true attacks, including novel
attacks the supervised model missed (`apache2`, `mscan`,
`processtable`, `httptunnel`) and stealthy ones (`guess_passwd`,
`buffer_overflow`, `warezmaster`, `xterm`). Flows flagged by *both*
models hit 97% precision. Each alert carries `alert_source`
(`supervised` / `anomaly` / `both`) and `ensemble_score` for ranking.

## 📁 Project structure

```
ai-network-intrusion-detection/
├── config.yaml            # all knobs: data, model, thresholds, paths
├── config_multiclass.yaml # same pipeline, 5-class attack-type labels
├── config_anomaly.yaml    # IsolationForest anomaly detector (normal-only training)
├── requirements.txt
├── app.py                 # Streamlit dashboard
├── data/
│   ├── download_nsl_kdd.py
│   └── README.md
├── src/
│   ├── data_loader.py     # NSL-KDD loading + synthetic data generator
│   ├── features.py        # preprocessing pipeline + feature importance
│   ├── train.py           # train & persist the model
│   ├── train_anomaly.py   # train the IsolationForest anomaly model
│   ├── train_autoencoder.py # train the MLP autoencoder anomaly model
│   ├── anomaly_models.py  # custom estimators (picklable from any process)
│   ├── explain.py         # SHAP per-alert explanations
│   ├── evaluate.py        # metrics + reports/
│   ├── detect.py          # NIDSDetector: batch / stdin / live scoring
│   └── utils.py
├── tests/
│   └── test_pipeline.py   # fast smoke tests (synthetic data)
├── models/                # trained bundles (*.joblib, git-ignored)
└── reports/               # metrics.json, confusion matrix (git-ignored)
```

## 🧠 Model details

- **Algorithm:** RandomForest (150 trees, balanced class weights)
- **Preprocessing:** one-hot encoding for `protocol_type`/`service`/`flag`,
  standard scaling for the 38 numeric features
- **Task:** binary — `normal` (0) vs `attack` (1); set
  `features.binary_labels: false` in `config.yaml` for 5-class
  (normal/dos/probe/r2l/u2r) mode
- **Top signals** the model learns: `serror_rate`, `srv_serror_rate`,
  `num_failed_logins`, `count`, `dst_host_serror_rate` — classic
  flood/scan fingerprints

## 🔍 Detection usage

```bash
# Score a CSV of flows (NSL-KDD columns, header row)
python -m src.detect --input flows.csv --output alerts.csv --threshold 0.8

# Score a single flow from JSON
echo '{"duration":0,"protocol_type":"tcp","service":"http","flag":"SF",
  "src_bytes":500,"dst_bytes":1200,"count":10}' | python -m src.detect --stdin

# Live capture demo (needs root + scapy)
sudo python -m src.detect --live --seconds 60
```

Alerts are ranked by attack probability and written to CSV.

## 🧪 Tests

```bash
python tests/test_pipeline.py
# or
python -m pytest tests/ -q
```

## ⚙️ Configuration

Everything lives in `config.yaml`: dataset URLs, synthetic sizes,
RandomForest hyperparameters, model paths, and the alert threshold.
No code changes needed to retune.

## 🔮 Ideas to extend

- Combine supervised + anomaly scores into a single ensemble detector
  (e.g. max of the two normalized scores) and wire both into `detect.py` / `app.py`
- Swap in XGBoost/LightGBM or a deep autoencoder for anomaly detection
- Stream flows from Kafka / a SIEM instead of CSV
- Add SHAP explanations per alert in the dashboard
- Export to ONNX for low-latency inline scoring

## 📄 Data license note

NSL-KDD is a public research dataset derived from KDD Cup '99.
Synthetic data is generated locally and has no license restrictions.
