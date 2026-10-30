"""Data loading utilities for the NIDS project.

Supports the NSL-KDD dataset (41 flow features + label) and a synthetic
fallback generator so the pipeline runs end-to-end without any downloads.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# NSL-KDD schema: 41 features, then the label, then a difficulty score.
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = [
    "http", "ftp", "smtp", "ssh", "telnet", "dns", "dhcp", "snmp",
    "pop3", "imap", "https", "ftp_data", "other",
]
FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"]


def download_if_missing(url: str, dest: str | os.PathLike) -> Path:
    """Download ``url`` to ``dest`` unless the file already exists."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print("Done.")
    return dest


def load_nsl_kdd(path: str | os.PathLike) -> pd.DataFrame:
    """Load an NSL-KDD train/test file (no header row in the raw files)."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    # NSL-KDD labels look like "normal." / "neptune." (trailing dot).
    df["label"] = df["label"].astype(str).str.rstrip(".")
    return df


def binarize_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Map every non-'normal' label to 1 (attack) and 'normal' to 0."""
    df = df.copy()
    df[label_col] = (df[label_col] != "normal").astype(int)
    return df


def multiclass_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Collapse NSL-KDD's ~38 attack names into 4 attack categories + normal.

    Includes the novel attack types that appear only in the test set
    (NSL-KDD's generalization benchmark), categorized per the standard
    Tavallaee taxonomy instead of lumping them into one class.
    """
    mapping = {
        "normal": "normal",
        # DoS
        "neptune": "dos", "smurf": "dos", "back": "dos", "teardrop": "dos",
        "pod": "dos", "land": "dos",
        "apache2": "dos", "mailbomb": "dos", "processtable": "dos",
        "udpstorm": "dos",
        # Probe
        "satan": "probe", "ipsweep": "probe", "portsweep": "probe",
        "nmap": "probe",
        "mscan": "probe", "saint": "probe",
        # R2L
        "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l",
        "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
        "warezclient": "r2l", "spy": "r2l",
        "snmpgetattack": "r2l", "snmpguess": "r2l", "httptunnel": "r2l",
        "named": "r2l", "sendmail": "r2l", "worm": "r2l",
        "xlock": "r2l", "xsnoop": "r2l",
        # U2R
        "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
        "perl": "u2r",
        "xterm": "u2r", "ps": "u2r", "sqlattack": "u2r",
    }
    df = df.copy()
    unmapped = set(df[label_col].unique()) - set(mapping)
    if unmapped:
        print(f"WARNING: unmapped labels -> 'unknown': {sorted(unmapped)}")
        df[label_col] = df[label_col].map(mapping).fillna("unknown")
    else:
        df[label_col] = df[label_col].map(mapping)
    return df


def generate_synthetic(
    n: int = 10_000, attack_ratio: float = 0.35, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic flow records with the NSL-KDD schema.

    Attack flows are drawn from shifted distributions (high error rates,
    failed logins, root activity...) so a classifier can actually learn
    the boundary. Useful for demos, CI and offline development.
    """
    rng = np.random.default_rng(seed)
    is_attack = rng.random(n) < attack_ratio

    def mix(normal_fn, attack_fn, **kw):
        out = normal_fn(size=n, **kw)
        out[is_attack] = attack_fn(size=is_attack.sum(), **kw)
        return out

    data = {
        "duration": mix(rng.exponential, rng.exponential, scale=1.0),
        "protocol_type": rng.choice(PROTOCOLS, n, p=[0.7, 0.2, 0.1]),
        "service": rng.choice(SERVICES, n),
        "flag": np.where(
            is_attack,
            rng.choice(["S0", "REJ", "RSTO", "SH"], is_attack.sum() * 0 + n),
            rng.choice(["SF", "S1"], n),
        ),
        "src_bytes": mix(lambda size, **k: rng.exponential(500, size),
                         lambda size, **k: rng.exponential(5000, size)),
        "dst_bytes": mix(lambda size, **k: rng.exponential(800, size),
                         lambda size, **k: rng.exponential(300, size)),
        "land": np.zeros(n, dtype=int),
        "wrong_fragment": (rng.random(n) < np.where(is_attack, 0.25, 0.01)).astype(int),
        "urgent": np.zeros(n, dtype=int),
        "hot": mix(lambda size, **k: rng.poisson(0.2, size),
                   lambda size, **k: rng.poisson(3.0, size)).astype(int),
        "num_failed_logins": mix(
            lambda size, **k: rng.poisson(0.05, size),
            lambda size, **k: rng.poisson(2.5, size)).astype(int),
        "logged_in": (rng.random(n) < np.where(is_attack, 0.3, 0.85)).astype(int),
        "num_compromised": mix(lambda size, **k: rng.poisson(0.02, size),
                              lambda size, **k: rng.poisson(1.5, size)).astype(int),
        "root_shell": (rng.random(n) < np.where(is_attack, 0.15, 0.0)).astype(int),
        "su_attempted": (rng.random(n) < np.where(is_attack, 0.1, 0.0)).astype(int),
        "num_root": mix(lambda size, **k: rng.poisson(0.05, size),
                        lambda size, **k: rng.poisson(2.0, size)).astype(int),
        "num_file_creations": mix(lambda size, **k: rng.poisson(0.1, size),
                                  lambda size, **k: rng.poisson(1.2, size)).astype(int),
        "num_shells": (rng.random(n) < np.where(is_attack, 0.12, 0.0)).astype(int),
        "num_access_files": mix(lambda size, **k: rng.poisson(0.05, size),
                                lambda size, **k: rng.poisson(1.0, size)).astype(int),
        "num_outbound_cmds": np.zeros(n, dtype=int),
        "is_host_login": np.zeros(n, dtype=int),
        "is_guest_login": (rng.random(n) < np.where(is_attack, 0.08, 0.01)).astype(int),
        "count": mix(lambda size, **k: rng.poisson(8, size).astype(float),
                     lambda size, **k: rng.poisson(80, size).astype(float)),
        "srv_count": mix(lambda size, **k: rng.poisson(6, size).astype(float),
                         lambda size, **k: rng.poisson(60, size).astype(float)),
        "serror_rate": np.clip(mix(lambda size, **k: rng.beta(1, 20, size),
                                   lambda size, **k: rng.beta(8, 2, size)), 0, 1),
        "srv_serror_rate": np.clip(mix(lambda size, **k: rng.beta(1, 20, size),
                                       lambda size, **k: rng.beta(8, 2, size)), 0, 1),
        "rerror_rate": np.clip(mix(lambda size, **k: rng.beta(1, 30, size),
                                   lambda size, **k: rng.beta(5, 5, size)), 0, 1),
        "srv_rerror_rate": np.clip(mix(lambda size, **k: rng.beta(1, 30, size),
                                       lambda size, **k: rng.beta(5, 5, size)), 0, 1),
        "same_srv_rate": np.clip(mix(lambda size, **k: rng.beta(15, 2, size),
                                     lambda size, **k: rng.beta(4, 4, size)), 0, 1),
        "diff_srv_rate": np.clip(mix(lambda size, **k: rng.beta(2, 15, size),
                                     lambda size, **k: rng.beta(4, 4, size)), 0, 1),
        "srv_diff_host_rate": np.clip(rng.beta(2, 20, n), 0, 1),
        "dst_host_count": rng.integers(1, 255, n).astype(float),
        "dst_host_srv_count": rng.integers(1, 255, n).astype(float),
        "dst_host_same_srv_rate": np.clip(rng.beta(10, 3, n), 0, 1),
        "dst_host_diff_srv_rate": np.clip(rng.beta(2, 15, n), 0, 1),
        "dst_host_same_src_port_rate": np.clip(rng.beta(5, 5, n), 0, 1),
        "dst_host_srv_diff_host_rate": np.clip(rng.beta(2, 20, n), 0, 1),
        "dst_host_serror_rate": np.clip(mix(lambda size, **k: rng.beta(1, 25, size),
                                            lambda size, **k: rng.beta(6, 3, size)), 0, 1),
        "dst_host_srv_serror_rate": np.clip(mix(lambda size, **k: rng.beta(1, 25, size),
                                                lambda size, **k: rng.beta(6, 3, size)), 0, 1),
        "dst_host_rerror_rate": np.clip(rng.beta(1, 30, n), 0, 1),
        "dst_host_srv_rerror_rate": np.clip(rng.beta(1, 30, n), 0, 1),
        "label": np.where(is_attack, "attack", "normal"),
        "difficulty": rng.integers(0, 21, n),
    }
    return pd.DataFrame(data)


def load_datasets(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test frames, preferring NSL-KDD, else synthetic fallback."""
    dcfg = cfg["data"]
    train_p, test_p = Path(dcfg["train_path"]), Path(dcfg["test_path"])
    try:
        if not train_p.exists():
            download_if_missing(dcfg["nsl_kdd_train_url"], train_p)
        if not test_p.exists():
            download_if_missing(dcfg["nsl_kdd_test_url"], test_p)
        train_df, test_df = load_nsl_kdd(train_p), load_nsl_kdd(test_p)
        print(f"Loaded NSL-KDD: train={len(train_df):,} test={len(test_df):,}")
    except Exception as exc:  # offline? fall back gracefully
        if not dcfg.get("use_synthetic_fallback", True):
            raise
        print(f"Could not load NSL-KDD ({exc}); using synthetic data.")
        seed = dcfg.get("random_state", 42)
        train_df = generate_synthetic(dcfg.get("synthetic_n_train", 20000), seed=seed)
        test_df = generate_synthetic(dcfg.get("synthetic_n_test", 5000), seed=seed + 1)

    if cfg["features"].get("binary_labels", True):
        train_df, test_df = binarize_labels(train_df), binarize_labels(test_df)
    else:
        train_df, test_df = multiclass_labels(train_df), multiclass_labels(test_df)
    return train_df, test_df
