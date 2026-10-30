"""Feature engineering: turn raw flow records into a model-ready matrix."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_columns(df: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str]]:
    """Split columns into (categorical, numeric) feature lists."""
    fcfg = cfg["features"]
    label_col = fcfg["label_col"]
    drop = set(fcfg.get("drop_cols", [])) | {label_col}
    categorical = [c for c in fcfg["categorical"] if c in df.columns]
    numeric = [c for c in df.columns if c not in drop and c not in categorical]
    return categorical, numeric


def build_preprocessor(
    categorical: list[str], numeric: list[str]
) -> ColumnTransformer:
    """One-hot encode categoricals, standard-scale numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )


def build_pipeline(cfg: dict, categorical: list[str], numeric: list[str]) -> Pipeline:
    """Full sklearn pipeline: preprocessing + RandomForest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    mcfg = cfg["model"]
    clf = RandomForestClassifier(
        n_estimators=mcfg.get("n_estimators", 150),
        max_depth=mcfg.get("max_depth", 25),
        min_samples_split=mcfg.get("min_samples_split", 2),
        random_state=mcfg.get("random_state", 42),
        n_jobs=mcfg.get("n_jobs", -1),
        class_weight=mcfg.get("class_weight", "balanced_subsample"),
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(categorical, numeric)),
            ("classifier", clf),
        ]
    )


def top_features(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Return the most important features from a fitted pipeline."""
    pre = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["classifier"]
    names = pre.get_feature_names_out()
    importances = clf.feature_importances_
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n)
