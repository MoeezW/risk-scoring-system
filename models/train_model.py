from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import yaml

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


# ----------------------------
# Config objects
# ----------------------------

@dataclass(frozen=True)
class DataConfig:
    db_path: Path
    table_name: str
    id_column: str
    target_column: str


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int


@dataclass(frozen=True)
class TrainingConfig:
    use_class_weight_balanced: bool


@dataclass(frozen=True)
class ModelSpec:
    name: str
    type: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class Config:
    data: DataConfig
    split: SplitConfig
    training: TrainingConfig
    models: List[ModelSpec]


# ----------------------------
# Utilities
# ----------------------------

def load_config(project_root: Path) -> Config:
    cfg_path = project_root / "models" / "model_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = DataConfig(
        db_path=project_root / raw["data"]["db_path"],
        table_name=raw["data"]["table_name"],
        id_column=raw["data"]["id_column"],
        target_column=raw["data"]["target_column"],
    )

    split = SplitConfig(
        test_size=float(raw["split"]["test_size"]),
        random_state=int(raw["split"]["random_state"]),
    )

    training = TrainingConfig(
        use_class_weight_balanced=bool(raw["training"]["use_class_weight_balanced"])
    )

    models = [
        ModelSpec(name=m["name"], type=m["type"], params=m.get("params", {}))
        for m in raw["models"]
    ]

    return Config(data=data, split=split, training=training, models=models)


def load_table_from_sqlite(db_path: Path, table_name: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
    finally:
        con.close()
    return df


def infer_feature_types(df: pd.DataFrame, id_column: str, target_column: str) -> Tuple[List[str], List[str]]:
    """
    Decide which columns are categorical vs numeric.

    In this dataset:
      - sex, education, marriage are integer-coded categories
      - everything else is numeric

    We'll treat only those three as categorical by default,
    and treat the rest as numeric.
    """
    categorical = [c for c in ["sex", "education", "marriage"] if c in df.columns]
    all_cols = [c for c in df.columns if c not in [id_column, target_column]]
    numeric = [c for c in all_cols if c not in categorical]
    return categorical, numeric


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """
    ColumnTransformer applies different preprocessing to different column subsets.

    - Numeric: StandardScaler (helps LogisticRegression a lot)
    - Categorical: OneHotEncoder (turns codes into one-hot columns)
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",  # drop any columns not explicitly listed
    )

    return preprocessor


def make_estimator(model_type: str, params: Dict[str, Any], use_class_weight_balanced: bool):
    """
    Instantiate the actual ML estimator (model) based on config.
    """
    if model_type == "logistic_regression":
        # class_weight='balanced' can improve recall on minority class
        class_weight = "balanced" if use_class_weight_balanced else None
        return LogisticRegression(class_weight=class_weight, **params)

    if model_type == "random_forest":
        class_weight = "balanced" if use_class_weight_balanced else None
        # RandomForest supports class_weight too (balanced or balanced_subsample)
        return RandomForestClassifier(class_weight=class_weight, **params)

    raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model on test set and return key metrics.

    We prefer AUC and Average Precision for imbalanced classification.
    We'll also log precision/recall/F1 at the default 0.5 threshold.
    """
    # Probability of class 1 (default)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
    }
    return metrics


def save_confusion_matrix(fig_path: Path, y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No default", "Default"])
    plt.yticks([0, 1], ["No default", "Default"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]  # repo root
    cfg = load_config(project_root)

    # Where we'll save the "best model"
    artifacts_dir = project_root / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data from SQLite
    df = load_table_from_sqlite(cfg.data.db_path, cfg.data.table_name)

    # Split into X/y
    y = df[cfg.data.target_column].astype(int)
    X = df.drop(columns=[cfg.data.target_column])

    # Drop ID column from features (keep it for reference elsewhere, not for model learning)
    if cfg.data.id_column in X.columns:
        X = X.drop(columns=[cfg.data.id_column])

    # Infer categorical and numeric columns
    categorical_cols, numeric_cols = infer_feature_types(
        df=df.drop(columns=[cfg.data.target_column]),
        id_column=cfg.data.id_column,
        target_column=cfg.data.target_column,
    )

    # If we dropped id from X, remove it from numeric if present
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    # Train/test split with stratification to keep default rate consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=y,
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Default rate train:", y_train.mean(), "test:", y_test.mean())
    print("Categorical cols:", categorical_cols)
    print("Numeric cols:", numeric_cols)

    # Set MLflow experiment
    mlflow.set_experiment("credit_risk_scoring")

    best_run = None
    best_score = -np.inf
    best_model = None
    best_metrics = None

    for spec in cfg.models:
        preprocessor = build_preprocessor(categorical_cols, numeric_cols)
        estimator = make_estimator(spec.type, spec.params, cfg.training.use_class_weight_balanced)

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        run_name = spec.name
        with mlflow.start_run(run_name=run_name) as run:
            # Log metadata/params
            mlflow.log_param("model_name", spec.name)
            mlflow.log_param("model_type", spec.type)
            mlflow.log_param("use_class_weight_balanced", cfg.training.use_class_weight_balanced)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("n_features_raw", X_train.shape[1])
            mlflow.log_param("n_categorical_cols", len(categorical_cols))
            mlflow.log_param("n_numeric_cols", len(numeric_cols))

            # Log hyperparams from config
            for k, v in spec.params.items():
                mlflow.log_param(f"hp_{k}", v)

            # Fit
            model.fit(X_train, y_train)

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Log classification report as artifact
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            report_txt = classification_report(y_test, pred, digits=4)
            report_path = artifacts_dir / f"{spec.name}_classification_report.txt"
            report_path.write_text(report_txt, encoding="utf-8")
            mlflow.log_artifact(str(report_path))

            # Log confusion matrix figure
            cm_path = artifacts_dir / f"{spec.name}_confusion_matrix.png"
            save_confusion_matrix(cm_path, y_test, pred)
            mlflow.log_artifact(str(cm_path))

            # Log model to MLflow
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Choose best by ROC AUC (you could also choose avg_precision)
            score = metrics["roc_auc"]
            print(f"{spec.name} ROC AUC: {score:.4f}")

            if score > best_score:
                best_score = score
                best_run = run.info.run_id
                best_model = model
                best_metrics = metrics

    # Save best model pipeline (preprocess + estimator together)
    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    best_model_path = artifacts_dir / "best_model.pkl"
    # sklearn includes joblib internally; importing joblib is safe
    import joblib
    joblib.dump(best_model, best_model_path)

    # Save metadata about best run
    best_info = {
        "best_run_id": best_run,
        "selection_metric": "roc_auc",
        "best_score": float(best_score),
        "best_metrics": best_metrics,
    }
    (artifacts_dir / "best_model_info.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")

    print("\nBest model saved to:", best_model_path)
    print("Best model info saved to:", artifacts_dir / "best_model_info.json")
    print("Best ROC AUC:", best_score)
    print("Done ✅")


if __name__ == "__main__":
    main()
