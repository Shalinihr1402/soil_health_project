"""Train, compare, and save crop-recommendation classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "crop_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
TARGET_COLUMN = "label"
RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the project crop dataset."""
    dataset = pd.read_csv(DATASET_PATH)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' target column.")

    features = dataset.drop(columns=TARGET_COLUMN)
    target = dataset[TARGET_COLUMN]
    if features.empty or target.empty:
        raise ValueError("Dataset must contain feature rows and crop labels.")
    if features.isnull().any().any() or target.isnull().any():
        raise ValueError("Dataset contains missing values; clean it before training.")
    return features, target


def build_models() -> dict[str, Any]:
    """Return supported classifiers; XGBoost is included only when installed."""
    models: dict[str, Any] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        pass

    return models


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate consistent weighted classification metrics."""
    predictions = model.predict(x_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
    }


def extract_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, float | str]]:
    """Return normalized feature importance when the selected model supports it."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    return [
        {"feature": name, "importance": round(float(value) * 100, 2)}
        for name, value in sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    ]


def train_and_save() -> dict[str, Any]:
    """Train candidates, select the highest F1 model, and save model plus metadata."""
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    trained_models: dict[str, Any] = {}
    comparisons: dict[str, dict[str, float]] = {}
    for model_name, candidate in build_models().items():
        try:
            candidate.fit(x_train, y_train)
            comparisons[model_name] = evaluate_model(candidate, x_test, y_test)
            trained_models[model_name] = candidate
        except Exception as error:
            print(f"Skipped {model_name}: {error}")

    if not trained_models:
        raise RuntimeError("No candidate model could be trained.")

    best_name = max(
        comparisons,
        key=lambda name: (comparisons[name]["f1_score"], comparisons[name]["accuracy"]),
    )
    best_model = trained_models[best_name]
    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "selected_model": best_name,
        "feature_names": list(features.columns),
        "model_comparison": comparisons,
        "feature_importance": extract_feature_importance(best_model, list(features.columns)),
        "training_rows": int(len(features)),
        "test_rows": int(len(x_test)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    training_metadata = train_and_save()
    print(f"Selected model: {training_metadata['selected_model']}")
    print("Model comparison:")
    for name, metrics in training_metadata["model_comparison"].items():
        print(f"  {name}: {metrics}")
    print(f"Saved model to {MODEL_PATH.name} and metadata to {METADATA_PATH.name}.")
