import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "heart_disease_clean.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def build_pipeline(model, feature_columns):
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            feature_columns
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "The target must contain exactly two classes."
        )

    # Keep a held-out test set for final evaluation.
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = {
        "logistic_regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    # Store experiment metadata in a local SQLite database.
    tracking_db = ROOT / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db}"
    )

    mlflow.set_experiment(
        "heart-disease-classification"
    )

    results = []
    best_pipeline = None
    best_model_name = None
    best_cv_auc = -np.inf

    for model_name, estimator in models.items():
        pipeline = build_pipeline(
            estimator,
            list(X.columns)
        )

        with mlflow.start_run(
            run_name=model_name
        ):
            # Cross-validation is performed only on training data.
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                error_score="raise",
            )

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            probabilities = (
                pipeline.predict_proba(X_test)[:, 1]
            )

            metrics = {
                "cv_accuracy": float(
                    np.mean(cv_results["test_accuracy"])
                ),
                "cv_precision": float(
                    np.mean(cv_results["test_precision"])
                ),
                "cv_recall": float(
                    np.mean(cv_results["test_recall"])
                ),
                "cv_roc_auc": float(
                    np.mean(cv_results["test_roc_auc"])
                ),
                "test_accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "test_precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),
                "test_recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),
                "test_roc_auc": float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                    )
                ),
            }

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "random_state",
                42
            )

            mlflow.log_param(
                "cv_folds",
                5
            )

            mlflow.log_params(
                estimator.get_params()
            )

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            results.append({
                "model": model_name,
                **metrics,
            })

            print(f"\nModel: {model_name}")
            print(json.dumps(metrics, indent=2))

            # Select using CV AUC, not test-set AUC.
            if metrics["cv_roc_auc"] > best_cv_auc:
                best_cv_auc = metrics["cv_roc_auc"]
                best_pipeline = pipeline
                best_model_name = model_name

    # Save the entire preprocessing + estimator pipeline.
    model_path = (
        MODEL_DIR / "heart_disease_pipeline.joblib"
    )

    joblib.dump(best_pipeline, model_path)

    metadata = {
        "selected_model": best_model_name,
        "feature_columns": list(X.columns),
        "selection_metric": "mean cross-validation ROC-AUC",
        "selection_score": float(best_cv_auc),
        "target_mapping": (
            "0 = absent, 1 = present; verify against "
            "the downloaded dataset variant"
        ),
    }

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nSelected model:", best_model_name)
    print("Saved model:", model_path)
    print("Comparison saved to reports/model_comparison.csv")


if __name__ == "__main__":
    main()