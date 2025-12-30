from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    model_type: str,
    model_parameters: Dict[str, Any],
) -> Any:
    """
    Trains a machine learning model based on the specified problem type and model type.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        problem_type (str): The type of problem ('regression' or 'classification').
        model_type (str): The type of model to train ('ridge', 'knn', 'random_forest').
        model_parameters (Dict[str, Any]): Model-specific hyperparameters.

    Returns:
        Any: The trained model or best estimator from Grid Search.
    """

    Model = {
        "regression": {
            "ridge": Ridge,
            "knn": KNeighborsRegressor,
            "random_forest": RandomForestRegressor,
        },
        "classification": {
            "logistic_regression": LogisticRegression,
            "knn": KNeighborsClassifier,
            "random_forest": RandomForestClassifier,
        },
    }
    model = Model.get(problem_type, {}).get(model_type, None)
    if model is None:
        raise ValueError(
            f"The model_type {model_type} is not defined for problem_type {problem_type}"
        )

    # Use Grid Search if parameter grid exists.
    if "param_grid" in model_parameters:
        grid_search = GridSearchCV(
            model(),
            model_parameters["param_grid"],
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        model = model(**model_parameters)
        model.fit(X_train, y_train)
        return model


def evaluate_model(
    model: any, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str
) -> Dict[str, float]:
    """
    Evaluates a trained model using specified metrics.

    Args:
        model (any): A trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target variable.
        problem_type (str): The type of problem ('regression' or 'classification').

    Returns:
        Dict[str, float]: A dictionary containing the performance metrics
        For regression: MAE, RMSE, R2.
        For classification: Accuracy, F1-score, Recall, Precision.
    """
    metrics = {
        "regression": {
            "mae": mean_absolute_error,
            "mse": mean_squared_error,
            "r2": r2_score,
        },
        "classification": {
            "accuracy": accuracy_score,
            "f1_score": f1_score,
            "recall": recall_score,
            "precision": precision_score,
        },
    }

    # Make predictions
    y_pred = model.predict(X_test)

    is_multiclass = np.unique(y_test).shape[0] > 2

    # Calculate metrics
    results = {}
    for metric_name, metric_func in metrics[problem_type].items():
        if problem_type == "classification" and metric_name in ["f1_score", "recall", "precision"]:
            results[metric_name] = metric_func(
                y_test, y_pred, average="macro" if is_multiclass else "binary"
            )
        else:
            results[metric_name] = metric_func(y_test, y_pred)

    return results


def report_performance(
    metrics: Dict[str, float], shap_values: np.ndarray = None
) -> None:
    """
    Prints the performance of the model in a nicely formatted string.

    Args:
        metrics (Dict[str, float]): Dictionary containing performance metrics.
        shap_values (np.ndarray): Array containing the shap values.

    Returns:
         None
    """
    print("===== Model Performance =====")
    for key, value in metrics.items():
        print(f"\t{key.upper()}:\t{value:.4f}")
    print()
    if shap_values is not None:
        print("===== SHAP values =====")
        print(f"Shape of shap values: {shap_values.shape}")
        print(shap_values)
