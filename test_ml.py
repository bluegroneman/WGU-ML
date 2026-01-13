import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import compute_model_metrics, inference
from ml.model import train_model as train_model_fn


def _load_data():
    """Helper to load the census.csv from the repo's data directory."""
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "data", "census.csv")
    assert os.path.exists(data_path), f"Data file not found at {data_path}"
    return pd.read_csv(data_path)


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_process_data_returns_expected_types_and_shapes():
    """
    Ensure process_data returns numpy arrays and proper encoders with consistent shapes
    for both training and test splits.
    """
    data = _load_data()
    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=0, stratify=data["salary"]
    )

    X_train, y_train, enc, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _enc2, _lb2 = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=enc,
        lb=lb,
    )

    # Types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(enc, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

    # Shapes: rows should match input sizes
    assert X_train.shape[0] == len(train_df)
    assert y_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert y_test.shape[0] == len(test_df)

    # Feature dimension should be consistent between train and test
    assert X_train.shape[1] == X_test.shape[1]

    # Labels should be binary 0/1
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_train_model_uses_logistic_regression_algorithm():
    """
    Verify that train_model returns a LogisticRegression instance
    """
    data = _load_data().sample(n=1000, random_state=1)  # subset for speed
    train_df, _ = train_test_split(
        data, test_size=0.2, random_state=1, stratify=data["salary"]
    )
    X_train, y_train, enc, lb = process_data(
        train_df, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model_fn(X_train, y_train)
    assert isinstance(model, LogisticRegression)


def test_inference_and_metrics_behave_as_expected():
    """
    Check that inference returns the expected type/values and that
    compute_model_metrics returns expected values on a known small example.
    """
    # Known, simple example for metrics
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    # Expected: TP=1, FP=0, FN=1 => P=1.0, R=0.5, F1=2*1*0.5/(1+0.5)=2/3
    assert precision == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert recall == pytest.approx(0.5, rel=1e-6, abs=1e-6)
    assert f1 == pytest.approx(2.0 / 3.0, rel=1e-6, abs=1e-6)

    # End-to-end: train, predict, and verify predictions are binary and sized correctly
    data = _load_data().sample(n=1000, random_state=2)  # subset for speed
    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=2, stratify=data["salary"]
    )
    X_train, y_train, enc, lb = process_data(
        train_df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=enc,
        lb=lb,
    )
    model = train_model_fn(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})
