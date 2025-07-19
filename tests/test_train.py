import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.utils import load_config, load_data

def test_config_file_loading():
    config = load_config()
    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)

def test_model_creation():
    config = load_config()
    X, y = load_data()
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_model_accuracy():
    config = load_config()
    X, y = load_data()
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.8 
