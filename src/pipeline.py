import numpy as np

from config import BUS_CLASS_COL, CARS_FILE, FEATURE_SLICE, ONE_HOT_SLICE, RANDOM_STATE
from src.data.dataset import build_dataset, split_train_test
from src.data.features import build_features
from src.data.loader import extract_data_and_target, load_cars
from src.models.decision_tree import DecisionTreeModel
from src.models.linear_regression import LinearRegressionModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.metrics import jaccard_similarity
from src.models.mlp import MLPModel


def _prepare_data():
    cars = load_cars(CARS_FILE)
    data, target = extract_data_and_target(cars)
    features = build_features(data)
    dataset = build_dataset(features, target)
    train, test = split_train_test(dataset, random_state=RANDOM_STATE)
    return train, test


def _report(label, train_score, test_score, train_jaccard=None, test_jaccard=None):
    print(f"{label}:")
    print(f"  train score:  {train_score:.4f}")
    print(f"  test score:   {test_score:.4f}")
    if train_jaccard is not None:
        print(f"  jaccard train: {train_jaccard:.4f}")
    if test_jaccard is not None:
        print(f"  jaccard test:  {test_jaccard:.4f}")


def linear_regression_analysis(train, test):
    """Multi-output linear regression over the one-hot encoded targets."""
    X_tr, X_ts = train[:, FEATURE_SLICE], test[:, FEATURE_SLICE]
    y_tr = train[:, ONE_HOT_SLICE].astype(int)
    y_ts = test[:, ONE_HOT_SLICE].astype(int)

    reg = LinearRegressionModel().fit(X_tr, y_tr)
    train_score = reg.score(X_tr, y_tr)
    test_score = reg.score(X_ts, y_ts)

    # Threshold the fitted response for the "bus" output and measure Jaccard.
    u = X_tr @ reg.model.coef_[0, :].reshape(-1, 1)
    bus_pred = (u > 0.25) * 1
    train_jaccard = jaccard_similarity(train[:, BUS_CLASS_COL], bus_pred[:, 0])

    _report("Linear Regression", train_score, test_score, train_jaccard)
    return train_score, test_score, train_jaccard


def logistic_regression_analysis(train, test):
    """Binary logistic regression on the 'bus' class column."""
    X_tr, X_ts = train[:, FEATURE_SLICE], test[:, FEATURE_SLICE]
    y_tr = train[:, BUS_CLASS_COL].astype(int)
    y_ts = test[:, BUS_CLASS_COL].astype(int)

    clf = LogisticRegressionModel().fit(X_tr, y_tr)
    train_score = clf.score(X_tr, y_tr)
    test_score = clf.score(X_ts, y_ts)
    train_jaccard = jaccard_similarity(train[:, BUS_CLASS_COL], clf.predict(X_tr))

    _report("Logistic Regression", train_score, test_score, train_jaccard)
    return train_score, test_score, train_jaccard


def decision_tree_analysis(train, test):
    """Decision tree classifier on the 'bus' class column."""
    X_tr, X_ts = train[:, FEATURE_SLICE], test[:, FEATURE_SLICE]
    y_tr = train[:, BUS_CLASS_COL].astype(int)
    y_ts = test[:, BUS_CLASS_COL].astype(int)

    clf = DecisionTreeModel().fit(X_tr, y_tr)
    train_score = clf.score(X_tr, y_tr)
    test_score = clf.score(X_ts, y_ts)
    train_jaccard = jaccard_similarity(train[:, BUS_CLASS_COL], clf.predict(X_tr))
    test_jaccard = jaccard_similarity(test[:, BUS_CLASS_COL], clf.predict(X_ts))

    _report("Decision Tree", train_score, test_score, train_jaccard, test_jaccard)
    return train_score, test_score, train_jaccard, test_jaccard


def mlp_analysis(train, test):
    """MLP classifier on the 'bus' class column."""
    X_tr, X_ts = train[:, FEATURE_SLICE], test[:, FEATURE_SLICE]
    y_tr = train[:, BUS_CLASS_COL].astype(int)
    y_ts = test[:, BUS_CLASS_COL].astype(int)

    clf = MLPModel().fit(X_tr, y_tr)
    train_score = clf.score(X_tr, y_tr)
    test_score = clf.score(X_ts, y_ts)

    _report("MLP", train_score, test_score)
    return train_score, test_score


def run_pipeline():
    print(f"Loading data from {CARS_FILE} ...")
    train, test = _prepare_data()
    print(f"Dataset matrix shape: {(train.shape[0] + test.shape[0], train.shape[1])}")
    print(f"Train rows: {train.shape[0]}  Test rows: {test.shape[0]}\n")

    results = {}
    results["linear_regression"] = linear_regression_analysis(train, test)
    results["logistic_regression"] = logistic_regression_analysis(train, test)
    results["decision_tree"] = decision_tree_analysis(train, test)
    results["mlp"] = mlp_analysis(train, test)
    return results
