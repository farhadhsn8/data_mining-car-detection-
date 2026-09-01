import numpy as np
import pandas as pd
import pytest

from src.data.dataset import build_dataset, one_hot_encode, split_train_test
from src.data.features import build_features
from src.data.loader import extract_data_and_target


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "x1": [0, 10, 5],
            "y1": [0, 5, 5],
            "x2": [10, 30, 25],
            "y2": [20, 25, 35],
            "label": ["bus", "sedan", "truck"],
        }
    )


def test_extract_data_and_target(sample_frame):
    data, target = extract_data_and_target(sample_frame)
    assert data.shape == (3, 4)
    assert target.tolist() == ["bus", "sedan", "truck"]


def test_build_features_shape():
    data = np.array([[0, 0, 10, 20], [10, 5, 30, 25]])
    features = build_features(data)
    assert features.shape == (2, 9)
    assert np.allclose(features[:, 0], 1)


def test_build_features_geometry():
    data = np.array([[0, 0, 10, 20]])
    features = build_features(data)
    assert features[0, 5] == 10   # width
    assert features[0, 6] == 20   # height
    assert features[0, 7] == 60   # perimeter
    assert features[0, 8] == 200  # area


def test_one_hot_encode(sample_frame):
    _, target = extract_data_and_target(sample_frame)
    encoded = one_hot_encode(target)
    assert encoded.shape == (3, 6)
    assert encoded.sum(axis=1).tolist() == [1, 1, 1]


def test_build_dataset_shape(sample_frame):
    data, target = extract_data_and_target(sample_frame)
    dataset = build_dataset(build_features(data), target)
    assert dataset.shape == (3, 15)


def test_split_train_test_ratio():
    dataset = np.arange(100).reshape(50, 2)
    train, test = split_train_test(dataset, train_ratio=0.8, random_state=0)
    assert train.shape[0] == 40
    assert test.shape[0] == 10
