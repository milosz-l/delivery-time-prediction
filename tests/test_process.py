import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.config import Location, ProcessConfig
from src.process import create_preprocessor, drop_columns, get_X_y


@pytest.fixture
def data():
    return pd.DataFrame({"X1": [1, 2], "X2": [3, 4], "A": ["a1", "a2"], "B": ["b1", "b2"], "Y": [5, 9]})


def test_drop_columns(data):
    res = drop_columns.fn(data, columns=["X1"])
    expected = pd.DataFrame({"X2": [3, 4], "A": ["a1", "a2"], "B": ["b1", "b2"], "Y": [5, 9]})
    assert_frame_equal(res, expected)


def test_get_X_y(data):
    X, y = get_X_y.fn(data, label="Y")
    X_expected = pd.DataFrame({"X1": [1, 2], "X2": [3, 4], "A": ["a1", "a2"], "B": ["b1", "b2"]})
    Y_expected = pd.Series([5, 9], name="Y")
    assert_frame_equal(X, X_expected)
    assert_series_equal(y, Y_expected)


def test_preprocessor():
    # training data
    data = pd.DataFrame(
        {"X1": [1, 2, 3], "X2": [3, 99, 100], "A": ["a1", "a2", "a1"], "B": ["b1", "b1", "b1"], "Y": [5, 9, 10]}
    )
    test_config = ProcessConfig()
    test_config.label = "Y"
    test_config.one_hot_columns = ["A", "B"]
    test_config.min_max_columns = ["X1"]

    X, Y = get_X_y.fn(data, label=test_config.label)
    preprocessor = create_preprocessor.fn(X, test_config)

    print("X before transformation:\n", X)
    X_train_transformed = preprocessor.fit_transform(X)
    print("X after transformation:\n", X_train_transformed)
    print("------ preprocessor fit_transform on training data info: ------")
    print(preprocessor.output_indices_)
    print("transformers:")
    print(preprocessor.transformers_)
    print(preprocessor.get_feature_names_out())

    # test data
    test_data = pd.DataFrame(
        {
            "X2": [3, 99, 100, 100],
            "A": ["a1", "a2", "a1", "a2"],
            "B": ["b1", "b1", "b1", "b2"],
            "X1": [0, 2, 3, 4],
            "X3": [5, 6, 7, 8],
        }
    )
    print("test data before transformation:")
    print(test_data)
    test_data_transformed = preprocessor.transform(test_data)
    print("test data after transformation:")
    print(test_data_transformed)
    print(f"type = {type(test_data_transformed)}")
    print("------ preprocessor transform on test_data info: ------")
    print(preprocessor.output_indices_)
    print("transformers:")
    print(preprocessor.transformers_)
    print(preprocessor.get_feature_names_out())
    test_data_transformed = pd.DataFrame(test_data_transformed)
    test_data_transformed.columns = preprocessor.get_feature_names_out()
    test_data_transformed = test_data_transformed[sorted(test_data_transformed.columns)]

    expected_test_data_after_transformation = np.array([[1, 1, 0, -0.5], [1, 0, 1, 0.5], [1, 1, 0, 1], [0, 0, 1, 1.5]])
    expected_test_data_after_transformation = pd.DataFrame(expected_test_data_after_transformation)
    expected_test_data_after_transformation.columns = ["cat__B_b1", "cat__A_a1", "cat__A_a2", "num__X1"]
    expected_test_data_after_transformation = expected_test_data_after_transformation[
        sorted(expected_test_data_after_transformation.columns)
    ]
    assert_frame_equal(test_data_transformed, expected_test_data_after_transformation)
