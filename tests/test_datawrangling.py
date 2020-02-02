"""Test module for src/datawrangling.py."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.preprocessing import StandardScaler
from src.datawrangling import DataWrangling

TEST_DATA = pd.read_excel("src/data/data.xlsx")


def test_data():
    """Test data loading."""
    obj = DataWrangling("src/data/data.xlsx")
    obj.load_data()
    test_data_copy = TEST_DATA.copy(deep=True)
    test_data_copy.columns = list(map(str.lower, test_data_copy.columns))
    output = obj.data
    assert isinstance(output, pd.DataFrame)
    assert_frame_equal(output, test_data_copy)


def test_standardize_features():
    """Test feature standardization."""
    obj = DataWrangling("src/data/data.xlsx")
    obj.load_data()

    ss = StandardScaler()
    ss.fit(TEST_DATA)
    expected = ss.transform(TEST_DATA)
    output = obj.standardize_features(obj.data, list(obj.data.columns))
    print(output)
    assert isinstance(output, np.ndarray)
    assert np.array_equal(output, expected)


def test_standardize_test_set():
    """Test test set standardization using means and variances of training set."""
    obj = DataWrangling("src/data/data.xlsx")
    obj.load_data()
    _ = obj.standardize_features(obj.data, list(obj.data.columns))

    ss = StandardScaler()
    ss.fit(TEST_DATA)
    means, variances = ss.mean_, ss.var_
    test_set = TEST_DATA.iloc[:150]
    test_set.columns = list(map(str.lower, test_set.columns))
    centered_expected = test_set - means
    expected = centered_expected / (variances**0.5)

    output = obj.standardize_test_set(obj.data.iloc[:150], list(obj.data.columns))
    assert isinstance(output, pd.DataFrame)
    assert_frame_equal(output, expected)


def test_one_hot_encode_features():
    """Test one-hot encoding of categorical variables."""
    obj = DataWrangling("src/data/data.xlsx")

    input_df = pd.DataFrame([{'a': 1}, {'a': 2}, {'a': 3}])
    expected = pd.DataFrame([{
        'a__1': 1,
        'a__2': 0,
        'a__3': 0
    }, {
        'a__1': 0,
        'a__2': 1,
        'a__3': 0.
    }, {
        'a__1': 0,
        'a__2': 0,
        'a__3': 1
    }], dtype=np.uint8)

    output = obj.one_hot_encode_features(input_df, ['a'])

    assert_frame_equal(output, expected)


def test_split_train_test():
    """Test splitting of full data into train and test sets."""
    obj = DataWrangling()
    x_set = [[1, 2], [3, 4], [5, 6]]
    labels = [1, 2, 3]
    features = ['a', 'b']
    expected = {
        "x_train": pd.DataFrame([[3, 4], [1, 2]], columns=features),
        "x_test": pd.DataFrame([[5, 6]], columns=features),
        "y_train": pd.Series([2, 1]),
        "y_test": pd.Series([3])
    }

    output = obj.split_train_test(x_set, labels, features, 1 / 3)

    assert_frame_equal(expected['x_train'], output['x_train'])
    assert_frame_equal(expected['x_test'], output['x_test'])
    assert_series_equal(expected['y_train'], expected['y_train'])
    assert_series_equal(expected['y_test'], expected['y_test'])


def test_class_properties():
    """Test property-decoraed class methods."""
    obj = DataWrangling("src/data/data.xlsx")
    obj.load_data()
    input_df = pd.DataFrame([{'a': val} for val in range(1, 6)])
    expected_means = [sum(val for val in range(1, 6)) / 5]
    expected_vars = [2.]
    _ = obj.standardize_features(input_df, list(input_df.columns))

    assert obj.train_means == expected_means
    assert obj.train_vars == expected_vars

    obj.data = input_df

    assert_frame_equal(obj.data, input_df)
