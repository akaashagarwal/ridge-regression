"""Test module for src/algorithm.py."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.algorithm import RidgeRegression

TEST_DATA = pd.read_excel("src/data/data.xlsx")


def test_ridge_regression_fit():
    """Test if model weights and predictions are correct."""
    obj = RidgeRegression(learning_rate=0.01, reg_strength=100, max_iter=200)
    x_train, x_test, y_train, _ = train_test_split([[1, 2], [1.5, 3], [3, 4], [3.5, 4.5], [5, 6]],
                                                   [1, 2, 3, 4, 5],
                                                   test_size=1 / 3,
                                                   random_state=41,
                                                   shuffle=True)
    expected_weights = np.array([0.04845848, 0.20013845, 0.24859693])
    expected_y_hats = np.array([1.09445694, 1.64326155])
    obj.fit(pd.DataFrame(np.array(x_train).reshape(-1, 2)), pd.Series(y_train))
    output_weights = obj.weights

    assert all(
        round(expected_weights[idx], 5) == round(output_weights[idx], 5)
        for idx in range(output_weights.shape[0]))

    y_hats = obj.predict(pd.DataFrame(x_test))

    assert all(
        round(y_hats[idx], 5) == round(expected_y_hats[idx], 5) for idx in range(y_hats.shape[0]))
