"""Utiity module for package."""
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    """Return mean absolute percentage error of y_pred."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    abs_errors = np.abs((y_true - y_pred) / y_true)
    return np.mean(abs_errors) * 100
