"""Test module for src/utils.py."""
from src.utils import mean_absolute_percentage_error


def test_mape():
    """Test if expected mean absolute percentage error is returned."""
    y_true, y_pred = [1, 2, 3], [1, 4, 3]
    expected = (1 / 3) * 100
    result = mean_absolute_percentage_error(y_true, y_pred)
    assert round(result, 5) == round(expected, 5)
