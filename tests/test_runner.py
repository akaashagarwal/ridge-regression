"""Test module for src/runner.py."""
from src.runner import predict


def test_predict():
    """Test full system end to end."""
    expected = (round(18.04574010314292, 1), round(18.434709712808793, 1))
    output = predict()
    output_1, output_2 = output

    assert expected[0] == round(output_1, 1)
    assert expected[1] == round(output_2, 1)
