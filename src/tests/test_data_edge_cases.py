import numpy as np
import pytest

from train import load_and_clean_data


def test_clean_data_removes_nans() -> None:
    """Test that rows with empty/NaN values are properly dropped."""
    x_raw, y_raw = load_and_clean_data("data/data_with_nans.csv")

    # It should only keep the 1st and 4th rows
    assert len(x_raw) == 2
    assert len(y_raw) == 2
    assert np.array_equal(x_raw, [100000.0, 50000.0])
    assert np.array_equal(y_raw, [5000.0, 8000.0])


def test_clean_data_coerces_invalid_strings() -> None:
    """Test that text in numeric columns is removed gracefully."""
    x_raw, y_raw = load_and_clean_data("data/data_with_strings.csv")

    # It should only keep the 1st and 4th rows (the valid ones)
    assert len(x_raw) == 2
    assert len(y_raw) == 2
    assert x_raw[0] == 100000.0
    assert y_raw[0] == 5000.0
    assert x_raw[1] == 50000.0
    assert y_raw[1] == 8000.0


def test_missing_file_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that providing a non-existent file causes a clean exit."""
    with pytest.raises(SystemExit) as excinfo:
        load_and_clean_data("data/does_not_exist.csv")

    assert excinfo.value.code == 1

    # Check that it printed a helpful error message to the terminal
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_missing_columns_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that a CSV with the wrong headers causes a clean exit."""
    with pytest.raises(SystemExit) as excinfo:
        load_and_clean_data("data/wrong_headers.csv")

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "must contain 'km' and 'price'" in captured.out


def test_insufficient_data_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that less than 2 valid rows causes a clean exit."""
    with pytest.raises(SystemExit) as excinfo:
        load_and_clean_data("data/one_row.csv")

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Insufficient valid data" in captured.out
