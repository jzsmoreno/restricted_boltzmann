# python -m pytest tests/test_report.py -v
import os

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless testing

import numpy as np
import pytest

from restricted_boltzmann.tools import RBMReportGenerator


@pytest.fixture
def reporter(tmp_path):
    """Create an RBMReportGenerator writing to a temporary directory."""
    return RBMReportGenerator(str(tmp_path))


@pytest.fixture
def activations():
    np.random.seed(42)
    return np.random.rand(10, 5)


@pytest.fixture
def input_data():
    np.random.seed(42)
    return np.random.rand(10, 64)  # 8x8 = 64 (perfect square)


def test_generate_normal(reporter, activations, input_data):
    """Normal case: square input, multiple samples."""
    reporter.generate(activations, input_data, filename="test_normal.html", num_samples=3)
    output = os.path.join(reporter.folder_path, "test_normal.html")
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_generate_nonsquare(reporter, activations):
    """Non-square input (like the example notebook: 3600 features)."""
    input_data_nonsquare = np.random.rand(10, 3600)  # not a perfect square
    reporter.generate(
        activations, input_data_nonsquare, filename="test_nonsquare.html", num_samples=2
    )
    output = os.path.join(reporter.folder_path, "test_nonsquare.html")
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_generate_single_sample(reporter):
    """Single sample (correlation edge case)."""
    activations_single = np.random.rand(1, 5)
    input_data_single = np.random.rand(1, 64)
    reporter.generate(
        activations_single, input_data_single, filename="test_single.html", num_samples=1
    )
    output = os.path.join(reporter.folder_path, "test_single.html")
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_generate_nan_handling(reporter, activations, input_data):
    """NaN in activations should be handled without error."""
    activations_nan = activations.copy()
    activations_nan[0, 0] = np.nan
    reporter.generate(activations_nan, input_data, filename="test_nan.html", num_samples=2)
    output = os.path.join(reporter.folder_path, "test_nan.html")
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_validate_mismatched_samples(reporter):
    """Validation: mismatched sample counts should raise ValueError."""
    with pytest.raises(ValueError):
        reporter.generate(np.random.rand(5, 3), np.random.rand(10, 64), filename="test_bad.html")


def test_validate_empty_data(reporter):
    """Validation: empty data should raise ValueError."""
    with pytest.raises(ValueError):
        reporter.generate(np.empty((0, 3)), np.empty((0, 64)), filename="test_bad.html")


def test_validate_num_samples_zero(reporter, activations, input_data):
    """Validation: num_samples <= 0 should raise ValueError."""
    with pytest.raises(ValueError):
        reporter.generate(activations, input_data, filename="test_bad.html", num_samples=0)


def test_num_samples_clamped(reporter, activations, input_data):
    """num_samples larger than data should be clamped to the data size."""
    reporter.generate(activations, input_data, filename="test_clamp.html", num_samples=100)
    output = os.path.join(reporter.folder_path, "test_clamp.html")
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_pad_to_square_perfect_square(reporter):
    """_pad_to_square: 3600 is already a perfect square, should be unchanged."""
    sample = np.random.rand(3600)
    padded = reporter._pad_to_square(sample)
    assert padded.size == 3600


def test_pad_to_square_pads(reporter):
    """_pad_to_square: 10 should pad to 16."""
    sample = np.random.rand(10)
    padded = reporter._pad_to_square(sample)
    assert padded.size == 16
    # Original values preserved in the first 10 positions
    np.testing.assert_array_equal(padded[:10], sample)
    # Remaining positions are zero-padded
    np.testing.assert_array_equal(padded[10:], np.zeros(6))


def test_calc_sparsity_custom_threshold(reporter):
    """_calc_sparsity with a custom threshold."""
    sp = reporter._calc_sparsity(np.array([[0.001, 0.5, 0.9]]), threshold=0.01)
    assert sp == pytest.approx(33.333, abs=0.01)
