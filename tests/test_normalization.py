import numpy as np
import pandas as pd
import pytest

from xrd_preprocessing import (
    QRangeNormalizer,
    QRangeValueNormalizer,
    normalize_profile_by_q_range,
    normalize_profile_by_q_range_value,
)


def test_normalize_profile_by_default_q_range_integral():
    q = np.linspace(6.0, 8.0, 401)
    intensity = 2.0 + 0.5 * q

    normalized, area = normalize_profile_by_q_range(q, intensity)

    mask = (q >= 6.7) & (q <= 7.1)
    assert area > 0
    np.testing.assert_allclose(np.trapezoid(normalized[mask], q[mask]), 1.0)


def test_q_range_normalizer_updates_profile_and_adds_scale_columns():
    q = np.linspace(6.0, 8.0, 401)
    intensity = np.ones_like(q) * 4.0
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})

    out = QRangeNormalizer().fit_transform(df)

    assert "radial_profile_data_raw" not in out.columns
    assert "q_range_normalization_area" in out.columns
    np.testing.assert_allclose(
        np.trapezoid(out["radial_profile_data"].iloc[0][(q >= 6.7) & (q <= 7.1)], q[(q >= 6.7) & (q <= 7.1)]),
        1.0,
    )


def test_q_range_normalizer_can_save_initial_profile():
    q = np.linspace(6.0, 8.0, 401)
    intensity = np.ones_like(q) * 4.0
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})

    out = QRangeNormalizer(save_initial_data=True).fit_transform(df)

    assert "radial_profile_data_raw" in out.columns
    np.testing.assert_allclose(out["radial_profile_data_raw"].iloc[0], intensity)
    assert not np.allclose(out["radial_profile_data"].iloc[0], intensity)


def test_q_range_normalizer_can_write_to_custom_output_column():
    q = np.linspace(6.0, 8.0, 401)
    intensity = np.ones_like(q) * 4.0
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})

    out = QRangeNormalizer(output_column="radial_profile_data_norm").fit_transform(df)

    assert "radial_profile_data_norm" in out.columns
    np.testing.assert_allclose(out["radial_profile_data"].iloc[0], intensity)


def test_q_range_normalizer_requires_q_and_profile_columns():
    df = pd.DataFrame({"radial_profile_data": [np.ones(10)]})

    with pytest.raises(ValueError, match="Missing required column\\(s\\): q_range"):
        QRangeNormalizer().transform(df)


def test_q_range_normalizer_raises_when_band_is_missing():
    q = np.linspace(1.0, 2.0, 20)
    intensity = np.ones_like(q)
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})

    with pytest.raises(ValueError, match="has <2 points"):
        QRangeNormalizer().transform(df)


def test_normalize_profile_by_q_range_value_median_sets_window_median_to_one():
    q = np.linspace(6.0, 8.0, 401)
    intensity = 2.0 + 0.5 * q

    normalized, value = normalize_profile_by_q_range_value(q, intensity)

    mask = (q >= 6.7) & (q <= 7.1)
    assert value > 0
    np.testing.assert_allclose(np.median(normalized[mask]), 1.0)


def test_q_range_value_normalizer_updates_profile_and_adds_value_columns():
    q = np.linspace(6.0, 8.0, 401)
    intensity = np.ones_like(q) * 4.0
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})

    out = QRangeValueNormalizer(statistic="median").fit_transform(df)

    assert "q_range_normalization_value" in out.columns
    assert "q_range_normalization_statistic" in out.columns
    assert out["q_range_normalization_value"].iloc[0] == 4.0
    assert out["q_range_normalization_statistic"].iloc[0] == "median"
    np.testing.assert_allclose(out["radial_profile_data"].iloc[0], 1.0)


def test_q_range_value_normalizer_can_save_initial_profile():
    q = np.linspace(6.0, 8.0, 401)
    intensity = np.ones_like(q) * 4.0
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})

    out = QRangeValueNormalizer(save_initial_data=True).fit_transform(df)

    assert "radial_profile_data_raw" in out.columns
    np.testing.assert_allclose(out["radial_profile_data_raw"].iloc[0], intensity)
