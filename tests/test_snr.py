import numpy as np
import pandas as pd
import pytest

from xrd_preprocessing import (
    SNRFilter,
    SNRTransformer,
    calculate_snr,
    snr_filter_statistics,
)
from xrd_preprocessing.numeric import trapz_compat
from xrd_preprocessing.snr import _calculate_poisson_snr


def test_calculate_snr_poisson_default():
    q = np.linspace(0.1, 4.0, 200)
    y = np.sin(q) + 2.0
    sigma = np.ones_like(q) * 0.2
    result = calculate_snr(q, y, sigma=sigma)
    assert result["method"] == "poisson"
    assert np.isfinite(result["snr_db"])


def test_calculate_snr_poisson_exact_db():
    q = np.linspace(0.1, 4.0, 30)
    y = np.ones_like(q) * 10.0
    sigma = np.ones_like(q)

    result = calculate_snr(q, y, sigma=sigma)

    assert result["snr_linear"] == pytest.approx(10.0)
    assert result["snr_db"] == pytest.approx(20.0)


def test_snr_transformer_poisson():
    q = np.linspace(0.1, 4.0, 30)
    y = np.ones_like(q) * 10
    sigma = np.ones_like(q)
    df = pd.DataFrame(
        {"q_range": [q], "radial_profile_data": [y], "radial_profile_sigma": [sigma]}
    )
    out = SNRTransformer(snr_method="poisson").transform(df)
    assert out["snr_method_used"].iloc[0] == "poisson"
    assert out["snr_db"].iloc[0] > 0
    assert "snr" not in out.columns


def test_trapz_compat_manual_fallback(monkeypatch):
    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.delattr(np, "trapz", raising=False)

    assert trapz_compat(np.asarray([1.0]), np.asarray([0.0])) == 0.0
    assert trapz_compat(
        np.asarray([1.0, 3.0, 5.0]),
        np.asarray([0.0, 1.0, 2.0]),
    ) == pytest.approx(6.0)


def test_calculate_poisson_snr_rejects_scalar_and_invalid_sigma():
    with pytest.raises(ValueError, match="sigma must be a 1D array"):
        _calculate_poisson_snr(np.asarray([1.0, 2.0]), np.asarray(1.0))

    with pytest.raises(ValueError, match="finite positive sigma"):
        _calculate_poisson_snr(
            np.asarray([1.0, np.nan, 2.0]),
            np.asarray([0.0, 1.0, np.nan]),
        )


def test_snr_transformer_rejects_scalar_and_short_profiles():
    with pytest.raises(ValueError, match="at least two intensity points"):
        SNRTransformer().transform(
            pd.DataFrame(
                {
                    "radial_profile_data": [1.0],
                    "radial_profile_sigma": [[1.0, 1.0]],
                }
            )
        )
    with pytest.raises(ValueError, match="at least two intensity points"):
        SNRTransformer().transform(
            pd.DataFrame(
                {
                    "radial_profile_data": [[1.0]],
                    "radial_profile_sigma": [[1.0]],
                }
            )
        )


def test_snr_transformer_invalid_method():
    with pytest.raises(ValueError, match="snr_method"):
        SNRTransformer(snr_method="bad")
    with pytest.raises(ValueError, match="snr_method"):
        SNRTransformer(snr_method="auto")


def test_calculate_snr_poisson_missing_sigma():
    q = np.linspace(0.1, 4.0, 30)
    y = np.ones_like(q) * 10
    with pytest.raises(ValueError, match="requires radial_profile_sigma"):
        calculate_snr(q, y, snr_method="poisson")


def test_snr_filter_rejects_drop_false():
    with pytest.raises(ValueError, match="drop=False"):
        SNRFilter(drop=False)


def test_snr_filter_drops_low_snr_rows_by_default():
    df = pd.DataFrame(
        {
            "sample_id": ["bad", "edge", "good", "nan"],
            "snr_db": [19.0, 20.0, 23.0, np.nan],
        }
    )

    filt = SNRFilter()
    out = filt.transform(df)

    assert out["snr_db"].tolist() == [20.0, 23.0]
    assert "snr_pass" not in out.columns
    assert filt.stats_["rows_in"] == 4
    assert filt.stats_["rows_pass"] == 2
    assert filt.stats_["rows_fail"] == 2


def test_snr_filter_statistics_audits_before_and_after_frames():
    df = pd.DataFrame(
        {
            "sample_id": ["bad", "edge", "good", "nan"],
            "snr_db": [19.0, 20.0, 23.0, np.nan],
        }
    )
    out = SNRFilter().transform(df)

    stats = snr_filter_statistics(df, out)

    assert stats["filter_type"] == "snr"
    assert stats["rows_in"] == 4
    assert stats["rows_pass"] == 2
    assert stats["rows_fail"] == 2
    assert stats["min_snr_db_observed"] == 19.0
    assert stats["max_snr_db_observed"] == 23.0
    assert stats["failed_ids"] == ["bad", "nan"]
