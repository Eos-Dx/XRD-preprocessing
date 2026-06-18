import numpy as np
import pandas as pd
import pytest

from xrd_preprocessing import SNRFilter, SNRTransformer, calculate_snr


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
    out = SNRTransformer(method="poisson").transform(df)
    assert out["snr_method_used"].iloc[0] == "poisson"
    assert out["snr_db"].iloc[0] > 0
    assert "snr" not in out.columns


def test_snr_transformer_invalid_method():
    with pytest.raises(ValueError, match="snr_method"):
        SNRTransformer(method="bad")
    with pytest.raises(ValueError, match="snr_method"):
        SNRTransformer(method="residual")
    with pytest.raises(ValueError, match="snr_method"):
        SNRTransformer(method="auto")


def test_calculate_snr_poisson_missing_sigma():
    q = np.linspace(0.1, 4.0, 30)
    y = np.ones_like(q) * 10
    result = calculate_snr(q, y, method="poisson")
    assert result["method"] == "poisson_missing_sigma"
    assert np.isnan(result["snr_db"])


def test_snr_filter_marks_and_keeps_rows_when_drop_false():
    df = pd.DataFrame({"snr_db": [19.0, 20.0, 23.0]})

    out = SNRFilter(drop=False).transform(df)

    assert out["snr_pass"].tolist() == [False, True, True]
    assert out["snr_min_db"].tolist() == [20.0, 20.0, 20.0]
    assert len(out) == 3


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
    assert out["snr_pass"].tolist() == [True, True]
    assert filt.stats_["rows_in"] == 4
    assert filt.stats_["rows_pass"] == 2
    assert filt.stats_["rows_fail"] == 2
    assert filt.stats_["failed_ids"] == ["bad", "nan"]
