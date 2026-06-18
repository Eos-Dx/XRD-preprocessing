import numpy as np
import pandas as pd
import pytest

from xrd_preprocessing import RadialProfileSnapshot


def test_radial_profile_snapshot_saves_q_and_profile_arrays():
    q = np.linspace(2.0, 23.0, 5)
    profile = np.arange(5, dtype=float)
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [profile]})

    out = RadialProfileSnapshot("after_integration").fit_transform(df)

    assert "q_range_after_integration" in out.columns
    assert "radial_profile_data_after_integration" in out.columns
    np.testing.assert_allclose(out["q_range_after_integration"].iloc[0], q)
    np.testing.assert_allclose(
        out["radial_profile_data_after_integration"].iloc[0],
        profile,
    )
    assert out["radial_profile_data_after_integration"].iloc[0] is not profile


def test_radial_profile_snapshot_can_be_disabled():
    df = pd.DataFrame({"q_range": [[1, 2]], "radial_profile_data": [[3, 4]]})

    out = RadialProfileSnapshot("stage", enabled=False).fit_transform(df)

    assert "q_range_stage" not in out.columns
    assert "radial_profile_data_stage" not in out.columns


def test_radial_profile_snapshot_requires_profile_columns():
    df = pd.DataFrame({"q_range": [[1, 2]]})

    with pytest.raises(ValueError, match="radial_profile_data"):
        RadialProfileSnapshot("stage").fit_transform(df)
