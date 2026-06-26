import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/Users/sad/dev/xrd-analysis/src")

from xrd_preprocessing import (  # noqa: E402
    FaultyPixelDetector,
    QRangeNormalizer,
    SNRTransformer,
    perform_azimuthal_integration,
)
try:
    from xrdanalysis.data_processing.azimuthal_integration import (  # noqa: E402
        perform_azimuthal_integration as old_perform_azimuthal_integration,
    )
    from xrdanalysis.data_processing.transformers import (  # noqa: E402
        ColumnNormalizer as OldColumnNormalizer,
        SNRTransformer as OldSNRTransformer,
    )
except ModuleNotFoundError as exc:
    pytestmark = pytest.mark.skip(
        reason=f"xrd-analysis parity dependency unavailable: {exc.name}"
    )


def fake_poni() -> str:
    return """# Fake PONI for parity tests
poni_version: 2.1
Detector: Detector
Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [16, 16], "orientation": 3}
Distance: 0.1
Poni1: 0.0008
Poni2: 0.0008
Rot1: 0
Rot2: 0
Rot3: 0
Wavelength: 1e-10
"""


def fake_image() -> np.ndarray:
    y, x = np.indices((16, 16))
    image = np.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 20.0) + 0.1
    image[1, 1] = 0.0
    image[3, 4] = 5.0
    return image.astype(np.float32)


def test_azimuthal_integration_matches_xrd_analysis_on_fake_poni_image():
    row = pd.Series(
        {
            "measurement_data": fake_image(),
            "ponifile": fake_poni(),
            "interpolation_q_range": None,
            "azimuthal_range": None,
            "sample_thickness_mm": 11.0,
        }
    )
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[1, 1] = 1

    old = old_perform_azimuthal_integration(
        row,
        column="measurement_data",
        npt=32,
        mask=mask,
        mode="1D",
        calibration_mode="poni",
        error_model=None,
    )
    new = perform_azimuthal_integration(
        row,
        column="measurement_data",
        npt=32,
        mask=mask,
        mode="1D",
        calibration_mode="poni",
        error_model=None,
        thickness_reference_mm=11.0,
    )

    np.testing.assert_allclose(new[0], old[0], rtol=0, atol=0)
    np.testing.assert_allclose(new[1], old[1], rtol=0, atol=0)
    assert new[2] is None
    assert old[2] is None
    assert new[3] == old[3]


def test_snr_transformer_matches_xrd_analysis():
    q = np.linspace(0.1, 4.0, 80)
    intensity = np.sin(q * 2.0) + 2.0 + 0.05 * np.cos(q * 9.0)
    sigma = np.full_like(q, 0.2)
    df = pd.DataFrame(
        {
            "q_range": [q],
            "radial_profile_data": [intensity],
            "radial_profile_sigma": [sigma],
        }
    )

    old = OldSNRTransformer(snr_method="poisson").transform(df)
    new = SNRTransformer(snr_method="poisson").transform(df)
    for column in ("noise_std", "snr_linear", "snr_db"):
        np.testing.assert_allclose(new[column], old[column], rtol=0, atol=0)
    assert new["snr_method_used"].tolist() == old["snr_method_used"].tolist()
    assert "snr" not in new.columns
    assert "radial_profile_data_snr" not in new.columns
    assert "radial_profile_residual" not in new.columns


def test_q_range_normalizer_matches_xrd_analysis_integral_mode():
    q = np.linspace(2.0, 23.0, 900)
    intensity = 3.0 + np.sin(q)
    df = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})

    old = OldColumnNormalizer(
        column="radial_profile_data",
        norm="integral",
        q_min=6.7,
        q_max=7.1,
    ).transform(df)
    new = QRangeNormalizer(
        output_column="radial_profile_data",
        q_min=6.7,
        q_max=7.1,
    ).transform(df)

    np.testing.assert_allclose(
        new["radial_profile_data"].iloc[0],
        old["radial_profile_data"].iloc[0],
        rtol=0,
        atol=0,
    )


def test_faulty_pixel_detector_processes_each_row_image_independently():
    images = [np.ones((6, 6), dtype=float) for _ in range(2)]
    images[0][1, 1] = 0.0
    images[0][2, 2] = 501.0
    images[1][3, 3] = 0.0
    images[1][4, 1] = 501.0

    df = pd.DataFrame(
        {
            "measurement_data": images,
            "ponifile": [fake_poni()] * 2,
        }
    )

    out = FaultyPixelDetector(exclude_beam_center_radius=None).fit_transform(df)

    assert [1, 1] not in out["faulty_pixel_mask"].iloc[0].tolist()
    assert [2, 2] in out["faulty_pixel_mask"].iloc[0].tolist()
    assert [3, 3] not in out["faulty_pixel_mask"].iloc[1].tolist()
    assert [4, 1] in out["faulty_pixel_mask"].iloc[1].tolist()
    assert "detector" not in out.columns
