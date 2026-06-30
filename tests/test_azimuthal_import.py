import numpy as np
import pandas as pd
import pytest

from xrd_preprocessing import AzimuthalIntegration, perform_azimuthal_integration
import xrd_preprocessing.azimuthal as azimuthal_module


def fake_poni() -> str:
    return """# Fake PONI for tests
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
    return (np.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 20.0) + 0.1).astype(np.float32)


def test_azimuthal_integration_default_npt_is_100():
    integrator = AzimuthalIntegration()

    assert integrator.npt == 100


def test_perform_azimuthal_integration_fake_poni_image():
    row = pd.Series(
        {
            "measurement_data": fake_image(),
            "ponifile": fake_poni(),
            "interpolation_q_range": None,
            "azimuthal_range": None,
            "sample_thickness_mm": 11.0,
        }
    )
    radial, intensity, sigma, distance = perform_azimuthal_integration(
        row,
        npt=32,
        calibration_mode="poni",
        thickness_reference_mm=11.0,
    )

    assert radial.shape == (32,)
    assert intensity.shape == (32,)
    assert sigma is None
    assert distance == 0.1
    assert np.isfinite(intensity).all()


def test_azimuthal_integration_transform_fake_poni_image():
    df = pd.DataFrame(
        {
            "measurement_data": [fake_image()],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
            "sample_thickness_mm": [11.0],
        }
    )
    out = AzimuthalIntegration(
        npt=32,
        calibration_mode="poni",
        thickness_reference_mm=11.0,
    ).fit_transform(df)

    assert out["q_range"].iloc[0].shape == (32,)
    assert out["radial_profile_data"].iloc[0].shape == (32,)
    assert out["radial_profile_sigma"].iloc[0] is None
    assert out["calculated_distance"].iloc[0] == 0.1
    assert bool(out["thickness_adjustment_applied"].iloc[0]) is True
    assert bool(out["thickness_adjustment_reliable"].iloc[0]) is True


def test_azimuthal_integration_uses_row_mask_column():
    image = fake_image()
    df = pd.DataFrame(
        {
            "measurement_data": [image],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
            "faulty_pixel_mask": [np.asarray([[8, 8]])],
            "sample_thickness_mm": [11.0],
        }
    )

    out = AzimuthalIntegration(
        npt=32,
        calibration_mode="poni",
        mask_column="faulty_pixel_mask",
        thickness_reference_mm=11.0,
    ).fit_transform(df)

    assert out["q_range"].iloc[0].shape == (32,)
    assert np.isfinite(out["radial_profile_data"].iloc[0]).all()
    assert out["azimuthal_mask_source"].iloc[0] == "faulty_pixel_mask"
    assert out["azimuthal_mask_pixels"].iloc[0] == 1


def test_azimuthal_integration_accepts_faulty_pixel_coordinate_mask():
    image = fake_image()
    df = pd.DataFrame(
        {
            "measurement_data": [image],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
            "faulty_pixel_mask": [np.asarray([[8, 8], [8, 9]])],
            "sample_thickness_mm": [11.0],
        }
    )

    out = AzimuthalIntegration(
        npt=32,
        calibration_mode="poni",
        mask_column="faulty_pixel_mask",
        thickness_reference_mm=11.0,
    ).fit_transform(df)

    assert out["q_range"].iloc[0].shape == (32,)
    assert out["azimuthal_mask_source"].iloc[0] == "faulty_pixel_mask"
    assert out["azimuthal_mask_pixels"].iloc[0] == 2


def test_azimuthal_integration_keeps_per_row_mask_counts():
    image = fake_image()
    masks = [np.asarray([[8, 8]]), np.asarray([[8, 8], [8, 9]])]
    df = pd.DataFrame(
        {
            "measurement_data": [image, image],
            "ponifile": [fake_poni(), fake_poni()],
            "interpolation_q_range": [None, None],
            "azimuthal_range": [None, None],
            "faulty_pixel_mask": masks,
            "sample_thickness_mm": [11.0, 11.0],
        }
    )

    out = AzimuthalIntegration(
        npt=32,
        calibration_mode="poni",
        mask_column="faulty_pixel_mask",
        thickness_reference_mm=11.0,
    ).fit_transform(df)

    assert out["azimuthal_mask_pixels"].tolist() == [1, 2]


def test_azimuthal_integration_missing_thickness_raises():
    df = pd.DataFrame(
        {
            "measurement_data": [fake_image()],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
        }
    )

    with pytest.raises(ValueError, match="thickness_reference_mm must be set explicitly"):
        AzimuthalIntegration(npt=32, calibration_mode="poni").fit_transform(df)


def test_azimuthal_integration_missing_sample_thickness_raises():
    df = pd.DataFrame(
        {
            "measurement_data": [fake_image()],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
        }
    )

    with pytest.raises(ValueError, match="Missing required thickness column: sample_thickness_mm"):
        AzimuthalIntegration(
            npt=32,
            calibration_mode="poni",
            thickness_reference_mm=11.0,
        ).fit_transform(df)


def test_azimuthal_integration_applies_thickness_to_poni_distance():
    df = pd.DataFrame(
        {
            "measurement_data": [fake_image()],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
            "sample_thickness_mm": [25.0],
        }
    )

    out = AzimuthalIntegration(
        npt=32,
        calibration_mode="poni",
        thickness_reference_mm=11.0,
    ).fit_transform(df)

    assert bool(out["thickness_adjustment_applied"].iloc[0]) is True
    assert bool(out["thickness_adjustment_reliable"].iloc[0]) is True
    assert out["sample_thickness_mm"].iloc[0] == 25.0
    assert out["thickness_reference_mm"].iloc[0] == 11.0
    assert out["calculated_distance"].iloc[0] == pytest.approx(0.093)


def test_azimuthal_integration_can_use_row_reference_thickness_column():
    df = pd.DataFrame(
        {
            "measurement_data": [fake_image()],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
            "sample_thickness_mm": [25.0],
            "calibrant_thickness_mm": [15.0],
        }
    )

    out = AzimuthalIntegration(
        npt=32,
        calibration_mode="poni",
        thickness_reference_column="calibrant_thickness_mm",
    ).fit_transform(df)

    assert out["sample_thickness_mm"].iloc[0] == 25.0
    assert out["thickness_reference_mm"].iloc[0] == 15.0
    assert out["thickness_reference_source"].iloc[0] == "calibrant_thickness_mm"
    assert out["calculated_distance"].iloc[0] == pytest.approx(0.095)


def test_azimuthal_integration_accepts_sample_and_reference_sequences():
    image = fake_image()
    df = pd.DataFrame(
        {
            "measurement_data": [image, image],
            "ponifile": [fake_poni(), fake_poni()],
            "interpolation_q_range": [None, None],
            "azimuthal_range": [None, None],
        }
    )

    out = AzimuthalIntegration(
        npt=32,
        calibration_mode="poni",
        sample_thickness_mm=[25.0, 10.0],
        thickness_reference_mm=[15.0, 40.0],
    ).fit_transform(df)

    assert out["sample_thickness_mm"].tolist() == [25.0, 10.0]
    assert out["thickness_reference_mm"].tolist() == [15.0, 40.0]
    assert out["calculated_distance"].tolist() == pytest.approx([0.095, 0.115])


def test_azimuthal_integration_reference_thickness_column_missing_raises():
    df = pd.DataFrame(
        {
            "measurement_data": [fake_image()],
            "ponifile": [fake_poni()],
            "interpolation_q_range": [None],
            "azimuthal_range": [None],
            "sample_thickness_mm": [25.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required thickness reference column"):
        AzimuthalIntegration(
            npt=32,
            calibration_mode="poni",
            thickness_reference_column="calibrant_thickness_mm",
        ).fit_transform(df)


def test_row_values_validation_branches():
    assert azimuthal_module._row_values(None, n_rows=2, name="x") is None
    assert azimuthal_module._row_values(1.5, n_rows=2, name="x") == [1.5, 1.5]
    assert azimuthal_module._row_values([1, 2], n_rows=2, name="x") == [1.0, 2.0]
    with pytest.raises(TypeError, match="numeric"):
        azimuthal_module._row_values("bad", n_rows=2, name="x")
    with pytest.raises(ValueError, match="finite"):
        azimuthal_module._row_values(np.nan, n_rows=2, name="x")
    with pytest.raises(ValueError, match="one-dimensional"):
        azimuthal_module._row_values([[1]], n_rows=1, name="x")
    with pytest.raises(ValueError, match="length"):
        azimuthal_module._row_values([1], n_rows=2, name="x")
    with pytest.raises(ValueError, match="finite"):
        azimuthal_module._row_values([1, np.nan], n_rows=2, name="x")


def test_adjust_poni_distance_handles_missing_distance():
    text, adjusted = azimuthal_module._adjust_poni_distance(
        "Poni1: 0\n",
        sample_thickness_mm=12,
        reference_thickness_mm=10,
    )

    assert text == "Poni1: 0\n"
    assert adjusted is None


def test_resolve_thickness_error_and_warning_branches():
    row = pd.Series({"sample_thickness_mm": 11.0, "calibrant_thickness_mm": 10.0})

    with pytest.raises(ValueError, match="disabled"):
        azimuthal_module._resolve_thickness(
            row,
            thickness_adjustment=False,
            require_thickness_adjustment=True,
            sample_thickness_column="sample_thickness_mm",
            thickness_reference_mm=10.0,
            thickness_reference_column=None,
        )
    with pytest.raises(ValueError, match="thickness_reference_mm"):
        azimuthal_module._resolve_thickness(
            row,
            thickness_adjustment=True,
            require_thickness_adjustment=True,
            sample_thickness_column="sample_thickness_mm",
            thickness_reference_mm=None,
            thickness_reference_column=None,
        )
    with pytest.raises(ValueError, match="Missing required thickness reference"):
        azimuthal_module._resolve_thickness(
            row,
            thickness_adjustment=True,
            require_thickness_adjustment=True,
            sample_thickness_column="sample_thickness_mm",
            thickness_reference_mm=None,
            thickness_reference_column="missing",
        )
    with pytest.raises(ValueError, match="Invalid thickness reference"):
        azimuthal_module._resolve_thickness(
            pd.Series({"sample_thickness_mm": 11.0, "calibrant_thickness_mm": np.nan}),
            thickness_adjustment=True,
            require_thickness_adjustment=True,
            sample_thickness_column="sample_thickness_mm",
            thickness_reference_mm=None,
            thickness_reference_column="calibrant_thickness_mm",
        )

    with pytest.warns(RuntimeWarning, match="Missing required thickness column"):
        sample, meta = azimuthal_module._resolve_thickness(
            row,
            thickness_adjustment=True,
            require_thickness_adjustment=False,
            sample_thickness_column="missing",
            thickness_reference_mm=10.0,
            thickness_reference_column=None,
        )
    assert sample is None
    assert meta["thickness_adjustment_reliable"] is False
