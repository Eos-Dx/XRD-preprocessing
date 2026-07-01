import numpy as np
import pandas as pd
import pytest

from xrd_preprocessing import (
    FaultyPixelDetector,
    create_mask,
    detect_faulty_pixels,
    faulty_pixel_statistics,
)
from xrd_preprocessing.faulty_pixels import (
    _beam_center_pixels,
    _detector_config_float,
    _find_image_column,
)


def test_create_mask_bounds_safe():
    mask = create_mask([(0, 0), (2, 2), (99, 99)], size=(3, 3))
    assert mask.sum() == 2
    assert mask[0, 0] == 1
    assert mask[2, 2] == 1


def test_create_mask_none_and_invalid_image_errors():
    assert create_mask(None) is None

    detector = FaultyPixelDetector()
    try:
        detector.detect(np.asarray([1.0, 2.0]))
    except ValueError as exc:
        assert "2D numeric image" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_detector_config_and_beam_center_invalid_cases():
    assert _detector_config_float("{'pixel1': 0.1}", "pixel1") == 0.1
    assert _detector_config_float("{}", "pixel1") is None
    assert _beam_center_pixels("Poni1: 1\nPoni2: 1\n") is None
    assert (
        _beam_center_pixels(
            "Detector_config: {'pixel1': 0, 'pixel2': 0.1}\nPoni1: 1\nPoni2: 1\n"
        )
        is None
    )


def test_detect_faulty_pixels_marks_invalid_and_bright_values():
    image = np.ones((5, 5))
    image[1, 2] = 501
    image[4, 0] = 0
    image[3, 3] = -1
    pixels = detect_faulty_pixels(image)
    assert (1, 2) in pixels
    assert (4, 0) not in pixels
    assert (3, 3) in pixels


def test_value_above_absolute_limit_is_detected():
    y, x = np.indices((21, 21))
    image = 10.0 + 0.2 * x + 0.1 * y
    image[10, 10] = 501.0

    pixels = detect_faulty_pixels(image)

    assert (10, 10) in pixels
    assert len(pixels) == 1


def test_pixel_below_absolute_guard_is_not_faulty():
    image = np.ones((21, 21))
    image[10, 10] = 500.0

    pixels = detect_faulty_pixels(image)

    assert (10, 10) not in pixels
    assert pixels == set()


def test_faulty_pixel_detector_transform():
    images = [np.ones((4, 4)) for _ in range(3)]
    images[0][2, 1] = 501
    images[1][1, 2] = -1
    images[2][3, 3] = np.nan
    df = pd.DataFrame(
        {
            "measurement_data": images,
        }
    )
    out = FaultyPixelDetector().fit_transform(df)
    stats = faulty_pixel_statistics(out)

    assert out.columns.tolist() == ["measurement_data", "faulty_pixel_mask"]
    assert [2, 1] in out["faulty_pixel_mask"].iloc[0].tolist()
    assert [1, 2] in out["faulty_pixel_mask"].iloc[1].tolist()
    assert [3, 3] in out["faulty_pixel_mask"].iloc[2].tolist()
    assert stats["n_images"] == 3
    assert stats["total_faulty_pixels"] == 3


def test_find_image_column_fallback_and_errors():
    frame = pd.DataFrame(
        {
            "empty": [None],
            "bad": ["not-image"],
            "candidate": [np.ones((2, 2))],
        }
    )
    assert _find_image_column(frame, "missing") == "candidate"

    with pytest.raises(ValueError, match="No 2D numeric image column"):
        _find_image_column(pd.DataFrame({"bad": ["not-image"]}), "missing")


def test_faulty_pixel_detector_requires_configured_image_column():
    with pytest.raises(KeyError, match="measurement_data"):
        FaultyPixelDetector().fit_transform(pd.DataFrame({"candidate": [np.ones((2, 2))]}))


def test_grouped_faulty_pixels_are_all_detected():
    image = np.ones((20, 20))
    image[5:8, 5:8] = 501.0

    pixels = detect_faulty_pixels(image)

    assert len(pixels) == 9
    assert (5, 5) in pixels
    assert (7, 7) in pixels


def test_low_background_noise_is_not_mass_faulty_pixels():
    rng = np.random.default_rng(42)
    image = rng.normal(0.05, 0.03, size=(80, 80))
    image[20, 20] = 501.0

    out = FaultyPixelDetector(detect_negative_pixels=False).fit_transform(
        pd.DataFrame({"measurement_data": [image]})
    )
    pixels = {tuple(pixel) for pixel in out["faulty_pixel_mask"].iloc[0]}

    assert (20, 20) in pixels
    assert len(pixels) < 10


def test_faulty_pixel_detector_is_row_wise():
    images = [np.ones((4, 4)) for _ in range(2)]
    images[0][1, 1] = -1
    images[1][2, 2] = -1

    df = pd.DataFrame(
        {
            "measurement_data": images,
        }
    )
    out = FaultyPixelDetector().fit_transform(df)

    assert [1, 1] in out["faulty_pixel_mask"].iloc[0].tolist()
    assert [2, 2] in out["faulty_pixel_mask"].iloc[1].tolist()
    assert [1, 1] not in out["faulty_pixel_mask"].iloc[1].tolist()


def test_zero_pixels_are_not_faulty_by_default():
    image = np.ones((5, 5))
    image[2, 2] = 0
    out = FaultyPixelDetector().fit_transform(pd.DataFrame({"measurement_data": [image]}))
    assert out["faulty_pixel_mask"].iloc[0].tolist() == []


def test_faulty_pixel_detector_flags_can_disable_detection_rules():
    image = np.ones((4, 4))
    image[1, 1] = -1
    image[2, 2] = 501

    detector = FaultyPixelDetector(
        detect_negative_pixels=False,
        detect_bright_pixels=False,
    )
    assert detector.detect(image) == set()

    out = detector.fit_transform(pd.DataFrame({"measurement_data": [image]}))
    assert out["faulty_pixel_mask"].iloc[0].tolist() == []


def test_beam_center_exclusion_requires_explicit_opt_in():
    image = np.ones((9, 9))
    image[4, 4] = -1
    image[1, 1] = -1
    poni = """Detector_config: {"pixel1": 0.1, "pixel2": 0.1}
Poni1: 0.4
Poni2: 0.4
"""
    default = FaultyPixelDetector().fit_transform(
        pd.DataFrame({"measurement_data": [image], "ponifile": [poni]})
    )
    out = FaultyPixelDetector(exclude_beam_center_radius=0.2).fit_transform(
        pd.DataFrame({"measurement_data": [image], "ponifile": [poni]})
    )

    assert [4, 4] in default["faulty_pixel_mask"].iloc[0].tolist()
    assert [4, 4] not in out["faulty_pixel_mask"].iloc[0].tolist()
    assert [1, 1] in out["faulty_pixel_mask"].iloc[0].tolist()


def test_faulty_pixel_detector_can_use_custom_output_column():
    image = np.ones((4, 4))
    image[1, 1] = -1

    out = FaultyPixelDetector(mask_column="mask").fit_transform(
        pd.DataFrame({"measurement_data": [image]})
    )

    assert "faulty_pixel_mask" not in out.columns
    assert [1, 1] in out["mask"].iloc[0].tolist()
    assert faulty_pixel_statistics(out, mask_column="mask")["total_faulty_pixels"] == 1
