import numpy as np
import pandas as pd

from xrd_preprocessing import (
    FAULTY_REASON_NEGATIVE,
    FAULTY_REASON_NONFINITE,
    FAULTY_REASON_SATURATED,
    FaultyPixelDetector,
    HotPixelDetector,
    create_faulty_pixel_reason_map,
    create_mask,
    detect_faulty_pixels,
)


def test_create_mask_bounds_safe():
    mask = create_mask([(0, 0), (2, 2), (99, 99)], size=(3, 3))
    assert mask.sum() == 2
    assert mask[0, 0] == 1
    assert mask[2, 2] == 1


def test_create_faulty_pixel_reason_map():
    image = np.ones((4, 4))
    image[0, 0] = -1
    image[1, 1] = np.nan
    image[2, 2] = 500

    reason_map = create_faulty_pixel_reason_map(image, hot_pixels=[(2, 2)])

    assert reason_map[0, 0] == FAULTY_REASON_NEGATIVE
    assert reason_map[1, 1] == FAULTY_REASON_NONFINITE
    assert reason_map[2, 2] == FAULTY_REASON_SATURATED
    assert reason_map[3, 3] == 0


def test_detect_faulty_pixels_marks_invalid_and_hot_values():
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
    assert "faulty_pixel_mask" in out.columns
    assert "invalid_pixel_mask" in out.columns
    assert "suspected_hot_pixel_mask" in out.columns
    assert "pyfai_faulty_pixel_mask" in out.columns
    assert "faulty_pixel_reason_map" in out.columns
    assert "faulty_pixel_reason_counts" in out.columns
    assert [2, 1] in out["faulty_pixel_mask"].iloc[0].tolist()
    assert [2, 1] in out["suspected_hot_pixel_mask"].iloc[0].tolist()
    assert [1, 2] in out["faulty_pixel_mask"].iloc[1].tolist()
    assert [1, 2] in out["invalid_pixel_mask"].iloc[1].tolist()
    assert [3, 3] in out["faulty_pixel_mask"].iloc[2].tolist()
    assert [3, 3] in out["invalid_pixel_mask"].iloc[2].tolist()
    assert out["pyfai_faulty_pixel_mask"].iloc[0][2, 1] == 1
    assert out["faulty_pixel_reason_map"].iloc[0][2, 1] == FAULTY_REASON_SATURATED
    assert out["faulty_pixel_reason_map"].iloc[2][3, 3] == FAULTY_REASON_NONFINITE
    assert out["faulty_pixel_reason_counts"].iloc[2]["nan_or_inf"] == 1


def test_faulty_pixel_detector_reason_map_marks_negative_and_saturated():
    image = np.ones((5, 5))
    image[1, 1] = -0.5
    image[2, 2] = np.nan
    image[3, 3] = 501.0

    out = FaultyPixelDetector().fit_transform(
        pd.DataFrame({"measurement_data": [image]})
    )
    reason_map = out["faulty_pixel_reason_map"].iloc[0]

    assert reason_map[1, 1] == FAULTY_REASON_NEGATIVE
    assert reason_map[2, 2] == FAULTY_REASON_NONFINITE
    assert reason_map[3, 3] == FAULTY_REASON_SATURATED


def test_grouped_faulty_pixels_are_all_detected():
    image = np.ones((20, 20))
    image[5:8, 5:8] = 501.0

    pixels = detect_faulty_pixels(image)

    assert len(pixels) == 9
    assert (5, 5) in pixels
    assert (7, 7) in pixels


def test_low_background_noise_is_not_mass_hot_pixels():
    rng = np.random.default_rng(42)
    image = rng.normal(0.05, 0.03, size=(80, 80))
    image[20, 20] = 501.0

    out = FaultyPixelDetector().fit_transform(
        pd.DataFrame({"measurement_data": [image]})
    )
    pixels = {tuple(pixel) for pixel in out["suspected_hot_pixel_mask"].iloc[0]}

    assert (20, 20) in pixels
    assert len(pixels) < 10


def test_legacy_hot_pixel_detector_alias_still_works():
    image = np.ones((4, 4))
    image[1, 1] = 501.0

    out = HotPixelDetector().fit_transform(pd.DataFrame({"measurement_data": [image]}))

    assert [1, 1] in out["faulty_pixel_mask"].iloc[0].tolist()


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


def test_beam_center_can_be_excluded_from_poni():
    image = np.ones((9, 9))
    image[4, 4] = -1
    image[1, 1] = -1
    poni = """Detector_config: {"pixel1": 0.1, "pixel2": 0.1}
Poni1: 0.4
Poni2: 0.4
"""
    out = FaultyPixelDetector(exclude_beam_center_radius=0.2).fit_transform(
        pd.DataFrame({"measurement_data": [image], "ponifile": [poni]})
    )

    assert [4, 4] not in out["faulty_pixel_mask"].iloc[0].tolist()
    assert [1, 1] in out["faulty_pixel_mask"].iloc[0].tolist()
    assert out["faulty_pixel_reason_map"].iloc[0][4, 4] == 0
    assert out["faulty_pixel_reason_map"].iloc[0][1, 1] == FAULTY_REASON_NEGATIVE
    assert out["faulty_pixel_reason_counts"].iloc[0]["negative"] == 1
