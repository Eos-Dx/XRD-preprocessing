"""Minimal faulty-pixel detection for GFRM detector frames."""

from __future__ import annotations

import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin

Pixel = tuple[int, int]

FAULTY_REASON_OK = 0
FAULTY_REASON_NEGATIVE = -1
FAULTY_REASON_NONFINITE = -2
FAULTY_REASON_SATURATED = -3

FAULTY_REASON_CODES = {
    FAULTY_REASON_NEGATIVE: "negative",
    FAULTY_REASON_NONFINITE: "nan_or_inf",
    FAULTY_REASON_SATURATED: "saturated_or_hot",
}


def _as_2d_image(value, context: str = "image") -> np.ndarray:
    image = np.asarray(value, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"{context} must be a 2D numeric image.")
    return image


def _pixel_array(pixels: set[Pixel]) -> np.ndarray:
    return np.array(sorted(pixels), dtype=int) if pixels else np.empty((0, 2), dtype=int)


def create_mask(
    faulty_pixels: Iterable[Pixel] | np.ndarray | None,
    size: tuple[int, int] = (256, 256),
) -> np.ndarray | None:
    """Create a pyFAI mask: 1 excludes a pixel, 0 keeps it."""
    if faulty_pixels is None:
        return None

    mask = np.zeros(size, dtype=np.uint8)
    for y, x in np.asarray(list(faulty_pixels), dtype=int).reshape(-1, 2):
        if 0 <= y < size[0] and 0 <= x < size[1]:
            mask[y, x] = 1
    return mask


def create_faulty_pixel_reason_map(
    image: np.ndarray,
    *,
    dead_pixels: Iterable[Pixel] | np.ndarray | None = None,
    hot_pixels: Iterable[Pixel] | np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
    negative_threshold: float = 0.0,
) -> np.ndarray:
    """Create reason map: 0 OK, -1 negative, -2 NaN/inf, -3 hot."""
    image = _as_2d_image(image)
    reason_map = np.zeros(image.shape, dtype=np.int16)

    if hot_pixels is not None:
        for y, x in np.asarray(list(hot_pixels), dtype=int).reshape(-1, 2):
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                reason_map[y, x] = FAULTY_REASON_SATURATED

    if dead_pixels is not None:
        for y, x in np.asarray(list(dead_pixels), dtype=int).reshape(-1, 2):
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                reason_map[y, x] = FAULTY_REASON_NEGATIVE

    reason_map[image < negative_threshold] = FAULTY_REASON_NEGATIVE
    reason_map[~np.isfinite(image)] = FAULTY_REASON_NONFINITE
    if exclude_mask is not None:
        reason_map[np.asarray(exclude_mask, dtype=bool)] = FAULTY_REASON_OK
    return reason_map


def count_faulty_pixel_reasons(reason_map: np.ndarray) -> dict[str, int]:
    """Count non-OK faulty-pixel reason codes in one reason map."""
    reason_map = np.asarray(reason_map)
    return {
        name: int(np.sum(reason_map == code))
        for code, name in FAULTY_REASON_CODES.items()
    }


def _find_image_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        _as_2d_image(df[preferred].dropna().iloc[0], preferred)
        return preferred

    for column in df.columns:
        series = df[column].dropna()
        if not len(series):
            continue
        try:
            if np.asarray(series.iloc[0]).ndim == 2:
                _as_2d_image(series.iloc[0], column)
                return column
        except (TypeError, ValueError):
            continue
    raise ValueError(f"No 2D numeric image column found in {list(df.columns)!r}.")


def _float_field(text: str, key: str) -> float | None:
    match = re.search(rf"^{key}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _beam_center_pixels(poni_text: str) -> tuple[int, int] | None:
    poni1 = _float_field(poni_text, "Poni1")
    poni2 = _float_field(poni_text, "Poni2")
    match = re.search(r"^Detector_config:\s*(.+)$", poni_text, re.MULTILINE)
    if poni1 is None or poni2 is None or match is None:
        return None
    try:
        config = json.loads(match.group(1).replace("'", '"'))
        pixel1 = float(config["pixel1"])
        pixel2 = float(config["pixel2"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if pixel1 <= 0 or pixel2 <= 0:
        return None
    return int(round(poni1 / pixel1)), int(round(poni2 / pixel2))


def _beam_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius_fraction: float,
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    center_y, center_x = center
    radius = float(radius_fraction) * max(shape)
    return np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2) <= radius


def detect_faulty_pixels(image: np.ndarray, **kwargs) -> set[Pixel]:
    """Return invalid and suspected-hot pixel coordinates for one image."""
    return FaultyPixelDetector(**kwargs).detect(image)


def detect_hot_pixels(image: np.ndarray, **kwargs) -> set[Pixel]:
    """Compatibility alias for detect_faulty_pixels."""
    return detect_faulty_pixels(image, **kwargs)


class FaultyPixelDetector(TransformerMixin, BaseEstimator):
    """Frame-local detector for GFRM preprocessing.

    The detector is intentionally simple. It does not estimate a permanent
    detector map from one image. It only marks pixels that are invalid for the
    current frame or high enough to create spikes after azimuthal integration.
    """

    def __init__(
        self,
        image_column: str = "measurement_data",
        negative_threshold: float = 0.0,
        detect_negative_pixels: bool = True,
        detect_local_hot_pixels: bool = True,
        local_hot_min_value: float = 500.0,
        exclude_beam_center_radius: float | None = 0.04,
        poni_column: str = "ponifile",
    ):
        self.image_column = image_column
        self.negative_threshold = float(negative_threshold)
        self.detect_negative_pixels = bool(detect_negative_pixels)
        self.detect_local_hot_pixels = bool(detect_local_hot_pixels)
        self.local_hot_min_value = float(local_hot_min_value)
        self.exclude_beam_center_radius = exclude_beam_center_radius
        self.poni_column = poni_column

    def fit(self, df: pd.DataFrame, y=None):
        _ = df
        _ = y
        self.is_fitted_ = True
        return self

    def _get_beam_center_pixels(self, poni_text: str) -> tuple[int, int] | None:
        return _beam_center_pixels(str(poni_text))

    def _invalid_pixels(self, image: np.ndarray) -> set[Pixel]:
        mask = ~np.isfinite(image)
        if self.detect_negative_pixels:
            mask |= image < self.negative_threshold
        return {(int(y), int(x)) for y, x in zip(*np.where(mask))}

    def _suspected_hot_pixels(self, image: np.ndarray) -> set[Pixel]:
        if not self.detect_local_hot_pixels:
            return set()

        hot = (
            np.isfinite(image)
            & (image >= self.negative_threshold)
            & (image > self.local_hot_min_value)
        )
        return {(int(y), int(x)) for y, x in zip(*np.where(hot))}

    def detect_by_type(
        self,
        image: np.ndarray,
        exclude_mask: np.ndarray | None = None,
    ) -> dict[str, set[Pixel]]:
        image = _as_2d_image(image)
        invalid = self._invalid_pixels(image)
        hot = self._suspected_hot_pixels(image) - invalid
        if exclude_mask is not None:
            invalid = {(y, x) for y, x in invalid if not exclude_mask[y, x]}
            hot = {(y, x) for y, x in hot if not exclude_mask[y, x]}
        return {
            "invalid": invalid,
            "hot": hot,
        }

    def detect(self, image: np.ndarray) -> set[Pixel]:
        typed = self.detect_by_type(image)
        return typed["invalid"] | typed["hot"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        image_column = _find_image_column(out, self.image_column)

        faulty_masks: list[np.ndarray] = []
        pyfai_masks: list[np.ndarray] = []
        invalid_masks: list[np.ndarray] = []
        hot_masks: list[np.ndarray] = []
        reason_maps: list[np.ndarray] = []
        reason_counts: list[dict[str, int]] = []

        for _, row in out.iterrows():
            image = _as_2d_image(row[image_column], image_column)
            exclude_mask = None
            if (
                self.exclude_beam_center_radius is not None
                and self.exclude_beam_center_radius > 0
                and self.poni_column in out.columns
            ):
                center = _beam_center_pixels(str(row[self.poni_column]))
                if center is not None:
                    exclude_mask = _beam_mask(
                        image.shape,
                        center,
                        self.exclude_beam_center_radius,
                    )
            typed = self.detect_by_type(image, exclude_mask=exclude_mask)
            invalid = typed["invalid"]
            hot = typed["hot"]
            faulty = invalid | hot

            faulty_mask = _pixel_array(faulty)
            invalid_mask = _pixel_array(invalid)
            hot_mask = _pixel_array(hot)
            reason_map = create_faulty_pixel_reason_map(
                image,
                dead_pixels=invalid_mask,
                hot_pixels=hot_mask,
                exclude_mask=exclude_mask,
                negative_threshold=self.negative_threshold,
            )

            faulty_masks.append(faulty_mask)
            pyfai_masks.append(create_mask(faulty_mask, image.shape))
            invalid_masks.append(invalid_mask)
            hot_masks.append(hot_mask)
            reason_maps.append(reason_map)
            reason_counts.append(count_faulty_pixel_reasons(reason_map))

        out["faulty_pixel_mask"] = faulty_masks
        out["pyfai_faulty_pixel_mask"] = pyfai_masks
        out["invalid_pixel_mask"] = invalid_masks
        out["suspected_hot_pixel_mask"] = hot_masks
        out["faulty_pixel_reason_map"] = reason_maps
        out["faulty_pixel_reason_counts"] = reason_counts

        self.stats_ = {
            "image_column": image_column,
            "n_images": int(len(out)),
            "faulty_pixels_per_row": [int(len(mask)) for mask in faulty_masks],
            "invalid_pixels_per_row": [int(len(mask)) for mask in invalid_masks],
            "suspected_hot_pixels_per_row": [int(len(mask)) for mask in hot_masks],
            "total_faulty_pixels": int(sum(len(mask) for mask in faulty_masks)),
            "total_invalid_pixels": int(sum(len(mask) for mask in invalid_masks)),
            "total_suspected_hot_pixels": int(sum(len(mask) for mask in hot_masks)),
            "detect_negative_pixels": self.detect_negative_pixels,
            "detect_local_hot_pixels": self.detect_local_hot_pixels,
            "hot_pixel_rule": (
                f"finite and >= {self.negative_threshold:g} "
                f"and > {self.local_hot_min_value:g}"
            ),
            "local_hot_min_value": self.local_hot_min_value,
            "beam_center_excluded": self.exclude_beam_center_radius is not None,
            "beam_center_radius_frac": (
                self.exclude_beam_center_radius
                if self.exclude_beam_center_radius is not None
                else 0
            ),
        }
        return out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        _ = y
        return self.transform(df)


HotPixelDetector = FaultyPixelDetector
