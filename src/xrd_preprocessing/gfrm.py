"""Bruker GFRM archive extraction and photon-conversion utilities."""

from __future__ import annotations

from pathlib import Path
import tarfile
from typing import Any

import numpy as np


def extract_gfrm_archive(
    archive_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Extract a tar archive containing GFRM files and return extraction root."""
    archive_path = Path(archive_path)
    if output_dir is None:
        output_dir = archive_path.with_suffix("").with_suffix("")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            target_path = (output_dir / member.name).resolve()
            if not target_path.is_relative_to(output_dir.resolve()):
                raise ValueError(f"Unsafe archive member path: {member.name!r}")
        archive.extractall(output_dir, filter="data")
    return output_dir


def parse_gfrm_header(gfrm_path: str | Path) -> dict[str, str]:
    """Parse Bruker GFRM 80-character header records."""
    gfrm_path = Path(gfrm_path)
    with gfrm_path.open("rb") as handle:
        first_block = handle.read(512)

    def field(buffer: bytes, key: str) -> str | None:
        for idx in range(0, len(buffer), 80):
            chunk = buffer[idx : idx + 80].decode("latin-1")
            if chunk.startswith(key):
                return chunk.split(":", 1)[1].strip()
        return None

    header_blocks = int(field(first_block, "HDRBLKS") or 15)
    with gfrm_path.open("rb") as handle:
        header_bytes = handle.read(512 * header_blocks)

    header: dict[str, str] = {}
    for idx in range(0, len(header_bytes), 80):
        chunk = header_bytes[idx : idx + 80].decode("latin-1")
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        key = key.strip()
        if key and key not in header:
            header[key] = value.strip()
    return header


def _first_number(value: Any, cast: type = int) -> Any:
    if value is None:
        return None
    try:
        return cast(str(value).split()[0])
    except (IndexError, ValueError, TypeError):
        return None


def parse_bruker_header_preview(gfrm_path: str | Path) -> dict[str, Any]:
    """Read lightweight Bruker frame metadata without decoding image bytes."""
    header = parse_gfrm_header(gfrm_path)
    return {
        "format": _first_number(header.get("FORMAT"), int),
        "version": _first_number(header.get("VERSION"), int),
        "hdrblks": _first_number(header.get("HDRBLKS"), int),
        "nrows": _first_number(header.get("NROWS"), int),
        "ncols": _first_number(header.get("NCOLS"), int),
        "npixelb": header.get("NPIXELB"),
        "noverfl": header.get("NOVERFL"),
        "linear": header.get("LINEAR"),
        "header": header,
    }


def _parse_eos_photon_metadata(
    header: dict[str, Any],
    gfrm_path: str | Path | None = None,
) -> dict[str, Any]:
    source = f" from {gfrm_path!s}" if gfrm_path is not None else ""
    try:
        nexp = [float(value) for value in str(header["NEXP"]).split()]
    except KeyError as exc:
        raise ValueError(f"Missing NEXP header{source}.") from exc
    except ValueError as exc:
        raise ValueError(f"Cannot parse NEXP header{source}: {header.get('NEXP')!r}") from exc
    if len(nexp) < 3:
        raise ValueError(f"NEXP must contain at least 3 values{source}: {header['NEXP']}")

    try:
        ccdparm = [float(value) for value in str(header["CCDPARM"]).split()]
    except KeyError as exc:
        raise ValueError(f"Missing CCDPARM header{source}.") from exc
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse CCDPARM header{source}: {header.get('CCDPARM')!r}"
        ) from exc
    if len(ccdparm) < 5:
        raise ValueError(
            f"CCDPARM must contain 5 values{source}: {header['CCDPARM']}"
        )

    readnoise, e_per_adu, e_per_photon, bias, full_scale = ccdparm[:5]
    if e_per_adu <= 0:
        raise ValueError(f"Invalid e_per_ADU{source}: {e_per_adu}")
    if e_per_photon <= 0:
        raise ValueError(f"Invalid e_per_photon{source}: {e_per_photon}")

    baseline_adu = float(nexp[2])
    gain_adu_per_photon = float(e_per_photon / e_per_adu)
    return {
        "baseline_adu": baseline_adu,
        "NEXP_raw": header.get("NEXP"),
        "CCDPARM_raw": header.get("CCDPARM"),
        "readnoise": float(readnoise),
        "e_per_ADU": float(e_per_adu),
        "e_per_photon": float(e_per_photon),
        "bias": float(bias),
        "full_scale": float(full_scale),
        "gain_adu_per_photon": gain_adu_per_photon,
        "photon_formula": (
            "photons = (adu - baseline_adu) / (e_per_photon / e_per_ADU)"
        ),
        "negative_values_preserved": True,
        "warning": (
            "EOS-normalized estimated photons. "
            "Not direct photon-counting detector events."
        ),
    }


def gfrm_conversion_metadata(gfrm_path: str | Path) -> dict[str, Any]:
    """Return header-derived EOS photon-conversion constants for GFRM frames."""
    header = parse_gfrm_header(gfrm_path)
    return {**_parse_eos_photon_metadata(header, gfrm_path), "header": header}


def read_gfrm_with_fabio(gfrm_path: str | Path):
    """Decode a Bruker GFRM frame with FabIO.

    FabIO owns the detector-frame decoding: FORMAT 86/100, compression,
    underflow/overflow tables, and Bruker raster order. This function does not
    parse image bytes manually.
    """
    gfrm_path = Path(gfrm_path)
    try:
        import fabio
    except Exception as exc:
        raise ImportError("fabio is required to read GFRM files") from exc

    try:
        return fabio.open(str(gfrm_path))
    except Exception as open_error:
        preview = parse_bruker_header_preview(gfrm_path)
        frame_format = preview.get("format")
        try:
            if frame_format == 100:
                from fabio.bruker100image import Bruker100Image

                return Bruker100Image().read(str(gfrm_path))
            if frame_format == 86:
                from fabio.brukerimage import BrukerImage

                return BrukerImage().read(str(gfrm_path))
        except Exception as fallback_error:
            raise ValueError(
                f"FabIO failed to decode Bruker FORMAT {frame_format} "
                f"frame {gfrm_path!s}."
            ) from fallback_error
        raise ValueError(
            f"FabIO failed to decode unsupported Bruker FORMAT {frame_format} "
            f"frame {gfrm_path!s}."
        ) from open_error


def validate_gfrm_array(
    array: np.ndarray,
    metadata: dict[str, Any],
    gfrm_path: str | Path,
) -> list[str]:
    """Validate decoded GFRM array and return non-fatal warnings."""
    path = Path(gfrm_path)
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Decoded GFRM data is not a NumPy array: {path!s}")
    if array.ndim != 2:
        raise ValueError(f"Decoded GFRM data must be 2D: {path!s}, ndim={array.ndim}")
    if array.size == 0:
        raise ValueError(f"Decoded GFRM data is empty: {path!s}")
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Decoded GFRM data must be numeric: {path!s}, {array.dtype}")

    warnings: list[str] = []
    nrows = metadata.get("nrows")
    ncols = metadata.get("ncols")
    if nrows and ncols and array.shape != (nrows, ncols):
        if array.shape == (ncols, nrows):
            warnings.append(
                f"Decoded shape {array.shape} is transposed versus header "
                f"({nrows}, {ncols})."
            )
        else:
            warnings.append(
                f"Decoded shape {array.shape} does not match header "
                f"({nrows}, {ncols})."
            )
    return warnings


def decode_gfrm(gfrm_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode Bruker .gfrm with FabIO.

    Returns FabIO-reconstructed detector counts / ADU-like units.
    This function does not convert to photons.
    """
    metadata = parse_bruker_header_preview(gfrm_path)
    image = read_gfrm_with_fabio(gfrm_path)
    array = np.asarray(image.data)
    validate_gfrm_array(array, metadata, gfrm_path)
    header = dict(image.header)
    header["_fabio_class"] = type(image).__name__
    return array, header


def read_gfrm_adu(gfrm_path: str | Path) -> np.ndarray:
    """Read FabIO-decoded detector counts / ADU-like units from a GFRM file."""
    adu, _ = decode_gfrm(gfrm_path)
    return adu.astype(np.int64)


def gfrm_to_photons(
    gfrm_path: str | Path,
    *,
    mask_bad_row: bool = True,
    mask_last_row: bool | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert Bruker .gfrm to EOS-normalized estimated photons.

    Formula:
        photons = (adu - baseline_adu) / gain_adu_per_photon

    where:
        baseline_adu = NEXP third value
        gain_adu_per_photon = CCDPARM[2] / CCDPARM[1]

    Negative values are preserved. Row 511 is masked as NaN by default.
    This is an EOS-normalized estimate based on Bruker CCD calibration fields,
    not direct photon-counting detector events.
    """
    gfrm_path = Path(gfrm_path)
    if mask_last_row is not None:
        mask_bad_row = mask_last_row

    adu, header = decode_gfrm(gfrm_path)
    conversion_metadata = _parse_eos_photon_metadata(header, gfrm_path)
    photons = (
        adu.astype(np.float64) - conversion_metadata["baseline_adu"]
    ) / conversion_metadata["gain_adu_per_photon"]
    masked_row_511 = bool(mask_bad_row and photons.shape[0] > 511)
    if masked_row_511:
        photons[511, :] = np.nan
    negative_pixel_count = int(np.sum(photons < 0))

    metadata = {
        **conversion_metadata,
        "source_file": str(gfrm_path),
        "source_path": str(gfrm_path),
        "fabio_class": header.get("_fabio_class"),
        "shape": tuple(photons.shape),
        "dtype": str(photons.dtype),
        "FORMAT": header.get("FORMAT"),
        "VERSION": header.get("VERSION"),
        "HDRBLKS": header.get("HDRBLKS"),
        "NPIXELB": header.get("NPIXELB"),
        "NOVERFL": header.get("NOVERFL"),
        "LINEAR": header.get("LINEAR"),
        "mask_bad_row": bool(mask_bad_row),
        "mask_last_row": bool(mask_bad_row),
        "masked_row_511": masked_row_511,
        "negative_pixel_count": negative_pixel_count,
        "header": header,
    }
    return photons, metadata


def save_gfrm_as_npy(
    gfrm_path: str | Path,
    npy_path: str | Path | None = None,
    *,
    mask_last_row: bool = True,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    """Convert GFRM to photons and save the resulting NumPy array."""
    gfrm_path = Path(gfrm_path)
    if npy_path is None:
        npy_path = gfrm_path.with_name(f"{gfrm_path.stem}_photons.npy")
    npy_path = Path(npy_path)

    photons, metadata = gfrm_to_photons(
        gfrm_path,
        mask_last_row=mask_last_row,
    )
    np.save(npy_path, photons)
    metadata = {**metadata, "npy_path": str(npy_path)}
    return photons, npy_path, metadata


def read_gfrm_as_photons(
    gfrm_path: str | Path,
    *,
    save: bool = True,
    npy_path: str | Path | None = None,
    mask_last_row: bool = True,
) -> tuple[np.ndarray, Path | None, dict[str, Any]]:
    """Read GFRM as photons, optionally materializing ``*_photons.npy``."""
    if save:
        return save_gfrm_as_npy(
            gfrm_path,
            npy_path=npy_path,
            mask_last_row=mask_last_row,
        )
    photons, metadata = gfrm_to_photons(
        gfrm_path,
        mask_last_row=mask_last_row,
    )
    return photons, None, metadata
