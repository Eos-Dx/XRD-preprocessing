#!/usr/bin/env python
"""Convert Bruker GADDS .gfrm detector frames to NumPy .npy arrays.

Usage:
    pip install fabio numpy
    python convert_gfrm_to_npy.py -i ./dataset -o ./npy_output -r
    python convert_gfrm_to_npy.py -i ./frame.gfrm -o ./npy_output --preview-png

The saved array is the FabIO-decoded detector image, preserving orientation and
decoded counts. It is not EOS photon-normalized data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_SRC = Path(__file__).resolve().parent / "src"
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from xrd_preprocessing.gfrm import (  # noqa: E402
    parse_bruker_header_preview,
    read_gfrm_with_fabio,
    validate_gfrm_array,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_json_safe(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def iter_input_files(input_path: str | Path, recursive: bool) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() != ".gfrm":
            raise ValueError(f"Input file is not .gfrm: {input_path!s}")
        return [input_path]
    if input_path.is_dir():
        pattern = "**/*.gfrm" if recursive else "*.gfrm"
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path does not exist: {input_path!s}")


def _relative_output_path(path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        relative = Path(path.name)
    else:
        relative = path.relative_to(input_root)
    return output_root / relative.with_suffix(".npy")


def _linear_values(linear: Any) -> tuple[float, float] | None:
    if linear is None:
        return None
    parts = str(linear).split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _save_preview_png(array: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    finite = array[np.isfinite(array)]
    image = np.log1p(np.maximum(array.astype(float), 0.0))
    vmax = float(np.percentile(np.log1p(np.maximum(finite, 0.0)), 99.5))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(image, cmap="inferno", origin="upper", vmin=0, vmax=vmax)
    ax.set_title(output_path.stem)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def convert_one(
    path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    options: argparse.Namespace | SimpleNamespace,
) -> dict[str, Any]:
    path = Path(path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_path = _relative_output_path(path, input_root, output_root)
    header_path = output_path.with_name(f"{output_path.stem}_header.json")
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    preview_path = output_path.with_suffix(".png")

    metadata = parse_bruker_header_preview(path)
    image = read_gfrm_with_fabio(path)
    array = np.asarray(image.data)
    warnings = validate_gfrm_array(array, metadata, path)

    if getattr(options, "force_linear_scaling", False):
        linear = _linear_values(metadata.get("linear"))
        if linear is not None:
            scale, offset = linear
            array = array.astype(float) * scale + offset
            warnings.append(f"Applied forced LINEAR scaling: scale={scale}, offset={offset}.")

    finite = array[np.isfinite(array)]
    output_metadata = {
        "source_file": str(path),
        "detected_format": metadata.get("format"),
        "version": metadata.get("version"),
        "hdrblks": metadata.get("hdrblks"),
        "nrows": metadata.get("nrows"),
        "ncols": metadata.get("ncols"),
        "npixelb": metadata.get("npixelb"),
        "noverfl": metadata.get("noverfl"),
        "linear": metadata.get("linear"),
        "fabio_class": type(image).__name__,
        "output_file": str(output_path),
        "output_shape": tuple(array.shape),
        "output_dtype": str(array.dtype),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "warnings": warnings,
    }

    print(
        f"{path}: FORMAT={metadata.get('format')} VERSION={metadata.get('version')} "
        f"HDRBLKS={metadata.get('hdrblks')} NROWS={metadata.get('nrows')} "
        f"NCOLS={metadata.get('ncols')} NPIXELB={metadata.get('npixelb')} "
        f"NOVERFL={metadata.get('noverfl')} LINEAR={metadata.get('linear')}"
    )
    for warning in warnings:
        print(f"WARNING: {path}: {warning}")

    if getattr(options, "dry_run", False):
        return output_metadata

    if output_path.exists() and not getattr(options, "overwrite", False):
        raise FileExistsError(f"Output exists; use --overwrite: {output_path!s}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)

    if getattr(options, "save_header", True):
        save_json_safe(metadata["header"], header_path)
    save_json_safe(output_metadata, metadata_path)

    if getattr(options, "preview_png", False):
        _save_preview_png(array, preview_path)

    return output_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Bruker GADDS .gfrm frames to FabIO-decoded .npy arrays."
    )
    parser.add_argument("-i", "--input", required=True, help="Input .gfrm file or directory.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    parser.add_argument("--save-header", dest="save_header", action="store_true", default=True)
    parser.add_argument("--no-save-header", dest="save_header", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without writing.")
    parser.add_argument("--preview-png", action="store_true", help="Save log-scaled PNG preview.")
    parser.add_argument(
        "--force-linear-scaling",
        action="store_true",
        help="Advanced: apply LINEAR scale/offset after FabIO decode.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_root = Path(args.output)
    input_root = input_path if input_path.is_dir() else input_path
    files = iter_input_files(input_path, args.recursive)
    if not files:
        print(f"No .gfrm files found: {input_path!s}")
        return 1

    failed: list[tuple[Path, str]] = []
    for path in files:
        try:
            convert_one(path, input_root, output_root, args)
        except Exception as exc:
            failed.append((path, str(exc)))
            print(f"ERROR: {path}: {exc}")

    print(f"Converted/inspected {len(files) - len(failed)} of {len(files)} .gfrm files.")
    if failed:
        print("Failed files:")
        for path, error in failed:
            print(f"- {path}: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
