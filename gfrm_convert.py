#!/usr/bin/env python
"""Convert Bruker GFRM files as decoded counts, EOS photons, or both."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_SRC = Path(__file__).resolve().parent / "src"
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from xrd_preprocessing.gfrm import decode_gfrm, gfrm_to_photons  # noqa: E402


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def iter_gfrm_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".gfrm":
            raise ValueError(f"Input file is not .gfrm: {input_path!s}")
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("**/*.gfrm" if recursive else "*.gfrm"))
    raise FileNotFoundError(f"Input path does not exist: {input_path!s}")


def output_base(path: Path, input_root: Path, output_root: Path) -> Path:
    relative = Path(path.name) if input_root.is_file() else path.relative_to(input_root)
    return output_root / relative.with_suffix("")


def decoded_metadata(path: Path, adu: np.ndarray, header: dict[str, Any]) -> dict[str, Any]:
    finite = adu[np.isfinite(adu)]
    return {
        "source_file": str(path),
        "fabio_class": header.get("_fabio_class"),
        "shape": tuple(adu.shape),
        "dtype": str(adu.dtype),
        "FORMAT": header.get("FORMAT"),
        "VERSION": header.get("VERSION"),
        "HDRBLKS": header.get("HDRBLKS"),
        "NPIXELB": header.get("NPIXELB"),
        "NOVERFL": header.get("NOVERFL"),
        "LINEAR": header.get("LINEAR"),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "unit": "FabIO-decoded detector counts / ADU-like units",
        "not_photons": True,
    }


def save_one(
    path: Path,
    input_root: Path,
    output_root: Path,
    *,
    mode: str,
    mask_bad_row: bool,
    overwrite: bool,
) -> None:
    base = output_base(path, input_root, output_root)

    if mode in {"adu", "both"}:
        adu, header = decode_gfrm(path)
        adu_path = base.with_name(f"{base.name}_decoded_counts.npy")
        header_path = base.with_name(f"{base.name}_header.json")
        metadata_path = base.with_name(f"{base.name}_metadata.json")
        for output_path in [adu_path, header_path, metadata_path]:
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {output_path!s}")
        adu_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(adu_path, adu)
        save_json(header, header_path)
        save_json(decoded_metadata(path, adu, header), metadata_path)

    if mode in {"photons", "both"}:
        photons, metadata = gfrm_to_photons(path, mask_bad_row=mask_bad_row)
        photons_path = base.with_name(f"{base.name}_photons.npy")
        metadata_path = base.with_name(f"{base.name}_photon_metadata.json")
        for output_path in [photons_path, metadata_path]:
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {output_path!s}")
        photons_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(photons_path, photons)
        save_json(metadata, metadata_path)

    print(f"{path}: saved mode={mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Bruker GFRM files to decoded counts and/or EOS photons."
    )
    parser.add_argument("-i", "--input", required=True, help="Input .gfrm file or directory.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search recursively.")
    parser.add_argument(
        "--mode",
        choices=["adu", "photons", "both"],
        default="adu",
        help="Output mode.",
    )
    parser.add_argument("--mask-bad-row", dest="mask_bad_row", action="store_true", default=True)
    parser.add_argument("--no-mask-bad-row", dest="mask_bad_row", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_root = Path(args.output)
    input_root = input_path if input_path.is_dir() else input_path
    files = iter_gfrm_files(input_path, args.recursive)
    if not files:
        print(f"No .gfrm files found: {input_path!s}")
        return 1

    failed: list[tuple[Path, str]] = []
    for path in files:
        try:
            save_one(
                path,
                input_root,
                output_root,
                mode=args.mode,
                mask_bad_row=args.mask_bad_row,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failed.append((path, str(exc)))
            print(f"ERROR: {path}: {exc}")

    print(f"Processed {len(files) - len(failed)} of {len(files)} .gfrm files.")
    if failed:
        print("Failed files:")
        for path, error in failed:
            print(f"- {path}: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
