import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import convert_gfrm_to_npy
import gfrm_convert
import xrd_preprocessing.gfrm as gfrm_module
from xrd_preprocessing import (
    decode_gfrm,
    extract_gfrm_archive,
    gfrm_conversion_metadata,
    gfrm_photon_statistics,
    gfrm_to_photons,
    parse_bruker_header_preview,
    parse_gfrm_header,
    read_gfrm_adu,
    read_gfrm_as_photons,
    read_gfrm_with_fabio,
    save_gfrm_as_npy,
    validate_gfrm_array,
)


ARCHIVE = Path("examples/data/gfrm_measurements.tar.gz")
WATER_20MM = (
    Path("GFRM_measurements")
    / "Water"
    / "20260608_112438_Water_20mm"
    / "20260608_112438_Water_20mm_Main.gfrm"
)


def test_gfrm_archive_layout_is_flattened():
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        names = archive.getnames()

    forbidden_root = "Baseline_" "measurments_with_Bruker"
    assert len([name for name in names if name.endswith(".gfrm")]) == 14
    assert all(forbidden_root not in name for name in names)
    assert all("/Water/" not in name or "_Main/" not in name for name in names)
    assert "GFRM_measurements/Calibrants/AgBh/AgBh.gfrm" in names
    assert str(WATER_20MM) in names


def test_extract_gfrm_archive_and_convert_water_20mm(tmp_path):
    root = extract_gfrm_archive(ARCHIVE, tmp_path / "gfrm")
    gfrm_path = root / WATER_20MM

    header = parse_gfrm_header(gfrm_path)
    decoded_counts, decoded_header = decode_gfrm(gfrm_path)
    metadata = gfrm_conversion_metadata(gfrm_path)
    adu = read_gfrm_adu(gfrm_path)
    photons, conversion = gfrm_to_photons(gfrm_path)

    assert gfrm_path.exists()
    assert header["HDRBLKS"] == "15"
    assert decoded_header["_fabio_class"] == "Bruker100Image"
    assert decoded_counts.shape == (512, 768)
    assert metadata["baseline_adu"] == conversion["baseline_adu"]
    assert metadata["gain_adu_per_photon"] == conversion["gain_adu_per_photon"]
    assert adu.shape == photons.shape == (512, 768)
    assert np.isnan(photons[511]).all()
    assert int(np.sum(photons < 0)) == 1596
    assert int(np.sum(photons == 0)) == 636
    stats = gfrm_photon_statistics(photons, conversion)
    assert stats["negative_pixel_count"] == 1596
    assert conversion["fabio_class"] == "Bruker100Image"
    assert conversion["NEXP_raw"] == header["NEXP"]
    assert conversion["CCDPARM_raw"] == header["CCDPARM"]
    assert conversion["masked_row_511"] is True


def test_fabio_is_primary_gfrm_decoder(tmp_path):
    root = extract_gfrm_archive(ARCHIVE, tmp_path / "gfrm")
    gfrm_path = root / WATER_20MM

    metadata = parse_bruker_header_preview(gfrm_path)
    image = read_gfrm_with_fabio(gfrm_path)
    array = np.asarray(image.data)
    warnings = validate_gfrm_array(array, metadata, gfrm_path)

    assert type(image).__name__ == "Bruker100Image"
    assert metadata["format"] == 100
    assert metadata["nrows"] == 512
    assert metadata["ncols"] == 768
    assert array.shape == (512, 768)
    assert warnings == []


def test_gfrm_validation_rejects_invalid_arrays():
    metadata = {"nrows": 2, "ncols": 2}

    with pytest.raises(TypeError, match="NumPy array"):
        validate_gfrm_array([[1, 2]], metadata, "bad.gfrm")
    with pytest.raises(ValueError, match="must be 2D"):
        validate_gfrm_array(np.ones(2), metadata, "bad.gfrm")
    with pytest.raises(ValueError, match="empty"):
        validate_gfrm_array(np.ones((0, 2)), metadata, "bad.gfrm")
    with pytest.raises(TypeError, match="numeric"):
        validate_gfrm_array(np.array([["x"]], dtype=object), metadata, "bad.gfrm")


def test_gfrm_validation_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="transposed"):
        validate_gfrm_array(
            np.ones((3, 2)),
            {"nrows": 2, "ncols": 3},
            "frame.gfrm",
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_gfrm_array(
            np.ones((4, 4)),
            {"nrows": 2, "ncols": 3},
            "frame.gfrm",
        )


def test_gfrm_metadata_parser_rejects_missing_or_invalid_headers():
    with pytest.raises(ValueError, match="Missing NEXP"):
        gfrm_module._parse_eos_photon_metadata({"CCDPARM": "1 1 1 1 1"})
    with pytest.raises(ValueError, match="NEXP must contain"):
        gfrm_module._parse_eos_photon_metadata(
            {"NEXP": "1 2", "CCDPARM": "1 1 1 1 1"}
        )
    with pytest.raises(ValueError, match="Cannot parse NEXP"):
        gfrm_module._parse_eos_photon_metadata(
            {"NEXP": "bad 2 3", "CCDPARM": "1 1 1 1 1"}
        )
    with pytest.raises(ValueError, match="Invalid e_per_ADU"):
        gfrm_module._parse_eos_photon_metadata(
            {"NEXP": "1 2 3", "CCDPARM": "1 0 1 1 1"}
        )
    with pytest.raises(ValueError, match="Invalid e_per_photon"):
        gfrm_module._parse_eos_photon_metadata(
            {"NEXP": "1 2 3", "CCDPARM": "1 1 0 1 1"}
        )


def test_read_gfrm_as_photons_default_does_not_save(monkeypatch):
    monkeypatch.setattr(
        "xrd_preprocessing.gfrm.gfrm_to_photons",
        lambda path, mask_bad_row=True: (np.ones((2, 2)), {"path": str(path)}),
    )

    photons, npy_path, metadata = read_gfrm_as_photons("fake.gfrm")

    assert npy_path is None
    assert photons.shape == (2, 2)
    assert metadata == {"path": "fake.gfrm"}


def test_gfrm_to_photons_preserves_negative_photons(tmp_path):
    root = extract_gfrm_archive(ARCHIVE, tmp_path / "gfrm")
    photons, _ = gfrm_to_photons(root / WATER_20MM)

    assert float(np.nanmin(photons)) < 0.0


def test_eos_photon_formula_with_artificial_adu(monkeypatch):
    header = {
        "FORMAT": "100",
        "VERSION": "18",
        "HDRBLKS": "15",
        "NPIXELB": "1 1",
        "NOVERFL": "0 0 0",
        "LINEAR": "1.0 0.0",
        "NEXP": "1 0 64 0 2",
        "CCDPARM": "212.549000 9.800000 406.744700 0.000000 1965720.000000",
        "_fabio_class": "FakeBruker100Image",
    }
    adu = np.array([[64.0, 105.5045612244898], [0.0, 64.0]])

    monkeypatch.setattr(
        "xrd_preprocessing.gfrm.decode_gfrm",
        lambda _path: (adu, header),
    )

    photons, metadata = gfrm_to_photons("fake.gfrm", mask_bad_row=False)

    assert metadata["baseline_adu"] == 64.0
    assert metadata["gain_adu_per_photon"] == 41.5045612244898
    np.testing.assert_allclose(photons[0], [0.0, 1.0])
    assert photons[1, 0] < 0
    assert photons.dtype == np.float64
    assert metadata["negative_values_preserved"] is True
    assert metadata["masked_row_511"] is False


def test_gfrm_to_photons_masks_bad_row(monkeypatch):
    header = {
        "NEXP": "1 0 64 0 2",
        "CCDPARM": "212.549000 9.800000 406.744700 0.000000 1965720.000000",
        "_fabio_class": "FakeBruker100Image",
    }
    adu = np.full((512, 2), 64.0)

    monkeypatch.setattr(
        "xrd_preprocessing.gfrm.decode_gfrm",
        lambda _path: (adu, header),
    )

    photons, metadata = gfrm_to_photons("fake.gfrm")

    assert np.isnan(photons[511]).all()
    assert metadata["masked_row_511"] is True


def test_save_and_read_gfrm_as_photons(tmp_path):
    root = extract_gfrm_archive(ARCHIVE, tmp_path / "gfrm")
    gfrm_path = root / WATER_20MM
    target = tmp_path / "water.npy"

    saved, saved_path, saved_metadata = save_gfrm_as_npy(gfrm_path, target)
    read, read_path, read_metadata = read_gfrm_as_photons(gfrm_path)
    saved_again, saved_again_path, saved_again_metadata = read_gfrm_as_photons(
        gfrm_path,
        save=True,
        npy_path=target,
    )

    assert saved_path == target
    assert read_path is None
    assert saved_again_path == target
    assert saved_metadata["npy_path"] == str(target)
    assert "npy_path" not in read_metadata
    assert saved_again_metadata["npy_path"] == str(target)
    np.testing.assert_allclose(saved, np.load(target), rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(read, np.load(target), rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(saved_again, np.load(target), rtol=0, atol=0, equal_nan=True)


def test_convert_gfrm_to_npy_cli_helpers(tmp_path):
    root = extract_gfrm_archive(ARCHIVE, tmp_path / "gfrm")
    gfrm_path = root / WATER_20MM
    output_root = tmp_path / "out"
    options = SimpleNamespace(
        dry_run=False,
        force_linear_scaling=False,
        overwrite=False,
        preview_png=False,
        save_header=True,
    )

    metadata = convert_gfrm_to_npy.convert_one(
        gfrm_path,
        root,
        output_root,
        options,
    )
    output_path = output_root / WATER_20MM.with_suffix(".npy")
    header_path = output_path.with_name(f"{output_path.stem}_header.json")
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")

    assert output_path.exists()
    assert header_path.exists()
    assert metadata_path.exists()
    assert metadata["detected_format"] == 100
    assert metadata["fabio_class"] == "Bruker100Image"
    assert tuple(metadata["output_shape"]) == (512, 768)
    np.testing.assert_array_equal(np.load(output_path), read_gfrm_adu(gfrm_path))


def test_gfrm_convert_cli_helpers_save_both_modes(tmp_path):
    root = extract_gfrm_archive(ARCHIVE, tmp_path / "gfrm")
    gfrm_path = root / WATER_20MM
    output_root = tmp_path / "gfrm_convert"

    gfrm_convert.save_one(
        gfrm_path,
        root,
        output_root,
        mode="both",
        mask_bad_row=True,
        overwrite=False,
    )

    base = output_root / WATER_20MM.with_suffix("")
    adu_path = base.with_name(f"{base.name}_decoded_counts.npy")
    photons_path = base.with_name(f"{base.name}_photons.npy")
    header_path = base.with_name(f"{base.name}_header.json")
    metadata_path = base.with_name(f"{base.name}_metadata.json")
    photon_metadata_path = base.with_name(f"{base.name}_photon_metadata.json")

    assert adu_path.exists()
    assert photons_path.exists()
    assert header_path.exists()
    assert metadata_path.exists()
    assert photon_metadata_path.exists()
    np.testing.assert_array_equal(np.load(adu_path), read_gfrm_adu(gfrm_path))
    photons, _ = gfrm_to_photons(gfrm_path)
    np.testing.assert_allclose(np.load(photons_path), photons, equal_nan=True)
