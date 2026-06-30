# GFRM Converter

`xrd_preprocessing.gfrm` reads Bruker `.gfrm` detector frames with FabIO.

FabIO is the primary GFRM decoder. It handles Bruker `FORMAT 86` / `FORMAT 100`
frame decoding, including compression plus underflow/overflow tables. We do not
manually parse image bytes with `np.fromfile`.

## API

```python
from xrd_preprocessing import decode_gfrm, gfrm_to_photons

adu, header = decode_gfrm("frame.gfrm")
photons, metadata = gfrm_to_photons("frame.gfrm")
```

## Steps

### Step 1: Bruker Decode

`decode_gfrm(path)` does only this:

```python
img = fabio.open(str(path))
adu = np.asarray(img.data)
header = dict(img.header)
```

It validates that the FabIO result is a non-empty 2D numeric array. The output
is FabIO-reconstructed detector counts / ADU-like units. It is not photons.

### Step 2: EOS Photon Conversion

`gfrm_to_photons(path)` first calls `decode_gfrm(path)`, then extracts EOS
photon-conversion constants:

   - `baseline_adu` from `NEXP`;
   - `gain_adu_per_photon` from `CCDPARM`.

Conversion:

```text
photons = (ADU - baseline_adu) / gain_adu_per_photon
gain_adu_per_photon = e_per_photon / e_per_ADU
```

For Water 20mm:

```text
NEXP = 1 0 64 0 2
baseline_adu = 64.0
CCDPARM = 212.549000 9.800000 406.744700 0.000000 1965720.000000
gain_adu_per_photon = 406.744700 / 9.800000 = 41.5045612244898
photons = (adu - 64.0) / 41.5045612244898
```

Negative values are preserved. Row `511` is masked as `NaN` by product default
via `mask_bad_row=True`.

FabIO returns decoded detector counts. It does not apply our EOS photon
normalization convention. The EOS photon image is an estimate based on Bruker
CCD calibration fields. It is not direct photon-counting detector output.

## Bruker Frame Format Notes

The local Bruker frame-format PDF says a `.gfrm` extension is not enough to
identify the internal format. The reader must inspect the header:

```text
FORMAT   86 or 100
VERSION  header version
HDRBLKS  header size in 512-byte blocks
NPIXELB  bytes per image pixel; FORMAT 100 also stores underflow byte size
NOVERFL  FORMAT 100: underflow count, 1-byte overflow count, 2-byte overflow count
LINEAR   scale and offset, commonly 1.0 0.0 or BOOSTER 0.1 0.0
```

For `FORMAT 100`, image decode requires:

```text
compressed image block
underflow table
overflow table 1
overflow table 2
baseline restoration from NEXP
optional LINEAR scaling
```

This is why `read_gfrm_adu` delegates decode to FabIO. Our code only adds EOS
metadata extraction and optional photon conversion after FabIO has reconstructed
the detector-count image.

## CLI

Use `gfrm_convert.py` for explicit mode selection:

```bash
python gfrm_convert.py -i ./dataset -o ./out -r --mode adu
python gfrm_convert.py -i ./dataset -o ./out -r --mode photons
python gfrm_convert.py -i ./dataset -o ./out -r --mode both
```

Mode `adu` writes:

```text
file_stem_decoded_counts.npy
file_stem_header.json
file_stem_metadata.json
```

Mode `photons` writes:

```text
file_stem_photons.npy
file_stem_photon_metadata.json
```

Use `convert_gfrm_to_npy.py` only when faithful GFRM to NumPy detector-count
conversion is needed:

```bash
python convert_gfrm_to_npy.py -i ./dataset -o ./npy_output -r
python convert_gfrm_to_npy.py -i ./frame.gfrm -o ./npy_output --preview-png
```

The CLI writes:

```text
frame.npy
frame_header.json
frame_metadata.json
```

`frame.npy` is the FabIO-decoded detector image. It is not divided by gain and
not baseline-subtracted.

## Metadata

Returned metadata includes:

```text
source_path
source_file
fabio_class
baseline_adu
NEXP_raw
CCDPARM_raw
e_per_ADU
e_per_photon
gain_adu_per_photon
shape
dtype
mask_bad_row
masked_row_511
negative_pixel_count
header
```

## Out of Scope

The GFRM converter does not classify faulty pixels and does not write detector
masks. Faulty-pixel classification is a separate step.

Faulty-pixel classification is documented in [`faulty_pixels.md`](faulty_pixels.md).
