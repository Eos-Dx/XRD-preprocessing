"""Lightweight XRD preprocessing API."""

from .azimuthal import AzimuthalIntegration, perform_azimuthal_integration
from .filters import (
    ColumnValueFilter,
    MetadataFilter,
    PatientFilter,
    PatientSpecimenValidityFilter,
    SNRFilter,
)
from .h5 import h5_to_df
from .faulty_pixels import (
    FAULTY_REASON_CODES,
    FAULTY_REASON_NEGATIVE,
    FAULTY_REASON_NONFINITE,
    FAULTY_REASON_OK,
    FAULTY_REASON_SATURATED,
    FaultyPixelDetector,
    HotPixelDetector,
    count_faulty_pixel_reasons,
    create_faulty_pixel_reason_map,
    create_mask,
    detect_faulty_pixels,
    detect_hot_pixels,
)
from .normalization import QRangeNormalizer, normalize_profile_by_q_range
from .gfrm import (
    decode_gfrm,
    extract_gfrm_archive,
    gfrm_conversion_metadata,
    gfrm_to_photons,
    parse_bruker_header_preview,
    parse_gfrm_header,
    read_gfrm_adu,
    read_gfrm_as_photons,
    read_gfrm_with_fabio,
    save_gfrm_as_npy,
    validate_gfrm_array,
)
from .snr import SNRTransformer, calculate_snr
from .snapshots import RadialProfileSnapshot

__all__ = [
    "AzimuthalIntegration",
    "ColumnValueFilter",
    "FAULTY_REASON_CODES",
    "FAULTY_REASON_NEGATIVE",
    "FAULTY_REASON_NONFINITE",
    "FAULTY_REASON_OK",
    "FAULTY_REASON_SATURATED",
    "FaultyPixelDetector",
    "HotPixelDetector",
    "MetadataFilter",
    "PatientFilter",
    "PatientSpecimenValidityFilter",
    "QRangeNormalizer",
    "RadialProfileSnapshot",
    "SNRFilter",
    "SNRTransformer",
    "calculate_snr",
    "count_faulty_pixel_reasons",
    "create_faulty_pixel_reason_map",
    "create_mask",
    "detect_faulty_pixels",
    "detect_hot_pixels",
    "decode_gfrm",
    "extract_gfrm_archive",
    "gfrm_conversion_metadata",
    "gfrm_to_photons",
    "h5_to_df",
    "normalize_profile_by_q_range",
    "parse_bruker_header_preview",
    "parse_gfrm_header",
    "perform_azimuthal_integration",
    "read_gfrm_adu",
    "read_gfrm_as_photons",
    "read_gfrm_with_fabio",
    "save_gfrm_as_npy",
    "validate_gfrm_array",
]
