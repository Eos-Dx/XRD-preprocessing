"""Product preprocessing transformers.

Import from this package for new code. The historical
``xrd_preprocessing.product_transformers`` module remains as a compatibility
facade.
"""

from .h5 import (
    H5BlobDataFrameTransformer,
    H5MeasurementSetAuditTransformer,
    H5SessionSelectorTransformer,
    H5ToDataFrameTransformer,
)
from .labels import PairedGroupFilter, ProductStatusGroupFilter
from .metadata import (
    ConstantQRangeTransformer,
    DropColumnsTransformer,
    JoblibWriterTransformer,
    ProductColumnBuilder,
    RequiredColumnsTransformer,
)
from .profiles import SimpleRadialProfileTransformer

__all__ = [
    "ConstantQRangeTransformer",
    "DropColumnsTransformer",
    "H5BlobDataFrameTransformer",
    "H5MeasurementSetAuditTransformer",
    "H5SessionSelectorTransformer",
    "H5ToDataFrameTransformer",
    "JoblibWriterTransformer",
    "PairedGroupFilter",
    "ProductColumnBuilder",
    "ProductStatusGroupFilter",
    "RequiredColumnsTransformer",
    "SimpleRadialProfileTransformer",
]
