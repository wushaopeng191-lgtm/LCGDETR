# Ultralytics-based paper experiment modules, AGPL-3.0.
"""Custom modules preserved from the paper experiment source."""

from .block import C2f_FAT, C2f_LGLB, C2f_MAC, DKCSPELAN, SpatialSepC3, WCSPELAN
from .transformer import (
    AIFI_ConvGLU,
    LCGAConvGLU,
    TransformerEncoderLayer_DHSA,
    TransformerEncoderLayer_DPB,
    TransformerEncoderLayer_LCGA,
    TransformerEncoderLayer_MSMHSA,
    TransformerEncoderLayer_TSSA,
)

__all__ = (
    "WCSPELAN", "DKCSPELAN", "SpatialSepC3", "LCGAConvGLU",
    "TransformerEncoderLayer_LCGA", "AIFI_ConvGLU",
    "C2f_FAT", "C2f_LGLB", "C2f_MAC",
    "TransformerEncoderLayer_DPB", "TransformerEncoderLayer_TSSA",
    "TransformerEncoderLayer_MSMHSA", "TransformerEncoderLayer_DHSA",
)
