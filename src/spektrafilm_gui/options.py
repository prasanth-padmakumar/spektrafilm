from __future__ import annotations

from enum import Enum


class RGBColorSpaces(Enum):
    sRGB = "sRGB"
    DCI_P3 = "DCI-P3"
    DisplayP3 = "Display P3"
    AdobeRGB = "Adobe RGB (1998)"
    ITU_R_BT2020 = "ITU-R BT.2020"
    ProPhotoRGB = "ProPhoto RGB"
    ACES2065_1 = "ACES2065-1"


class RGBtoRAWMethod(Enum):
    hanatos2026 = "hanatos2026"
    hanatos2025 = "hanatos2025"
    arctic2026beta02 = "arctic2026beta02"
    arctic2026beta04 = "arctic2026beta04"
    mallett2019 = "mallett2019"

class GrainModels(Enum):
    layered = "layered"
    rms = "rms"
    monte_carlo = "monte_carlo"


class ExportEngines(Enum):
    python = "python"
    rust_f64 = "rust_f64"


class RawWhiteBalance(Enum):
    as_shot = "as_shot"
    daylight = "daylight"
    tungsten = "tungsten"
    custom = "custom"


class AutoExposureMethods(Enum):
    center_weighted = "center_weighted"
    matrix = "matrix"
    multi_zone = "multi_zone"
    partial = "partial"
    highlight_weighted = "highlight_weighted"
    hybrid = "hybrid"
    median = "median"
    average = "average"


class AutoLevelsModes(Enum):
    off = "Off"
    monochromatic_contrast = "Enhance Monochromatic Contrast"
    per_channel_contrast = "Per Channel Contrast with Snap Neutral Midtones"


class NapariInterpolationModes(Enum):
    nearest = "nearest"
    linear = "linear"
    cubic = "cubic"
    spline16 = "spline16"
    spline36 = "spline36"
    lanczos = "lanczos"
    blackman = "blackman"


class DiffusionFilterFamilies(Enum):
    glimmerglass = "glimmerglass"
    black_pro_mist = "black_pro_mist"
    pro_mist = "pro_mist"
    cinebloom = "cinebloom"


class InputGamutCompressAlgorithms(Enum):
    xy = "xy"
    oklch = "oklch"


class OutputGamutCompressAlgorithms(Enum):
    off = "off"
    oklch = "oklch"
    aces_rgc = "aces_rgc"
    oklrab = "oklrab"
    jzazbz = "jzazbz"
    cam16ucs = "cam16ucs"
