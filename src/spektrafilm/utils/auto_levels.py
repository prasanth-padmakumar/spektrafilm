from __future__ import annotations

import numpy as np


_AUTO_LEVELS_CLIP_FRACTION = 0.005
_NEUTRAL_MIDTONE_LUMINANCE_RANGE = (0.1, 0.8)
_NEUTRAL_RELATIVE_TOLERANCE = 0.05
_NEUTRAL_ABSOLUTE_TOLERANCE = 0.01
_HIGHLIGHT_SHOULDER_START = 0.97
_HIGHLIGHT_OUTPUT_CEILING = 0.995


def apply_auto_levels(rgb: np.ndarray, mode: str) -> np.ndarray:
    """Apply percentile-based automatic contrast to the output RGB image."""
    image = np.asarray(rgb)
    if mode == "Off":
        return image
    if mode not in (
        "Enhance Monochromatic Contrast",
        "Per Channel Contrast with Snap Neutral Midtones",
    ):
        raise ValueError(f"unsupported auto-levels mode: {mode!r}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("auto levels expects an image with shape (height, width, 3)")
    if image.size == 0:
        raise ValueError("auto levels cannot process an empty image")

    working = np.asarray(image, dtype=np.float64)
    if mode == "Enhance Monochromatic Contrast":
        low, high = np.nanpercentile(
            working,
            (
                _AUTO_LEVELS_CLIP_FRACTION * 100.0,
                (1.0 - _AUTO_LEVELS_CLIP_FRACTION) * 100.0,
            ),
        )
        levels = _stretch(working, low, high)
    else:
        low, high = np.nanpercentile(
            working,
            (
                _AUTO_LEVELS_CLIP_FRACTION * 100.0,
                (1.0 - _AUTO_LEVELS_CLIP_FRACTION) * 100.0,
            ),
            axis=(0, 1),
        )
        levels = _stretch(working, low, high)
        levels = _snap_neutral_midtones(working, levels)

    levels = np.maximum(levels, 0.0)
    levels = _roll_off_highlights(levels)
    return levels.astype(image.dtype, copy=False)


def _stretch(image: np.ndarray, low: np.ndarray | float, high: np.ndarray | float) -> np.ndarray:
    span = np.asarray(high) - np.asarray(low)
    safe_span = np.where(span > 1e-12, span, 1.0)
    stretched = (image - low) / safe_span
    return np.where(span > 1e-12, stretched, image)


def _roll_off_highlights(levels: np.ndarray) -> np.ndarray:
    """Compress highlights smoothly so the auto white point does not clip."""
    above_knee = levels > _HIGHLIGHT_SHOULDER_START
    if not np.any(above_knee):
        return levels

    shoulder_range = _HIGHLIGHT_OUTPUT_CEILING - _HIGHLIGHT_SHOULDER_START
    distance = levels[above_knee] - _HIGHLIGHT_SHOULDER_START
    levels = levels.copy()
    levels[above_knee] = (
        _HIGHLIGHT_SHOULDER_START
        + shoulder_range * -np.expm1(-distance / shoulder_range)
    )
    return levels


def _snap_neutral_midtones(
    source: np.ndarray,
    leveled: np.ndarray,
) -> np.ndarray:
    source_mean = np.mean(source, axis=-1)
    source_span = np.max(source, axis=-1) - np.min(source, axis=-1)
    neutral_tolerance = np.maximum(
        _NEUTRAL_ABSOLUTE_TOLERANCE,
        np.abs(source_mean) * _NEUTRAL_RELATIVE_TOLERANCE,
    )
    neutral_midtones = (
        (source_mean >= _NEUTRAL_MIDTONE_LUMINANCE_RANGE[0])
        & (source_mean <= _NEUTRAL_MIDTONE_LUMINANCE_RANGE[1])
        & (source_span <= neutral_tolerance)
    )
    if not np.any(neutral_midtones):
        return leveled

    neutral_means = np.mean(leveled[neutral_midtones], axis=0)
    if np.any(neutral_means <= 1e-12):
        return leveled

    neutral_target = float(np.mean(neutral_means))
    gains = neutral_target / neutral_means
    return leveled * gains
