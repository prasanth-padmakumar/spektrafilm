from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.utils.auto_levels import apply_auto_levels


def test_auto_levels_off_returns_input_unchanged() -> None:
    image = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)

    result = apply_auto_levels(image, "Off")

    assert result is image
    np.testing.assert_array_equal(result, image)


def test_monochromatic_contrast_uses_a_common_rgb_stretch() -> None:
    ramp = np.linspace(0.1, 0.8, 400, dtype=np.float64).reshape(20, 20)
    image = np.stack((ramp, ramp * 0.8, ramp * 0.6), axis=-1)

    result = apply_auto_levels(image, "Enhance Monochromatic Contrast")

    assert result.dtype == image.dtype
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    common_low, common_high = np.percentile(
        image,
        (0.5, 99.5),
    )
    outside_shoulder = np.all((result > 0.0) & (result <= 0.97), axis=-1)
    np.testing.assert_allclose(
        (result[..., 0] - result[..., 1])[outside_shoulder],
        ((image[..., 0] - image[..., 1]) / (common_high - common_low))[
            outside_shoulder
        ],
        atol=1e-12,
    )


def test_per_channel_levels_snap_detected_neutral_midtones() -> None:
    ramp = np.linspace(0.2, 0.7, 400, dtype=np.float64).reshape(20, 20)
    image = np.stack((ramp, ramp, ramp), axis=-1)
    image[0, 0] = (0.95, 0.15, 0.1)
    image[0, 1] = (0.1, 0.95, 0.15)
    image[0, 2] = (0.15, 0.1, 0.95)

    result = apply_auto_levels(
        image,
        "Per Channel Contrast with Snap Neutral Midtones",
    )
    neutral_candidates = (
        (np.mean(image, axis=-1) >= 0.1)
        & (np.mean(image, axis=-1) <= 0.8)
        & ((np.max(image, axis=-1) - np.min(image, axis=-1)) <= 0.01)
    )
    channel_means = np.mean(result[neutral_candidates], axis=0)

    assert np.max(channel_means) - np.min(channel_means) < 1e-12
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


@pytest.mark.parametrize(
    "mode",
    [
        "Enhance Monochromatic Contrast",
        "Per Channel Contrast with Snap Neutral Midtones",
    ],
)
def test_auto_levels_rolls_highlights_off_instead_of_clipping_to_white(
    mode: str,
) -> None:
    values = np.linspace(0.2, 1.0, 400, dtype=np.float64).reshape(20, 20)
    image = np.stack((values, values, values), axis=-1)

    result = apply_auto_levels(image, mode)
    brightest = result[:, -4:, :]

    assert np.max(result) < 1.0
    assert np.max(result) > 0.98
    assert np.all(np.diff(brightest[..., 0], axis=1) > 0.0)


@pytest.mark.parametrize(
    ("image", "mode", "message"),
    [
        (np.zeros((2, 2)), "Enhance Monochromatic Contrast", "shape"),
        (np.zeros((2, 2, 3)), "unknown", "unsupported"),
        (np.zeros((0, 2, 3)), "Enhance Monochromatic Contrast", "empty"),
    ],
)
def test_auto_levels_rejects_invalid_input(
    image: np.ndarray,
    mode: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_auto_levels(image, mode)
