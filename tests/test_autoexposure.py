from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.utils.autoexposure import (
    _metered_value,
    measure_autoexposure_ev,
    measure_raw_autoexposure_ev,
)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (1, 8),
        (4, 7),
        (8, 4),
    ],
)
def test_matrix_meter_uses_full_field_mean_when_grid_does_not_fit(
    shape: tuple[int, int],
) -> None:
    field = np.linspace(0.1, 0.9, num=np.prod(shape)).reshape(shape)

    assert _metered_value(field, "matrix") == pytest.approx(np.mean(field))


def test_matrix_meter_returns_finite_compensation_for_single_pixel_raw() -> None:
    raw = np.full((1, 1, 1), 0.1386374592308157)

    exposure_ev = measure_raw_autoexposure_ev(raw, method="matrix")

    assert exposure_ev == pytest.approx(-np.log2(raw.item()))


def test_hybrid_meter_blends_spatial_and_highlight_exposure_in_stops() -> None:
    field = np.full((100, 100, 1), 0.18, dtype=np.float64)
    field[10:40, 10:40, :] = 0.9

    spatial_ev = measure_raw_autoexposure_ev(field, method="multi_zone")
    highlight_ev = measure_raw_autoexposure_ev(field, method="highlight_weighted")
    hybrid_ev = measure_raw_autoexposure_ev(field, method="hybrid")

    assert highlight_ev < hybrid_ev < spatial_ev
    assert hybrid_ev == pytest.approx(0.65 * spatial_ev + 0.35 * highlight_ev)


def test_hybrid_meter_uses_same_blend_for_display_rgb_metering() -> None:
    image = np.full((40, 40, 3), 0.18, dtype=np.float64)
    image[4:16, 4:16, :] = 0.9

    spatial_ev = measure_autoexposure_ev(
        image,
        apply_cctf_decoding=False,
        method="multi_zone",
    )
    highlight_ev = measure_autoexposure_ev(
        image,
        apply_cctf_decoding=False,
        method="highlight_weighted",
    )
    hybrid_ev = measure_autoexposure_ev(
        image,
        apply_cctf_decoding=False,
        method="hybrid",
    )

    assert hybrid_ev == pytest.approx(0.65 * spatial_ev + 0.35 * highlight_ev)
