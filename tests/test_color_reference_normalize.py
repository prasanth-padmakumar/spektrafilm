from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spektrafilm.runtime.params_schema import IOParams
from spektrafilm.runtime.services.color_reference import ColorReferenceService


def _service(*, normalize_black_white: bool) -> ColorReferenceService:
    film = SimpleNamespace(
        info=SimpleNamespace(type="negative"),
        data=SimpleNamespace(density_curves=np.array([[2.0, 2.0, 2.0]])),
    )
    service = ColorReferenceService(
        film,
        film_render=None,
        print_profile=None,
        print_render=None,
        black_correction=False,
        white_correction=False,
        black_level=0.01,
        white_level=0.98,
        io_params=IOParams(scan_film=True, normalize_black_white=normalize_black_white),
    )

    def cmy_to_log_xyz(density: np.ndarray) -> np.ndarray:
        y = 1.0 - np.mean(density, axis=-1, keepdims=True) / 3.0
        xyz = np.repeat(y, 3, axis=-1)
        return np.log10(xyz)

    service.cmy_to_log_xyz = cmy_to_log_xyz
    return service


def test_normalize_black_white_maps_modeled_negative_endpoints_to_full_range() -> None:
    service = _service(normalize_black_white=True)
    input_xyz = np.array([[
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        [1.0, 1.0, 1.0],
    ]])

    corrected = service.black_white_xyz_correction(input_xyz)

    np.testing.assert_allclose(corrected[0, :, 1], [0.0, 0.5, 1.0], atol=1e-9)
    np.testing.assert_allclose(corrected[0, 1], [0.5, 0.5, 0.5], atol=1e-9)


def test_normalize_black_white_is_opt_in() -> None:
    service = _service(normalize_black_white=False)
    input_xyz = np.array([[[0.2, 0.3, 0.4]]])

    corrected = service.black_white_xyz_correction(input_xyz)

    np.testing.assert_array_equal(corrected, input_xyz)
