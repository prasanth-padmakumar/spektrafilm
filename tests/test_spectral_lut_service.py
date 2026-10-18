import numpy as np
import pytest

from spektrafilm.data.profiles_loader import Hanatos2025SensitivityAdaptation
from spektrafilm.runtime.services import spectral_lut_compute as spectral_lut_compute_module


pytestmark = pytest.mark.unit


def test_filming_tc_lut_recomputes_when_spectral_gaussian_blur_changes(monkeypatch) -> None:
    calls: list[float] = []

    def fake_compute_hanatos2025_tc_lut(sensitivity, adaptation, gamut_compress=None):
        del sensitivity, gamut_compress
        calls.append(float(adaptation.spectral_gaussian_blur))
        return np.full((2, 2, 3), adaptation.spectral_gaussian_blur + 1.0, dtype=float)

    monkeypatch.setattr(
        spectral_lut_compute_module,
        'compute_hanatos2025_tc_lut',
        fake_compute_hanatos2025_tc_lut,
    )

    service = spectral_lut_compute_module.SpectralLUTService(lut_resolution=17)
    sensitivity = np.ones((4, 3), dtype=float)
    adaptation = Hanatos2025SensitivityAdaptation(
        window_params=np.empty((0,), dtype=float),
        surface_params=np.empty((0, 3), dtype=float),
        spectral_gaussian_blur=0.0,
        reference_illuminant='D55',
        apply_window=True,
        apply_surface=True,
    )

    service.set_hanatos2025_adaptation(adaptation)
    first = service.get_filming_tc_lut(sensitivity)

    adaptation.spectral_gaussian_blur = 4.0
    service.set_hanatos2025_adaptation(adaptation)
    second = service.get_filming_tc_lut(sensitivity)

    assert calls == [0.0, 4.0]
    assert np.array_equal(first, second) is False


def test_arctic2026beta02_tc_lut_cache_tracks_sensitivity_and_illuminant(monkeypatch) -> None:
    calls: list[tuple[str, np.ndarray, np.ndarray]] = []

    def fake_compute(sensitivity, reference_illuminant, *, neutral_sensitivity, gamut_compress):
        del gamut_compress
        calls.append(
            (
                reference_illuminant,
                np.array(sensitivity, copy=True),
                np.array(neutral_sensitivity, copy=True),
            )
        )
        return np.full((2, 2, 3), len(calls), dtype=float)

    monkeypatch.setattr(
        spectral_lut_compute_module,
        "compute_arctic2026beta02_tc_lut",
        fake_compute,
    )

    service = spectral_lut_compute_module.SpectralLUTService(lut_resolution=17)
    sensitivity = np.ones((4, 3), dtype=float)
    neutral_sensitivity = np.full((4, 3), 2.0, dtype=float)

    first = service.get_arctic2026beta02_tc_lut(
        sensitivity, "D55", neutral_sensitivity=neutral_sensitivity
    )
    cached = service.get_arctic2026beta02_tc_lut(
        sensitivity, "D55", neutral_sensitivity=neutral_sensitivity
    )
    other_illuminant = service.get_arctic2026beta02_tc_lut(
        sensitivity, "D65", neutral_sensitivity=neutral_sensitivity
    )

    assert len(calls) == 2
    assert first is cached
    assert other_illuminant is not first


def test_hanatos2026_tc_lut_cache_tracks_window_but_ignores_surface(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_compute(sensitivity, adaptation, gamut_compress=None):
        del sensitivity, gamut_compress
        calls.append(bool(adaptation.apply_window))
        return np.full((2, 2, 3), float(adaptation.apply_window), dtype=float)

    monkeypatch.setattr(
        spectral_lut_compute_module,
        "compute_hanatos2026_tc_lut",
        fake_compute,
    )
    service = spectral_lut_compute_module.SpectralLUTService(lut_resolution=17)
    sensitivity = np.ones((4, 3), dtype=float)
    adaptation = Hanatos2025SensitivityAdaptation(
        window_params=np.array([420.0, 15.0, 680.0, 70.0]),
        surface_params=np.ones((3, 15)),
        reference_illuminant="D55",
        apply_window=True,
        apply_surface=True,
    )
    service.set_hanatos2025_adaptation(adaptation)

    first = service.get_hanatos2026_tc_lut(sensitivity)
    adaptation.apply_surface = False
    service.set_hanatos2025_adaptation(adaptation)
    same = service.get_hanatos2026_tc_lut(sensitivity)
    adaptation.apply_window = False
    service.set_hanatos2025_adaptation(adaptation)
    no_window = service.get_hanatos2026_tc_lut(sensitivity)

    assert calls == [True, False]
    assert first is same
    assert no_window is not first