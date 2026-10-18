from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import spektrafilm.runtime.stages.filming as filming_module


def test_rgb_to_film_raw_passes_linear_sensitivity_to_hanatos2025(monkeypatch) -> None:
    """`_rgb_to_film_raw` should pass `10**log_sensitivity` (linear sensitivity)
    to `rgb_to_raw_hanatos2025`. With color_filter='none' the sensitivity is
    passed through unmodified. Hanatos2025 adaptation is handled separately via
    the LUT service's `tc_lut`, not by mutating sensitivity in this method.
    """
    captured: dict[str, np.ndarray] = {}
    log_sensitivity = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=float)

    def fake_rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        *,
        color_space=None,
        apply_cctf_decoding=None,
        reference_illuminant=None,
        tc_lut=None,
    ):
        del color_space, apply_cctf_decoding, reference_illuminant, tc_lut
        captured['sensitivity'] = np.asarray(sensitivity, dtype=float)
        return np.ones(rgb.shape, dtype=float)

    monkeypatch.setattr(filming_module, 'rgb_to_raw_hanatos2025', fake_rgb_to_raw_hanatos2025)

    stage = object.__new__(filming_module.FilmingStage)
    setattr(stage, '_film', SimpleNamespace(
        info=SimpleNamespace(reference_illuminant='D55'),
        data=SimpleNamespace(log_sensitivity=log_sensitivity),
    ))
    setattr(stage, '_camera', SimpleNamespace(color_filter='none'))
    setattr(stage, '_settings', SimpleNamespace(rgb_to_raw_method='hanatos2025'))
    setattr(stage, '_lut_service', SimpleNamespace(get_filming_tc_lut=lambda sensitivity: None))

    rgb = np.ones((1, 1, 3), dtype=float)

    getattr(stage, '_rgb_to_film_raw')(rgb)

    np.testing.assert_allclose(captured['sensitivity'], 10.0 ** log_sensitivity)


def test_rgb_to_film_raw_applies_color_filter_to_sensitivity(monkeypatch) -> None:
    """A selected camera color filter multiplies its spectral transmittance into
    the sensitivity (per wavelength, broadcast across channels) before spectral
    upsampling -- no renormalization."""
    captured: dict[str, np.ndarray] = {}
    log_sensitivity = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=float)
    transmittance = np.array([0.5, 0.25], dtype=float)

    def fake_rgb_to_raw_hanatos2025(rgb, sensitivity, **kwargs):
        captured['sensitivity'] = np.asarray(sensitivity, dtype=float)
        return np.ones(rgb.shape, dtype=float)

    monkeypatch.setattr(filming_module, 'rgb_to_raw_hanatos2025', fake_rgb_to_raw_hanatos2025)
    monkeypatch.setattr(filming_module, 'color_filter_transmittance', lambda key: transmittance)

    stage = object.__new__(filming_module.FilmingStage)
    setattr(stage, '_film', SimpleNamespace(
        info=SimpleNamespace(reference_illuminant='D55'),
        data=SimpleNamespace(log_sensitivity=log_sensitivity),
    ))
    setattr(stage, '_camera', SimpleNamespace(color_filter='hoya_y2'))
    setattr(stage, '_settings', SimpleNamespace(rgb_to_raw_method='hanatos2025'))
    setattr(stage, '_lut_service', SimpleNamespace(get_filming_tc_lut=lambda sensitivity: None))

    getattr(stage, '_rgb_to_film_raw')(np.ones((1, 1, 3), dtype=float))

    np.testing.assert_allclose(captured['sensitivity'], 10.0 ** log_sensitivity * transmittance[:, None])


def test_rgb_to_film_raw_dispatches_arctic_with_unfiltered_white_balance(monkeypatch) -> None:
    captured: dict[str, np.ndarray | str] = {}
    log_sensitivity = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=float)
    transmittance = np.array([0.5, 0.25], dtype=float)
    tc_lut = np.ones((2, 2, 3), dtype=float)

    def fake_rgb_to_raw_arctic2026beta02(rgb, sensitivity, **kwargs):
        captured["sensitivity"] = np.asarray(sensitivity, dtype=float)
        captured["color_space"] = kwargs["color_space"]
        captured["tc_lut"] = kwargs["tc_lut"]
        return np.ones(rgb.shape, dtype=float)

    def fake_get_arctic_tc_lut(sensitivity, reference_illuminant, *, neutral_sensitivity):
        captured["neutral_sensitivity"] = np.asarray(neutral_sensitivity, dtype=float)
        captured["reference_illuminant"] = reference_illuminant
        return tc_lut

    monkeypatch.setattr(
        filming_module, "rgb_to_raw_arctic2026beta02", fake_rgb_to_raw_arctic2026beta02
    )
    monkeypatch.setattr(filming_module, "color_filter_transmittance", lambda _key: transmittance)

    stage = object.__new__(filming_module.FilmingStage)
    stage._film = SimpleNamespace(
        info=SimpleNamespace(reference_illuminant="D55"),
        data=SimpleNamespace(log_sensitivity=log_sensitivity),
    )
    stage._camera = SimpleNamespace(color_filter="hoya_y2")
    stage._settings = SimpleNamespace(rgb_to_raw_method="arctic2026beta02")
    stage._lut_service = SimpleNamespace(get_arctic2026beta02_tc_lut=fake_get_arctic_tc_lut)

    stage._rgb_to_film_raw(
        np.ones((1, 1, 3), dtype=float),
        color_space="ProPhoto RGB",
        apply_cctf_decoding=False,
    )

    np.testing.assert_allclose(
        captured["sensitivity"], 10.0 ** log_sensitivity * transmittance[:, None]
    )
    np.testing.assert_allclose(captured["neutral_sensitivity"], 10.0 ** log_sensitivity)
    assert captured["reference_illuminant"] == "D55"
    assert captured["color_space"] == "ProPhoto RGB"
    assert captured["tc_lut"] is tc_lut


def test_rgb_to_film_raw_dispatches_hanatos2026(monkeypatch) -> None:
    captured: dict[str, object] = {}
    log_sensitivity = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])
    tc_lut = np.ones((2, 2, 3), dtype=float)

    def fake_get_hanatos_tc_lut(sensitivity):
        captured["sensitivity"] = np.asarray(sensitivity)
        return tc_lut

    def fake_rgb_to_raw_hanatos2026(rgb, sensitivity, **kwargs):
        captured["rgb"] = rgb
        captured["method_sensitivity"] = np.asarray(sensitivity)
        captured["kwargs"] = kwargs
        return np.ones(rgb.shape, dtype=float)

    monkeypatch.setattr(
        filming_module, "rgb_to_raw_hanatos2026", fake_rgb_to_raw_hanatos2026
    )
    stage = object.__new__(filming_module.FilmingStage)
    stage._film = SimpleNamespace(
        info=SimpleNamespace(reference_illuminant="D55"),
        data=SimpleNamespace(log_sensitivity=log_sensitivity),
    )
    stage._camera = SimpleNamespace(color_filter="none")
    stage._settings = SimpleNamespace(rgb_to_raw_method="hanatos2026")
    stage._lut_service = SimpleNamespace(
        get_hanatos2026_tc_lut=fake_get_hanatos_tc_lut
    )

    rgb = np.ones((1, 1, 3), dtype=float)
    stage._rgb_to_film_raw(rgb, color_space="ProPhoto RGB")

    np.testing.assert_allclose(captured["sensitivity"], 10.0 ** log_sensitivity)
    np.testing.assert_allclose(captured["method_sensitivity"], 10.0 ** log_sensitivity)
    assert captured["rgb"] is rgb
    assert captured["kwargs"]["reference_illuminant"] == "D55"
    assert captured["kwargs"]["tc_lut"] is tc_lut
