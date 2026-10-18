from dataclasses import replace

import numpy as np
import pytest

from spektrafilm.model.stocks import FilmStocks
from spektrafilm.runtime.params_builder import digest_params, init_params
from spektrafilm.runtime.params_schema import IOParams
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.runtime.stages.scanning import _adapt_projected_slide_xyz_to_display
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.persistence import gui_state_from_dict, gui_state_to_dict
from spektrafilm_gui.state import (
    PROJECT_DEFAULT_GUI_STATE,
    clone_gui_state,
    gui_state_from_params,
)


pytestmark = pytest.mark.unit


def _scan_density(*, film_stock: str, scan_film: bool, projector_view: bool):
    params = init_params(film_profile=film_stock)
    params.io.scan_film = scan_film
    params.io.project_positive_as_slide = projector_view
    params.settings.use_scanner_lut = False
    params.settings.use_enlarger_lut = False
    params.io.output_gamut_compress = replace(
        params.io.output_gamut_compress,
        algorithm="off",
    )
    params.print_render.glare.active = False
    params.scanner.lens_blur = 0.0
    params.scanner.unsharp_mask = (0.0, 0.0)
    params.camera.auto_exposure = False
    pipeline = SimulationPipeline(digest_params(params))
    density = np.broadcast_to(
        np.array([0.2, 0.45, 0.7], dtype=np.float64),
        (8, 8, 3),
    ).copy()
    return pipeline._scanning_stage.scan(density)


def test_projector_appearance_transform_preserves_near_black_and_white():
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = np.stack((np.zeros(3), white))

    mapped = _adapt_projected_slide_xyz_to_display(xyz, white)

    assert np.all(np.isfinite(mapped))
    np.testing.assert_allclose(mapped[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(mapped[1], white, rtol=0.02, atol=0.02)


def test_projector_view_changes_only_direct_positive_scans():
    slide_stock = FilmStocks.kodak_ektachrome_100.value
    regular = _scan_density(
        film_stock=slide_stock,
        scan_film=True,
        projector_view=False,
    )
    projected = _scan_density(
        film_stock=slide_stock,
        scan_film=True,
        projector_view=True,
    )
    assert not np.allclose(regular, projected)

    negative_scan = _scan_density(
        film_stock=FilmStocks.kodak_portra_400.value,
        scan_film=True,
        projector_view=False,
    )
    negative_scan_toggle = _scan_density(
        film_stock=FilmStocks.kodak_portra_400.value,
        scan_film=True,
        projector_view=True,
    )
    np.testing.assert_allclose(negative_scan, negative_scan_toggle)

    positive_print = _scan_density(
        film_stock=slide_stock,
        scan_film=False,
        projector_view=False,
    )
    positive_print_toggle = _scan_density(
        film_stock=slide_stock,
        scan_film=False,
        projector_view=True,
    )
    np.testing.assert_allclose(positive_print, positive_print_toggle)


def test_projector_view_setting_round_trips_through_gui_state():
    slide_stock = FilmStocks.kodak_ektachrome_100.value
    params = init_params(film_profile=slide_stock)
    params.io.project_positive_as_slide = True
    state = gui_state_from_params(
        params,
        film_stock=slide_stock,
        print_paper=PROJECT_DEFAULT_GUI_STATE.selection.print_paper,
    )

    assert state.simulation.io.project_positive_as_slide is True
    assert build_params_from_state(state).io.project_positive_as_slide is True

    round_tripped = gui_state_from_dict(gui_state_to_dict(state))
    assert round_tripped.simulation.io.project_positive_as_slide is True

    legacy_state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    legacy_data = gui_state_to_dict(legacy_state)
    legacy_data["simulation"].pop("project_positive_as_slide")
    loaded_legacy = gui_state_from_dict(legacy_data)
    assert loaded_legacy.simulation.io.project_positive_as_slide is False


def test_projector_view_defaults_off_in_runtime_io():
    assert IOParams().project_positive_as_slide is False
