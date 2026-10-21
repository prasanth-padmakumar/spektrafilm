from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from spektrafilm_gui.persistence import (
    clear_saved_default_gui_state,
    gui_state_from_dict,
    gui_state_to_dict,
    load_default_gui_state,
    load_gui_state_from_path,
    save_default_gui_state,
    save_gui_state_to_path,
)
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE
from spektrafilm_gui.state_bridge import apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import create_widget_bundle

from .helpers import make_test_gui_state


def test_gui_state_round_trip_preserves_tuple_fields() -> None:
    state = make_test_gui_state()
    state.input_image.io.crop_size = (0.25, 0.4)
    state.grain.rms_granularity = (6.0, 7.0, 12.0)
    state.simulation.io.auto_levels = 'Per Channel Contrast with Snap Neutral Midtones'
    state.gui_only.display.gray_18_canvas = True
    state.gui_only.display.white_padding = 0.18
    state.special.scan_diagnostics = True

    restored = gui_state_from_dict(gui_state_to_dict(state))

    assert restored == state
    assert isinstance(restored.input_image.io.crop_size, tuple)
    assert isinstance(restored.grain.rms_granularity, tuple)
    assert restored.simulation.io.auto_levels == 'Per Channel Contrast with Snap Neutral Midtones'
    assert restored.special.scan_diagnostics is True


def test_legacy_special_state_defaults_scan_diagnostics_off() -> None:
    restored = gui_state_from_dict({'special': {'film_channel_swap': [0, 1, 2]}})

    assert restored.special.scan_diagnostics is False


def test_save_and_load_gui_state_file(tmp_path: Path) -> None:
    state = make_test_gui_state()
    state.simulation.enlarger.print_exposure = 1.4
    state.chemistry = replace(state.chemistry, gamma_factor=1.2)
    state.gui_only.display.gray_18_canvas = True
    state.gui_only.display.white_padding = 0.12
    destination = tmp_path / 'gui_state.json'

    save_gui_state_to_path(state, destination)
    restored = load_gui_state_from_path(destination)

    assert restored == state


def test_load_default_gui_state_uses_factory_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        'spektrafilm_gui.persistence.default_gui_state_path',
        lambda: tmp_path / 'missing.json',
    )

    restored = load_default_gui_state()

    assert restored == PROJECT_DEFAULT_GUI_STATE
    assert restored is not PROJECT_DEFAULT_GUI_STATE


def test_save_default_and_clear_saved_default(monkeypatch, tmp_path: Path) -> None:
    default_path = tmp_path / 'gui_default_state.json'
    monkeypatch.setattr(
        'spektrafilm_gui.persistence.default_gui_state_path',
        lambda: default_path,
    )
    state = make_test_gui_state()
    state.simulation.io.output_color_space = 'ACES2065-1'

    saved_path = save_default_gui_state(state)
    loaded_state = load_default_gui_state()

    assert saved_path == default_path
    assert loaded_state == state

    clear_saved_default_gui_state()

    assert not default_path.exists()


def test_save_current_gui_as_default_preserves_saving_color_space(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    monkeypatch.setattr(
        'spektrafilm_gui.persistence.default_gui_state_path',
        lambda: tmp_path / 'gui_default_state.json',
    )
    widgets = create_widget_bundle()
    apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=widgets)
    widgets.simulation.saving_color_space.value = 'Adobe RGB (1998)'

    current_state = collect_gui_state(widgets=widgets)
    save_default_gui_state(current_state)
    restored = load_default_gui_state()

    assert current_state.simulation.workflow.saving_color_space == 'Adobe RGB (1998)'
    assert restored.simulation.workflow.saving_color_space == 'Adobe RGB (1998)'


def test_gui_state_from_dict_fills_missing_section_from_defaults() -> None:
    data = gui_state_to_dict(PROJECT_DEFAULT_GUI_STATE)
    del data['simulation']

    restored = gui_state_from_dict(data)

    assert restored.simulation == PROJECT_DEFAULT_GUI_STATE.simulation


def test_gui_state_from_dict_fills_missing_field_from_defaults() -> None:
    data = gui_state_to_dict(PROJECT_DEFAULT_GUI_STATE)
    del data['display']['output_interpolation']
    del data['simulation']['print_exposure']

    restored = gui_state_from_dict(data)

    assert restored.gui_only.display.output_interpolation == PROJECT_DEFAULT_GUI_STATE.gui_only.display.output_interpolation
    assert restored.simulation.enlarger.print_exposure == PROJECT_DEFAULT_GUI_STATE.simulation.enlarger.print_exposure


def test_gui_state_from_dict_ignores_unknown_fields() -> None:
    data = gui_state_to_dict(PROJECT_DEFAULT_GUI_STATE)
    data['display']['legacy_dropped_field'] = 'gone'
    data['unknown_top_level_section'] = {'foo': 1}

    restored = gui_state_from_dict(data)

    assert restored == PROJECT_DEFAULT_GUI_STATE


def test_gui_state_from_dict_flat_gui_only_sections_override_nested_copy() -> None:
    data = gui_state_to_dict(PROJECT_DEFAULT_GUI_STATE)
    data['gui_only'] = {
        'display': {
            'white_padding': 0.9,
            'settings': {'preview_max_size': 128},
        },
        'load_raw': {
            'white_balance': 'camera',
            'temperature': 4000.0,
            'tint': 0.8,
            'lens_correction': True,
        },
    }
    data['display']['white_padding'] = 0.12
    data['display']['preview_max_size'] = 2048
    data['load_raw']['white_balance'] = 'daylight'

    restored = gui_state_from_dict(data)

    assert restored.gui_only.display.white_padding == 0.12
    assert restored.gui_only.display.settings.preview_max_size == 2048
    assert restored.gui_only.load_raw.white_balance == 'daylight'