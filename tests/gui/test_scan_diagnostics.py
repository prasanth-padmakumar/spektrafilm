import json
from types import SimpleNamespace

import numpy as np

from spektrafilm_gui.controller_runtime import SimulationRequest
from spektrafilm_gui.scan_diagnostics import (
    parse_rust_calibration,
    parse_rust_stage_statistics,
    write_scan_diagnostics,
)


def test_rust_scan_params_are_normalized_to_effective_python_settings(monkeypatch):
    from spektrafilm_gui import params_mapper

    params = SimpleNamespace(
        settings=SimpleNamespace(neutral_print_filters_from_database=True),
        debug=SimpleNamespace(deactivate_stochastic_effects=True),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            m_filter_neutral=65.0,
            y_filter_neutral=55.0,
        ),
        film_render=SimpleNamespace(grain=SimpleNamespace(active=True)),
        print_render=SimpleNamespace(glare=SimpleNamespace(active=True)),
    )

    def apply_filters(target):
        target.enlarger.c_filter_neutral = 0.0
        target.enlarger.m_filter_neutral = 67.8
        target.enlarger.y_filter_neutral = 67.5
        return target

    monkeypatch.setattr(params_mapper, "apply_database_neutral_print_filters", apply_filters)

    prepared = params_mapper.prepare_rust_scan_params(params)

    assert prepared is params
    assert (
        prepared.enlarger.c_filter_neutral,
        prepared.enlarger.m_filter_neutral,
        prepared.enlarger.y_filter_neutral,
    ) == (0.0, 67.8, 67.5)
    assert prepared.film_render.grain.active is False
    assert prepared.print_render.glare.active is False


def test_scan_request_diagnostics_are_opt_in():
    request = SimulationRequest(
        mode_label="Preview",
        image=np.zeros((1, 1, 3)),
        params={},
        output_color_space="sRGB",
        use_display_transform=False,
    )

    assert request.scan_diagnostics is False


def test_scan_diagnostics_writes_stats_hashes_and_params(tmp_path):
    source = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 18.0
    output = source * 0.5
    params = json.dumps({"film_render": {"grain": {"active": False}}}, sort_keys=True)
    stage_statistics = {
        "filming_density": {
            "mean": 0.56,
            "channel_mean": [0.58, 0.60, 0.50],
        }
    }

    path = write_scan_diagnostics(
        engine="python",
        input_image=source,
        output_image=output,
        serialized_params=params,
        stage_statistics=stage_statistics,
        workspace_root=tmp_path,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "python_scan.log"
    assert record["engine"] == "python"
    assert len(record["input_sha256"]) == 64
    assert len(record["parameters_sha256"]) == 64
    assert record["input"]["shape"] == [2, 3, 3]
    assert np.isclose(record["input"]["mean"], np.mean(source, dtype=np.float64))
    assert len(record["input"]["samples"]) == 6
    assert np.isclose(record["output"]["mean"], np.mean(output, dtype=np.float64))
    assert record["parameters"]["film_render"]["grain"]["active"] is False
    assert record["stages"] == stage_statistics


def test_rust_scan_diagnostics_capture_cli_output(tmp_path):
    engine_output = (
        'Pipeline: 12 ms\n'
        'SCAN_STAGE_STATS {"stage":"filming_density","mean":0.56,"channel_mean":[0.58,0.60,0.50],"samples":[]}\n'
        'SCAN_CALIBRATION {"print_calibration":{"selected_exposure_factor":1.02}}'
    )
    path = write_scan_diagnostics(
        engine="rust_f64",
        input_image=np.zeros((1, 1, 3), dtype=np.float64),
        output_image=np.ones((1, 1, 3), dtype=np.float64),
        serialized_params="{}",
        engine_output=engine_output,
        stage_statistics=parse_rust_stage_statistics(engine_output),
        calibration=parse_rust_calibration(engine_output),
        workspace_root=tmp_path,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "rust_f64_scan.log"
    assert record["engine_output"] == engine_output
    assert record["stages"]["filming_density"]["mean"] == 0.56
    assert record["calibration"]["print_calibration"]["selected_exposure_factor"] == 1.02
