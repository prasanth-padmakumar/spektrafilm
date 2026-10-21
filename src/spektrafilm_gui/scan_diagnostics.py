from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


_LOG_NAMES = {
    "python": "python_scan.log",
    "rust_f64": "rust_f64_scan.log",
}
_MAX_PERCENTILE_SAMPLES = 1_000_000
_RUST_STAGE_PREFIX = "SCAN_STAGE_STATS "
_RUST_CALIBRATION_PREFIX = "SCAN_CALIBRATION "


def image_statistics(image: np.ndarray) -> dict[str, object]:
    array = np.asarray(image)
    flat = array.reshape(-1)
    invalid_count = int(np.count_nonzero(~np.isfinite(array)))
    finite_sample = flat[::_sample_stride(flat.size)]
    finite_sample = finite_sample[np.isfinite(finite_sample)]
    percentiles = (
        np.percentile(finite_sample, [1, 99])
        if finite_sample.size
        else np.array([np.nan, np.nan])
    )

    def finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    channels = array[..., :3] if array.ndim >= 3 and array.shape[-1] >= 3 else None
    samples = []
    if array.ndim == 3 and array.shape[-1] >= 3 and array.shape[0] and array.shape[1]:
        fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
        coordinates = sorted(
            {
                (round((array.shape[1] - 1) * x), round((array.shape[0] - 1) * y))
                for x in fractions
                for y in fractions
            },
            key=lambda point: (point[1], point[0]),
        )
        samples = [
            {
                "x": x,
                "y": y,
                "rgb": [finite_or_none(value) for value in array[y, x, :3]],
            }
            for x, y in coordinates
        ]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": finite_or_none(np.nanmin(array)) if invalid_count < array.size else None,
        "max": finite_or_none(np.nanmax(array)) if invalid_count < array.size else None,
        "mean": finite_or_none(np.nanmean(array)) if invalid_count < array.size else None,
        "p01_sampled": finite_or_none(percentiles[0]),
        "p99_sampled": finite_or_none(percentiles[1]),
        "channel_mean": (
            [finite_or_none(np.nanmean(channels[..., index])) for index in range(3)]
            if channels is not None
            else None
        ),
        "invalid_count": invalid_count,
        "samples": samples,
    }


def _sample_stride(size: int) -> int:
    return max(1, size // _MAX_PERCENTILE_SAMPLES)


def parse_rust_stage_statistics(engine_output: str) -> dict[str, dict[str, object]]:
    stages: dict[str, dict[str, object]] = {}
    for line in engine_output.splitlines():
        if not line.startswith(_RUST_STAGE_PREFIX):
            continue
        stage = json.loads(line[len(_RUST_STAGE_PREFIX) :])
        name = stage.pop("stage")
        stages[name] = stage
    return stages


def parse_rust_calibration(engine_output: str) -> dict[str, object] | None:
    for line in engine_output.splitlines():
        if line.startswith(_RUST_CALIBRATION_PREFIX):
            return json.loads(line[len(_RUST_CALIBRATION_PREFIX) :])
    return None


def write_scan_diagnostics(
    *,
    engine: str,
    input_image: np.ndarray,
    output_image: np.ndarray,
    serialized_params: str,
    engine_output: str = "",
    stage_statistics: dict[str, dict[str, object]] | None = None,
    calibration: dict[str, object] | None = None,
    workspace_root: Path | None = None,
) -> Path:
    """Write one full-resolution scan's settings and image summaries."""
    try:
        filename = _LOG_NAMES[engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported diagnostics engine: {engine}") from exc

    input_rgb = np.ascontiguousarray(np.asarray(input_image, dtype=np.float64)[..., :3])
    input_hash = hashlib.sha256(input_rgb.tobytes()).hexdigest()
    params_hash = hashlib.sha256(serialized_params.encode("utf-8")).hexdigest()
    parameters = json.loads(serialized_params)
    workspace = workspace_root or Path(__file__).resolve().parents[3]
    path = workspace / filename
    record = {
        "engine": engine,
        "run_id": f"{engine}-{uuid4().hex}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hash,
        "parameters_sha256": params_hash,
        "input": image_statistics(input_rgb),
        "output": image_statistics(output_image),
        "stages": stage_statistics or {},
        "calibration": calibration or {},
        "parameters": parameters,
        "engine_output": engine_output,
    }
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
