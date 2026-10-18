import numpy as np
import pytest

from spektrafilm.model.diffusion import apply_halation_um
from spektrafilm.model.grain import _layer_grain_params, layer_particle_model
from spektrafilm.runtime.params_schema import HalationParams
from spektrafilm.runtime.services.resize import ResizingService


pytestmark = pytest.mark.unit


class _IOParams:
    def __init__(self, *, upscale_factor=1.0, crop=False):
        self.upscale_factor = upscale_factor
        self.crop = crop
        self.crop_center = (0.5, 0.5)
        self.crop_size = (0.5, 0.5)


@pytest.mark.parametrize(
    ("film_format_mm", "shape", "upscale_factor", "expected_pixel_size_um"),
    [
        (35.0, (100, 200, 3), 1.0, 175.0),
        (70.0, (100, 200, 3), 1.0, 350.0),
        (35.0, (200, 400, 3), 1.0, 87.5),
        (35.0, (100, 200, 3), 2.0, 87.5),
    ],
)
def test_film_format_and_raster_set_physical_pixel_size(
    film_format_mm, shape, upscale_factor, expected_pixel_size_um
):
    service = ResizingService(_IOParams(upscale_factor=upscale_factor), film_format_mm)
    output = service.crop_and_rescale(np.zeros(shape, dtype=np.float32))

    assert service.pixel_size_um == pytest.approx(expected_pixel_size_um)
    assert output.shape[:2] == (
        round(shape[0] * upscale_factor),
        round(shape[1] * upscale_factor),
    )


def test_crop_preserves_pitch_and_upscale_changes_it():
    service = ResizingService(_IOParams(upscale_factor=2.0, crop=True), 35.0)
    output = service.crop_and_rescale(np.zeros((100, 200, 3), dtype=np.float32))

    assert output.shape[:2] == (200, 200)
    assert service.pixel_size_um == pytest.approx(87.5)


@pytest.mark.parametrize(
    ("pixel_size_um", "expected_ratio"),
    [(10.0, 1.0), (5.0, 0.25)],
)
def test_grain_particle_count_tracks_physical_pixel_area(pixel_size_um, expected_ratio):
    density_curves_layers = np.linspace(0.0, 2.0, 16)[:, None, None]
    density_curves_layers = np.repeat(density_curves_layers, 3, axis=2)
    density_max_layers = np.nanmax(density_curves_layers, axis=0)

    _, _, _, particles_per_pixel = _layer_grain_params(
        density_max_layers,
        density_min=(0.03, 0.03, 0.03),
        rms_granularity=(8.0, 8.0, 8.0),
        particle_scale_sublayers=(1.0,),
        uniformity=(0.97, 0.97, 0.97),
        n_ch=3,
        pixel_size_um=pixel_size_um,
        density_curves_layers=density_curves_layers,
    )
    reference = _layer_grain_params(
        density_max_layers,
        density_min=(0.03, 0.03, 0.03),
        rms_granularity=(8.0, 8.0, 8.0),
        particle_scale_sublayers=(1.0,),
        uniformity=(0.97, 0.97, 0.97),
        n_ch=3,
        pixel_size_um=10.0,
        density_curves_layers=density_curves_layers,
    )[3]

    np.testing.assert_allclose(particles_per_pixel / reference, expected_ratio)


def test_grain_sampling_handles_very_large_physical_pixels():
    density = np.full((128, 128), 1.0, dtype=np.float64)
    density_max = 2.0
    particles_per_pixel = 1e8
    uniformity = 0.97

    sampled = layer_particle_model(
        density,
        density_max=density_max,
        n_particles_per_pixel=particles_per_pixel,
        grain_uniformity=uniformity,
    )

    expected_std = np.sqrt(
        density[0, 0] * (density_max - uniformity * density[0, 0])
        / particles_per_pixel
    )
    assert np.all(np.isfinite(sampled))
    assert sampled.mean() == pytest.approx(1.0, abs=expected_std * 0.1)
    assert sampled.std() == pytest.approx(expected_std, rel=0.1)


def test_halation_radius_tracks_pixel_pitch_without_changing_energy():
    halation = HalationParams(
        scatter_amount=0.0,
        halation_amount=1.0,
        halation_strength=(1.0, 1.0, 1.0),
        halation_first_sigma_um=(20.0, 20.0, 20.0),
        halation_n_bounces=1,
        halation_renormalize=False,
    )
    center = 64
    outputs = []
    for pixel_size_um in (4.0, 2.0):
        impulse = np.zeros((129, 129, 3), dtype=np.float64)
        impulse[center, center, :] = 1.0
        outputs.append(apply_halation_um(impulse, halation, pixel_size_um))

    def spatial_variance(image):
        channel = image[:, :, 1]
        total = channel.sum()
        coords = np.arange(channel.shape[0], dtype=np.float64) - center
        return (channel * coords[None, :] ** 2).sum() / total

    variance_at_4um = spatial_variance(outputs[0])
    variance_at_2um = spatial_variance(outputs[1])
    assert variance_at_2um / variance_at_4um == pytest.approx(4.0, rel=0.03)
    assert outputs[0].sum() == pytest.approx(outputs[1].sum(), rel=0.001)
