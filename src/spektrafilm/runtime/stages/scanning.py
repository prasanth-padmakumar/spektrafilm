from __future__ import annotations

import colour
import numpy as np
from opt_einsum import contract
from colour.appearance import InductionFactors_CIECAM02

from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.model.diffusion import apply_gaussian_blur, apply_unsharp_mask, match_channels
from spektrafilm.model.develop import compute_density_spectral
from spektrafilm.model.glare import add_glare
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.conversions import density_to_light
from spektrafilm.utils.gamut_compression import compress_rgb
from spektrafilm.utils.auto_levels import apply_auto_levels


_PROJECTOR_SURROUND = InductionFactors_CIECAM02(F=0.8, c=0.525, N_c=0.8)
_DISPLAY_SURROUND = colour.VIEWING_CONDITIONS_CIECAM02["Average"]
_PROJECTOR_ADAPTING_LUMINANCE = 15.0
_PROJECTOR_BACKGROUND_Y = 2.0
_DISPLAY_ADAPTING_LUMINANCE = 64.0
_DISPLAY_BACKGROUND_Y = 20.0


def _adapt_projected_slide_xyz_to_display(
    xyz: np.ndarray,
    white_xyz: np.ndarray,
) -> np.ndarray:
    """Adapt projected-slide appearance in a dark surround to display viewing.

    The fixed CAM16-UCS conditions are representative, not a calibrated
    projector model.
    """
    appearance = colour.XYZ_to_CAM16UCS(
        xyz,
        XYZ_w=white_xyz,
        L_A=_PROJECTOR_ADAPTING_LUMINANCE,
        Y_b=_PROJECTOR_BACKGROUND_Y,
        surround=_PROJECTOR_SURROUND,
    )
    return np.asarray(colour.CAM16UCS_to_XYZ(
        appearance,
        XYZ_w=white_xyz,
        L_A=_DISPLAY_ADAPTING_LUMINANCE,
        Y_b=_DISPLAY_BACKGROUND_Y,
        surround=_DISPLAY_SURROUND,
    ))


class ScanningStage:
    def __init__(
        self,
        film,
        film_render_params,
        print_profile,
        print_render_params,
        scanner_params,
        io_params,
        settings_params,
        lut_service,
        color_reference_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print_profile
        self._print_render = print_render_params
        self._scanner = scanner_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._color_reference_service = color_reference_service
        
        self.cmy_to_log_xyz = self._return_callable_cmy_to_log_xyz()
        
        # communicate to the color reference service the callable to convert cmy densities to log xyz
        self._color_reference_service.cmy_to_log_xyz = self.cmy_to_log_xyz
        
    # public methods

    def scan(self, density_channels: np.ndarray) -> np.ndarray:
        rgb = self._density_to_rgb(density_channels, use_lut=self._settings.use_scanner_lut)
        rgb = self._apply_blur_and_unsharp(rgb)
        rgb = self._apply_cctf_encoding(rgb)
        rgb = apply_auto_levels(rgb, self._io.auto_levels)
        return rgb

    # private methods

    def _density_to_rgb(self, density_channels: np.ndarray, *, use_lut: bool) -> np.ndarray:
        if self._io.scan_film:
            glare = None
            density_min = -match_channels(self._film_render.grain.density_min, density_channels.shape[-1])
            density_max = np.nanmax(self._film.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
        else:
            glare = self._print_render.glare
            density_min = np.nanmin(self._print.data.density_curves, axis=0)
            density_max = np.nanmax(self._print.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        log_xyz = self._lut_service.spectral_compute_scanner(
            density_channels,
            spectral_calculation=self.cmy_to_log_xyz,
            data_min=density_min,
            data_max=density_max,
            use_lut=use_lut,
        )
        xyz = 10 ** log_xyz
        xyz = self._color_reference_service.black_white_xyz_correction(xyz)
        illuminant_xyz = contract("k,kl->l", scan_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        xyz = add_glare(xyz, illuminant_xyz, glare)
        if (
            self._io.project_positive_as_slide
            and self._io.scan_film
            and self._film.info.type == "positive"
        ):
            xyz = _adapt_projected_slide_xyz_to_display(xyz, illuminant_xyz)
        rgb = colour.XYZ_to_RGB(
            xyz,
            colourspace=self._io.output_color_space,
            apply_cctf_encoding=False,
            illuminant=illuminant_xy,
        )
        # Output gamut compression. Compresses chromaticities the
        # simulation reached that fall outside the output primaries
        # cube; for perceptual algorithms (oklch / oklrab / jzazbz /
        # cam16ucs) the spec's lightness_compression also pulls
        # super-bright pixels back into the cube via a one-sided soft
        # roll-off on the perceptual lightness axis (black stays at 0).
        # With both in place the output is in [0, 1] without a
        # downstream clip; see n100 / n110 for the design and b40 for
        # the smoothness analysis.
        rgb = compress_rgb(
            rgb, self._io.output_gamut_compress,
            output_color_space=self._io.output_color_space,
        )
        return rgb

    def _return_callable_cmy_to_log_xyz(self):
        if self._io.scan_film:
            channel_density = self._film.data.channel_density
            base_density = self._film.data.base_density
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
        else:
            channel_density = self._print.data.channel_density
            base_density = self._print.data.base_density
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        def cmy_to_log_xyz(density_cmy: np.ndarray) -> np.ndarray:
            density_spectral = compute_density_spectral(
                channel_density,
                density_cmy,
                base_density,
                base_density_params=self._film_render.base,
            )
            light = density_to_light(density_spectral, scan_illuminant)
            xyz = contract("ijk,kl->ijl", light, STANDARD_OBSERVER_CMFS[:]) / normalization
            return np.log10(np.fmax(xyz, 0.0) + 1e-10)
        return cmy_to_log_xyz

    def _apply_blur_and_unsharp(self, rgb: np.ndarray) -> np.ndarray:
        rgb = apply_gaussian_blur(rgb, self._scanner.lens_blur)
        sigma, amount = self._scanner.unsharp_mask
        if sigma > 0 and amount > 0:
            rgb = apply_unsharp_mask(rgb, sigma=sigma, amount=amount)
        return rgb

    def _apply_cctf_encoding(self, rgb: np.ndarray) -> np.ndarray:
        if self._io.output_cctf_encoding:
            rgb = colour.RGB_to_RGB(
                rgb,
                self._io.output_color_space,
                self._io.output_color_space,
                apply_cctf_decoding=False,
                apply_cctf_encoding=True,
            )
        return rgb
