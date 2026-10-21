from __future__ import annotations

import copy
from time import perf_counter

import numpy as np

from spektrafilm.model.density_curves import (
    refresh_density_curves_from_model,
    select_development_time,
)
from spektrafilm.utils.morph_curves import apply_print_curves_morph_with_layers
from spektrafilm.runtime.services import (
    EnlargerService,
    ResizingService,
    SpectralLUTService,
    ColorReferenceService,
)
from spektrafilm.runtime.stages import FilmingStage, PrintingStage, ScanningStage
from spektrafilm.runtime.topology import Node, Tap, run_topology
from spektrafilm.utils.timings import format_timings



class SimulationPipeline:
    """Thin runtime orchestrator that composes stage objects around a
    tap-based topology dispatcher.

    The pipeline declares its stages as a list of :class:`Node` objects via
    :meth:`_build_topology`. Calling :meth:`process` walks the topology in
    declared order, firing each node whose input taps are present in state,
    and returns the value at the requested ``collect`` tap.
    """

    def __init__(self, params, update_params=False):
        self._params = copy.deepcopy(params)

        self.camera = self._params.camera
        self.film = self._params.film
        self.film_render = self._params.film_render
        self.enlarger = self._params.enlarger
        self.print = self._params.print
        self.print_render = self._params.print_render
        self.scanner = self._params.scanner
        self.io = self._params.io
        self.debug = self._params.debug
        self.settings = self._params.settings
        self.taps = self._params.taps

        # The parametric density_curves_model is the source of truth. Sample it
        # onto each profile's log_exposure grid so every downstream consumer
        # (develop, grain, couplers, Dmax probes) reads model-derived curves.
        # Done before services/stages capture the profiles below.
        for profile in (self.film, self.print):
            if profile is not None:
                refresh_density_curves_from_model(profile.data, profile.info.type)
        # A BW development-time family carries one curve / base+fog per
        # development time; collapse the film to the one chemistry.development_time
        # selects, so every downstream consumer sees a single curve and a 1-D base.
        if self.film is not None:
            select_development_time(self.film.data, self.film_render.chemistry.development_time)
            # Bake the film chemistry morph (gamma / developer exhaustion) into the
            # curves, regenerating the grain sublayers from the same morphed params
            # so grain stays consistent. Replaces the old density_curve_gamma; an
            # inactive/identity chemistry reproduces the fitted curves exactly.
            film_data = self.film.data
            film_model = film_data.density_curves_model
            if film_model is not None and film_model.n_layers > 0:
                total, layers = apply_print_curves_morph_with_layers(
                    film_data.log_exposure,
                    film_model,
                    self.film_render.chemistry,
                    profile_type=self.film.info.type,
                )
                film_data.density_curves = total
                film_data.density_curves_layers = layers if film_model.n_layers > 1 else None
        # The print can likewise be a BW development-time family (e.g. a BW print
        # film); collapse it to the chosen development. Its morph is applied in
        # the printing stage, so only the family selection happens here.
        if self.print is not None:
            select_development_time(self.print.data, self.print_render.chemistry.development_time)

        self.timings = {}
        self._last_elapsed_time = None

        self._resize_service = ResizingService(self.io, self.camera.film_format_mm)
        if not update_params:
            self._lut_service = SpectralLUTService(self.settings.lut_resolution)
        self._enlarger_service = EnlargerService(self.enlarger)
        self._color_reference_service = ColorReferenceService(self.film, self.film_render,
                                                              self.print, self.print_render,
                                                              self.scanner.black_correction, self.scanner.white_correction,
                                                              self.scanner.black_level, self.scanner.white_level,
                                                              self.io)

        self._filming_stage = FilmingStage(
            self.film,
            self.film_render,
            self.camera,
            self.io,
            self.settings,
            self._lut_service,
            self._resize_service,
            self._enlarger_service,
            self._color_reference_service,
        )
        self._printing_stage = PrintingStage(
            self.film,
            self.film_render,
            self.print,
            self.print_render,
            self.enlarger,
            self.settings,
            self._lut_service,
            self._enlarger_service,
            self._resize_service,
            self._color_reference_service,
        )
        self._scanning_stage = ScanningStage(
            self.film,
            self.film_render,
            self.print,
            self.print_render,
            self.scanner,
            self.io,
            self.settings,
            self._lut_service,
            self._color_reference_service,
        )

        # timing communication
        self._filming_stage.timings = self.timings
        self._printing_stage.timings = self.timings
        self._scanning_stage.timings = self.timings
        self._lut_service.timings = self.timings

        self._topology: list[Node] = self._build_topology()
        self._diagnostic_callback = None

    def process(
        self,
        image,
        *,
        inject: str | None = None,
        collect: str | None = None,
        on_tap=None,
        on_diagnostic=None,
    ):
        """Run the pipeline.

        Defaults run end-to-end (``rgb_in`` → ``rgb_out``). Pass
        ``inject`` / ``collect`` to enter or exit at a named tap.
        Call-site kwargs override the persistent :class:`TapsParams`
        defaults.
        """
        inject = inject or self.taps.inject or Tap.RGB_IN
        collect = collect or self.taps.collect or Tap.RGB_OUT

        self.timings.clear()
        start = perf_counter()
        previous_diagnostic_callback = self._diagnostic_callback
        self._diagnostic_callback = on_diagnostic
        try:
            return run_topology(
                self._topology,
                inject,
                collect,
                image,
                on_fire=self._record_node_timing,
                on_tap=on_tap,
            )
        finally:
            self._diagnostic_callback = previous_diagnostic_callback
            self._last_elapsed_time = perf_counter() - start

    def get_timings(self):
        return self.timings

    def scan_diagnostics(self):
        from spektrafilm.config import SPECTRAL_SHAPE
        from spektrafilm.model.illuminants import standard_illuminant
        from spektrafilm.runtime.stages.printing import _exposure_factor
        from spektrafilm.utils.autoexposure import measure_raw_autoexposure_ev
        from spektrafilm.utils.conversions import density_to_light

        film_stage = self._filming_stage
        printing_stage = self._printing_stage
        gray_rgb = np.full((1, 1, 3), 0.184, dtype=np.float64)
        raw_unscaled = film_stage._rgb_to_film_raw(gray_rgb)
        auto_ev = (
            float(measure_raw_autoexposure_ev(raw_unscaled, self.camera.auto_exposure_method))
            if self.camera.auto_exposure
            else 0.0
        )
        auto_scale = 2.0**auto_ev if self.camera.auto_exposure else 1.0
        raw_scaled = raw_unscaled * auto_scale
        comp_scale = 2.0**self.camera.exposure_compensation_ev
        raw_comp_unscaled = film_stage._rgb_to_film_raw(gray_rgb * comp_scale)
        raw_comp_scaled = raw_comp_unscaled * auto_scale
        density_unscaled = film_stage._raw_to_density_spectral(raw_unscaled)
        density_comp_unscaled = film_stage._raw_to_density_spectral(raw_comp_unscaled)
        density_midgray = film_stage._raw_to_density_spectral(raw_scaled)
        density_midgray_comp = film_stage._raw_to_density_spectral(raw_comp_scaled)

        sensitivity = np.nan_to_num(10.0 ** self.print.data.log_sensitivity)
        source_illuminant = standard_illuminant(self.enlarger.illuminant)
        print_illuminant = self._enlarger_service.enlarger_filtered_illuminant(source_illuminant)
        factor_midgray = float(_exposure_factor(sensitivity, print_illuminant, density_midgray).item())
        factor_midgray_comp = float(
            _exposure_factor(sensitivity, print_illuminant, density_midgray_comp).item()
        )
        factor_midgray_unscaled = float(
            _exposure_factor(sensitivity, print_illuminant, density_unscaled).item()
        )
        factor_midgray_comp_unscaled = float(
            _exposure_factor(sensitivity, print_illuminant, density_comp_unscaled).item()
        )
        if self.enlarger.normalize_print_exposure and self.enlarger.print_exposure_compensation:
            selected_factor = factor_midgray_comp
        elif self.enlarger.normalize_print_exposure:
            selected_factor = factor_midgray
        elif self.enlarger.print_exposure_compensation:
            selected_factor = factor_midgray_comp / factor_midgray
        else:
            selected_factor = 1.0

        preflash = printing_stage._compute_raw_preflash(source_illuminant, sensitivity)
        bw_correction = float(
            self._color_reference_service.black_white_printing_exposure_correction()
        )
        raw_midgray_by_channel = np.einsum(
            "w,wc->c",
            density_to_light(density_midgray, print_illuminant).reshape(-1),
            sensitivity,
        )
        raw_comp_midgray_by_channel = np.einsum(
            "w,wc->c",
            density_to_light(density_midgray_comp, print_illuminant).reshape(-1),
            sensitivity,
        )
        def as_list(value):
            def clean(item):
                if isinstance(item, list):
                    return [clean(child) for child in item]
                numeric = float(item)
                return numeric if np.isfinite(numeric) else None

            return clean(np.asarray(value).tolist())

        return {
            "midgray": {
                "rgb": [0.184, 0.184, 0.184],
                "raw_before_autoexposure": as_list(raw_unscaled.reshape(-1, 3)[0]),
                "auto_exposure_method": self.camera.auto_exposure_method,
                "auto_exposure_ev": auto_ev,
                "auto_exposure_scale": auto_scale,
                "raw_after_autoexposure": as_list(raw_scaled.reshape(-1, 3)[0]),
                "density_spectral_before_autoexposure": as_list(
                    np.asarray(density_unscaled).reshape(-1)
                ),
                "exposure_compensation_ev": float(self.camera.exposure_compensation_ev),
                "compensated_raw_before_autoexposure": as_list(raw_comp_unscaled.reshape(-1, 3)[0]),
                "compensated_raw_after_autoexposure": as_list(raw_comp_scaled.reshape(-1, 3)[0]),
                "density_spectral": as_list(np.asarray(density_midgray).reshape(-1)),
                "compensated_density_spectral": as_list(
                    np.asarray(density_midgray_comp).reshape(-1)
                ),
            },
            "print_calibration": {
                "illuminant_name": self.enlarger.illuminant,
                "wavelengths_nm": as_list(SPECTRAL_SHAPE.wavelengths),
                "filter_values": {
                    "c": float(self.enlarger.c_filter_neutral),
                    "m": float(self.enlarger.m_filter_neutral),
                    "y": float(self.enlarger.y_filter_neutral),
                    "m_shift": float(self.enlarger.m_filter_shift),
                    "y_shift": float(self.enlarger.y_filter_shift),
                },
                "source_illuminant": as_list(source_illuminant),
                "filtered_illuminant": as_list(print_illuminant),
                "print_sensitivity": as_list(sensitivity),
                "film_channel_density": as_list(self.film.data.channel_density),
                "film_base_density": as_list(self.film.data.base_density),
                "raw_midgray_by_channel": as_list(raw_midgray_by_channel),
                "raw_compensated_midgray_by_channel": as_list(raw_comp_midgray_by_channel),
                "factor_midgray": factor_midgray,
                "factor_midgray_comp": factor_midgray_comp,
                "factor_midgray_after_autoexposure": factor_midgray,
                "factor_midgray_comp_after_autoexposure": factor_midgray_comp,
                "factor_midgray_before_autoexposure": factor_midgray_unscaled,
                "factor_midgray_comp_before_autoexposure": factor_midgray_comp_unscaled,
                "selected_exposure_factor": float(selected_factor),
                "normalize_print_exposure": bool(self.enlarger.normalize_print_exposure),
                "print_exposure_compensation": bool(self.enlarger.print_exposure_compensation),
                "print_exposure": float(self.enlarger.print_exposure),
                "black_white_correction": bw_correction,
                "effective_print_multiplier": float(
                    selected_factor * self.enlarger.print_exposure * bw_correction
                ),
                "preflash_raw": as_list(preflash),
                "preflash_exposure": float(self.enlarger.preflash_exposure),
                "use_enlarger_lut": bool(self.settings.use_enlarger_lut),
                "lut_resolution": int(self.settings.lut_resolution),
                "diffusion_filter_active": bool(self.enlarger.diffusion_filter.active),
                "wavelength_count": int(len(print_illuminant)),
                "non_finite_counts": {
                    "source_illuminant": int(np.count_nonzero(~np.isfinite(source_illuminant))),
                    "filtered_illuminant": int(np.count_nonzero(~np.isfinite(print_illuminant))),
                    "print_sensitivity": int(np.count_nonzero(~np.isfinite(sensitivity))),
                    "film_channel_density": int(
                        np.count_nonzero(~np.isfinite(self.film.data.channel_density))
                    ),
                    "film_base_density": int(
                        np.count_nonzero(~np.isfinite(self.film.data.base_density))
                    ),
                },
            },
        }

    def get_total_elapsed_time(self):
        return self._last_elapsed_time

    def format_timings(self):
        return format_timings(
            self.get_timings(),
            total_elapsed_time=self.get_total_elapsed_time(),
        )

    def print_timings(self):
        print(self.format_timings())

    def update(self, params):
        """Update params and re-initialize stages that depend on them."""
        self.__init__(params, update_params=True)

    def soft_update(self,
                    exposure_compensation_ev=None,
                    print_exposure=None,
                    c_filter_neutral=None,
                    m_filter_neutral=None,
                    y_filter_neutral=None,
                    film_density_curves=None,
                    print_density_curves=None,):
        invalidates_print_balance_reference = False
        if exposure_compensation_ev is not None:
            self.camera.exposure_compensation_ev = exposure_compensation_ev
            invalidates_print_balance_reference = True
        if print_exposure is not None:
            self.enlarger.print_exposure = print_exposure
        if c_filter_neutral is not None:
            self.enlarger.c_filter_neutral = c_filter_neutral
        if m_filter_neutral is not None:
            self.enlarger.m_filter_neutral = m_filter_neutral
        if y_filter_neutral is not None:
            self.enlarger.y_filter_neutral = y_filter_neutral
        if film_density_curves is not None:
            self.film.data.density_curves = film_density_curves
            invalidates_print_balance_reference = True
        if print_density_curves is not None:
            self.print.data.density_curves = print_density_curves
        if invalidates_print_balance_reference:
            (
                self._enlarger_service.density_spectral_midgray,
                self._enlarger_service.density_spectral_midgray_comp,
            ) = self._filming_stage._compute_density_spectral_midgray_to_balance_print()

    # private methods

    def _build_topology(self) -> list[Node]:
        f, p, s = self._filming_stage, self._printing_stage, self._scanning_stage
        common = [
            Node((Tap.RGB_IN,),     (Tap.RGB_PRE,),    self._preprocess, "preprocess"),
            Node((Tap.RGB_PRE,),    (Tap.LOG_E_FILM,), f.expose,          "filming.expose"),
            Node((Tap.LOG_E_FILM,), (Tap.CMY_FILM,),   f.develop,        "filming.develop"),
        ]
        if self.io.scan_film:
            return common + [
                Node((Tap.CMY_FILM,), (Tap.RGB_OUT,), s.scan, "scanning.scan_film"),
            ]
        return common + [
            Node((Tap.CMY_FILM,),    (Tap.LOG_E_PRINT,), self._expose_print, "printing.expose"),
            Node((Tap.LOG_E_PRINT,), (Tap.CMY_PRINT,),   p.develop, "printing.develop"),
            Node((Tap.CMY_PRINT,),   (Tap.RGB_OUT,),     s.scan,    "scanning.scan_print"),
        ]

    def _expose_print(self, cmy_film_density):
        return self._printing_stage.expose(
            cmy_film_density,
            on_diagnostic=self._diagnostic_callback,
        )

    def _preprocess(self, image):
        image = np.double(np.array(image)[:, :, 0:3])
        image = self._resize_service.crop_and_rescale(image)
        return image

    def _record_node_timing(self, node: Node, elapsed: float) -> None:
        self.timings[node.label] = self.timings.get(node.label, 0.0) + elapsed
