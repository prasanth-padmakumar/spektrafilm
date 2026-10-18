from __future__ import annotations

import numpy as np
from skimage.transform import rescale

from spektrafilm.utils.crop_resize import crop_image


class ResizingService:
    """Shared resize/crop operations used by runtime stages.

    ``pixel_size_um`` is the physical film-plane size of an output pixel.
    It is derived from the selected film format's long edge and the input
    raster's long edge, then adjusted for any output rescaling. Cropping does
    not change the physical sampling pitch.
    """

    def __init__(self, io_params, film_format_mm: float):
        self._io = io_params
        self.film_format_mm = film_format_mm
        self.pixel_size_um: float | None = None
        # Downsampled copy of the full frame, captured before any crop/rescale.
        # Autoexposure meters this so the metered exposure reflects the whole
        # captured frame and does not shift when the crop changes.
        self.full_frame_preview: np.ndarray | None = None

    def crop_and_rescale(self, image: np.ndarray) -> np.ndarray:
        input_long_edge_pixels = max(image.shape[0:2])
        output_scale = self._io.upscale_factor
        if self.film_format_mm <= 0:
            raise ValueError("film_format_mm must be greater than zero")
        if input_long_edge_pixels <= 0:
            raise ValueError("input image dimensions must be greater than zero")
        if output_scale <= 0:
            raise ValueError("upscale_factor must be greater than zero")
        self.pixel_size_um = (
            self.film_format_mm * 1000.0
            / (input_long_edge_pixels * output_scale)
        )

        # Stash the metering preview from the full frame, before cropping and
        # rescaling, so autoexposure is independent of the crop.
        self.full_frame_preview = _compute_small_preview(image)

        if self._io.crop:
            image = crop_image(image, center=self._io.crop_center, size=self._io.crop_size)

        if output_scale != 1.0:
            image = rescale(
                image,
                output_scale,
                channel_axis=2,
                order=3,
            )
        return image

# helper
def _compute_small_preview(image: np.ndarray,
                    max_size: int = 256) -> np.ndarray:
    if max(image.shape[0:2]) > max_size:
        scale_factor = max_size / max(image.shape[0:2])
        return rescale(
            image,
            scale_factor,
            channel_axis=2,
            order=0,
        )
    return image
    