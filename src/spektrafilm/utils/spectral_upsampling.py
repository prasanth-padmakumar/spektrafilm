import numpy as np
import struct
import colour
import scipy
import importlib.resources
from functools import lru_cache
from opt_einsum import contract
import scipy.interpolate
import scipy.special

from spektrafilm.data.profiles_loader import Hanatos2025SensitivityAdaptation
from spektrafilm.utils.fast_interp_lut import apply_lut_cubic_2d
from spektrafilm.config import SPECTRAL_SHAPE, STANDARD_OBSERVER_CMFS
from spektrafilm.model.illuminants import standard_illuminant

_SQRT2PI = float(np.sqrt(2.0 * np.pi))
_HANATOS2025_MAX_CORRECTION_STOPS = 2.0  # matches max_correction_stops used during the fit
ARCTIC2026BETA02_MIDGRAY = 0.184
ARCTIC2026BETA02_SCENE_ILLUMINANT = "D65"

################################################################################
# LUT generatation of irradiance spectra for any xy chromaticity
# Thanks to hanatos for providing luts and sample code to develop this. I am grateful.

def _load_coeffs_lut(filename='hanatos_irradiance_xy_coeffs_250304.lut'):
    # load lut of coefficients for efficient computations of irradiance spectra
    # formatting
    header_fmt = '=4i'
    header_len = struct.calcsize(header_fmt)
    pixel_fmt = '=4f'
    pixel_len = struct.calcsize(pixel_fmt)

    package = importlib.resources.files('spektrafilm.data.luts.spectral_upsampling')
    resource = package / filename
    with resource.open("rb") as file:
        header = file.read(header_len)
        _, _, width, height = struct.Struct(header_fmt).unpack_from(header)
        px = [[0] * height for _ in range(width)]
        for j in range(height):
            for i in range(width):
                data = file.read(pixel_len)
                px[i][j] = struct.Struct(pixel_fmt).unpack_from(data)
    return np.array(px)

def _tri2quad(tc):
    # converts triangular coordinates into square coordinates.
    # for better sampling of the visible locus of xy chromaticities.
    # the lut is represented in triangular coordinates
    tc = np.array(tc)
    tx = tc[...,0]
    ty = tc[...,1]
    y = ty / np.fmax(1.0 - tx, 1e-10)
    x = (1.0 - tx)*(1.0 - tx)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    return np.stack((x,y), axis=-1)

def _quad2tri(xy):
    # converts square coordinates into triangular coordinates
    x = xy[...,0]
    y = xy[...,1]
    tx = 1 - np.sqrt(x)
    ty = y * np.sqrt(x)
    return np.stack((tx,ty), axis=-1)

def _fetch_coeffs(tc, lut_coeffs):
    # find the coefficients for spectral upsampling of given rgb coordinates
    # if color_space!='ITU-R BT.2020' or apply_cctf_decoding:
    #     rgb = colour.RGB_to_RGB(rgb, input_colourspace=color_space, apply_cctf_decoding=apply_cctf_decoding,
    #                                     output_colourspace='ITU-R BT.2020', apply_cctf_encoding=False)
    #     rgb = np.clip(rgb,0,1)
    # xyz = colour.RGB_to_XYZ(rgb, colourspace='ITU-R BT.2020', apply_cctf_decoding=False)
    # b = np.sum(xyz, axis=-1)
    # xy = xyz[...,0:2] / b[...,None]
    # tc = _tri2quad(xy)
    coeffs = np.zeros(np.concatenate((tc.shape[:-1],[4])))
    x = np.linspace(0,1,lut_coeffs.shape[0])
    for i in np.arange(4):
        coeffs[...,i] = scipy.interpolate.RegularGridInterpolator((x,x), lut_coeffs[:,:,i], method='cubic')(tc)
    return coeffs

def _compute_spectra_from_coeffs(coeffs, smooth_steps=1):
    wl = SPECTRAL_SHAPE.wavelengths
    wl_up = np.linspace(360,800,441) # upsampled wl for finer initial calculation 0.5 nm
    x = (coeffs[...,0,None] * wl_up + coeffs[...,1,None])*  wl_up  + coeffs[...,2,None]
    y = 1.0 / np.sqrt(x * x + 1.0)
    spectra = 0.5 * x * y +  0.5
    spectra /= coeffs[...,3][...,None]
    
    # gaussian smooth with smooth_step*sigmas and downsample
    step = np.mean(np.diff(wl))
    spectra = scipy.ndimage.gaussian_filter(spectra, step*smooth_steps, axes=-1)
    def interp_slice(a, wl, wl_up):
        return np.interp(wl, wl_up, a)
    spectra = np.apply_along_axis(interp_slice, axis=-1, wl=wl, wl_up=wl_up, arr=spectra)
    return spectra

def compute_lut_spectra(lut_size=128, smooth_steps=1, lut_coeffs_filename='hanatos_irradiance_xy_coeffs_250304.lut'):
    v = np.linspace(0,1,lut_size)
    tx,ty = np.meshgrid(v,v, indexing='ij')
    tc = np.stack((tx,ty), axis=-1)
    lut_coeffs = _load_coeffs_lut(lut_coeffs_filename)
    coeffs = _fetch_coeffs(tc, lut_coeffs)
    lut_spectra = _compute_spectra_from_coeffs(coeffs, smooth_steps=smooth_steps)
    lut_spectra = np.array(lut_spectra, dtype=np.half)
    return lut_spectra

def _load_hanatos2025_spectra_lut(filename='irradiance_xy_tc.npy'):
    data_path = importlib.resources.files('spektrafilm.data.luts.spectral_upsampling').joinpath(filename)
    with data_path.open('rb') as file:
        spectra_lut = np.double(np.load(file))
    return spectra_lut

def _illuminant_to_xy(illuminant_label):
    illu = standard_illuminant(illuminant_label)
    xyz = np.zeros((3))
    for i in np.arange(3):
        xyz[i] = np.sum(illu * STANDARD_OBSERVER_CMFS[:][:,i])
    xy = xyz[0:2] / np.sum(xyz)
    return xy

def _rgb_to_tc_b(rgb, color_space='ITU-R BT.2020', apply_cctf_decoding=False, reference_illuminant='D55'):
    # source_cs = colour.RGB_COLOURSPACES[color_space]
    # target_cs = source_cs.copy()
    # target_cs.whitepoint = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    # adapted_rgb = colour.RGB_to_RGB(rgb, input_colourspace=source_cs,
    #                                 output_colourspace=target_cs,
    #                                 adaptation_transform='Bradford')    
    # illu_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][reference_illuminant]
    illu_xy = _illuminant_to_xy(reference_illuminant)
    # CAT16 (Li et al. 2017) replaces the cone-primary instabilities
    # of CAT02 around blue and violet; it is the adaptation matrix
    # CIECAM16 and the spektrafilm output-side CAM16-UCS algorithm use
    # internally, so the input chromaticity projection stays in the
    # same adaptation family as the output gamut compression.
    xyz = colour.RGB_to_XYZ(rgb, colourspace=color_space,
                            apply_cctf_decoding=apply_cctf_decoding,
                            illuminant=illu_xy,
                            chromatic_adaptation_transform='CAT16')
    b = np.sum(xyz, axis=-1)
    xy = xyz[...,0:2] / np.fmax(b[...,None], 1e-10)
    # Previously: xy = np.clip(xy, 0, 1). Removed because the input gamut
    # compression baked into the tc_lut at build time (see
    # gamut_compression.remap_tc_lut_for_compression) now handles OOG
    # chromaticities correctly via the LUT itself. _tri2quad below already
    # clamps each output component to [0, 1] for the (rare) inputs that
    # also fall outside the lower xy triangle, so the LUT lookup
    # coordinates remain in-bounds.
    tc = _tri2quad(xy)
    b = np.nan_to_num(b)
    return tc, b

################################################################################
# hanatos2025 sensitivity adaptation


def _radial_mobius_warp_xy(
    xy: np.ndarray,
    center: tuple[float, float],
    alpha: float,
) -> np.ndarray:
    cx, cy = center
    dx = xy[..., 0] - cx
    dy = xy[..., 1] - cy
    d = np.sqrt(dx * dx + dy * dy)
    scale = 1.0 / (1.0 + alpha * d)
    wxy = np.zeros_like(xy)
    wxy[..., 0] = cx + dx * scale
    wxy[..., 1] = cy + dy * scale
    return wxy
    
def hanika_sigmoid(z: np.ndarray, max_val: float) -> np.ndarray:
    """Algebraic sigmoid matching Jakob & Hanika 2019, bounded to [-max_val, max_val]."""
    return z / np.sqrt(1.0 + (z / max_val) ** 2)


def poly2d_deg4(tc: np.ndarray,
                params: np.ndarray,
                center_tc: tuple[float, float] = (0.0, 0.0),) -> np.ndarray:
    x = tc[..., 0] - center_tc[0]
    y = tc[..., 1] - center_tc[1]
    x2 = x * x
    y2 = y * y
    xy_term = x * y
    x3 = x2 * x
    y3 = y2 * y
    _, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 = params
    # note that we want center_tc to be zero correction -> not adding c0
    return (
        c1 * x + c2 * y + c3 * x2 + c4 * y2 + c5 * xy_term
        + c6 * x3 + c7 * y3 + c8 * (x2 * y) + c9 * (x * y2)
        + c10 * (x2 * x2) + c11 * (y2 * y2) + c12 * (x3 * y) + c13 * (x2 * y2) + c14 * (x * y3)
    )

def locked_logistic_rising(
    x: np.ndarray, mu: float, sigma: float, nu: float,
) -> np.ndarray:
    """Rising flex-and-sigma-locked logistic.

    Locked through (μ, ½) with maximum slope `S = 1/(σ·√(2π))`. ν > 0
    bends the tails (ν=1 ≈ erf in shape; ν<1 fattens the lower tail,
    ν>1 thins it). Inputs may broadcast freely.
    """
    nu = np.maximum(np.asarray(nu, dtype=float), 1e-5)
    S = 1.0 / (sigma * _SQRT2PI)
    k = (2.0 * nu * S) / (1.0 - 2.0 ** (-nu))
    exponent = np.clip(-k * (x - mu), -500.0, 500.0)
    nu_log2 = nu * np.log(2.0)
    log_q = np.where(
        nu_log2 > 50.0,
        nu_log2 + np.log1p(-np.exp(-nu_log2)),
        np.log(np.expm1(nu_log2)),
    )
    log_denom = np.logaddexp(0.0, log_q + exponent)
    return np.exp(-(1.0 / nu) * log_denom)


def eval_poly4_warp_log_exposure_surface(params, illuminant_xy) -> np.ndarray:
    """Evaluate a degree-4 polynomial surface after warping in xy, then mapping back to tc."""
    # Warp xy coordinate with the mobius warp, which is a radial compression towards the illuminant chromaticity.
    # This is to better sample the spectra of chromaticities close to the illuminant
    # and stabilize the surface for large chromaticity shifts.
    surface_size = HANATOS2025_SPECTRA_LUT.shape[0]
    tc_base = np.linspace(0,1,surface_size)
    tc = np.stack(np.meshgrid(tc_base, tc_base, indexing='ij'), axis=-1)
    xy = _quad2tri(tc)
    tc_center = _tri2quad(illuminant_xy)
    surface_log_exposure_correction = np.zeros((surface_size, surface_size,3))
    for i in range(3):
        poly_params = params[i, :-1]
        alpha = float(params[i, -1])
        xy_warped = _radial_mobius_warp_xy(xy, illuminant_xy, alpha=alpha)
        tc_warped = _tri2quad(xy_warped)
        raw = poly2d_deg4(tc_warped, poly_params, center_tc=tc_center)
        surface_log_exposure_correction[...,i] = hanika_sigmoid(raw, _HANATOS2025_MAX_CORRECTION_STOPS)
    return surface_log_exposure_correction

def eval_poly4_log_exposure_surface(params, illuminant_xy) -> np.ndarray:
    surface_size = HANATOS2025_SPECTRA_LUT.shape[0]
    tc_base = np.linspace(0,1,surface_size)
    tc = np.stack(np.meshgrid(tc_base, tc_base, indexing='ij'), axis=-1)
    tc_center = _tri2quad(illuminant_xy)
    surface_log_exposure_correction = np.zeros((surface_size, surface_size,3))
    for i in range(3):
        poly_params = params[i]
        raw = poly2d_deg4(tc, poly_params, center_tc=tc_center)
        surface_log_exposure_correction[...,i] = hanika_sigmoid(raw, _HANATOS2025_MAX_CORRECTION_STOPS)
    return surface_log_exposure_correction

def eval_logiflex8_spectral_bandpass(params: np.ndarray) -> np.ndarray:
    """8-param logiflex window: erf6 skeleton + (nu_uv, nu_ir) shared across channels.

    params: (c_uv_base, sigma_uv, c_ir_base, sigma_ir, c_uv_b, c_ir_r, nu_uv, nu_ir).
    """
    wavelengths = SPECTRAL_SHAPE.wavelengths
    (c_uv_base, sigma_uv, c_ir_base, sigma_ir,
     c_uv_b, c_ir_r, nu_uv, nu_ir) = params
    cuv = np.array([c_uv_base, c_uv_base, c_uv_b], dtype=float)
    cir = np.array([c_ir_r, c_ir_base, c_ir_base], dtype=float)
    w = wavelengths[:, None]
    edge_uv = locked_logistic_rising(w, cuv[None, :], sigma_uv, nu_uv)
    edge_ir = 1.0 - locked_logistic_rising(w, cir[None, :], sigma_ir, nu_ir)
    window = edge_uv * edge_ir
    return window

def eval_erf4_spectral_bandpass(params: np.ndarray) -> np.ndarray:
    """4-param erf window: common bandpass replicated across all channels.

    params: (c_uv, sigma_uv, c_ir, sigma_ir).
    """
    _SQRT2 = float(np.sqrt(2.0))
    wavelengths = SPECTRAL_SHAPE.wavelengths
    c_uv, sigma_uv, c_ir, sigma_ir = (float(v) for v in params)
    edge_uv = 0.5 * (1.0 + scipy.special.erf((wavelengths - c_uv) / (sigma_uv * _SQRT2)))
    edge_ir = 0.5 * (1.0 - scipy.special.erf((wavelengths - c_ir) / (sigma_ir * _SQRT2)))
    common = edge_uv * edge_ir
    return np.repeat(common[:, None], 3, axis=1)

def eval_spectral_bandpass_window(params: np.ndarray, model: str = 'erf4') -> np.ndarray:
    if model == 'logiflex8':
        return eval_logiflex8_spectral_bandpass(params)
    if model == 'erf4':
        return eval_erf4_spectral_bandpass(params)
    raise ValueError(f"Unknown spectral bandpass model: {model}")
    
def eval_log_exposure_correction_surface(params, illuminant_xy, model='poly4') -> np.ndarray:
    if model == 'poly4_warp_xy':
        return eval_poly4_warp_log_exposure_surface(params, illuminant_xy)
    if model == 'poly4':
        return eval_poly4_log_exposure_surface(params, illuminant_xy)
    raise ValueError(f"Unknown log exposure correction surface model: {model}")

################################################################################
# From [Mallett2019]

MALLETT2019_BASIS = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019.copy().align(SPECTRAL_SHAPE)
def rgb_to_raw_mallett2019(RGB, sensitivity,
                           color_space='sRGB', apply_cctf_decoding=True,
                           reference_illuminant='D65'):
    """
    Converts an RGB color to a raw sensor response using the method described in Mallett et al. (2019).

    Parameters
    ----------
    RGB : array_like
        RGB color values.
    illuminant : array_like
        Illuminant spectral distribution.
    sensitivity : array_like
        Camera sensor spectral sensitivities.
    color_space : str, optional
        The color space of the input RGB values. Default is 'sRGB'.
    apply_cctf_decoding : bool, optional
        Whether to apply the color component transfer function (CCTF) decoding. Default is True.

    Returns
    -------
    raw : ndarray
        Raw sensor response.
    """
    illuminant = standard_illuminant(reference_illuminant)[:]
    basis_set_with_illuminant = np.array(MALLETT2019_BASIS[:])*np.array(illuminant)[:, None]
    lrgb = colour.RGB_to_RGB(RGB, color_space, 'sRGB',
                    apply_cctf_decoding=apply_cctf_decoding,
                    apply_cctf_encoding=False)
    raw  = contract('ijk,lk,lm->ijm', lrgb, basis_set_with_illuminant, sensitivity)
    raw = np.nan_to_num(raw)
    raw = np.ascontiguousarray(raw)
    
    raw_midgray  = np.einsum('k,km->m', illuminant*0.184, sensitivity) # use 0.184 as midgray reference
    return raw / raw_midgray[1] # normalize with green channel

################################################################################
# Using hanatos irradiance spectra generation

HANATOS2025_SPECTRA_LUT = _load_hanatos2025_spectra_lut()
HANATOS2025_NO_ADAPTATION = Hanatos2025SensitivityAdaptation(apply_surface=False, apply_window=False)

def compute_hanatos2025_tc_lut(sensitivity, hanatos2025_adaptation, gamut_compress=None):
    """Compute the LUT of spectra for given sensitivities and hanatos2025 adaptation parameters.
    Both window and surface preserve white balance.

    Parameters
    ----------
    sensitivity, hanatos2025_adaptation :
        As before.
    gamut_compress :
        Optional ``spektrafilm.utils.gamut_compression.InputGamutCompressSpec``.
        When provided and ``gamut_compress.active``, the returned
        LUT has the input gamut compression baked into it via remap-resample
        (see n100 §3.1): a runtime lookup ``new_lut[xy]`` returns the same
        raw value the uncompressed LUT would have returned for
        ``compress(xy)``. The per-pixel hot path therefore stays
        compression-agnostic.
    """

    spectra_lut = HANATOS2025_SPECTRA_LUT

    if hanatos2025_adaptation.spectral_gaussian_blur > 0:
        spectra_lut = scipy.ndimage.gaussian_filter(spectra_lut,
                        (0, 0, hanatos2025_adaptation.spectral_gaussian_blur))

    if hanatos2025_adaptation.apply_window and hanatos2025_adaptation.window_params is not None:
        window = eval_spectral_bandpass_window(hanatos2025_adaptation.window_params)
        illuminant = standard_illuminant(hanatos2025_adaptation.reference_illuminant)
        normalization = (
            np.sum(sensitivity * illuminant[:, None] * window, axis=0)
            / np.sum(sensitivity * illuminant[:, None], axis=0)
        )
        window = window / normalization
        raw_lut = contract('ijl,lm->ijm', spectra_lut, sensitivity * window)
    else:
        raw_lut = contract('ijl,lm->ijm', spectra_lut, sensitivity)

    if hanatos2025_adaptation.apply_surface and hanatos2025_adaptation.surface_params is not None:
        xy_illu = _illuminant_to_xy(hanatos2025_adaptation.reference_illuminant)
        surface = eval_log_exposure_correction_surface(hanatos2025_adaptation.surface_params,
                                                       illuminant_xy=xy_illu)
        raw_lut *= 2**surface

    if gamut_compress is not None and gamut_compress.active:
        # Bake compression into the LUT at build time. The reference
        # illuminant matches what _rgb_to_tc_b uses at runtime, so the
        # compression's achromatic axis is the film's white.
        from spektrafilm.utils.gamut_compression import remap_tc_lut_for_compression
        ref_xy = _illuminant_to_xy(hanatos2025_adaptation.reference_illuminant)
        raw_lut = remap_tc_lut_for_compression(raw_lut, ref_xy, gamut_compress)

    return raw_lut

def rgb_to_raw_hanatos2025(rgb, sensitivity,
                           color_space,
                           apply_cctf_decoding,
                           reference_illuminant,
                           tc_lut=None
                           ):
    tc_raw, b = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=reference_illuminant,
    )
    if tc_lut is None: # fallback to on-the-fly computation if tc_lut not provided
        tc_lut = compute_hanatos2025_tc_lut(sensitivity, HANATOS2025_NO_ADAPTATION)
    raw = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw *= b[...,None] # scale the raw back with the scale factor
    # note that sensitivities are already normalized in balancing such that raw_midgray is 1, so no need to normalize here
    return raw


def compute_hanatos2026_tc_lut(sensitivity, hanatos2025_adaptation, gamut_compress=None):
    """Compute Hanatos 2026's sensitivity-integrated LUT.

    Hanatos 2026 applies the film's UV/IR adaptation window to the reconstructed
    irradiance spectra and normalizes each channel against the reference-
    illuminant-weighted sensitivity. Unlike Hanatos 2025, it does not apply the
    fitted exposure surface or spectral Gaussian blur.
    """
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    spectra_lut = HANATOS2025_SPECTRA_LUT
    raw_sensitivity = sensitivity

    window_params = hanatos2025_adaptation.window_params
    if (
        hanatos2025_adaptation.apply_window
        and window_params is not None
        and np.asarray(window_params).size == 4
    ):
        window_params = np.asarray(window_params, dtype=np.float64)
        if window_params[1] > 0.0 and window_params[3] > 0.0:
            window = eval_erf4_spectral_bandpass(window_params)
            if sensitivity.shape[1] == 1:
                window = window[:, :1]
            illuminant = standard_illuminant(
                hanatos2025_adaptation.reference_illuminant
            )
            denominator = np.sum(sensitivity * illuminant[:, None], axis=0)
            numerator = np.sum(
                sensitivity * illuminant[:, None] * window, axis=0
            )
            normalization = np.maximum(
                numerator / np.maximum(denominator, 1e-10),
                1e-10,
            )
            raw_sensitivity = sensitivity * window / normalization[None, :]

    raw_lut = contract("ijl,lm->ijm", spectra_lut, raw_sensitivity)

    if gamut_compress is not None and gamut_compress.active:
        from spektrafilm.utils.gamut_compression import remap_tc_lut_for_compression

        reference_xy = _illuminant_to_xy(
            hanatos2025_adaptation.reference_illuminant
        )
        raw_lut = remap_tc_lut_for_compression(
            raw_lut, reference_xy, gamut_compress
        )
    return raw_lut


def rgb_to_raw_hanatos2026(
    rgb,
    sensitivity,
    color_space,
    apply_cctf_decoding,
    reference_illuminant,
    tc_lut=None,
):
    tc_raw, brightness = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=reference_illuminant,
    )
    if tc_lut is None:
        tc_lut = compute_hanatos2026_tc_lut(
            sensitivity,
            Hanatos2025SensitivityAdaptation(
                apply_window=False,
                apply_surface=False,
                reference_illuminant=reference_illuminant,
            ),
        )
    raw = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw *= brightness[..., None]
    return raw


@lru_cache(maxsize=1)
def load_arctic2026beta02_spectra_lut():
    """Load Arctic 2026 beta02's D65-recovered reflectance LUT."""
    resource = (
        importlib.resources.files("spektrafilm.data.luts.spectral_upsampling")
        .joinpath("arctic2026beta02", "reflectance_xy_tc.npy")
    )
    with resource.open("rb") as file:
        spectra_lut = np.asarray(np.load(file, allow_pickle=False), dtype=np.float64)
    expected_shape = (192, 192, len(SPECTRAL_SHAPE.wavelengths))
    if spectra_lut.shape != expected_shape:
        raise ValueError(
            f"Arctic 2026 beta02 LUT must have shape {expected_shape}, "
            f"got {spectra_lut.shape}"
        )
    if not np.all(np.isfinite(spectra_lut)):
        raise ValueError("Arctic 2026 beta02 LUT contains non-finite values")
    return spectra_lut

@lru_cache(maxsize=1)
def load_arctic2026beta04_spectra_lut():
    resource = (importlib.resources.files("spektrafilm.data.luts.spectral_upsampling")
                .joinpath("arctic2026beta04", "reflectance_xy_tc.npy"))
    with resource.open("rb") as file:
        spectra_lut = np.asarray(np.load(file, allow_pickle=False), dtype=np.float64)
    expected_shape = (192, 192, len(SPECTRAL_SHAPE.wavelengths))
    if spectra_lut.shape != expected_shape:
        raise ValueError(f"Arctic 2026 beta04 LUT must have shape {expected_shape}, got {spectra_lut.shape}")
    return spectra_lut


def compute_arctic2026beta02_tc_lut(
    sensitivity,
    reference_illuminant,
    *,
    neutral_sensitivity=None,
    gamut_compress=None,
    spectra_lut=None,
):
    """Build the film-raw LUT from Arctic beta02 reflectance spectra.

    The reflectance table is projected under D65, but is relit by the film's
    reference illuminant before integration. Each channel is white-balanced
    against the unfiltered film sensitivity, then anchored so 18.4% D65 gray
    maps to unit raw exposure.
    """
    if spectra_lut is None:
        spectra_lut = load_arctic2026beta02_spectra_lut()
    spectra_lut = np.asarray(spectra_lut, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    if sensitivity.ndim != 2:
        raise ValueError("film sensitivity must be a wavelength-by-channel array")
    if spectra_lut.ndim != 3 or spectra_lut.shape[-1] != sensitivity.shape[0]:
        raise ValueError(
            "Arctic 2026 beta02 spectra and film sensitivity wavelength "
            "dimensions must match"
        )
    if neutral_sensitivity is None:
        neutral_sensitivity = sensitivity
    neutral_sensitivity = np.asarray(neutral_sensitivity, dtype=np.float64)
    if neutral_sensitivity.shape != sensitivity.shape:
        raise ValueError(
            "neutral_sensitivity must have the same shape as sensitivity"
        )

    illuminant = standard_illuminant(reference_illuminant)
    if illuminant.shape != (sensitivity.shape[0],):
        raise ValueError(
            "reference illuminant and film sensitivity wavelength dimensions "
            "must match"
        )
    relit_spectra = spectra_lut * illuminant[None, None, :]
    raw_lut = contract("ijl,lm->ijm", relit_spectra, sensitivity)

    scene_xy = _illuminant_to_xy(ARCTIC2026BETA02_SCENE_ILLUMINANT)
    neutral_tc, midgray_brightness = _rgb_to_tc_b(
        np.full((1, 1, 3), ARCTIC2026BETA02_MIDGRAY, dtype=np.float64),
        color_space="sRGB",
        apply_cctf_decoding=False,
        reference_illuminant=ARCTIC2026BETA02_SCENE_ILLUMINANT,
    )
    neutral_spectrum = apply_lut_cubic_2d(spectra_lut, neutral_tc)[0, 0]
    raw_neutral = contract(
        "l,lm->m", neutral_spectrum * illuminant, neutral_sensitivity
    )
    if not np.all(np.isfinite(raw_neutral)) or np.any(raw_neutral <= 0):
        raise ValueError(
            "Arctic 2026 beta02 produced a non-positive or non-finite neutral "
            "film exposure"
        )
    if not np.all(np.isfinite(midgray_brightness)) or np.any(midgray_brightness <= 0):
        raise ValueError("Arctic 2026 beta02 midgray brightness is invalid")

    raw_lut /= raw_neutral[None, None, :] * midgray_brightness[..., None]

    if gamut_compress is not None and gamut_compress.active:
        from spektrafilm.utils.gamut_compression import remap_tc_lut_for_compression

        raw_lut = remap_tc_lut_for_compression(raw_lut, scene_xy, gamut_compress)
    return raw_lut


def rgb_to_raw_arctic2026beta02(
    rgb,
    sensitivity,
    color_space,
    apply_cctf_decoding,
    reference_illuminant,
    tc_lut=None,
):
    """Convert RGB to film raw using the D65-projected Arctic beta02 LUT."""
    if tc_lut is None:
        tc_lut = compute_arctic2026beta02_tc_lut(
            sensitivity, reference_illuminant
        )
    tc_raw, brightness = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=ARCTIC2026BETA02_SCENE_ILLUMINANT,
    )
    raw = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw *= brightness[..., None]
    return raw

def rgb_to_raw_arctic2026beta04(rgb, sensitivity, color_space,
                                apply_cctf_decoding, reference_illuminant,
                                tc_lut=None):
    if tc_lut is None:
        tc_lut = compute_arctic2026beta02_tc_lut(
            sensitivity, reference_illuminant,
            spectra_lut=load_arctic2026beta04_spectra_lut())
    tc_raw, brightness = _rgb_to_tc_b(
        rgb, color_space=color_space, apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=ARCTIC2026BETA02_SCENE_ILLUMINANT)
    return apply_lut_cubic_2d(tc_lut, tc_raw) * brightness[..., None]


def rgb_to_smooth_spectrum(rgb, color_space, apply_cctf_decoding, reference_illuminant):
    # direct interpolation of the spectra lut, to be used only for smooth spectra close to white
    tc_w, b_w = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=reference_illuminant,
    )
    v = np.linspace(0, 1, HANATOS2025_SPECTRA_LUT.shape[0])
    spectrum_w = scipy.interpolate.RegularGridInterpolator((v,v), HANATOS2025_SPECTRA_LUT)(tc_w)
    spectrum_w *= b_w
    return spectrum_w.flatten()


if __name__=='__main__':
    lut_coeffs = _load_coeffs_lut()
    coeffs = _fetch_coeffs(np.array([[1,1]]) ,lut_coeffs)
    spectra = _compute_spectra_from_coeffs(coeffs)
    lut_spectra = compute_lut_spectra(lut_size=128)
