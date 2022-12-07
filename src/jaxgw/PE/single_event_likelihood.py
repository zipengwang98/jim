from jax import jit
from jaxgw.PE.detector_projection import get_detector_response
from jaxgw.PE.utils import inner_product
import jax.numpy as jnp


# def single_detector_likelihood(
#     waveform_model, intrinsic_params, data_f, PSD, detector, gmst, epoch
# ):

#     extrinsic = jnp.array([0.3, -0.4, 400.0, 0.02, 0.1, 0.5, 0.2, 1.2, 0.3])
#     params = jnp.append(intrinsic_params, extrinsic)
#     waveform_detector = get_detector_response(
#         waveform_model, params, data_f, detector, gmst, epoch
#     )
#     index = jnp.where(jnp.abs(waveform_detector) > 0)

#     optimal_SNR = inner_product(
#         waveform_detector[index], waveform_detector[index], data_f[index], PSD[index]
#     )
#     return optimal_SNR


def single_detector_likelihood(
    waveform_model, params, data, data_f, PSD, detector, gmst, epoch
):
    waveform = get_detector_response(
        waveform_model, params, data_f, detector, gmst, epoch
    )
    match_filter_SNR = inner_product(waveform, data, data_f, PSD)
    optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
    return match_filter_SNR - optimal_SNR / 2

