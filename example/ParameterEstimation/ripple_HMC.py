# Import GW packages
import numpy as np
from lal import GreenwichMeanSiderealTime
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import *
from jaxgw.PE.single_event_likelihood import single_detector_likelihood
from jaxgw.PE.detector_projection import make_detector_response, get_detector_response
from jaxgw.PE.generate_noise import generate_noise
import matplotlib.pyplot as plt

# from jaxgw.PE.utils import inner_product
# from jax import grad, vmap
# from functools import partial

# Import FlowMC stuff
from flowMC.sampler.HMC import HMC
from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
import jax
import jax.numpy as jnp

# from jax.scipy.special import logsumexp


#########################################
# Setting up the GW likleilhood
#########################################

# Detector Setup

f_sampling: 2048
duration: 8
fmin: 20

f_sampling = 1024
duration = 16
fmin = 20
ifos = ["H1", "L1", "V1"]

freqs, psd_dict, noise_dict = generate_noise(1234, f_sampling, duration, fmin, ifos)


H1 = get_H1()
L1 = get_L1()
V1 = get_V1()

f_ref = fmin
trigger_time = 1126259462.4
post_trigger_duration = 4
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)

# True parameters for injection

# Mc, eta, chi1, chi2, dist, tc, phic, inclination, polarization_angle, ra, dec,
m1 = 60
m2 = 50
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
true_params = jnp.array(
    [Mc, eta, 0.3, -0.4, 800, 0.0, 0.0, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3]
)

# We now can generate data

f_list = freqs[freqs > fmin]
psd_list = [psd_dict["H1"][freqs > fmin], psd_dict["L1"][freqs > fmin]]
waveform_generator = lambda f_, theta_: gen_IMRPhenomD_polar(f_, theta_, f_ref)

H1_signal = get_detector_response(
    waveform_generator, true_params, f_list, H1, gmst, epoch
)
H1_noise_psd = noise_dict["H1"][freqs > fmin]
H1_data = H1_noise_psd + H1_signal

L1_signal = get_detector_response(
    waveform_generator, true_params, f_list, L1, gmst, epoch
)
L1_noise_psd = noise_dict["L1"][freqs > fmin]
L1_data = L1_noise_psd + L1_signal


def logL(p):
    # Adding on the true ones
    extrinsic_variables = jnp.array(
        [800, 0.0, 0.0, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3]
    )
    params = jnp.concatenate((p, extrinsic_variables))
    print(params)
    logL_H1 = single_detector_likelihood(
        waveform_generator, params, H1_data, f_list, psd_list[0], H1, gmst, epoch
    )
    logL_L1 = single_detector_likelihood(
        waveform_generator, params, L1_data, f_list, psd_list[1], L1, gmst, epoch
    )

    return logL_H1 + logL_L1


# fake_params = jnp.array([Mc, eta, 0.3, -0.4])
# print(logL(fake_params))
# quit()


#########################################
# Purely HMC below
#########################################


# def dual_moon_pe(x):
#     """
#     Term 2 and 3 separate the distribution and smear it along the first and second dimension
#     """
#     term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
#     term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
#     term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
#     return -(term1 - logsumexp(term2) - logsumexp(term3))


n_dim = 4
n_chains = 15
n_local_steps = 3
n_global_steps = 30
step_size = 0.1
n_loop_training = 2
n_loop_production = 2
n_leapfrog = 10

HMC = HMC(
    logL,
    True,
    {
        "step_size": step_size,
        "n_leapfrog": n_leapfrog,
        "inverse_metric": jnp.ones(n_dim),
    },
)

rng_key_set = initialize_rng_keys(n_chains, seed=42)

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

model = RQSpline(n_dim, 4, [32, 32], 8)

print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    HMC,
    logL,
    model,
    n_loop_training=n_loop_training,
    n_loop_production=n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    use_global=False,
)

nf_sampler.sample(initial_position)
