# Import packages
import numpy as np
import jax.numpy as jnp
import jax
from lal import GreenwichMeanSiderealTime

import os
import sys
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar, gen_IMRPhenomD
from jaxgw.PE.detector_preset import *
from jaxgw.PE.single_event_likelihood import single_detector_likelihood
from jaxgw.PE.detector_projection import make_detector_response, get_detector_response
from jaxgw.PE.generate_noise import generate_noise
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from jaxgw.PE.utils import inner_product
from jax import grad, vmap
from functools import partial
import time

import argparse
import yaml


########################################
# Detector setup
########################################
f_sampling: 2048
duration: 8
fmin: 20

f_sampling = 1024
duration = 16
fmin = 20
ifos = ["H1", "L1", "V1"]

freqs, psd_dict, noise_dict = generate_noise(1234, f_sampling, duration, fmin, ifos)

########################################
# Generating parameters
########################################


# Mc, eta, chi1, chi2, dist, tc, phic, inclination, polarization_angle, ra, dec,
m1 = 60
m2 = 50
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
params = jnp.array(
    [Mc, eta, 0.3, -0.4, 800, 0.0, 0.0, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3,]
)

########################################
# Setting up the structure to calculate Fisher Matrices
########################################

H1 = get_H1()
L1 = get_L1()
V1 = get_V1()

f_ref = fmin
trigger_time = 1126259462.4
post_trigger_duration = 4
epoch = duration - post_trigger_duration
# gmst = GreenwichMeanSiderealTime(trigger_time)
gmst = 0.0

f_list = freqs[freqs > fmin]
psd_list = [psd_dict["H1"], psd_dict["L1"], psd_dict["V1"]]
waveform_generator = lambda f_, theta_: gen_IMRPhenomD_polar(f_, theta_, f_ref)

strain_real_H1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, H1, gmst, epoch
).real
strain_imag_H1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, H1, gmst, epoch
).imag

strain_real_L1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, L1, gmst, epoch
).real
strain_imag_L1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, L1, gmst, epoch
).imag


strain_real_V1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, V1, gmst, epoch
).real
strain_imag_V1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, V1, gmst, epoch
).imag


def get_strain_derivatives_H1(theta):
    hp_grad_real = vmap(partial(grad(strain_real_H1), theta))(f_list)
    hp_grad_imag = vmap(partial(grad(strain_imag_H1), theta))(f_list)
    return hp_grad_real + hp_grad_imag * 1j


def get_strain_derivatives_L1(theta):
    hp_grad_real = vmap(partial(grad(strain_real_L1), theta))(f_list)
    hp_grad_imag = vmap(partial(grad(strain_imag_L1), theta))(f_list)
    return hp_grad_real + hp_grad_imag * 1j


def get_strain_derivatives_V1(theta):
    hp_grad_real = vmap(partial(grad(strain_real_V1), theta))(f_list)
    hp_grad_imag = vmap(partial(grad(strain_imag_V1), theta))(f_list)
    return hp_grad_real + hp_grad_imag * 1j


# @jax.jit
def calc_SNR(p):
    h0_H1 = get_detector_response(waveform_generator, p, f_list, H1, gmst, epoch)
    h0_L1 = get_detector_response(waveform_generator, p, f_list, L1, gmst, epoch)
    h0_L1 = get_detector_response(waveform_generator, p, f_list, V1, gmst, epoch)
    SNR2_H1 = inner_product(h0_H1, h0_H1, f_list, PSD_vals_H1)
    SNR2_L1 = inner_product(h0_L1, h0_L1, f_list, PSD_vals_L1)
    SNR2_V1 = inner_product(h0_H1, h0_H1, f_list, PSD_vals_V1)

    return jnp.sqrt(SNR2_H1 + SNR2_L1 + SNR2_V1)


Nderivs = 11


# @jax.jit
def get_Fisher_H1(p):
    strain_grad = get_strain_derivatives_H1(p)
    F = jnp.zeros([Nderivs, Nderivs])
    for i in range(Nderivs):
        for j in range(Nderivs):
            F = F.at[i, j].set(
                inner_product(strain_grad[:, i], strain_grad[:, j], f_list, PSD_vals_H1)
            )
    return F


# @jax.jit
def get_Fisher_L1(p):
    strain_grad = get_strain_derivatives_L1(p)
    F = jnp.zeros([Nderivs, Nderivs])
    for i in range(Nderivs):
        for j in range(Nderivs):
            F = F.at[i, j].set(
                inner_product(strain_grad[:, i], strain_grad[:, j], f_list, PSD_vals_L1)
            )
    return F


# @jax.jit
def get_Fisher_V1(p):
    strain_grad = get_strain_derivatives_V1(p)
    F = jnp.zeros([Nderivs, Nderivs])
    for i in range(Nderivs):
        for j in range(Nderivs):
            F = F.at[i, j].set(
                inner_product(strain_grad[:, i], strain_grad[:, j], f_list, PSD_vals_V1)
            )
    return F


PSD_vals_H1 = jax.numpy.interp(f_list, freqs, psd_list[0])
PSD_vals_L1 = jax.numpy.interp(f_list, freqs, psd_list[1])
PSD_vals_V1 = jax.numpy.interp(f_list, freqs, psd_list[2])


SNR = calc_SNR(params)

F_H1 = get_Fisher_H1(params)
F_L1 = get_Fisher_L1(params)
F_V1 = get_Fisher_V1(params)
F = F_H1 + F_L1 + F_V1
cov = np.linalg.inv(F)

print("SNR:", SNR)
print("Fisher Diagonals", np.diag(F))
print("Covariance:", np.diag(cov))
