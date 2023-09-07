from jaxtyping import Array
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripple.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from ripple.waveforms.PPE_IMRPhenomD import gen_PPE_IMRPhenomD_hphc
import jax.numpy as jnp
from abc import ABC


class Waveform(ABC):
    def __init__(self):
        return NotImplemented

    def __call__(self, axis: Array, params: Array) -> Array:
        return NotImplemented


class RippleIMRPhenomD(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_PPE_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

class RipplePPEIMRPhenomD(Waveform):

    f_ref: float
    ppe_index: int

    def __init__(self, f_ref: float = 20.0, ppe_index: int = 0):
        self.f_ref = f_ref
        self.ppe_index = ppe_index

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        ppe_index = self.ppe_index
        ppes = jnp.zeros(15)
        ppes = ppes.at[ppe_index].set(params["ppe"])
        #ppes = [
        #    params["dpsi_m2"],
        #    params["dpsi_0"],
        #    params["dpsi_1"],
        #    params["dpsi_2"],
        #    params["dpsi_3"],
        #    params["dpsi_4"],
        #    params["dpsi_5l"],
        #    params["dpsi_6"],
        #    params["dpsi_6l"],
        #    params["dpsi_7"],
        #    params["dbeta_2"],
        #    params["dbeta_3"],
        #    params["dalpha_2"],
        #    params["dalpha_3"],
        #    params["dalpha_4"],
        #]
        hp, hc = gen_PPE_IMRPhenomD_hphc(frequency, theta, ppes, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output


class RippleIMRPhenomPv2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> Array:
        output = {}
        theta = [
            params["M_c"],
            params["eta"],
            params['s1_x'],
            params['s1_y'],
            params["s1_z"],
            params['s2_x'],
            params['s2_y'],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output
