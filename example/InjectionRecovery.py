from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
from ripple import ms_to_Mc_eta, Mc_eta_to_ms
import jax.numpy as jnp
import jax
from astropy.time import Time

from tap import Tap
import yaml
from tqdm import tqdm

from jax import config
config.update("jax_enable_x64", True)

class InjectionRecoveryParser(Tap):
    config: str 
    
    # Noise parameters
    seed: int = None
    f_sampling: int  = None
    duration: int = None
    fmin: float = None
    ifos: list[str]  = None

    # Injection parameters
    m1: float = None
    m2: float = None
    chi1: float = None
    chi2: float = None
    dist_mpc: float = None
    tc: float = None
    phic: float = None
    inclination: float = None
    polarization_angle: float = None
    ra: float = None
    dec: float = None

    # Sampler parameters
    n_dim: int = None
    n_chains: int = None
    n_loop_training: int = None
    n_loop_production: int = None
    n_local_steps: int = None
    n_global_steps: int = None
    learning_rate: float = None
    max_samples: int = None
    momentum: float = None
    num_epochs: int = None
    batch_size: int = None
    stepsize: float = None

    # Output parameters
    output_path: str = None
    downsample_factor: int = None


args = InjectionRecoveryParser().parse_args()

opt = vars(args)
yaml_var = yaml.load(open(opt['config'], 'r'), Loader=yaml.FullLoader)
opt.update(yaml_var)

# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

#Fetch injection parameters and inject signal

print("Injection signals")

freqs = jnp.linspace(args.fmin, args.f_sampling/2, args.duration*args.f_sampling//2)

Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
f_ref = 30.0
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = args.duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad


# tc prior hack
tc_low = args.tc - abs(args.tc) * 0.2
tc_up = args.tc + abs(args.tc) * 0.2
waveform = RippleIMRPhenomD(f_ref=f_ref)
prior = Uniform(
    xmin = [20, 0.125, -1., -1., 100., tc_low, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., 1., 1., 1600., tc_up, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": ("eta", lambda q: q/(1+q)**2),
                 #"cos_iota": ("iota",lambda cos_iota: jnp.arccos(cos_iota)),
                 #"sin_dec": ("dec",lambda sin_dec: jnp.arcsin(sin_dec))}
                 "cos_iota": ("iota",lambda cos_iota: jnp.arccos(jnp.arcsin(jnp.sin(cos_iota/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda sin_dec: jnp.arcsin(jnp.arcsin(jnp.sin(sin_dec/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
)
true_param = jnp.array([Mc, eta, args.chi1, args.chi2, args.dist_mpc, args.tc, args.phic, args.inclination, args.polarization_angle, args.ra, args.dec])
true_param = prior.add_name(true_param, with_transform=True)
detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle, "epoch": epoch, "t_c": args.tc}
h_sky = waveform(freqs, true_param)
key, subkey = jax.random.split(jax.random.PRNGKey(args.seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)

likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, args.duration, post_trigger_duration)

#test_param = jnp.array([4.09444581e+01,  2.28985473e-01,  5.71953570e-01, -7.70069604e-01,
#  1.22731078e+03, -2.22176576e-02,  1.20768179e+00, -1.89423792e-01,
#  5.90291631e-01,  3.52329394e+00, 7.09614722e-01])
print(Mc)
test_param = jnp.array([    
                        Mc, 
                        eta, 
                        args.chi1, 
                        args.chi2, 
                        args.dist_mpc, 
                        args.tc, 
                        args.phic, 
                        args.inclination, 
                        args.polarization_angle, 
                        args.ra, 
                        args.dec])
#max_q = 9.9993967e-1
#max_cos_iota = 2.21078245e-01
#max_sin_dec = 3.3862155e-1
#
#max_eta = max_q/(1 + max_q) **2
#max_iota = jnp.arccos(jnp.arcsin(jnp.sin(max_cos_iota/2*jnp.pi))*2/jnp.pi)
#max_dec = jnp.arcsin(jnp.arcsin(jnp.sin(max_sin_dec/2*jnp.pi))*2/jnp.pi)
#maximize_param = jnp.array([1.94841923e1,
#                            max_eta,
#                            -4.6570421e-1,
#                            -4.6607e-1,
#                            6.63507346e+2,
#                            -3.282078e-2,
#                            4.0630425,
#                            max_iota,
#                            1.161489,
#                            4.342504,
#                            max_dec])
log_likelihood = likelihood.evaluate(test_param,{})
print("log_likelihood:", log_likelihood)
#maximize_likelihood = likelihood.evaluate(maximize_param,{})
#print("log_likelihood:", maximize_likelihood)


mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix*3e-3}

jim = Jim(likelihood, 
        prior,
        n_loop_training=10,
        n_loop_production = 10,
        n_local_steps=300,
        n_global_steps=300,
        n_chains=500,
        n_epochs=300,
        learning_rate = 0.001,
        momentum = 0.9,
        batch_size = 50000,
        use_global=True,
        keep_quantile=0.,
        train_thinning = 40,
        local_sampler_arg = local_sampler_arg,
        seed = args.seed,
        )

sample = jim.maximize_likelihood([prior.xmin, prior.xmax], n_loops=2000)
print(sample)
key, subkey = jax.random.split(key)
jim.sample(subkey)
samples = jim.get_samples()
jim.print_summary()
chains, log_prob, local_accs, global_accs= jim.Sampler.get_sampler_state().values()
# print("Script complete and took: {} minutes".format((time.time()-total_time_start)/60))
jnp.savez(args.output_path + '.npz', chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)