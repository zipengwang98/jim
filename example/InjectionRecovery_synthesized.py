from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from jimgw.prior import Uniform, Composite, Sphere
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time
import random

import sys
args = sys.argv[1:]
if len(args) == 0:
    raise ValueError("No command-line arguments provided.")

index = int(args[0])
seed = index + 1234

jax.config.update("jax_enable_x64", True)


# some constants definitions
gate_keeping_posterior_low = 12 # likelihood less than this is discarded
gate_keeping_posterior_high = 1000 # likelihood above this is discarded
#seed = -42
f_min = 20.
f_max = 1024.
f_sampling = 4096.0
duration = 4
post_trigger_duration = 2
f_ref = f_min
trigger_time = 1126259462.4
epoch = duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
print("Injection signals")


# define freq data
freqs = jnp.linspace(f_min, f_sampling/2, int(duration*f_sampling))
freqs = freqs[(freqs>=f_min) & (freqs<=f_max)]

# define prior

#Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
#Mc_low = Mc - Mc * 0.2
#Mc_high = Mc + Mc * 0.2
Mc_prior = Uniform(10, 80, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1_prior = Sphere(naming="s1")
s2_prior = Sphere(naming="s2")
dL_prior = Uniform(10.0, 1500.0, naming=["d_L"])
t_c_prior = Uniform(-0.002, 0.002, naming=["t_c"])
phase_c_prior = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
cos_iota_prior = Uniform(
    -1.0,
    1.0,
    naming=["cos_iota"],
    transforms={
        "cos_iota": (
            "iota",
            lambda params: jnp.arccos(
                jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)
psi_prior = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
sin_dec_prior = Uniform(
    -1.0,
    1.0,
    naming=["sin_dec"],
    transforms={
        "sin_dec": (
            "dec",
            lambda params: jnp.arcsin(
                jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)

prior = Composite(
    [
        Mc_prior,
        q_prior,
        s1_prior,
        s2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]
)

InjectionWaveform = RippleIMRPhenomPv2(f_ref=f_ref)
bounds = jnp.array([[10, 80], 
                    [0.125, 1.0], 
                    [0.0, jnp.pi], #s1
                    [0.0, 2 * jnp.pi],
                    [0.0, 1.0], 
                    [0.0, jnp.pi], #s2
                    [0.0, 2 * jnp.pi],
                    [0.0, 1.0], 
                    [0.0, 2000.0], 
                    [-0.002, 0.002], 
                    [0.0, 2 * jnp.pi], 
                    [-1.0, 1.0], 
                    [0.0, jnp.pi], 
                    [0.0, 2 * jnp.pi], 
                    [-1.0, 1.0]])

def test_log_likelihood(true_param,seed, prior):
    true_param_transformed = prior.transform(true_param)
    detector_parameters = {
            "ra": true_param_transformed["ra"],
            "dec": true_param_transformed["dec"],
            "psi": true_param_transformed["psi"],
            "t_c": true_param_transformed["t_c"],
            "gmst": gmst,
            "epoch": epoch
    }
    #print("detector parameters:", detector_parameters)
    #print("transformed true parameters:", true_param_transformed)
    h_sky = InjectionWaveform(freqs, true_param_transformed)

    #injections
    key, subkey = jax.random.split(jax.random.PRNGKey(seed + 1901))
    H1.inject_signal(subkey, freqs, h_sky, detector_parameters)
    key, subkey = jax.random.split(key)
    L1.inject_signal(subkey, freqs, h_sky, detector_parameters)
    key, subkey = jax.random.split(key)
    V1.inject_signal(subkey, freqs, h_sky, detector_parameters)

    likelihood = HeterodynedTransientLikelihoodFD(
    [H1, L1, V1],
    prior=prior,
    bounds=bounds,
    waveform=RippleIMRPhenomPv2(),
    trigger_time=trigger_time,
    duration=duration,
    post_trigger_duration=post_trigger_duration,
    popsize=250,
    n_loops=1500,
    ref_params = true_param_transformed,
    n_bins =2500,
    )
    logL = likelihood.evaluate_original(true_param_transformed,{})
    prior_value = prior.log_prob(true_param)
    return logL, prior_value

# select the true_params that generates a log_prob larger than the gatekeeping value
pass_the_threshold = False
print("seed:",seed)
key_param_gen, subkey_param_gen = jax.random.split(jax.random.PRNGKey(seed + 1901))
while not pass_the_threshold:
    # generate a true params randomly
    # seed = random.randint(0, 10000000)

    true_param = prior.sample(subkey_param_gen,1)
    for param_key in true_param:
        true_param[param_key] = true_param[param_key][0] # to turn this 1 item array into numbers
    print("trial params",true_param)
    likelihood, prior_value = test_log_likelihood(true_param, seed, prior)
    posterior = likelihood + prior_value
    print('trial posterior', posterior)
    print('trial likelihood', likelihood)
    if (posterior>gate_keeping_posterior_low) and (posterior<gate_keeping_posterior_high):
        pass_the_threshold = True
    key_param_gen, subkey_param_gen = jax.random.split(key_param_gen)
    

#true_param = jnp.array([Mc, args.m2/args.m1, args.s1_theta, args.s1_phi, args.s1_mag, args.s2_theta, args.s2_phi, args.s2_mag, args.dist_mpc, args.tc, args.phic, args.inclination, args.polarization_angle, args.ra, args.dec])
#true_param = prior.add_name(true_param, )
#true_param_trans = prior.transform(true_param)
#print("true_param trans:", true_param_trans)
#detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle, "epoch": epoch, "t_c": args.tc}
#InjectionWaveform = RippleIMRPhenomPv2(f_ref)
true_param_transformed = prior.transform(true_param)
detector_param = {
        "ra": true_param_transformed["ra"],
        "dec": true_param_transformed["dec"],
        "psi": true_param_transformed["psi"],
        "t_c": true_param_transformed["t_c"],
        "gmst": gmst,
        "epoch": epoch
}
h_sky = InjectionWaveform(freqs, true_param_transformed)
key, subkey = jax.random.split(jax.random.PRNGKey(seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)

likelihood = HeterodynedTransientLikelihoodFD(
    [H1, L1, V1],
    prior=prior,
    bounds=bounds,
    waveform=RippleIMRPhenomPv2(),
    trigger_time=trigger_time,
    duration=duration,
    post_trigger_duration=post_trigger_duration,
    popsize=1000,
    n_loops=3000,
    n_bins =1500,
)
# likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax],  waveform = waveform, trigger_time = trigger_time, duration = args.duration, post_trigger_duration = post_trigger_duration)

n_dim = 15
mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-4)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-2)
mass_matrix = mass_matrix.at[3, 3].set(1e-2)
mass_matrix = mass_matrix.at[4, 4].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-2)
mass_matrix = mass_matrix.at[6, 6].set(1e-2)
mass_matrix = mass_matrix.at[7, 7].set(1e-3)
#mass_matrix = mass_matrix.at[8, 8].set(5)
mass_matrix = mass_matrix.at[9, 9].set(1e-2)
mass_matrix = mass_matrix.at[13, 13].set(1e-2)
mass_matrix = mass_matrix.at[14, 14].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix*1e-3}

n_chains = 1000
outdir_name = "./outdir/0328_50/"

jim = Jim(
    likelihood,
    prior,
    n_loop_pretraining=0,
    n_loop_training=100,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=300,
    n_chains=n_chains,
    n_epochs=100,
    learning_rate=0.001,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=10,
    output_thinning=30,
    num_layers=6,
    hidden_size=[64, 64],
    num_bins=8,
    n_loops_maximize_likelihood = 2000,
    local_sampler_arg=local_sampler_arg,
    outdir_name = outdir_name
)
#jim = Jim(likelihood, 
#        prior,
#        n_loop_training=args.n_loop_training,
#        n_loop_production = args.n_loop_production,
#        n_local_steps=args.n_local_steps,
#        n_global_steps=args.n_global_steps,
#        n_chains=args.n_chains,
#        n_epochs=args.num_epochs,
#        learning_rate = args.learning_rate,
#        max_samples = args.max_samples,
#        momentum = args.momentum,
#        batch_size = args.batch_size,
#        use_global=args.use_global,
#        keep_quantile= args.keep_quantile,
#        train_thinning = args.train_thinning,
#        output_thinning = args.output_thinning,
#        local_sampler_arg = local_sampler_arg,
#        seed = args.seed,
#        num_layers = args.num_layers,
#        hidden_size = args.hidden_size,
#        num_bins = args.num_bins
#        )

key, subkey = jax.random.split(key)
jim.sample(subkey)
samples = jim.get_samples()
chains_training, log_prob_training, local_accs_training, global_accs_training, loss_vals_training= jim.Sampler.get_sampler_state(training=True).values()
chains, log_prob, local_accs, global_accs,= jim.Sampler.get_sampler_state(training=False).values()

true_param_out = [true_param[key] for key in true_param.keys()]
true_param_out.append(posterior)

output_path = './outdir/injection_test_run5/injection_synthesized_priorflattened'
jnp.savez(output_path + '_production_'+str(index)+'.npz', 
          chains=chains, 
          log_prob=log_prob, 
          local_accs=local_accs, 
          global_accs=global_accs,
          true_params = true_param_out,
        )

jnp.savez(output_path + '_training_'+str(index)+'.npz', 
          chains=chains_training, 
          log_prob=log_prob_training, 
          local_accs=local_accs_training, 
          global_accs=global_accs_training,
          loss_vals = local_accs_training,
        )
jim.print_summary()
