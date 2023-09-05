import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomPv2

from jimgw.prior import Uniform
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]

H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

prior = Uniform(
    xmin=[10, 0.125, -1.0, -1.0, 0.0, -0.05, 0.0, -1, 0.0, 0.0, -1.0],
    xmax=[80.0, 1.0, 1.0, 1.0, 2000.0, 0.05, 2 * jnp.pi, 1.0, jnp.pi, 2 * jnp.pi, 1.0],
    naming=[
        "M_c",
        "q",
        "s1_z",
        "s2_z",
        "d_L",
        "t_c",
        "phase_c",
        "cos_iota",
        "psi",
        "ra",
        "sin_dec",
    ],
    transforms={
        "q": ("eta", lambda q: q / (1 + q) ** 2),
        "cos_iota": (
            "iota",
            lambda cos_iota: jnp.arccos(
                jnp.arcsin(jnp.sin(cos_iota / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        ),
        "sin_dec": (
            "dec",
            lambda sin_dec: jnp.arcsin(
                jnp.arcsin(jnp.sin(sin_dec / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        ),
    },  # sin and arcsin are for periodizing cos_iota and sin_dec, otherwise it might gives some nans because of numpy
)
likelihood = TransientLikelihoodFD([H1, L1], waveform=RippleIMRPhenomD(), trigger_time=gps, duration=4, post_trigger_duration=2)
# likelihood = HeterodynedTransientLikelihoodFD([H1, L1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=RippleIMRPhenomD(), trigger_time=gps, duration=4, post_trigger_duration=2)


mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}

jim = Jim(
    likelihood,
    prior,
    n_loop_training=10,
    n_loop_production=10,
    n_local_steps=300,
    n_global_steps=300,
    n_chains=500,
    n_epochs=300,
    learning_rate=0.001,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=40,
    local_sampler_arg=local_sampler_arg,
)

jim.maximize_likelihood([prior.xmin, prior.xmax])
jim.sample(jax.random.PRNGKey(42))
chains, log_prob, local_accs, global_accs= jim.Sampler.get_sampler_state().values()
jnp.savez( 'GW150914.npz', 
          chains=chains, 
          log_prob=log_prob, 
          local_accs=local_accs, 
          global_accs=global_accs)