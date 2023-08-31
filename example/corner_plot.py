import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.interpolate import interp1d

n_dim = 11

data = np.load("./zw_test_batch_out/injection_7.npz")
chains = data["chains"]
log_prob = data["log_prob"]
mask = (log_prob >-1000.)
print(sum(mask))
chains = chains[mask]
print(data.keys())
#print(chains[0])
#print(chains[1])
#print(chains[88])



chains = chains.reshape(-1, n_dim)
print(chains.shape)
# log_prob_raw = data["log_prob"].flatten()
# nancheck = np.isnan(log_prob_raw)
# for i in range(len(log_prob_raw)):
#     if nancheck[i] == True:
#         log_prob_raw[i] = 0
# log_prob = log_prob = np.array([log_prob_raw]).T

chains[:, 7] = np.arccos(np.arcsin(np.sin(chains[:, 7]/2*np.pi))*2/np.pi)
chains[:, 10] = np.arcsin(np.arcsin(np.sin(chains[:, 10]/2*np.pi))*2/np.pi)

#########################
# chains = np.concatenate((chains, log_prob), axis=1)
chains = chains[::50]
truths = data["true_params"]

#q_axis = np.linspace(0.1,1,100)
#eta = q_axis/(1+q_axis)**2
#q_interp = interp1d(eta,q_axis)
#truths[1] = q_interp(truths[1])

print("truths: ", truths)
figure = corner.corner(
    chains,
    truths = truths,
    
    labels=[
        "$M_c$",
        "$q$",
        "$\chi_1$",
        "$\chi_2$",
        "$D$",
        "$t_c$",
        "$\phi_c$",
        "$\iota$",
        "$\Psi$",
        "RA",
        "DEC",
        "log L",
        "m1",
        "m2",
        "Lambda_eff",
        "chi_eff",
#         "Log likelihood"
    ],
    smooth=True,
    show_titles=True,
)
#figure.savefig("./phenomp_corner.png")
figure.savefig("./phenomd_corner.png")