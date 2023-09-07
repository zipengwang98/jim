import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.interpolate import interp1d

n_dim = 12

data = np.load("./PPE_GW150914_14.npz")
chains = data["chains"]
log_prob = data["log_prob"]
mask = (log_prob >-5000.)
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

#s1_theta = chains[:,2]
#s1_phi = chains[:,3]
#s1_mag = chains[:,4]
#s2_theta = chains[:,5]
#s2_phi = chains[:,6]
#s2_mag = chains[:,7]
#
#
#
## s1x, s1y, and s1z
#chains[:, 2] = s1_mag * np.sin(s1_theta) * np.cos(s1_phi)
#chains[:, 3] = s1_mag * np.sin(s1_theta) * np.sin(s1_phi)
#chains[:, 4] = s1_mag * np.cos(s1_theta) 
#
#
## s2x, s2y, and s2z
#chains[:, 5] = s2_mag * np.sin(s2_theta) * np.cos(s2_phi)
#chains[:, 6] = s2_mag * np.sin(s2_theta) * np.sin(s2_phi)
#chains[:, 7] = s2_mag * np.cos(s2_theta) 
#
#chains[:, 11] = np.arccos(np.arcsin(np.sin(chains[:, 11]/2*np.pi))*2/np.pi)
#chains[:, 14] = np.arcsin(np.arcsin(np.sin(chains[:, 14]/2*np.pi))*2/np.pi)

#########################
# chains = np.concatenate((chains, log_prob), axis=1)
chains = chains[::50]
#truths = data["true_params"]

#q_axis = np.linspace(0.1,1,100)
#eta = q_axis/(1+q_axis)**2
#q_interp = interp1d(eta,q_axis)
#truths[1] = q_interp(truths[1])

#print("truths: ", truths)
figure = corner.corner(
    chains,
    #truths = truths,
    
    labels=[
        "$M_c$",
        "$q$",
        #"$s_1x$",
        #"$s_1y$",
        "$s_1z$",
        #"$s_2x$",
        #"$s_2y$",
        "$s_2z$",
        "$D$",
        "$t_c$",
        "$\phi_c$",
        "$\iota$",
        "$\Psi$",
        "RA",
        "DEC",
        "ppe",
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
figure.savefig("./PPE_plots/PPE_14_phenomd_corner.png")