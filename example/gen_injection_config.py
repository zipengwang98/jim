import numpy as np

def Mc_eta_to_ms(m):
    Mchirp, eta = m
    M = Mchirp / (eta ** (3 / 5))
    m2 = (M - np.sqrt(M ** 2 - 4 * M ** 2 * eta)) / 2
    m1 = M - m2
    return m1, m2

prior_range = np.array(
    [[10,50], # mc
     [0.5,1], # q
     [-0.5,0.5], # chi1
     [-0.5,0.5], # chi2
     [300,2000], # dist_mpc
     [-0.04,0.04], # tc
     [0,2*np.pi], # phic
     [-1,1], # cos_incl
     [0,np.pi], # polarization angle
     [0,2*np.pi], # ra
     [-1,1]] # sin_dec
     )

N_config = 10

mc = np.random.uniform(prior_range[0,0],prior_range[0,1],N_config)
q = np.random.uniform(prior_range[1,0],prior_range[1,1],N_config)
eta = q/(1+q)**2
m1,m2 = Mc_eta_to_ms(np.stack([mc,eta]))
chi1 = np.random.uniform(prior_range[2,0],prior_range[2,1],N_config)
chi2 = np.random.uniform(prior_range[3,0],prior_range[3,1],N_config)
dist_mpc = np.random.uniform(prior_range[4,0],prior_range[4,1],N_config)
tc = np.random.uniform(prior_range[5,0],prior_range[5,1],N_config)
phic = np.random.uniform(prior_range[6,0],prior_range[6,1],N_config)
cos_inclination = np.random.uniform(prior_range[7,0],prior_range[7,1],N_config)
inclination = np.arccos(np.arcsin(np.sin(cos_inclination/2*np.pi))*2/np.pi)
polarization_angle = np.random.uniform(prior_range[8,0],prior_range[8,1],N_config)
ra = np.random.uniform(prior_range[9,0],prior_range[9,1],N_config)
sin_dec = np.random.uniform(prior_range[10,0],prior_range[10,1],N_config)
dec = np.arcsin(np.arcsin(np.sin(sin_dec/2*np.pi))*2/np.pi)

directory = '/home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/configs/zw_test_batch/'

for i in range(N_config):
    f = open(directory+"injection_config_"+str(i)+".yaml","w")
    
    f.write('output_path: /home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/zw_test_batch_out/injection_'+str(i)+'\n')
    f.write('downsample_factor: 10\n')
    f.write('seed: '+str(np.random.randint(low=0,high=10000))+'\n')
    f.write('f_sampling: 2048\n')
    f.write('duration: 4\n')
    f.write('fmin: 30\n')
    f.write('ifos:\n')
    f.write('  - H1\n')
    f.write('  - L1\n')
    f.write('  - V1\n')

    f.write("m1: "+str(m1[i])+"\n")
    f.write("m2: "+str(m2[i])+"\n")
    f.write("chi1: "+str(chi1[i])+"\n")
    f.write("chi2: "+str(chi2[i])+"\n")
    f.write("dist_mpc: "+str(dist_mpc[i])+"\n")
    f.write("tc: "+str(tc[i])+"\n")
    f.write("phic: "+str(phic[i])+"\n")
    f.write("inclination: "+str(inclination[i])+"\n")
    f.write("polarization_angle: "+str(polarization_angle[i])+"\n")
    f.write("ra: "+str(ra[i])+"\n")
    f.write("dec: "+str(dec[i])+"\n")
    f.write("heterodyne_bins: 1001\n")

    #f.write("n_dim: 11\n")
    #f.write("n_chains: 1000\n")
    #f.write("n_loop_training: 20\n")
    #f.write("n_loop_production: 20\n")
    #f.write("n_local_steps: 200\n")
    #f.write("n_global_steps: 200\n")
    #f.write("learning_rate: 0.001\n")
    #f.write("max_samples: 50000\n")
    #f.write("momentum: 0.9\n")
    #f.write("num_epochs: 240\n")
    #f.write("batch_size: 50000\n")
    #f.write("stepsize: 0.01\n")

    f.close()
