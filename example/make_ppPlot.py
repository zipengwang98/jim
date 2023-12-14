import numpy as np
from scipy.optimize import minimize_scalar
import arviz as az
from scipy.interpolate import interp1d

q_axis = np.linspace(0.1,1,100)
eta = q_axis/(1+q_axis)**2
q_interp = interp1d(eta,q_axis)

def get_all_quantile(filename):
    data = np.load(filename)
    
    chains = data['chains']
    true_param = data['true_param']
    #true_param[1] = q_interp(true_param[1])
    true_param[11] = np.cos(true_param[11])
    true_param[14] = np.sin(true_param[14])

    def compute_percentile(value,data):
        return np.where(data<value)[0].shape[0]/data.size


    #def compute_percentile(value,data):
    #     f = lambda x : np.min(np.abs(az.hdi(data, hdi_prob=x)-value))
    #     result = minimize_scalar(f,bounds=[0.001,0.99],method='bounded')
    #     return result.x


    # Multi-modal HDI works only for data that are actuall multi-modal.
    # If it is not, it may actually screw up the result
    #def compute_percentile_multimodal(value,data):
    #    f = lambda x : np.min(np.abs(az.hdi(data, hdi_prob=x,multimodal=True,max_modes=4)-value))
    #    result = minimize_scalar(f,bounds=[0.001,0.99],method='bounded')
    #    return result.x
    
    result = []
    result_multimodal = []
    for i in range(15):
        #if i == 1:
        #    print(true_param[i],chains[:,:,i].reshape(-1)[::10])
        result.append(compute_percentile(true_param[i],chains[:,:,i].reshape(-1)[::10]))
        result_multimodal.append(1)#compute_percentile_multimodal(true_param[i],chains[:,:,i].reshape(-1)[::10]))

    mean_local_accs = data['local_accs'].mean()
    mean_global_accs = data['global_accs'].mean()

    return np.array(result), np.array(result_multimodal), true_param, mean_global_accs, mean_local_accs

directory = '/home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/zw_test_batch_out/'
#directory = '../../../data_storage/10_pv2_1109/'
result = []
result_multimodal = []
true_param = []
mean_global_accs = []
mean_local_accs = []
for i in range(10):
    print(i)
    name = directory+'injection_'+str(i)+'.npz'
    local_result = get_all_quantile(name)
    result.append(local_result[0])
    result_multimodal.append(local_result[1])
    true_param.append(local_result[2])
    mean_global_accs.append(local_result[3])
    mean_local_accs.append(local_result[4])

result = np.stack(result)
result_multimodal = np.stack(result_multimodal)
true_param = np.stack(true_param)
mean_global_accs = np.stack(mean_global_accs)
mean_local_accs = np.stack(mean_local_accs)

np.savez('/home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/ppPlots/combined_quantile_balance_LVK_10_1120',result=result, result_multimodal=result_multimodal, true_param=true_param, mean_global_accs=mean_global_accs, mean_local_accs= mean_local_accs)
