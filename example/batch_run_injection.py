import os

directory = '/home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/configs/zw_test_batch/'
for i in range(100):
    print("file number: " + str(i))
    print("\n\n")
    filename = "injection_config_"+str(i)+".yaml"
    os.system("python InjectionRecovery.py --config "+directory+filename)