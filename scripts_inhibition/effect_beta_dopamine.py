'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_dopamine

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'supermicro/simulate_beta_ZZZ_dop_effect_perturb/'),
        'from_diks':1,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Activation (beta)'}

obj=effect_dopamine.Main(**kwargs)
obj.do()