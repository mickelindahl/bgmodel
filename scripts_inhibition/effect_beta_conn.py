'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'supermicro/simulate_beta_ZZZ_conn_effect_perturb/'),
        'from_diks':0,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data')}

obj=effect_conns.Main(**kwargs)
obj.do()