'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'supermicro/simulate_slow_wave_ZZZ_neuclei_effect_perturb/'),
        'from_diks':1,
        'midpoint':3.5,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data')}

obj=effect_conns.Main(**kwargs)
obj.do()