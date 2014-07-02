'''
Created on Aug 12, 2013

@author: lindahlm
'''
from copy import deepcopy
from toolbox import misc
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_slow_wave2 as Builder
from toolbox.parallel_excecution import loop

import simulate_slow_wave
import pprint
pp=pprint.pprint


def perturbations():
    sim_time=40000.0
    size=20000.0
    threads=2

    
#     l=op.get()
    l=[]
    w=1.0
    v=0.25
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':v},
                    'GA_M2_gaba':{'weight':v},
                    'M1_M1_gaba':{'weight':v},
                    'M1_M2_gaba':{'weight':v},
                    'M2_M1_gaba':{'weight':v},
                    'M2_M2_gaba':{'weight':v},
                    },
            'node':{'C1':{'rate':w},
                    'C2':{'rate':w}}},
           '*',
            **{'name':'MsGa-MS-weight'+str(v)})]  
    
    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')

    ll=[]
    for amp in [
                [0.25, 0.9],
                [0.3, 0.9],
                [0.35, 0.9],
                [0.4, 0.9],
                [0.5, 0.9],
                [0.6, 0.9],
                [0.7, 0.9],
                ]: 
        d={'type':'oscillation2', 
           'params':{'p_amplitude_mod':amp[0],
                     'p_amplitude0':amp[1],
                     'freq_min':0.5,
                     'freq_max':1.5, 
                     'period':'uniform'}} 
        
        for i, _l in enumerate(l):
            _l=deepcopy(_l)
            dd={}
            for key in ['C1', 'C2', 'CF', 'CS']: 
                dd=misc.dict_update(dd, {'netw': {'input': {key:d} } })     
                      
            _l+=pl(dd,'=',**{'name':'amp_'+str(amp)})
                
            ll.append(_l)
    return ll, threads


p_list, threads=perturbations()
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)

for j in range(0,3):
    for i, p in enumerate(p_list):
#         
#         if i<n-4:
#             continue
#         
        from_disk=j

        fun=simulate_slow_wave.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, threads])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, threads])


loop(args_list, path, 7)
        