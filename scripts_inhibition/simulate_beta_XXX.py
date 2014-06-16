'''
Created on Aug 12, 2013

@author: lindahlm
'''
from copy import deepcopy
from toolbox import misc
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_beta as Builder
from toolbox.parallel_excecution import loop

import simulate_beta
import pprint
pp=pprint.pprint




def perturbations():
    sim_time=10000.0
    size=20000.0
    threads=4
    l=[]   
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.25},
                    'GA_GA_gaba':{'weight':0.25},
                    'GI_GA_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weight0.25-EA300'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EA':{'rate':300.0}}},
           '=',
            **{'name':''})   


    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')

    ll=[]
    for amp in [
                [0.1, 1],  [0.15, 1], 
                [0.20, 1], [0.25, 1],
                [0.30, 1], [0.35, 1],
                [0.40, 1], [0.45, 1],
                [0.3, 0.9],    [0.4, 0.9], 
                [0.35, 0.85],  [0.45, 0.85], 
                [0.4, 0.8],    [0.5, 0.8], 
                [0.45, 0.75],  [0.55, 0.75], 
                [0.15, 1.1], [0.3, 1.1], 
                [0.15, 1.2], [0.3, 1.2]
                ]: 
        d={'type':'oscillation2', 
           'params':{'p_amplitude_mod':amp[0],
                     'p_amplitude0':amp[1],
                     'freq': 20.}} 
        for i, _l in enumerate(l):
            _l=deepcopy(_l)
            dd={}
            for key in ['C1', 'C2', 'CF', 'CS']: 
                dd=misc.dict_update(dd, {'netw': {'input': {key:d} } })     
                      
            _l+=pl(dd,'=',**{'name':'amp_'+str(amp)})
        
#             if i in [4, 5,6]:
                
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

for j in range(2,3):
    for i, p in enumerate(p_list):
        
#         if i<5:
#             continue
#         
        from_disk=j

        fun=simulate_beta.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
        fun(*[Builder, from_disk, p, script_name, threads])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, threads])


loop(args_list, path, 10)
        