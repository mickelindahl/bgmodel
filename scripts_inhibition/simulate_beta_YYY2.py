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
    sim_time=5000.0
    size=20000.0
    threads=4
    l=[]


    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.2},
                    'GA_M2_gaba':{'weight':0.2},
                    'M1_M1_gaba':{'weight':0.2},
                    'M1_M2_gaba':{'weight':0.2},
                    'M2_M1_gaba':{'weight':0.2},
                    'M2_M2_gaba':{'weight':0.2}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.3},
                    'GA_M2_gaba':{'weight':0.3},
                    'GA_FS_gaba':{'weight':0.3},
                    'M1_M1_gaba':{'weight':0.3},
                    'M1_M2_gaba':{'weight':0.3},
                    'M2_M1_gaba':{'weight':0.3},
                    'M2_M2_gaba':{'weight':0.3},
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.3and*0.3'})]  

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                  'sim_stop':sim_time,
                  'threads':threads},
                  'netw':{'size':size}}, 
                  '=')


    ll=[]
    for amp in [0.85, 0.9, 0.95]: 
        d={'type':'oscillation', 
             'params':{'p_amplitude_mod':amp,
                     'freq': 20.}} 
        for _l in l:
            _l=deepcopy(_l)
            dd={}
            for key in ['C1', 'C2', 'CF', 'CS']: 
                dd=misc.dict_update(dd, {'netw': {'input': {key:d} } })     
                      
            _l+=pl(dd,'=',**{'name':'amp_'+str(amp)})
        
            ll.append(_l)
        

    return ll


p_list=perturbations()
for p in p_list:
    print p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

for j in range(0,3):
    for i, p in enumerate(p_list):
        from_disk=j

        fun=simulate_beta.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, script_name])


loop(args_list, path, 4)
        