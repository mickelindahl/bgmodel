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
import oscillation_perturbations3 as op
import pprint
pp=pprint.pprint


def perturbations():
    sim_time=40000.0
    size=20000.0
    threads=4

    
    l=op.get()

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')

    ll=[]
    for amp in [
                [0.15, 1.0], 
                [0.2, 1.0], 
                ]: 
        d={'type':'oscillation2', 
           'params':{'p_amplitude_mod':amp[0],
                     'p_amplitude0':amp[1],
                     'freq': 1.}} 
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

for j in range(2,3):
    for i, p in enumerate(p_list):
        
#         if i<5:
#             continue
#         
        from_disk=j

        fun=simulate_slow_wave.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, threads])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, threads])


loop(args_list, path, 6)
        