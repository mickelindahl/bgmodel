'''
Created on Aug 12, 2013

@author: lindahlm
'''
from copy import deepcopy
from inhibition_gather_results import process
from core import misc
from core.network.default_params import Perturbation_list as pl
from core.network.manager import Builder_beta as Builder
from core.parallel_excecution import loop

import numpy
import scripts_inhibition.base_oscillation_beta
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint


def perturbations():
    sim_time=10000.0
    size=20000.0
    threads=1

    freqs=[1., 2.0]

    path=('/home/mikael/results/papers/inhibition'+
       '/network/simulate_inhibition_ZZZ4/')
    l=op.get()[-3:]

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')


    
    damp=process(path, freqs)
    for key in sorted(damp.keys()):
        val=damp[key]
        print numpy.round(val, 2), key

    ll=[]
    for i, _l in enumerate(l):
        for j, _ in enumerate(freqs):
        
            amp=[numpy.round(damp[_l.name][j],2), 1]
            d={'type':'oscillation2', 
               'params':{'p_amplitude_mod':amp[0],
                         'p_amplitude0':amp[1],
                         'freq': 20.}} 
            _ll=deepcopy(_l)
            dd={}
            for key in ['C1', 'C2', 'CF', 'CS']: 
                dd=misc.dict_update(dd, {'netw': {'input': {key:d} } })     
                      
            _ll+=pl(dd,'=',**{'name':'amp_{0}-{1}'.format(*amp)})

            ll.append(_ll)
        

    return ll, threads


p_list, threads=perturbations()
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

for j in range(1,3):
    for i, p in enumerate(p_list):
        
#         if i<5:
#             continue
#         
        from_disk=j

        fun=simulate_beta.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, threads])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, threads])


loop(args_list, path, 6)
        