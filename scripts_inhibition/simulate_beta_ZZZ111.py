'''
Created on Aug 12, 2013

@author: lindahlm
'''
from copy import deepcopy
from inhibition_gather_results import process
from simulate import get_file_name
from toolbox import misc
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_beta as Builder
from toolbox.parallel_excecution import loop

import numpy
import simulate_beta
import oscillation_perturbations as op
import pprint
pp=pprint.pprint

 
if misc.determine_host()=='milner':
    type_of_run='mpi_milner'
else: 
    type_of_run='mpi_supermicro'
#     type_of_run='shared_memory'

def perturbations():
    sim_time=10000.0
    size=5000.0
    threads=5

    freqs=[0.5, 1.0, 1.5]

    path=('/home/mikael/results/papers/inhibition'+
       '/network/simulate_inhibition_ZZZ/')
    l=op.get()

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
    for j, _ in enumerate(freqs):
        for i, _l in enumerate(l):
            amp=[numpy.round(damp[_l.name][j],2), 1]
            d={'type':'oscillation2', 
               'params':{'p_amplitude_mod':amp[0],
                         'p_amplitude0':amp[1],
                         'freq': 20.}} 
            _l=deepcopy(_l)
            dd={}
            for key in ['C1', 'C2', 'CF', 'CS']: 
                dd=misc.dict_update(dd, {'netw': {'input': {key:d} } })     
                      
            _l+=pl(dd,'=',**{'name':'amp_{0}-{1}'.format(*amp)})

            ll.append(_l)
        

    return ll, threads


p_list, threads=perturbations()
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

# path=(home + '/results/papers/inhibition/network/'
#       +__file__.split('/')[-1][0:-3]+'/')
path=get_file_name(__file__.split('/')[-1][0:-3])
n=len(p_list)

j0=0
for j in range(j0,3):
    for i, p in enumerate(p_list):
        
# #         if i<n-9:
        if i!=0:
            continue


        script_name=(__file__.split('/')[-1][0:-3]
                     +'/script_'+str(i)+'_'+p.name+'_'+type_of_run)
        setup=simulate_beta.Setup(1000.0/20.0, threads)
        obj=simulate_beta.Main(**{'builder':Builder,
                                'from_disk':j,
                                'perturbation_list':p,
                                'script_name':script_name,
                                'setup':setup})
        obj.do()
        args_list.append([obj, script_name])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 1, 
     **{'type_of_run':type_of_run,
        'threads':threads,
        'i0':j0, 
        'debug':False})

        