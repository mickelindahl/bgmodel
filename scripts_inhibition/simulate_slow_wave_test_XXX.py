'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_slow_wave2 as Builder
from toolbox.parallel_excecution import loop

import simulate_slow_wave
import oscillation_perturbations as op
import pprint
pp=pprint.pprint


def perturbations():
    sim_time=10000.0
    size=5000.0
    threads=10

    
    l=op.get()[0:2]

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')

    return l, threads


p_list, threads=perturbations()
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)


for j in range(2,3):
    for i, p in enumerate(p_list):
                
        from_disk=j

        fun=simulate_slow_wave.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
        fun(*[Builder, from_disk, p, script_name, threads])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, threads])


loop(args_list, path, 2)
        