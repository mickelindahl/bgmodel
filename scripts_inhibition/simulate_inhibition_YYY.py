'''
Created on Aug 12, 2013

@author: lindahlm
'''

from inhibition_striatum import Setup
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_inhibition_striatum as Builder
from toolbox.parallel_excecution import loop

import inhibition_striatum
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint


def perturbations(rep,res):
    sim_time=rep*res*1000.0
    size=3000.0
    threads=16

    
    l=[op.get()[7]]

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')
        
    return l, threads



res, rep, low, upp=14, 5, 1, 2
p_list, threads=perturbations(rep, res)
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
        
# #         if i<n-9:
#         if i>18:
#             continue

        from_disk=j

        fun=inhibition_striatum.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               Setup(threads, res, rep, low, upp)])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, 
                           Setup(**{'threads':threads,
                            'resolution':res,
                            'repetition':rep,
                            'lower':low,
                            'upper':upp})])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 1)
        