'''
Created on Aug 12, 2013

@author: lindahlm
'''

from scripts_inhibition.base_inhibition_striatum import Setup
from core.network.default_params import Perturbation_list as pl
from core.network.manager import Builder_striatum as Builder
from core.parallel_excecution import loop

import scripts_inhibition.base_inhibition_striatum
import oscillation_perturbations41 as op
import pprint
pp=pprint.pprint


def perturbations(rep,res):
    sim_time=rep*res*1000.0
    size=3000.0
    threads=20

    
    l=op.get()

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')
        
    return l, threads



res, rep, low, upp=14, 1, 1, 3
p_list, threads=perturbations(rep, res)
for i, p in enumerate(p_list):
    if i<4:
        continue
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)



for j in range(0,3):
    for i, p in enumerate(p_list):
        
#         if i<n-12:
#         if i!=1:
#             continue
         
        from_disk=j

        fun=inhibition_striatum.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               Setup(**{'threads':threads,
#                                     'resolution':res,
#                                     'repetition':rep,
#                                     'lower':low,
#                                     'upper':upp})])
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