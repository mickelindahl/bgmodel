'''
Created on Aug 12, 2013

@author: lindahlm
'''

from scripts_inhibition.base_Go_NoGo_compete import Setup
from core.network.default_params import Perturbation_list as pl
from core.network.manager import Builder_Go_NoGo_with_nodop as Builder
from core.parallel_excecution import loop

import scripts_inhibition.base_Go_NoGo_compete
import oscillation_perturbations41 as op
import pprint
pp=pprint.pprint

from copy import deepcopy

def perturbations(rep,res, sizes):

    threads=20

    l=[]
    
#     l.append(op.get()[0])
#     l.append(op.get()[4::3])
    l=op.get()
    
    ll=[]
    
    

    for size in sizes:
        for i, _l in enumerate(l):
            _lt=deepcopy(_l)
            per=pl({'netw':{'size':size, 
                            'sub_sampling':{'M1':1,
                                            'M2':1},}},
                      '=', 
                      **{'name':''})
            _lt+=per
    
            _lt+=pl({'simu':{'threads':threads}},'=')
            ll.append(_lt)

    return ll, threads


sizes=[5000]
res, rep=5, 1
duration=[900.,100.0]
laptime=1000.0

l_mean_rate_slices= ['mean_rate_slices']
p_list, threads=perturbations(rep, res, sizes)
props_conn=[0.08]*len(sizes)*len(p_list)
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)

setups=[]
for i in range(len(p_list)):
    setups.append(Setup(**{'duration':duration,
            'laptime':laptime,
            'l_mean_rate_slices':l_mean_rate_slices,
            'threads':threads,
            'resolution':res,
            'repetition':rep,
            'proportion_connected':props_conn[i],
            'labels':['Only D1', 
                      'D1,D2',
                      'Only D1 no dop', 
                      'D1,D2 no dop']
            }))
               
               


for j in range(0, 3):
    for i, p in enumerate(p_list):
        
        
#         if i<3:
# #         if i!=1:
#             continue
        from_disk=j

        fun=Go_NoGo_compete.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               setups[i]])

        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, 
                           setups[i]])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 1)
        
