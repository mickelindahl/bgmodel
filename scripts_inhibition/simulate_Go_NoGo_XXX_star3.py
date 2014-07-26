'''
Created on Aug 12, 2013

@author: lindahlm
'''

from Go_NoGo_compete import Setup
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_Go_NoGo_with_lesion_FS as Builder
from toolbox.parallel_excecution import loop

import Go_NoGo_compete
import oscillation_perturbations3 as op1
# import oscillation_perturbations4 as op2
import pprint
pp=pprint.pprint

from copy import deepcopy

def perturbations(rep,res):

    threads=8

    l=[]
    
#     l.append(op.get()[0])
#     l.append(op.get()[4::3])

    l=op1.get()[-3:]
    
    ll=[]
#     w=0.4
    p_sizes=[0.1989460102,  
            0.1608005821,    
            0.122655154, 
            0.0845097259
              ]
    p_sizes=[p/p_sizes[0] for p in p_sizes]
    max_size=4000
    

    j=0
    for w, _lt in zip([0.25, 0.3, 0.35], l):
  
        for ss, p_size in zip([6.25, 
                               8.3 , 
                               12.5, 
                               25
                               ], p_sizes): 
            
            _lp=deepcopy(_lt)
            per=pl({'netw':{'size':int(p_size*max_size), 
                            'sub_sampling':{'M1':ss,
                                            'M2':ss},}},
                      '=', 
                      **{'name':'ss-'+str(ss)})
            _lp+=per
    
            _lp+=pl({'simu':{'threads':threads}},'=')
            ll.append(_lp)
        print j, ll[j]
        for _l in deepcopy([ll[j]]):
            _lr=deepcopy(_l)
            _lr.update_list(
                pl({'nest':{'GA_M1_gaba':{'weight':5*2.*w},
    #                         'GA_FS_gaba':{'weight':0.1}
                            }},'*', **{'name':'GA-XX-equal'}))
            ll.append(_lr)
     
            _lr=deepcopy(_l)
            _lr.update_list(
                pl({'nest':{'GA_M1_gaba':{'weight':5*4.*w},
    #                         'GA_FS_gaba':{'weight':0.1}
                            }},'*', **{'name':'GA-XX-equal'}))
            ll.append(_lr)
     
            _lr=deepcopy(_l)
            _lr.update_list(
                pl({'nest':{'GA_M1_gaba':{'weight':5*0.5*w},
    #                         'GA_FS_gaba':{'weight':0.1}
                            }},'*', **{'name':'GA-XX-equal'}))
            ll.append(_lr)
        j+=8
    
    return ll, threads



res, rep=5, 1
duration=[900.,100.0]
laptime=1000.0
l_mean_rate_slices= ['mean_rate_slices']
p_list, threads=perturbations(rep, res)
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)



for j in range(0, 3):
    for i, p in enumerate(p_list):
        
        
        if i<3:
#         if i!=1:
            continue
        from_disk=j

        fun=Go_NoGo_compete.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               Setup(**{'duration':duration,
#                        'l_mean_rate_slices':l_mean_rate_slices,
#                     'laptime':laptime,
#                      'threads':threads,
#                      'resolution':res,
#                      'repetition':rep})])

        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, 
                           Setup(**{'duration':duration,
                                    'laptime':laptime,
                                    'l_mean_rate_slices':l_mean_rate_slices,
                                    'threads':threads,
                                    'resolution':res,
                                    'repetition':rep})])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 2)
        