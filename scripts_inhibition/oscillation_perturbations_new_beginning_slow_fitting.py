'''
Created on Aug 12, 2013

@author: lindahlm
'''

from oscillation_perturbations_new_beginning_slow0 import get_solution 
from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def STN_ampa_gaba_input_magnitude():
    l=[
       [20, 188, 0.08], # 20 Hz 
#        [20, 210, 0.08], # 20 Hz 
#        [25, 290, 0.119], #25 Hz
#        [30, 430, 0.18],      #30 Hz
#        [35, 540, 0.215],     #35 Hz,
#        [40, 702, 0.28],     #40 Hz
#        [45, 830., 0.336],   #45 Hz 
#        [46, 876.7, 0.349],  #46 Hz
#        [50, 1000.8, 0.3957],     # 50 Hz
#        [55, 1159., 0.458],  #50 Hz
#        [80, 2102, 0.794] # 80 Hz] 
       ]
    
    return l


# def get_solution():
#     solution=get_solution_net()
#     
#     for key in solution.keys():
#         for tp in ['nest', 'conn']:
#             for key, d in solution[key]['conn'].items():
#                 l=key.split('_')
#                 i[0]+='p'
#                 solution[key]['conn']['_'.join(l)]=d.copy()
#             
#     return solution

def get():
    
    
    l=[]
    solution=get_solution()

    
#     misc.dict_update(solution, {'mul':d})   

    d={}
    misc.dict_update(d, solution['mul'])
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    misc.dict_update(d, solution['equal']) 
    dd={'conn': {'GAp_GA_gaba':{'fan_in': 5}, 
                 'GIp_GA_gaba':{'fan_in': 25}}}
    misc.dict_update(d, dd)   
    l[-1]+=pl(d, '=', **{'name':'control_sim'}) 
      
    d={}
    misc.dict_update(d, solution['mul'])
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    misc.dict_update(d, solution['equal']) 
    dd={'conn': {'GAp_GA_gaba':{'fan_in': 5}, 
                 'GIp_GA_gaba':{'fan_in': 25}},
        'node':{'EAp':{'rate':400.0}}}
    misc.dict_update(d, dd)   
    l[-1]+=pl(d, '=', **{'name':'control_sim'}) 
 
 
 
    d={}
    misc.dict_update(d, solution['mul'])   
    l+=[pl(d, '*', **{'name':''})]
       
    d={}
    misc.dict_update(d, solution['equal']) 
    dd={'conn': {'GAp_GA_gaba':{'fan_in': 5}, 
                 'GIp_GA_gaba':{'fan_in': 25}},
        'node':{'EAp':{'rate':800.0}}}
    misc.dict_update(d, dd)
    
    l[-1]+=pl(d, '=', **{'name':'EA800.'})       
 
    d={}
    misc.dict_update(d, solution['mul'])   
    l+=[pl(d, '*', **{'name':''})]
       
    d={}
    misc.dict_update(d, solution['equal']) 
    dd={'conn': {'GAp_GA_gaba':{'fan_in': 5}, 
                 'GIp_GA_gaba':{'fan_in': 25}},
        'node':{'EAp':{'rate':1200.0}}}
    misc.dict_update(d, dd)
    
    l[-1]+=pl(d, '=', **{'name':'EA800.'})      
 
#     d={}
#     misc.dict_update(d, solution['mul'])
#     l+=[pl(d, '*', **{'name':''})]
#       
#     d={}
#             # GI predominently connect to to GA 
# 
#     misc.dict_update(d, solution['equal']) 
#     dd={'conn': {'GAp_GA_gaba':{'fan_in': 15}, 
#                  'GIp_GA_gaba':{'fan_in': 15 }}}
#     misc.dict_update(d, dd)   
#     l[-1]+=pl(d, '=', **{'name':'fanin1515.'})    
# 
# 
# 
# 
#         
#     d={}
#     misc.dict_update(d, solution['mul'])
#     dd={'nest':{'ST_GA_ampa':{'weight': 1.},
#                 'GA_GA_gaba':{'weight': 1.},
#                 'GI_GA_gaba':{'weight': 1.}}}
# 
#     misc.dict_update(d, dd)
#     l+=[pl(d, '*', **{'name':''})]
#        
#     d={}
#     misc.dict_update(d, solution['equal']) 
#     dd={'conn': {'GAp_GA_gaba':{'fan_in': 5}, 
#                  'GIp_GA_gaba':{'fan_in': 25}}}
#     misc.dict_update(d, dd)
#     
#     l[-1]+=pl(d, '=', **{'name':'GAw_*1.'})         
 
 
    
# 
#     d={}
#     misc.dict_update(d, solution['mul'])
#     x=0.1
#     dd={'nest':{'ST_GA_ampa':{'weight':x},
#                'GA_GA_gaba':{'weight':x},
#                'GI_GA_gaba':{'weight':x}}}
#     
#     misc.dict_update(d, dd)
#     l+=[pl(d, '*', **{'name':''})]
#       
#     d={}
#     misc.dict_update(d, solution['equal']) 
#     l[-1]+=pl(d, '=', **{'name':'GAw_*.1'})   
#     
# 
#     d={}
#     misc.dict_update(d, solution['mul'])
#     dd={'node':{'EAp':{'rate':0.0}}}
#     misc.dict_update(d, dd)
#     l+=[pl(d, '*', **{'name':''})]
#       
#     d={}
#     misc.dict_update(d, solution['equal']) 
#     l[-1]+=pl(d, '=', **{'name':'EA0.'})   
# 
# 
#     d={}
#     misc.dict_update(d, solution['mul'])
# 
#     misc.dict_update(d, dd)
#     l+=[pl(d, '*', **{'name':''})]
#       
#     d={}
#     misc.dict_update(d, solution['equal']) 
#     dd={'node':{'EAp':{'rate':1200*0.25}}}
#     l[-1]+=pl(d, '=', **{'name':'EA+.'})   
        
    return l

l=get()
for i, p in enumerate(l):
    print i, p