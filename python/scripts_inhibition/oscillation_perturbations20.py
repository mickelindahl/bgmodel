'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution, update
# from oscillation_perturbations15 import STN_ampa_gaba_input_magnitude

def get():
     
     
    l=[]
    solution, s_mul, s_equal=get_solution()
     
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
     
    y=2.5
    
        
    w=numpy.arange(0.5, 1.2, 0.1)
    v=[0.1,0.25, 0.5]
    ll=[[v1,v2] for v1 in v for v2 in w]
    pp(l)
    for x, z in ll:
        print x,z
                  
        d={}
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
             
        d={}
        for keys in s_equal: update(solution, d, keys)  
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(y)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(y)}}})
                
                
        misc.dict_update(d,{'nest':{'MS':{'beta_I_GABAA_2': f_beta_rm(x)}}})
        for conn in ['M1_M1_gaba', 'M1_M2_gaba','M2_M1_gaba','M2_M2_gaba']:
            misc.dict_update(d,{'conn':{conn:{'beta_fan_in': f_beta_rm(x)}}}) 
                
        misc.dict_update(d,{'nest':{'GA_M1_gaba':{'weight': z}}})
        misc.dict_update(d,{'nest':{'GA_M2_gaba':{'weight': z}}})
        misc.dict_update(d,{'nest':{'GA_FS_gaba':{'weight': z}}})
        
        d['node']['EA']['rate']*=0.7
        l[-1]+=pl(d, '=', **{'name':('mod_GA_str_' + str(z)+'_MS_'+str(x))
                         })   
     
    return l
# def get():
#      
#      
#     l=[]
#     solution, s_mul, s_equal=get_solution()
#      
#     d0=0.8
#     f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
#      
#     y=2.5
#     v=[0.06125, 0.125, 0.25,0.5,1.,2.,4.,8.]
#     for x in v:
#           
#         d={}
#         for keys in s_mul: update(solution, d, keys)  
#         l+=[pl(d, '*', **{'name':''})]
#             
#         d={}
#         for keys in s_equal: update(solution, d, keys) 
#   
#         misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(y)}}})
#         misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(y)}}})
#                
#         misc.dict_update(d,{'nest':{'GA_M1_gaba':{'weight': x}}})
#         misc.dict_update(d,{'nest':{'GA_M2_gaba':{'weight': x}}})
#         
#         d['node']['EA']['rate']*=0.7
#           
#         l[-1]+=pl(d, '', **{'name':('mod_GA_MS_' + str(x))
#                              })    
#  
#      
#     return l

get()