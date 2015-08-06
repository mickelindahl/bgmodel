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


def get():
    

    
    l=[]
    solution, s_mul, s_equal=get_solution()
    
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
    
    # betas MS-MS
    v=[0.1,0.25, 0.5, 0.75]
    for x, y in [[v1,v2] for v1 in v for v2 in v]:
        
        d={}
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(2.5)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(2.5)}}})
        d['node']['EA']['rate']*=0.7
        
        misc.dict_update(d,{'nest':{'MS':{'beta_I_GABAA_2': f_beta_rm(x)}}})
        for conn in ['M1_M1_gaba', 'M1_M2_gaba','M2_M1_gaba','M2_M2_gaba']:
            misc.dict_update(d,{'conn':{conn:{'beta_fan_in': f_beta_rm(y)}}}) 
        
        l[-1]+=pl(d, '=', **{'name':'mod_MSg_'+str(x)+'_MSc_'+str(y)})    
            
    
    #beta CTX-STN
    
    
    return l

get()