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
    v=numpy.arange(1.5,3,0.25)
    for x in v:
        
        d={}
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})

        l[-1]+=pl(d, '=', **{'name':'mod_ST_beta_'+str(x)})    
            
    v1=numpy.arange(0.1, 1, 0.2)
    for x in v1:
        
        
        d={}
        for keys in s_mul: update(solution, d, keys) 

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys)
        
        d['node']['EA']['rate']*=x
        l[-1]+=pl(d, '=', **{'name':'mod_EA_'+str(x)})   
    
    v2=numpy.arange(1.1, 2, 0.2)
    for x in v2:
        
        
        d={}
        for keys in s_mul: update(solution, d, keys) 

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys)
        
        d['node']['EI']['rate']*=x
        l[-1]+=pl(d, '=', **{'name':'mod_EI_'+str(x)})  
    
    return l

get()