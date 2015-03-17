'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution, update


def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution()
    
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
   
    d={}
    x=2.5    
    for keys in s_mul: update(solution, d, keys)  
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    for keys in s_equal: update(solution, d, keys) 
    
    misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
    misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
    d['node']['EA']['rate']*=0.7    
    
    l[-1]+=pl(d, '=', **{'name':'mod_ST_beta_'+str(x)})    

    for y in numpy.arange(1,11,1):
        d={}
        
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
        d['node']['EA']['rate']*=0.7
        
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'fan_in0': y}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'fan_in0': y}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'fan_in0': y}}})        
        
        l[-1]+=pl(d, '=', **{'name':'mod_GI_M2_'+str(y)})   
    

    for y, EA_rate in zip(numpy.arange(25,125,25),
                          numpy.arange(0.9,1.3,0.1)):
        d={}
        
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
        d['node']['EA']['rate']*=EA_rate
        
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'fan_in0': y}}})
        
        
        l[-1]+=pl(d, '=', **{'name':'M2GA_'+str(y)+'_EArate_'+str(EA_rate)})    
            

    
    return l

get()