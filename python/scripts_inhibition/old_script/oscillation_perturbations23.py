'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum, update


def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution_slow_GP_striatum()
    
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
    x=2.5
    d={}
    for y in [12]+ list(numpy.arange(5,55,5)):
#         x=3.4
#         y=2.25
            
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        
        ratio=12./y
        
        dd={'nest':{'GA_M1_gaba':{'weight':0.8*ratio}, 
                   'GA_M2_gaba':{'weight':0.8*ratio}}}
        misc.dict_update(d,dd)            
         
        # Decreasing from 2 leads to ...
        # Increasing from 2 leads to ... 
        dd={'nest':{'GA_FS_gaba':{'weight':2.*ratio}}}
        misc.dict_update(d,dd)           
        
        # Just assumed to be 12 ms    
        dd={'nest':{'M1_low':{'GABAA_3_Tau_decay':12./ratio},  
                   'M2_low':{'GABAA_3_Tau_decay':12./ratio},
                   'FS_low':{'GABAA_2_Tau_decay':12./ratio},     
                   }}
        misc.dict_update(d,dd)  
        
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
        d['node']['EA']['rate']*=0.7
        
        l[-1]+=pl(d, '=', **{'name':'mod_GAtau_'+str(y)})    

    
    return l

get()