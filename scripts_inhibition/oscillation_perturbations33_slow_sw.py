'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum_2


def get():
    
    
    l=[]
    solution=get_solution_slow_GP_striatum_2()

    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    for y in [0.8, 0.9, 1.]:   
    
        d={}
        misc.dict_update(d, solution['mul'])
        
        misc.dict_update(d,{'node':{'CS':{'rate': y}}}) 
        
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 

        d['node']['EI']['rate']*=y*0.9
        d['node']['EA']['rate']*=y
        
        misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
          
        l[-1]+=pl(d, '=', **{'name':'all_CSEIEA_'+str(y)})  

  
    y=0.9
    d={}
    misc.dict_update(d, solution['mul'])
    
    misc.dict_update(d,{'node':{'CS':{'rate': y}}}) 
    
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    misc.dict_update(d, solution['equal']) 

    d['node']['EI']['rate']*=y*0.9
    d['node']['EA']['rate']*=0.0
    
    misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
      
    l[-1]+=pl(d, '=', **{'name':'all_CSEIEA_'+str(y)})  
    return l



get()