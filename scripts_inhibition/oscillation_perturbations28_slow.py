'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum, update


def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution_slow_GP_striatum()
    
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
    for z0 in numpy.arange(1, 6., 1):
        for z1 in numpy.arange(1, 6., 1):
            y=2.5
            x=2.5
            d={}
            for keys in s_mul: update(solution, d, keys)  

            
            l+=[pl(d, '*', **{'name':''})]
        
    
              
            d={}
            for keys in s_equal: update(solution, d, keys) 
            misc.dict_update(d,{'nest':{'ST_GA_ampa':{'delay':z0}}})
            misc.dict_update(d,{'nest':{'ST_GI_ampa':{'delay':z0}}})
            misc.dict_update(d,{'nest':{'GI_ST_gaba':{'delay':z1}}})

            
            
            misc.dict_update(d,{'nest':{'C1_M1_ampa':{'delay':y}}})
            misc.dict_update(d,{'nest':{'C1_M1_nmda':{'delay':y}}})            
            misc.dict_update(d,{'nest':{'C2_M2_ampa':{'delay':y}}})
            misc.dict_update(d,{'nest':{'C2_M2_nmda':{'delay':y}}})            
            misc.dict_update(d,{'nest':{'CF_FS_ampa':{'delay':y}}})
            misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
            misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
            d['node']['EA']['rate']*=0.7
            
        
            l[-1]+=pl(d, '=', **{'name':'CSTRdelay_'+str(y)})  
         


    for y in numpy.arange(1,8., 1.):
            z=2.5 # ms
            x=2.5
            d={}
            for keys in s_mul: update(solution, d, keys)  

            
            l+=[pl(d, '*', **{'name':''})]
        
    
              
            d={}
            for keys in s_equal: update(solution, d, keys) 
            misc.dict_update(d,{'nest':{'C1_M1_ampa':{'delay':z}}})
            misc.dict_update(d,{'nest':{'C1_M1_nmda':{'delay':z}}})            
            misc.dict_update(d,{'nest':{'C2_M2_ampa':{'delay':z}}})
            misc.dict_update(d,{'nest':{'C2_M2_nmda':{'delay':z}}})            
            misc.dict_update(d,{'nest':{'CF_FS_ampa':{'delay':z}}})
            
            misc.dict_update(d,{'nest':{'GA_M1_gaba':{'delay':y}}})
            misc.dict_update(d,{'nest':{'GA_M2_gaba':{'delay':y}}})            
            misc.dict_update(d,{'nest':{'GA_FS_gaba':{'delay':y}}})
            d['node']['EA']['rate']*=0.7
            
        
            l[-1]+=pl(d, '=', **{'name':'GASTRdelay_'+str(y)})  

    for y in numpy.arange(1, 11., 2):
        for z in numpy.arange(1, 11., 2):
            y=2.5
            x=2.5
            d={}
            for keys in s_mul: update(solution, d, keys)  

            
            l+=[pl(d, '*', **{'name':''})]
        
    
              
            d={}
            for keys in s_equal: update(solution, d, keys) 
            
            
            misc.dict_update(d,{'nest':{'C1_M1_ampa':{'delay':y}}})
            misc.dict_update(d,{'nest':{'C1_M1_nmda':{'delay':y}}})            
            misc.dict_update(d,{'nest':{'C2_M2_ampa':{'delay':y}}})
            misc.dict_update(d,{'nest':{'C2_M2_nmda':{'delay':y}}})            
            misc.dict_update(d,{'nest':{'CF_FS_ampa':{'delay':y}}})            
            misc.dict_update(d,{'nest':{'CS_ST_ampa':{'delay':z}}})            
            misc.dict_update(d,{'nest':{'CS_ST_nmda':{'delay':z}}})
            misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
            misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
            d['node']['EA']['rate']*=0.7
            
        
            l[-1]+=pl(d, '=', **{'name':'CSTRdelay_'+str(y)})     
         
    return l

get()