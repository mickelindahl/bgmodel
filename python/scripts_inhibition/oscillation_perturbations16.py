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
    

    v0=numpy.arange(1.25,3.75,0.5)
    v1=numpy.arange(2.5, 5.,0.5)
    v=[[x,y] for x in v0 for y in v1]
    for x, y in v:
         
        d={}
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
           
        d={}
        for keys in s_equal: update(solution, d, keys) 
 
        misc.dict_update(d,{'nest':{'ST':{'beta_I_GABAA_1': f_beta_rm(x)}}})        
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(y)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(y)}}})
         
        d['node']['EA']['rate']*=0.7
        l[-1]+=pl(d, '=', **{'name':('mod_ST_beta_'
                                     +str(x)+'_'+str(y))
                             })    

    
    return l

get()