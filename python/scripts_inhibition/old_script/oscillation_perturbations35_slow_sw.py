'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum_2


def get():
    
    
    l=[]
    solution=get_solution_slow_GP_striatum_2()

    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
 
        
    for a,b in zip([200., 300., 300., 400., 500.],
                    [a*0.25 for a in [0.75, 0.75, 0.5, 0.5, 0.5]]):

        z=1.2
        y=0.75
        d={}
        misc.dict_update(d, solution['mul'])
        
        misc.dict_update(d,{'node':{'CS':{'rate': y}}}) 
        misc.dict_update(d,{'node':{'CF':{'rate': z}}}) 
                
        misc.dict_update(d,{'nest':{'ST_GA_ampa':{'weight':b}}})        
        
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal'])   
          
         
        
        d['node']['EI']['rate']*=y
        d['node']['EA']['rate']=a

        
        misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
          
        l[-1]+=pl(d, '=', **{'name':'all_EIEACS_'+str(a)+'_STGAw_'+str(b)})  
   


    return l
get()