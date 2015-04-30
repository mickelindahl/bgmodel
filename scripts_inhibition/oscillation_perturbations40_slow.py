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

    for z in [1.05,1.1]:
        for a,b in zip([600., 700.,800.,900.0, 1000.0, 1100.0 ],
                        [a*0.25 for a in [0.75,0.75,0.75, 0.5,0.5,0.5]]):
    
       
#         z=0.8
            d={}
            misc.dict_update(d, solution['mul'])  
            misc.dict_update(d,{'node':{'CS':{'rate': z}}}) 
            misc.dict_update(d,{'nest':{'ST_GA_ampa':{'weight':b}}})  
            
            l+=[pl(d, '*', **{'name':''})]
              
            d={}
            misc.dict_update(d, solution['equal']) 
            
            dd={'conn': {'GA_GA_gaba':{'fan_in0': 15,'rule':'all-all' }, 
                         'GA_GI_gaba':{'fan_in0': 2,'rule':'all-all' },
                         'GI_GA_gaba':{'fan_in0': 15,'rule':'all-all' },
                         'GI_GI_gaba':{'fan_in0': 28,'rule':'all-all' }}}
            
            misc.dict_update(d, dd)
             
             
            d['node']['EI']['rate']*=z
            d['node']['EA']['rate']=a
    
            misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
         
            l[-1]+=pl(d, '=', **{'name':'exc'+str(z)
                                 +'_EIEACS_'+str(a)
                                 +'_STGAw_'+str(b)}) 
        
    return l


get()