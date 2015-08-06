'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum_2, update


def get():
    
    
    l=[]
    solution=get_solution_slow_GP_striatum_2()

    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    for y in [1.2, 1.3]:   
    
        d={}
        misc.dict_update(d, solution['mul'])
        
        misc.dict_update(d,{'node':{'CS':{'rate': y}}}) 
        
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        dd={'conn': {'GA_GA_gaba':{'fan_in0': 20,'rule':'all-all' }, 
                     'GA_GI_gaba':{'fan_in0': 2,' rule':'all-all' },
                     'GI_GA_gaba':{'fan_in0': 10,'rule':'all-all' },
                     'GI_GI_gaba':{'fan_in0': 28,'rule':'all-all' }}}
        
        misc.dict_update(d, dd)

        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'fan_in0': 2}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'fan_in0': 2}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'fan_in0': 2}}})        


#         d['node']['EA']['rate']*=0.9
         
#         misc.dict_update(d,{'conn':{'M2_GA_gaba':{'lesion': False}}})
#         misc.dict_update(d,{'conn':{'M2_GA_gaba':{'fan_in0': 25}}})

        d['node']['EI']['rate']*=y*0.9
        d['node']['EA']['rate']*=y
#         d['node']['CS']['rate']*=y
        
#            
        
        misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
          
        l[-1]+=pl(d, '=', **{'name':'all_EIEA_'+str(y)})  

    for a,b in zip([1.1, 1.2, 1.7, 2.3, 2.5],
                    [a*0.25 for a in [0.75, 0.5, 0.25, 0.1, 0.01]]):
        y=1.2   
    
        d={}
        misc.dict_update(d, solution['mul'])    
        misc.dict_update(d,{'node':{'CS':{'rate': y}}}) 

        misc.dict_update(d,{'nest':{'ST_GA_ampa':{'weight':b}}})        
        
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal'])   
        
        dd={'conn': {'GA_GA_gaba':{'fan_in0': 20,'rule':'all-all' }, 
                     'GA_GI_gaba':{'fan_in0': 2,' rule':'all-all' },
                     'GI_GA_gaba':{'fan_in0': 10,'rule':'all-all' },
                     'GI_GI_gaba':{'fan_in0': 28,'rule':'all-all' }}}
        
        misc.dict_update(d, dd)

        
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'fan_in0': 2}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'fan_in0': 2}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'fan_in0': 2}}})        


#         d['node']['EA']['rate']*=0.9
         
#         misc.dict_update(d,{'conn':{'M2_GA_gaba':{'lesion': False}}})
#         misc.dict_update(d,{'conn':{'M2_GA_gaba':{'fan_in0': 25}}})

        d['node']['EI']['rate']*=y*0.9
        d['node']['EA']['rate']*=y*a
#         d['node']['CS']['rate']*=y
        
        misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
          
        l[-1]+=pl(d, '=', **{'name':'all_EIEACS_'+str(a)+'_STGAw_'+str(b)})  

    return l



get()