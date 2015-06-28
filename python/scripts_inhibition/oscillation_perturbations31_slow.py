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
    
    for y in numpy.arange(2.,29.,4.):       
        d={}
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        dd={'conn': {'GA_GA_gaba':{'fan_in0': y,'rule':'all-all' }, 
                     'GA_GI_gaba':{'fan_in0': 2,'rule':'all-all' },
                     'GI_GA_gaba':{'fan_in0': 30-y,'rule':'all-all' },
                     'GI_GI_gaba':{'fan_in0': 28,'rule':'all-all' }}}
        
        misc.dict_update(d, dd)
          
        l[-1]+=pl(d, '=', **{'name':'GAGA_'+str(y)+'_GIGA_'+str(30-y)})  


    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
    for y in [1.8, 2.0]:       
        d={}
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        d['node']['EI']['rate']*=y
        d['node']['EA']['rate']*=y*0.7
        
        
        misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
    
        
        l[-1]+=pl(d, '=', **{'name':'mod_EIEA_'+str(y)})  
    
    
    for y, EA_rate in zip(numpy.arange(25,80,25)
                          numpy.arange(0.9,1.1,0.1)):
        d={}
        
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 

        d['node']['EA']['rate']*=EA_rate
        
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'fan_in0': y}}})
        
        
        l[-1]+=pl(d, '=', **{'name':'M2GA_'+str(y)+'_EArate_'+str(EA_rate)})    
            

    for y in numpy.arange(2, 5, 2):
    
        d={}
        
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
            
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'fan_in0': y}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'fan_in0': y}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'fan_in0': y}}})        
               
        
        l[-1]+=pl(d, '=', **{'name':'mod_GI_M2_'+str(y)})  


    for y in [1.8, 2.0]:   
    
        d={}
        misc.dict_update(d, solution['mul'])  
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
        
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'fan_in0': 25}}})

        d['node']['EI']['rate']*=y*0.9
        d['node']['EA']['rate']*=y
        
        
        misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
          
        l[-1]+=pl(d, '=', **{'name':'all_EIEA_'+str(y)})  

    return l

get()