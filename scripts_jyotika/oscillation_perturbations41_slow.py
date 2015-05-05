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
    misc.dict_update(d, solution['mul']) 
    l+=[pl(d, '*', **{'name':''})]
    misc.dict_update(d, solution['equal']) 
    l[-1]+=pl(d, '=', **{'name':'myname'}) 
    
            for y in numpy.arange(5.,29.,5.):    
#         z=0.8
                d={}
                misc.dict_update(d, solution['mul'])  
                misc.dict_update(d,{'node':{'CS':{'rate': z}}}) 
                misc.dict_update(d,{'nest':{'ST_GA_ampa':{'weight':b}}})  
                
                l+=[pl(d, '*', **{'name':''})]
                  
                d={}
                misc.dict_update(d, solution['equal']) 
                
                dd={'conn': {'GA_GA_gaba':{'fan_in0': y,'rule':'all-all' }, 
                             'GA_GI_gaba':{'fan_in0': 2,'rule':'all-all' },
                             'GI_GA_gaba':{'fan_in0': 30-y,'rule':'all-all' },
                             'GI_GI_gaba':{'fan_in0': 28,'rule':'all-all' }}}
                
                misc.dict_update(d, dd)
                 
                 
                d['node']['EI']['rate']*=z
                d['node']['EA']['rate']=a
        
                misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}})
             
                l[-1]+=pl(d, '=', **{'name':'exc'+str(z)
                                     +'_EIEACS_'+str(a)
                                     +'_STGAw_'+str(b)
                                     +'_GAGA_'+str(y)
                                     +'_GIGA_'+str(30-y)}) 
		print l[0]
		for e in sorted(l[0].list):
			print e
		raise
		
    for reset in [-52., -51., -50., -49.]:
        d={}
        misc.dict_update(d, solution['mul'])  
        misc.dict_update(d,{'node':{'CS':{'rate': 1.05}}}) 
        misc.dict_update(d,{'nest':{'ST_GA_ampa':{'weight':0.75}}})  
        
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        dd={'conn': {'GA_GA_gaba':{'fan_in0': 15,'rule':'all-all' }, 
                     'GA_GI_gaba':{'fan_in0': 2, 'rule':'all-all' },
                     'GI_GA_gaba':{'fan_in0': 15,'rule':'all-all' },
                     'GI_GI_gaba':{'fan_in0': 28,'rule':'all-all' }}}
        
        misc.dict_update(d, dd)
   
        misc.dict_update(d,{'nest':{'GA':{'V_reset': reset}}})
         
        d['node']['EI']['rate']*=1.05
        d['node']['EA']['rate']=600.
    
    return l


get()
