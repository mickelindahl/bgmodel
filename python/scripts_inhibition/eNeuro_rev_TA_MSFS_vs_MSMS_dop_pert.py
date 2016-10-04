'''
Created on Aug 12, 2013

@author: lindahlm
'''

from core.network.default_params import Perturbation_list as pl
from core import misc
from scripts_inhibition.base_perturbations import get_solution
from scripts_inhibition.fig_01_and_02_pert import get as _get

import numpy
import pprint

pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def get():
    
    
    l=[[],[]]
    labels=['sw', 'beta']
    for i, p in enumerate(_get('dictionary')):
                       
        for x in numpy.arange(0.5,0.0,-0.1):       
                   
            # TA-MS and FS dop as Glajch 2016
            d={} 
            d=misc.dict_update(p['mul'], d) 
            l[i]+=[pl(d, '*', **{'name':''})]
             
            # Dopamine effect on GA-MS
            d={'nest':{
                        'M1_low':{'beta_I_GABAA_2': f_beta_rm(x),
                                  'beta_I_GABAA_3': f_beta_rm(2.5)},
                        'M2_low':{'beta_I_GABAA_2': f_beta_rm(x),
                                  'beta_I_GABAA_3': f_beta_rm(2.4)},
                        'FS_low':{'beta_I_GABAA_2': f_beta_rm(1.5)},
                        'FS_low':{'beta_I_GABAA_3': f_beta_rm(1.5)},
                        },
               'conn':{'M1_M1_gaba':{'beta_fan_in': f_beta_rm(x)},
                       'M1_M2_gaba':{'beta_fan_in': f_beta_rm(x)},
                       'M2_M1_gaba':{'beta_fan_in': f_beta_rm(x)},
                       'M2_M2_gaba':{'beta_fan_in': f_beta_rm(x)},
                       }}
        
            pp(d)
         
            d=misc.dict_update(p['equal'], d) 
     
            s='MSMS_{0}_dop_{1}'.format(x,  labels[i] )
             
            l[i][-1]+=pl(d, '=', **{'name':s})   
            
       
       
   
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}

ld=get()
pp(ld)
