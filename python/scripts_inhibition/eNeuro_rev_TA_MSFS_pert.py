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
    labels=['beta', 'sw']
    for i, p in enumerate(_get('dictionary')):
    
        
        # TA-MS decay as Glajch 2016
        d={}
        d=misc.dict_update( p['mul'], d) 
         
        l[i]+=[pl(d, '*', **{'name':''})]
         
        d={'nest':{'M1_low':{'GABAA_3_Tau_decay':87.},
                   'M2_low':{'GABAA_3_Tau_decay':76.}}}
        d=misc.dict_update(p['equal'], d)  
 
        s='TA_MS_delay_{0}'.format( labels[i] )
         
        l[i][-1]+=pl(d, '=', **{'name':s})     
    
    
        # TA-FS decay as Glajch 2016
        d={}
        d=misc.dict_update( p['mul'], d)  
         
        l[i]+=[pl(d, '*', **{'name':''})]
         
        d={'nest':{'FS_low':{'GABAA_2_Tau_decay':66.}}}
        d=misc.dict_update(p['equal'], d)  
 
        s='TA_FS_delay_{0}'.format( labels[i] )
         
        l[i][-1]+=pl(d, '=', **{'name':s})     
     
         
        # TA-MS and TA-FS decay as Glajch 2016
        d={}
        misc.dict_update(d, p['mul']) 
         
        l[i]+=[pl(d, '*', **{'name':''})]
         
        d={'nest':{'M1_low':{'GABAA_3_Tau_decay':87.},
                   'M2_low':{'GABAA_3_Tau_decay':76.},
                   'FS_low':{'GABAA_2_Tau_decay':66.}}}
        d=misc.dict_update(p['equal'], d) 
 
        s='TA_MSFS_delay_{0}'.format( labels[i] )
         
        l[i][-1]+=pl(d, '=', **{'name':s})     
    
    
        # TA-MS and TA-FS decay as Glajch 2016
        d={}
        d=misc.dict_update( p['mul'], d) 
         
        l[i]+=[pl(d, '*', **{'name':''})]
         
        d={'nest':{'M1_low':{'GABAA_3_Tau_decay':87.},
                   'M2_low':{'GABAA_3_Tau_decay':76.},
                   'FS_low':{'GABAA_2_Tau_decay':66.},
                   'GA_M1_gaba':{'weight':0.08},
                   'GA_M2_gaba':{'weight':0.17}}}
         
        d=misc.dict_update(p['equal'], d ) 
 
        s='TA_MSFS_delay_MS_weight_{0}'.format( labels[i] )
         
        l[i][-1]+=pl(d, '=', **{'name':s})  
         
         
        # TA-MS and FS dop as Glajch 2016
        d={} 
        d=misc.dict_update(p['mul'], d) 
        l[i]+=[pl(d, '*', **{'name':''})]
         
        # Dopamine effect on GA-MS
        d={'nest':{
                    'M1_low':{'beta_I_GABAA_3': f_beta_rm(2.5)},
                    'M2_low':{'beta_I_GABAA_3': f_beta_rm(2.4)},
                    'FS_low':{'beta_I_GABAA_2': f_beta_rm(1.5)},
                    'FS_low':{'beta_I_GABAA_3': f_beta_rm(1.5)},
                    }}
     
        pp(d)
     
        d=misc.dict_update(p['equal'], d) 
 
        s='GA_MSFS_dop_{0}'.format( labels[i] )
         
        l[i][-1]+=pl(d, '=', **{'name':s})   
        
   
        # TA-MS and TA-FS decay as well as TA-MS and FS dop as Glajch 2016
        d={} 
        d=misc.dict_update(p['mul'], d) 
        l[i]+=[pl(d, '*', **{'name':''})]
        
        # Dopamine effect on GA-MS
        d={'nest':{
                   'M1_low':{'beta_I_GABAA_3': f_beta_rm(2.5),
                             'GABAA_3_Tau_decay':87.},
                   'M2_low':{'beta_I_GABAA_3': f_beta_rm(2.4),
                             'GABAA_3_Tau_decay':76.},
                   'FS_low':{'beta_I_GABAA_2': f_beta_rm(1.5),
                             'beta_I_GABAA_3': f_beta_rm(1.5),
                             'GABAA_2_Tau_decay':66.},
                   'GA_M1_gaba':{'weight':0.08},
                   'GA_M2_gaba':{'weight':0.17}
                    }}
    
        pp(d)
    
        d=misc.dict_update(p['equal'], d) 

        s='TA_MSFS_all_{0}'.format( labels[i] )
        
        l[i][-1]+=pl(d, '=', **{'name':s})   
   
   
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}

ld=get()
pp(ld)
