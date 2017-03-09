'''
Created on Aug 12, 2013
 
@author: lindahlm
'''
 
from core.network.default_params import Perturbation_list as pl
from core import misc
from scripts_inhibition.eNeuro_fig_01_and_02_pert import get as _get
 
import numpy
import pprint
 
pp=pprint.pprint
 
d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
 
 
def get():
    l=[[],[]]
    labels=['beta', 'sw']
    for i, p in enumerate(_get('dictionary')):
        EIr=p['equal']['node']['EI']['rate'] 
        EFr=p['equal']['node']['EF']['rate']
        EAr=p['equal']['node']['EA']['rate']
              
        
        for ESr in [1800., 2100., 2400.]:
            for TAMSdop in [2.5, 2.0]:
     
             
                d={}
                
                d=misc.dict_update(p['mul'], d) 
                 
                l[i]+=[pl(d, '*', **{'name':''})]
                          
                d={
                   # Dopamine GA-MS, GA-FS, GF-FS 
                   'nest':{
                           'GA_M1_gaba':{'weight':0.01*2},
                           'GA_M2_gaba':{'weight':0.02*2},
                           'M1_low':{'beta_I_GABAA_2': f_beta_rm(0.25),
                                     'beta_I_GABAA_3': f_beta_rm(TAMSdop)},
                           'M2_low':{'beta_I_GABAA_2': f_beta_rm(0.25),
                                     'beta_I_GABAA_3': f_beta_rm(TAMSdop)}
                           },
  
                   
                   # Tuning rate GI/GF and GA
                   'node':{
                            'EI':{'rate':EIr*1.},
                            'EF':{'rate':EFr*1.},
                            'EA':{'rate':EAr*1.},
                            'ES':{'rate':ESr}
                           },
                    # Activate GF to FS connection
                  'conn':{'M1_M1_gaba':{'beta_fan_in': f_beta_rm(0.25)},
                       'M1_M2_gaba':{'beta_fan_in': f_beta_rm(0.25)},
                       'M2_M1_gaba':{'beta_fan_in': f_beta_rm(0.25)},
                       'M2_M2_gaba':{'beta_fan_in': f_beta_rm(0.25)}
                       }
                   }
                 
                d=misc.dict_update(p['equal'], d) 
                s='TAMSdop_{0}_ESr_{1}_{2}'
                s=s.format( TAMSdop, ESr, labels[i] )
                 
                l[i][-1]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
