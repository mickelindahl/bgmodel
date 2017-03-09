'''
Created on Aug 12, 2013
 
@author: lindahlm
'''
 
from core.network.default_params import Perturbation_list as pl
from core import misc
from scripts_inhibition.base_perturbations import get_solution
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
        GPM1w=p['equal']['nest']['GA_M1_gaba']['weight']
        GPM2w=p['equal']['nest']['GA_M2_gaba']['weight']
              
        
        
        
        for MSMSdop in [0.5,0.4,0.3, 0.25]:
            for GAf, GIf, GPMSf in [
                                [1., 1.4, 2.],
                                [1., 1.6, 2.],
                                [1., 1.8, 2.],
				]:    
             
                d={}
                    'nest':{
                       'M1_low':{'beta_I_GABAA_3': f_beta_rm(2.6),
                                 'beta_I_GABAA_2': f_beta_rm(x),
                                 'GABAA_3_Tau_decay':87.},
                       'M2_low':{'beta_I_GABAA_3': f_beta_rm(2.5),
                                 'beta_I_GABAA_2': f_beta_rm(x),
                d=misc.dict_update(p['mul'], d) 
                 
                l[i]+=[pl(d, '*', **{'name':''})]
                               
                d={
                   # Dopamine GA-MS, GA-FS, GF-FS 
                   'nest':{
                           'GA_M1_gaba':{'weight':0.01*GPMSf},
                           'GA_M2_gaba':{'weight':0.02*GPMSf},
                           'M1_low':{'beta_I_GABAA_2': f_beta_rm(MSMSdop)},
                           'M2_low':{'beta_I_GABAA_2': f_beta_rm(MSMSdop)}
                           },
                   
                   # Tuning rate GI/GF and GA
                   'node':{
                            'EI':{'rate':EIr*GIf},
                            'EF':{'rate':EFr*GIf},
                            'EA':{'rate':EAr*GAf}
                           },
                    # Activate GF to FS connection
               'conn':{'M1_M1_gaba':{'beta_fan_in': f_beta_rm(MSMSdop)},
                       'M1_M2_gaba':{'beta_fan_in': f_beta_rm(MSMSdop)},
                       'M2_M1_gaba':{'beta_fan_in': f_beta_rm(MSMSdop)},
                       'M2_M2_gaba':{'beta_fan_in': f_beta_rm(MSMSdop)}
                       }
                   }
                 
                d=misc.dict_update(p['equal'], d) 
                s='GAf_{0}_GIf_{1}_GPMSf_{2}_{3}'
                s=s.format( GAf, GIf, GPMSf, labels[i] )
                 
                l[i][-1]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
