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
        EIr=p['equal']['node']['EI']['rate'] 
        EFr=p['equal']['node']['EF']['rate']
        EAr=p['equal']['node']['EA']['rate']
     
            
              
        for GPM1w, GPMSdop, GIMS_lesion, GIMS_fan in [
                                                        [0.08, 1.8, False, 0], 
                                                        [0.08, 2.0, False, 0], 
                                                        [0.08, 2.2, False, 0], 
                                                        [0.08, 2.4, False, 0],
                                                        [0.07, 2.4, False, 0],
                                                        [0.06, 2.4, False, 0],
                                                        [0.05, 2.4, False, 0],
                                                        [0.04, 2.4, False, 0],
                                                        [0.03, 2.4, False, 0],
                                                        [0.08, 2.4, True, 1],
                                                        [0.08, 2.4, True, 2],
                                                        [0.08, 2.4, True, 3],
                                                        [0.08, 2.4, True, 4],
                                                        ]:    
             
            for f_MSMS_w, x in [
                                  [0.25, 0.5],
#                                   [0.25, 0.1],
#                                   [0.5,  0.1],
                                  ]:
            
                d={'nest':{
                           # Weight GA and GF to FS
                           'GA_FS_gaba':{'weight':0.2},
                           'GF_FS_gaba':{'weight':0.8},
                           
                           # Weight MS-MS
                           'M1_M1_gaba':{'weight':f_MSMS_w},
                           'M1_M2_gaba':{'weight':f_MSMS_w},
                           'M2_M1_gaba':{'weight':f_MSMS_w},
                           'M2_M2_gaba':{'weight':f_MSMS_w},
                           
                           # IF curve GA
                           'GA':{
                                  'b':1.5,
                                  'C_m':1.5,
                                  'Delta_T':1.5
                                  }
                            },
                    'node':{
                            'CF':{'rate':1.8},
            #                                     'M1':{'rate':f_MS},  
            #                                     'M2':{'rate':f_MS},
                            
                        }}
                   
                d=misc.dict_update(p['mul'], d) 
                 
                l[i]+=[pl(d, '*', **{'name':''})]
                               
                d={
                   # 75-25 GI+GF-GA
                   'netw':{
                           'GA_prop':0.25,
                           'GI_prop':0.675, #<=0.9*0.75
                           'GF_prop':0.075,     
                           },
                   
            
                   # Dopamine GA-MS, GA-FS, GF-FS 
                   'nest':{
                           'M1_low':{'beta_I_GABAA_3': f_beta_rm(GPMSdop),
                                     'beta_I_GABAA_2': f_beta_rm(x),
                                     'GABAA_3_Tau_decay':87.},
                           'M2_low':{'beta_I_GABAA_3': f_beta_rm(GPMSdop),
                                     'beta_I_GABAA_2': f_beta_rm(x),
                                     'GABAA_3_Tau_decay':76.},
                           'FS_low':{'beta_I_GABAA_2': f_beta_rm(1.6),
                                     'beta_I_GABAA_3': f_beta_rm(1.6),
                                     'GABAA_2_Tau_decay':66.},
                           'GA_M1_gaba':{'weight':GPM1w},
                           'GA_M2_gaba':{'weight':GPM1w*2},
                           },
                   
                   # Tuning rate GI/GF and GA
                   'node':{
                            'EI':{'rate':EIr},
                            'EF':{'rate':EFr},
                            'EA':{'rate':EAr}
                            },
                   
                   # Activate GF to FS connection
                   'conn':{'GF_FS_gaba':{'lesion':False},
                           'GI_M1_gaba':{'lesion':GIMS_lesion,
                                         'fan_in0':GIMS_fan},
                           'GI_M2_gaba':{'lesion':GIMS_lesion,
                                         'fan_in0':GIMS_fan},
                           'M1_M1_gaba':{'beta_fan_in': f_beta_rm(x)},
                           'M1_M2_gaba':{'beta_fan_in': f_beta_rm(x)},
                           'M2_M1_gaba':{'beta_fan_in': f_beta_rm(x)},
                           'M2_M2_gaba':{'beta_fan_in': f_beta_rm(x)}
                           }
                   }
                 
                d=misc.dict_update(p['equal'], d) 
                s='w_{0}_dop_{1}_fan_{2}_{3}'
                s=s.format( GPM1w, GPMSdop, GIMS_fan,  labels[i] )
                 
                l[i][-1]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
