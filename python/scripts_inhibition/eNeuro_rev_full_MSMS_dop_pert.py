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
     
              
        for f_GAFS_w, f_GFFS_w, f_CF_rate in [
                                            [0.2,0.8, 1.8],
                                            [0.0,1.0, 1.2]
                                            ]:    
            
                for CSST_delay, CXMS_delay in [[
                                                12.5, 12,
                                                10., 12,
                                                7.5, 12,
                                                5.,  12,
                                                2.5, 12,
                                                2.5, 10,
                                                2.5, 8,
                                                2.5, 6,
                                                2.5, 4,
                                                2.5, 2,
                                                ]]:
                
                    for f_MSMS_w, f_MS, x in [
                                          [0.25, 1.2, 0.5],
                                          [0.25, 1.3, 0.5],
                                          [0.4,  1.4, 0.3],
                                          [0.4,  1.5, 0.3],
                                          [0.6,  1.6, 0.1],
                                          ]:
    
     
                        d={'nest':{
                                   # Weight GA and GF to FS
                                   'GA_FS_gaba':{'weight':f_GAFS_w},
                                   'GF_FS_gaba':{'weight':f_GFFS_w},
                                   
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
                                    'CF':{'rate':f_CF_rate},
                                    'M1':{'rate':f_MS},  
                                    'M2':{'rate':f_MS},
                                    
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
                                   'M1_low':{'beta_I_GABAA_3': f_beta_rm(2.5),
                                             'beta_I_GABAA_2': f_beta_rm(x),
                                             'GABAA_3_Tau_decay':87.},
                                   'M2_low':{'beta_I_GABAA_3': f_beta_rm(2.4),
                                             'beta_I_GABAA_2': f_beta_rm(x),
                                             'GABAA_3_Tau_decay':76.},
                                   'FS_low':{'beta_I_GABAA_2': f_beta_rm(1.6),
                                             'beta_I_GABAA_3': f_beta_rm(1.6),
                                             'GABAA_2_Tau_decay':66.},
                                   'GA_M1_gaba':{'weight':0.08},
                                   'GA_M2_gaba':{'weight':0.17},
                                   'CS_ST_ampa':{'delay':CSST_delay},
                                   'CS_ST_nmda':{'delay':CSST_delay},
                                   'C1_M1_ampa':{'delay':CXMS_delay},
                                   'C1_M1_nmda':{'delay':CXMS_delay},            
                                   'C2_M2_ampa':{'delay':CXMS_delay},
                                   'C2_M2_nmda':{'delay':CXMS_delay},            
                                   'CF_FS_ampa':{'delay':CXMS_delay},
                                   'CF_FS_nmda':{'delay':CXMS_delay}
                                   },
                           
                           # Tuning rate GI/GF and GA
                           'node':{
                                    'EI':{'rate':EIr},
                                    'EF':{'rate':EFr},
                                    'EA':{'rate':3*EAr}
                                    },
                           
                           # Activate GF to FS connection
                           'conn':{'GF_FS_gaba':{'lesion':False},
                                   'M1_M1_gaba':{'beta_fan_in': f_beta_rm(x)},
                                   'M1_M2_gaba':{'beta_fan_in': f_beta_rm(x)},
                                   'M2_M1_gaba':{'beta_fan_in': f_beta_rm(x)},
                                   'M2_M2_gaba':{'beta_fan_in': f_beta_rm(x)}
                                   }
                           }
                         
                        d=misc.dict_update(p['equal'], d) 
                        s='MS_{0}_MSMS_{1}_dop_{2}_GAFS_w_{3}_GFFS_w_{4}_CF_rate_{5}__delays_{6}-{7}_{8}'
                        s=s.format( f_MS, f_MSMS_w, x, 
                                    f_GAFS_w, f_GFFS_w, f_CF_rate,  
                                    CSST_delay, CXMS_delay, labels[i] )
                         
                        l[i][-1]+=pl(d, '=', **{'name':s})     
 
     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
