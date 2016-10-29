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
     
              
        for f_GAFS_w, f_GFFS_w, f_CF_rate in [
                                            [0.2,0.8, 1.8],
                                            [0.0,1.0, 1.2]
                                            ]:    
            
            for f_MS, f_GI, f_GA in [
                                     [1., 1., 1.],
                                     [1.02, 1., 1.],
                                     [1.04, 1., 1.],
                                     [1.06, 1., 1.],
                                     [1.08, 1., 1.],
                                     [1.1, 1., 1.],
                                     
                                     [1., 1., .95],
                                     [1.02, 1., .9],
                                     [1.04, 1., .85],
                                     [1.06, 1., .8],
                                     [1.08, 1., .75],
                                     [1.1, 1., .7],
                                     
                                     [1., .95,  1.],
                                     [1., .9, 1.],
                                     [1., .85, 1.],
                                     [1., .8, 1.],
                                     [1., .75, 1.],
                                     
                                     [1., .95,  .95],
                                     [1., .9, .9],
                                     [1., .85, .85],
                                     [1., .8, .8],
                                     [1., .75, .75],
                                     
                                     [1., .95,  .97],
                                     [1., .9, .94],
                                     [1., .85, .91],
                                     [1., .8, .88],
                                     [1., .75, .85],
                                     
                                     ]:
                     
                d={'nest':{
                           # Weight GA and GF to FS
                           'GA_FS_gaba':{'weight':f_GAFS_w},
                           'GF_FS_gaba':{'weight':f_GFFS_w},
                           
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
                                     'GABAA_3_Tau_decay':87.},
                           'M2_low':{'beta_I_GABAA_3': f_beta_rm(2.4),
                                     'GABAA_3_Tau_decay':76.},
                           'FS_low':{'beta_I_GABAA_2': f_beta_rm(1.6),
                                     'beta_I_GABAA_3': f_beta_rm(1.6),
                                     'GABAA_2_Tau_decay':66.},
                           'GA_M1_gaba':{'weight':0.08},
                           'GA_M2_gaba':{'weight':0.17}},
                   
                   # Tuning rate GI/GF and GA
                   'node':{
                            'EI':{'rate':f_GI*p['equal']['node']['EI']['rate']},
                            'EF':{'rate':f_GI*p['equal']['node']['EF']['rate']},
                            'EA':{'rate':f_GA*p['equal']['node']['EA']['rate']}
                            },
                   
                   # Activate GF to FS connection
                   'conn':{'GF_FS_gaba':{'lesion':False}}
                   }
                 
                d=misc.dict_update(p['equal'], d) 
         
                s='MS_{2}_GI_{2}_GA_{2}_GAFS_w_{3}_GFFS_w_{4}_CF_rate{5}_{6}'.format(f_MS, f_GI, f_GA ,
                                                                                     f_GAFS_w, f_GFFS_w, f_CF_rate,
                                                                                     labels[i] )
                 
                l[i][-1]+=pl(d, '=', **{'name':s})     
 
     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       