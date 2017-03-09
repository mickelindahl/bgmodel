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
                       
        for MSMSdop in [0.25,0.5]:
            for CFf, GPFSf, GAMSf in [
                                      [1.,1.,1.],
                                      
                                      [0.95,1.,1.],
                                      [0.9,1.,1.],
                                      [0.85,1.,1.],
                                      
                                      [1.,1.5,1.],
                                      [1.,2.,1.],
                                      
                                      [1.,1.,1.1],
                                      [1.,1.,1.2],
                                      [1.,1.,1.3],
                                      [1.,1.,1.4],
                                      
                                      [.9, 1.,1.2],
                                      [.9, 1.,1.4],
                                      [.85, 1.,1.2],
                                      [.85, 1.,1.4],
                                      
                                      [1., 2.,1.2],
                                      [1., 2.,1.4],
                                      
                                      ]:    
                
                d={
                   'nest':{
                           
                            'GA_M1_gaba':{'weight':GAMSf},
                            'GA_M2_gaba':{'weight':GAMSf},
                            
                            'GA_FS_gaba':{'weight':GPFSf},
                            'GF_FS_gaba':{'weight':GPFSf},
                           },
                    'node':{
                            'CF':{'rate':CFf},
                                }}
                
                d=misc.dict_update(p['mul'], d) 
                 
                l[i]+=[pl(d, '*', **{'name':''})]
                          
                d={'nest':{
                           'M1_low':{'beta_I_GABAA_2': f_beta_rm(MSMSdop)},
                           'M2_low':{'beta_I_GABAA_2': f_beta_rm(MSMSdop)}
                           }
                   }
                 
                d=misc.dict_update(p['equal'], d) 
                s='CFf_{0}_GPFSf_{1}_GAMSf_{2}_MSMSdop_{3}_{4}'
                s=s.format( CFf, GPFSf, GAMSf, MSMSdop, labels[i] )
                 
                l[ i ][ -1 ]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
