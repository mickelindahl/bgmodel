'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl

import pprint
pp=pprint.pprint


def get():
    
    l=[]
    for rate in [680., 700., 720.]:
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                        'GA_M2_gaba':{'weight':0.25},
                        'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25}
                        }},
               '*',
                **{'name':'MsGa-MS-weight0.25_C2r-'+str(rate)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':rate},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':''})  

        
    for w in [0.75, 0.5, 0.25]:
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                        'GA_M2_gaba':{'weight':0.25},
                        'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25},
                        'GI_ST_gaba':{'weight':w},
                        'ST_GI_ampa':{'weight':w}
                        }},
               '*',
                **{'name':'MsGa-MS-weight0.25_ST-GI-'+str(w)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':''})  
   
   
   
   
    
    return l
 