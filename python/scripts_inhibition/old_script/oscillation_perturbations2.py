'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl

import pprint
pp=pprint.pprint


def get():
    
    l=[]
    
    for u in [2.5, 5.]:
        v=0.25
        l+=[pl({'nest':{
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},
                        },},
               '*',
                **{'name':'MsGa-MS-weight'+str(v)})]  

        l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':v*u}, # Even out
                           'GA_M2_gaba':{'weight':v*u},
                           'GA_FS_gaba':{'weight':u},
                           'M1_low':{'GABAA_3_Tau_decay':1./5},  
                           'M2_low':{'GABAA_3_Tau_decay':1./5},
                           'FS_low':{'GABAA_2_Tau_decay':1./5},
                    
                    }},
           '*',
            **{'name':'fast-'+str(u)}) 

    for u in [0.4, 0.3]:     
        v=0.25
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v},
                        'GA_M2_gaba':{'weight':v},
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},
                        'ST_GA_ampa':{'weight':u},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25}
                        },
                        },
               '*',
                **{'name':'MsGa-MS-weight-StGa-'+str(u)})]  
    
    
    for u in [0.4, 0.3]:  
        for x in [2.5, 5.]:
            v=0.25
            l+=[pl({'nest':{
                            'M1_M1_gaba':{'weight':v},
                            'M1_M2_gaba':{'weight':v},
                            'M2_M1_gaba':{'weight':v},
                            'M2_M2_gaba':{'weight':v},
                            'ST_GA_ampa':{'weight':u},
                            'GA_GA_gaba':{'weight':0.25},
                            'GI_GA_gaba':{'weight':0.25}
                            },},
                   '*',
                    **{'name':'MsGa-MS-weight-StGa-'+str(u)})]  
    
            l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':v*x}, # Even out
                               'GA_M2_gaba':{'weight':v*x},
                               'GA_FS_gaba':{'weight':x},
                               'M1_low':{'GABAA_3_Tau_decay':1./5},  
                               'M2_low':{'GABAA_3_Tau_decay':1./5},
                               'FS_low':{'GABAA_2_Tau_decay':1./5},
                        
                        }},
               '*',
                **{'name':'fast-'+str(x)}) 
        
           
    return l
 