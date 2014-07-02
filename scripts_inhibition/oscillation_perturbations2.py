'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl

import pprint
pp=pprint.pprint


def get():
    
    l=[]

    for w in [0.95, 0.9, 0.85]:     
        v=0.25
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v},
                        'GA_M2_gaba':{'weight':v},
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},
                        },
                'node':{'C1':{'rate':w},
                        'C2':{'rate':w}}},
               '*',
                **{'name':'MsGa-MS-weight'+str(v)})]  
    
    for w in [0.95, 0.9, 0.85]:     
        v=0.25
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v*5},
                        'GA_M2_gaba':{'weight':v*5},
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},
                        
                        },
                'node':{'C1':{'rate':w},
                        'C2':{'rate':w}}},
               '*',
                **{'name':'MsGa-MS-weight'+str(v)})]  

        l[-1]+=pl({'nest':{
                           'GA_FS_gaba':{'weight':5.},
                           'M1_low':{'GABAA_3_Tau_decay':1./5*2}, # Even out 
                           'M2_low':{'GABAA_3_Tau_decay':1./5},
                           'FS_low':{'GABAA_2_Tau_decay':1./5},
                    
                    }},
           '*',
            **{'name':'fast'}) 

    for w in [0.95, 0.85]:
        for u in [0.4, 0.35, 0.3]:     
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
                    'node':{'C1':{'rate':w},
                            'C2':{'rate':w}}},
                   '*',
                    **{'name':'MsGa-MS-weight'+str(v)})]  

    for w in [0.95, 0.85]:
        for u in [0.4, 0.35, 0.3]:     
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
                    'node':{'C1':{'rate':w},
                            'C2':{'rate':w}}},
                   '*',
                    **{'name':'MsGa-MS-weight'+str(v)})]  
            l[-1]+=pl({'nest':{
                           'GA_FS_gaba':{'weight':5.},
                           'M1_low':{'GABAA_3_Tau_decay':1./5*2}, # Even out 
                           'M2_low':{'GABAA_3_Tau_decay':1./5},
                           'FS_low':{'GABAA_2_Tau_decay':1./5},
                    
                    }},
           '*',
            **{'name':'fast'}) 
   
    return l
 