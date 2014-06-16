'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl

import pprint
pp=pprint.pprint


def get():
    
    l=[]
    l+=[pl(**{'name':'no_pert'})]
    l+=[pl({'node':{'M1':{'model':'M1_high'},
                    'M2':{'model':'M2_high'},
                    'FS':{'model':'FS_high'}}},
           '=',
            **{'name':'high_rev'})]
    
    for v in [0.75, 0.5, 0.25, 0.1]:    
        l+=[pl({'nest':{'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v}}},
           '*',
            **{'name':'MS-MS-weight*'+str(v)})]

    for v in [0.75, 0.5, 0.25, 0.1]:     
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v},
                        'GA_M2_gaba':{'weight':v}
                        }},
               '*',
                **{'name':'GA-MS-weight*'+str(v)})]   

    for v in [0.75, 0.5, 0.25, 0.1]:     
    
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v},
                        'GA_M2_gaba':{'weight':v},
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},
                        
                        }},
               '*',
                **{'name':'MsGa-MS-weight'+str(v)})]  
    
    for v in [0.75, 0.5, 0.25, 0.1]:     
    
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v*5},
                        'GA_M2_gaba':{'weight':v*5},
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},
                        
                        }},
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
    
    l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.2},
                    'M1_M2_gaba':{'weight':0.2},
                    'M2_M1_gaba':{'weight':0.2},
                    'M2_M2_gaba':{'weight':0.2},
                     },
            'conn':{'M1_M1_gaba':{'fan_in0':140},
                    'M1_M2_gaba':{'fan_in0':140},
                    'M2_M1_gaba':{'fan_in0':140},
                    'M2_M2_gaba':{'fan_in0':140}}},
           '=',
            **{'name':'old-and-slow'})]     
  
   
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':2.},
                    'GA_M2_gaba':{'weight':2.},
                    'GA_FS_gaba':{'weight':2.},
                    'M1_M1_gaba':{'weight':0.2},
                    'M1_M2_gaba':{'weight':0.2},
                    'M2_M1_gaba':{'weight':0.2},
                    'M2_M2_gaba':{'weight':0.2},
                    'M1_low':{'GABAA_3_Tau_decay':12.},
                    'M2_low':{'GABAA_3_Tau_decay':12.},
                    'FS_low':{'GABAA_2_Tau_decay':12.},
                    },
            'conn':{'M1_M1_gaba':{'fan_in0':140},
                    'M1_M2_gaba':{'fan_in0':140},
                    'M2_M1_gaba':{'fan_in0':140},
                    'M2_M2_gaba':{'fan_in0':140}}},
           '=',
            **{'name':'old-and-fast'})]     
    
    return l
 