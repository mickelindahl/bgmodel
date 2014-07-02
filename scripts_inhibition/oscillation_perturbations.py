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
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0-100.},
                      'C2':{'rate':740.0-150.},
                      }},
           '=',
            **{'name':''})  


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


    ###############################
    # Investigate GA GI crosstalk
    ###############################
    v=0.25



    for GA_GA, GA_GI, GI_GI, GI_GA in [[30, 0, 30, 0],
                                       [26, 4, 26, 2],
                                       [22, 8, 22, 8],
                                       [18, 12, 18, 12],
                                       [14, 16, 14, 16]]:
#     for v in [0.75, 0.5, 0.25, 0.1]:     
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v},
                        'GA_M2_gaba':{'weight':v},
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},


                        }},
               '*',
                **{'name':'MsGa-MS-weight'+str(v)})]  

        l[-1]+=pl({'conn':{
                        'GA_GA_gaba':{'fan_in0': GA_GA}, 
                        'GA_GI_gaba':{'fan_in0': GA_GI},
                        'GI_GI_gaba':{'fan_in0': GI_GI},
                        'GI_GA_gaba':{'fan_in0': GI_GA},                        
                    
                    }},
           '=',
            **{'name':'ct-'+'{}-{}-{}-{}'.format(GA_GA, GA_GI, GI_GI, GI_GA)}) 
    
    # Half incomming  to GA
    for GA_GA, GA_GI, GI_GI, GI_GA in [[15, 0, 30, 0],
                                       [13, 2, 28, 2],
                                       [11, 4, 26, 4],
                                       [9, 6, 24, 6],
                                       [7, 8, 22, 8]]:
#     for v in [0.75, 0.5, 0.25, 0.1]:     
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':v},
                        'GA_M2_gaba':{'weight':v},
                        'M1_M1_gaba':{'weight':v},
                        'M1_M2_gaba':{'weight':v},
                        'M2_M1_gaba':{'weight':v},
                        'M2_M2_gaba':{'weight':v},
                        'ST_GA_ampa':{'weight':0.5},
                        }},
               '*',
                **{'name':'MsGa-MS-weight'+str(v)})]  
        
        l[-1]+=pl({'conn':{
                        'GA_GA_gaba':{'fan_in0': GA_GA}, 
                        'GA_GI_gaba':{'fan_in0': GA_GI},
                        'GI_GI_gaba':{'fan_in0': GI_GI},
                        'GI_GA_gaba':{'fan_in0': GI_GA},                        
                    
                    }},
           '=',
            **{'name':'ct-wh-'+'{}-{}-{}-{}'.format(GA_GA, GA_GI, GI_GI, GI_GA)}) 
    
    return l
 