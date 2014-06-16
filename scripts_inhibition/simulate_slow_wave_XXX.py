'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_slow_wave as Builder
from toolbox.parallel_excecution import loop

import simulate_slow_wave
import pprint
pp=pprint.pprint




def perturbations():
    sim_time=10000.0
    size=10000.0
    threads=8
    l=[]
    l+=[pl({'node':{'M1':{'model':'M1_high'},
                    'M2':{'model':'M2_high'},
                    'FS':{'model':'FS_high'}}},
           '=',
            **{'name':'high_rev'})]
    
    l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.5},
                    'M1_M2_gaba':{'weight':0.5},
                    'M2_M1_gaba':{'weight':0.5},
                    'M2_M2_gaba':{'weight':0.5}}},
           '*',
            **{'name':'weight*0.5'})]
    l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.33},
                    'M1_M2_gaba':{'weight':0.33},
                    'M2_M1_gaba':{'weight':0.33},
                    'M2_M2_gaba':{'weight':0.33}}},
           '*',
            **{'name':'weight*0.33'})]   
    l+=[pl(**{'name':'no_pert'})]
    l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.1},
                    'M1_M2_gaba':{'weight':0.1},
                    'M2_M1_gaba':{'weight':0.1},
                    'M2_M2_gaba':{'weight':0.1}}},
           '*',
            **{'name':'weight*0.1'})]       
    l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.05},
                    'M1_M2_gaba':{'weight':0.05},
                    'M2_M1_gaba':{'weight':0.05},
                    'M2_M2_gaba':{'weight':0.05}}},
           '*',
            **{'name':'weight*0.05'})]   
    l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.05},
                    'M1_M2_gaba':{'weight':0.05},
                    'M2_M1_gaba':{'weight':0.05},
                    'M2_M2_gaba':{'weight':0.05}}},
           '*',
            **{'name':'weight*0.05'})]   
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.5},
                    'GA_M2_gaba':{'weight':0.5}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.5'})]   
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_FS_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_FS_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.1},
                    'M1_M2_gaba':{'weight':0.1},
                    'M2_M1_gaba':{'weight':0.1},
                    'M2_M2_gaba':{'weight':0.1},
                    
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.1'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_FS_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.05},
                    'M1_M2_gaba':{'weight':0.05},
                    'M2_M1_gaba':{'weight':0.05},
                    'M2_M2_gaba':{'weight':0.05},
                  
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.05'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':5.},
                    'GA_M2_gaba':{'weight':5.},
                    'M1_low':{'GABAA_3_Tau_decay':1./5},
                    'M2_low':{'GABAA_3_Tau_decay':1./5},
                    
                    }},
           '*',
            **{'name':'weightGPe-faster*5'})] 
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':1},
                    'GA_M2_gaba':{'weight':1},
                    'M1_low':{'GABAA_3_Tau_decay':1./5},
                    'M2_low':{'GABAA_3_Tau_decay':1./5},
                    }},
           '*',
            **{'name':'weightGPe-faster'})] 
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.5},
                    'GA_M2_gaba':{'weight':0.5},
                    'GA_FS_gaba':{'weight':0.5},
                    'M1_M1_gaba':{'weight':0.5},
                    'M1_M2_gaba':{'weight':0.5},
                    'M2_M1_gaba':{'weight':0.5},
                    'M2_M2_gaba':{'weight':0.5}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.5and*0.5'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.5},
                    'M1_M2_gaba':{'weight':0.5},
                    'M2_M1_gaba':{'weight':0.5},
                    'M2_M2_gaba':{'weight':0.5}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.5'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.5},
                    'GA_M2_gaba':{'weight':0.5},
                    'GA_FS_gaba':{'weight':0.5},   
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.5and*0.25'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.5},
                    'GA_M2_gaba':{'weight':0.5},
                    'GA_FS_gaba':{'weight':0.5},
                    'M1_M1_gaba':{'weight':0.5},
                    'M1_M2_gaba':{'weight':0.5},
                    'M2_M1_gaba':{'weight':0.5},
                    'M2_M2_gaba':{'weight':0.5},
                    'GA_M1_gaba':{'weight':5.},
                    'GA_M2_gaba':{'weight':5.},
                    'M1_low':{'GABAA_3_Tau_decay':1./5},
                    'M2_low':{'GABAA_3_Tau_decay':1./5},
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.5and*0.5-fast'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_FS_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.5},
                    'M1_M2_gaba':{'weight':0.5},
                    'M2_M1_gaba':{'weight':0.5},
                    'M2_M2_gaba':{'weight':0.5},
                    'GA_M1_gaba':{'weight':5.},
                    'GA_M2_gaba':{'weight':5.},
                    'M1_low':{'GABAA_3_Tau_decay':1./5},
                    'M2_low':{'GABAA_3_Tau_decay':1./5},
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.5-fast'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.5},
                    'GA_M2_gaba':{'weight':0.5},
                    'GA_FS_gaba':{'weight':0.5},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'GA_M1_gaba':{'weight':5.},
                    'GA_M2_gaba':{'weight':5.},
                    'M1_low':{'GABAA_3_Tau_decay':1./5},
                    'M2_low':{'GABAA_3_Tau_decay':1./5},
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.5and*0.25-fast'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_FS_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'GA_M1_gaba':{'weight':5.},
                    'GA_M2_gaba':{'weight':5.},
                    'M1_low':{'GABAA_3_Tau_decay':1./5},
                    'M2_low':{'GABAA_3_Tau_decay':1./5},
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.3and*0.25-fast'})]   
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_FS_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.3},
                    'M1_M2_gaba':{'weight':0.3},
                    'M2_M1_gaba':{'weight':0.3},
                    'M2_M2_gaba':{'weight':0.3}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.3and*0.25'})]  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'GA_FS_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.35},
                    'M1_M2_gaba':{'weight':0.35},
                    'M2_M1_gaba':{'weight':0.35},
                    'M2_M2_gaba':{'weight':0.35},
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.35and*0.25'})]  


    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.3},
                    'GA_M2_gaba':{'weight':0.3},
                    'GA_FS_gaba':{'weight':0.3},
                    'M1_M1_gaba':{'weight':0.3},
                    'M1_M2_gaba':{'weight':0.3},
                    'M2_M1_gaba':{'weight':0.3},
                    'M2_M2_gaba':{'weight':0.3},
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.3and*0.3'})]  

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
            **{'name':'old'})]     
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':2.},
                    'GA_M2_gaba':{'weight':2.},
                    'GA_FS_gaba':{'weight':2.},
                    'M1_M1_gaba':{'weight':0.6},
                    'M1_M2_gaba':{'weight':0.6},
                    'M2_M1_gaba':{'weight':0.6},
                    'M2_M2_gaba':{'weight':0.6},
                    'M1_low':{'GABAA_3_Tau_decay':12.},
                    'M2_low':{'GABAA_3_Tau_decay':12.},
                    'FS_low':{'GABAA_2_Tau_decay':12.},
                    },
            'conn':{'M1_M1_gaba':{'fan_in0':140},
                    'M1_M2_gaba':{'fan_in0':140},
                    'M2_M1_gaba':{'fan_in0':140},
                    'M2_M2_gaba':{'fan_in0':140}}},
           '=',
            **{'name':'oldMSN*6'})]   

    l+=[pl({'nest':{
                    'M1_M1_gaba':{'weight':1.},
                    'M1_M2_gaba':{'weight':0.5},
                    'M2_M1_gaba':{'weight':0.5},
                    'M2_M2_gaba':{'weight':0.5},
                    },
            'conn':{'M1_M1_gaba':{'fan_in0':0.75},
                    'M1_M2_gaba':{'fan_in0':1.5},
                    'M2_M1_gaba':{'fan_in0':0.75},
                    'M2_M2_gaba':{'fan_in0':0.75}}},
           '*',
            **{'name':'scaled1'})]     
    l+=[pl({'nest':{
                    'M1_M1_gaba':{'weight':1.},
                    'M1_M2_gaba':{'weight':0.5},
                    'M2_M1_gaba':{'weight':0.5},
                    'M2_M2_gaba':{'weight':0.5},
                    },
            'conn':{'M1_M1_gaba':{'fan_in0':0.5},
                    'M1_M2_gaba':{'fan_in0':2.0},
                    'M2_M1_gaba':{'fan_in0':0.5},
                    'M2_M2_gaba':{'fan_in0':0.5}}},
           '*',
            **{'name':'scaled2'})]    

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                  'sim_stop':sim_time,
                  'threads':threads},
                  'netw':{'size':size}}, 
                  '=')
    return l
p_list=perturbations()
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

for j in range(1,3):
    for i, p in enumerate(p_list):
        from_disk=j

        fun=simulate_slow_wave.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, script_name])


loop(args_list, path, 6)
        