'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_final_beta2

def STN_ampa_gaba_input_magnitude():
    l=[
       [20, 188, 0.08], # 20 Hz 
       [25, 290, 0.119], #25 Hz
       [30, 430, 0.18],      #30 Hz
       [35, 540, 0.215],     #35 Hz,
       [40, 702, 0.28],     #40 Hz
       [45, 830., 0.336],   #45 Hz 
#        [46, 876.7, 0.349],  #46 Hz
       [50, 1000.8, 0.3957],     # 50 Hz
       [55, 1159., 0.458],  #50 Hz
#        [80, 2102, 0.794] # 80 Hz] 
       ]
    
    return l

def get():
    
    
    l=[]
    solution=get_solution_final_beta2()

    d={}
    misc.dict_update(d, solution['mul'])
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    misc.dict_update(d, solution['equal']) 
    l[-1]+=pl(d, '=', **{'name':'control_sim'}) 
    
    # Decreasing delay TA-striatum increases oscillations in MSN and FSN
    for y in [12]+ list(numpy.arange(5,55,5)):

        misc.dict_update(d, solution['mul'])
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        ratio=12./y
        
        dd={'nest':{'GA_M1_gaba':{'weight':0.8*ratio}, 
                   'GA_M2_gaba':{'weight':0.8*ratio}}}
        misc.dict_update(d,dd)            
         
        # Decreasing from 2 leads to ...
        # Increasing from 2 leads to ... 
        dd={'nest':{'GA_FS_gaba':{'weight':2.*ratio}}}
        misc.dict_update(d,dd)           
        
        # Just assumed to be 12 ms    
        dd={'nest':{'M1_low':{'GABAA_3_Tau_decay':12./ratio},  
                    'M2_low':{'GABAA_3_Tau_decay':12./ratio},
                    'FS_low':{'GABAA_2_Tau_decay':12./ratio},     
                   }}
        
        misc.dict_update(d,dd)  
        
        l[-1]+=pl(d, '=', **{'name':'mod_GAtau_'+str(y)}) 
    
    # Phase synch in GP and ST effected by delay from cortex
    for y in numpy.arange(2.5, 21., 2.5):

            d={}
            misc.dict_update(d, solution['mul']) 

            
            l+=[pl(d, '*', **{'name':''})]
              
            d={}
            misc.dict_update(d, solution['equal']) 
            misc.dict_update(d,{'nest':{'CS_ST_ampa':{'delay':y}}})
            misc.dict_update(d,{'nest':{'CS_ST_nmda':{'delay':y}}})            

        
            l[-1]+=pl(d, '=', **{'name':'CSSTdelay_'+str(y)})    
    
    # Effect on phase TI-TA when having MSN connect to TA
    for y, EA_rate in zip(numpy.arange(25,130,25),
                          numpy.arange(1,1.8,0.15)):
        d={}
        
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 

        d['node']['EA']['rate']*=EA_rate
        
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'M2_GA_gaba':{'fan_in0': y}}})
        
        
        l[-1]+=pl(d, '=', **{'name':'M2GA_'+str(y)+'_EArate_'+str(EA_rate)})
        
    # Effect of TI  back to str 
    for y in numpy.arange(1, 6, 1):
    
        d={}
        
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
            
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_FS_gaba':{'fan_in0': y}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M1_gaba':{'fan_in0': y}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'lesion': False}}})
        misc.dict_update(d,{'conn':{'GI_M2_gaba':{'fan_in0': y}}})        
               
        
        l[-1]+=pl(d, '=', **{'name':'mod_GI_M2_'+str(y)})  
    
    
    # 
    v=STN_ampa_gaba_input_magnitude()    
    for _, x, y in v:

        d={}
        misc.dict_update(d, solution['mul'])   
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal'])  
        
        misc.dict_update(d,{'node':{'CS':{'rate': x}}})
        misc.dict_update(d,{'nest':{'GI_ST_gaba':{'weight':y }}})
        
        l[-1]+=pl(d, '=', **{'name':'mod_ST_inp_'+str(x)+'_'+str(y)})  
        
    # Effect of connectiviity onto TA from TI and TA
    for y in numpy.arange(5.,29.,5.):       
        
        d={}
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        dd={'conn': {'GA_GA_gaba':{'fan_in0': y,'rule':'all-all' }, 
                     'GA_GI_gaba':{'fan_in0': 2,'rule':'all-all' },
                     'GI_GA_gaba':{'fan_in0': 30-y,'rule':'all-all' },
                     'GI_GI_gaba':{'fan_in0': 28,'rule':'all-all' }}}
        
        misc.dict_update(d, dd)
         
     
        l[-1]+=pl(d, '=', **{'name':'GAGA_'+str(y)+'_GIGA_'+str(30-y)})  
  
        
    return l


get()