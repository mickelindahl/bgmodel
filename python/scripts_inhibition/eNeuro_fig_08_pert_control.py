'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
from core import misc

import numpy
import pprint
pp=pprint.pprint

# from oscillation_perturbations8 import get_solution_final_beta2

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
    solution={'equal':{},
              'mul':{}}#get_solution_final_beta()

    d={}
    misc.dict_update(d, solution['mul'])
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    misc.dict_update(d, solution['equal']) 
    l[-1]+=pl(d, '=', **{'name':'control_sim'}) 
    
    # Decreasing delay TA-striatum increases oscillations in MSN and FSN
    for y in [12]+ list(numpy.arange(5,105,10)):

        d={}
        misc.dict_update(d, solution['mul'])
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        ratio=100./y
        
        dd={'nest':{'GA_M1_gaba':{'weight':0.4*ratio}, 
                    'GA_M2_gaba':{'weight':0.8*ratio}}}
        misc.dict_update(d,dd)            
         
        # Decreasing from 2 leads to ...
        # Increasing from 2 leads to ... 
        dd={'nest':{'GA_FS_gaba':{'weight':2./0.29*(17./66.)*ratio}}}
        misc.dict_update(d,dd)           
        
        # Just assumed to be 12 ms    
        dd={'nest':{'M1_low':{'GABAA_3_Tau_decay':87./ratio},  
                    'M2_low':{'GABAA_3_Tau_decay':76./ratio},
                    'FS_low':{'GABAA_2_Tau_decay':66./ratio},     
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
                          numpy.arange(1, 1.8, 0.15)):
        d={}
        misc.dict_update(d, solution['mul'])  
        misc.dict_update(d,{'node':{'EA':{'rate': EA_rate}}})

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
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
    
    # Connectivity GPe-STN
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
                     
                     'GF_GA_gaba':{'fan_in0': int((30-y)*0.1),'rule':'all-all' },
                     'GI_GA_gaba':{'fan_in0': (30-y)-int((30-y)*0.1),'rule':'all-all' },
                     }}
        
        misc.dict_update(d, dd)
         
     
        l[-1]+=pl(d, '=', **{'name':'GAGA_'+str(y)+'_GIGA_'+str(30-y)})  
  
    # Effect of MSN-MSN strength
    for y in [0.125, 0.25, 0.5, 1 , 2, 4, 8]:       
        
        d={}
        misc.dict_update(d, solution['mul'])  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        dd={'nest': {'M1_M1_gaba':{'weight': y }, 
                     'M1_M2_gaba':{'weight': y },
                     'M2_M1_gaba':{'weight': y },
                     'M2_M2_gaba':{'weight': y }}}
        
        misc.dict_update(d, dd)
        l[-1]+=pl(d, '=', **{'name':'MSMS_'+str(y)})    
        
    # Phase synch in GP and ST effected by delay from cortex to STR
    for y in numpy.arange(20., 1.5, -2.5):

            d={}
            misc.dict_update(d, solution['mul'])       
            l+=[pl(d, '*', **{'name':''})]
              
            d={}
            misc.dict_update(d, solution['equal']) 
            dd={'nest':{'C1_M1_ampa':{'delay':y},
                        'C1_M1_nmda':{'delay':y},            
                        'C2_M2_ampa':{'delay':y},
                        'C2_M2_nmda':{'delay':y},            
                        'CF_FS_ampa':{'delay':y},
                        'CF_FS_nmda':{'delay':y}}}            

            misc.dict_update(d, dd)
            l[-1]+=pl(d, '=', **{'name':'CXSTRdelay_'+str(y)})          

    # Delay STGP and GPST delay
    for y, z in zip(*[numpy.arange(1.,9.)]*2):

            d={}
            misc.dict_update(d, solution['mul'])       
            l+=[pl(d, '*', **{'name':''})]
           
            d={}
            misc.dict_update(d, solution['equal']) 
            
            dd={'nest':{'ST_GA_ampa':{'delay':y},
                        'ST_GI_ampa':{'delay':y},
                        'GI_ST_gaba':{'delay':z}}}
            misc.dict_update(d, dd)   
            l[-1]+=pl(d, '=', **{'name':'STGPdelay_'+str(y)+'_GPSTdelay_'+str(z)})      
              
    return l

    # Strength STN-GP 
    r=numpy.array([0.,200.,400.,600.,800., 1000.0])+200.
    w=numpy.array([304., 0.259, 0.225, 0.18, 0.134, 0.09])+0.46
    for y, z in zip(*[r,w]):

            d={}
            misc.dict_update(d, solution['mul'])       
            l+=[pl(d, '*', **{'name':''})]
           
            d={}
            misc.dict_update(d, solution['equal']) 
            
            dd={'nest':{'ST_GA_ampa':{'weight':z}},
                'node':{'GA':{'rate':y}}}
            misc.dict_update(d, dd)   
            l[-1]+=pl(d, '=', **{'name':'GAr_'+str(y)+'_STGAw_'+str(z)})      
              
    return l



get()