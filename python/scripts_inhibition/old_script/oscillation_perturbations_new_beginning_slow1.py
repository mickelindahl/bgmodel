'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def STN_ampa_gaba_input_magnitude():
    l=[
#        [20, 188, 0.08], # 20 Hz 
       [25, 290, 0.119], #25 Hz
       [30, 430, 0.18],      #30 Hz
       [35, 540, 0.215],     #35 Hz,
       [40, 702, 0.28],     #40 Hz
#        [45, 830., 0.336],   #45 Hz 
#        [46, 876.7, 0.349],  #46 Hz
#        [50, 1000.8, 0.3957],     # 50 Hz
#        [55, 1159., 0.458],  #50 Hz
#        [80, 2102, 0.794] # 80 Hz] 
       ]
    
    return l

def get_solution():


    solution={'mul':{},
              'equal':{}}
    
    #Decreasing from 0.25 leads to ...
    #Increasing from 0.25 leads to ...
    d={'nest':{'M1_M1_gaba':{'weight':0.25},
               'M1_M2_gaba':{'weight':0.25},
               'M2_M1_gaba':{'weight':0.25},
               'M2_M2_gaba':{'weight':0.25}}}
    misc.dict_update(solution,{'mul':d})    
    
    # GA firing rate needs to be maintained, around 12 Hz for sw in 
    # dopamine depleted rats). When decreasing/increasing
    # ST-GA one need to either compensate by changing inhibition from 
    # GPe and/or EA external input (assume that synapses from TA and TI
    # are of equal strength. 
    
    # This script is first run where ST-GA GP-GA and EA individually are
    # perturbed to determine each relative influence to TA firing rate. 
    # From this data then combinations fo changes that should results in
    # 12 Hz TA firing rate are created.  
    d={'nest':{'ST_GA_ampa':{'weight':0.25},
               'GA_GA_gaba':{'weight':0.25},
               'GI_GA_gaba':{'weight':0.25}}}
    misc.dict_update(solution, {'mul':d})    
    
    d={'node':{'EA':{'rate':200.0}}}
    misc.dict_update(solution,{'equal':d})              
    
    # C1 and C2 have been set such that MSN D1 and MSN D2 fires at 
    # low firing rates (0.1-0,2 Hz). 
    d={'node':{'C1':{'rate':560.0},
               'C2':{'rate':700.}}}
    misc.dict_update(solution,{'equal':d})              
           
    # Set such that GPe TI fires in accordance with slow wave sleep
    # in dopamine depleted rats.
    d={'node':{'EI':{'rate':1400.0}}} # Increase with 340, to get close to 24 Hz sw dopamine depleted rats TI
    misc.dict_update(solution,{'equal':d})           
    
    # Decrease weight since tau decay is 5 times stronger
    d={'nest':{'GA_M1_gaba':{'weight':0.8/5}, 
               'GA_M2_gaba':{'weight':0.8/5}}}
    misc.dict_update(solution,{'equal':d})            
     
    # Decrease weight since tau decay is 5 times stronger
    d={'nest':{'GA_FS_gaba':{'weight':2./5}}}
    misc.dict_update(solution,{'equal':d})           
    
    # Just assumed to be 12*5 ms    
    d={'nest':{'M1_low':{'GABAA_3_Tau_decay':12.*5},  
               'M2_low':{'GABAA_3_Tau_decay':12.*5},
               'FS_low':{'GABAA_2_Tau_decay':12.*5},     
               }}
    misc.dict_update(solution,{'equal':d})           

    #Dopamine such that STN increase above 50-100 %    
    x=2.5
    d={'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x),
                     'beta_I_NMDA_1': f_beta_rm(x)}}}
    misc.dict_update(solution,{'equal':d})            

    # Delay ctx striatum and ctx stn set to 2.5 ms Jaeger 2011
    y=2.5
    misc.dict_update(solution,{'equal':{'nest':{'C1_M1_ampa':{'delay':y}}}})
    misc.dict_update(solution,{'equal':{'nest':{'C1_M1_nmda':{'delay':y}}}})            
    misc.dict_update(solution,{'equal':{'nest':{'C2_M2_ampa':{'delay':y}}}})
    misc.dict_update(solution,{'equal':{'nest':{'C2_M2_nmda':{'delay':y}}}})            
    misc.dict_update(solution,{'equal':{'nest':{'CF_FS_ampa':{'delay':y}}}})     
    
    # Dopamine effect on MS-GI
    d={'equal':{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}}}
    misc.dict_update(solution,d)
    
    # GI predominently connect to to GA 
    d={'conn': {'GA_GA_gaba':{'fan_in0': 5}, 
                'GI_GA_gaba':{'fan_in0': 25 }}}
    
    
    return solution



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
    
   
        
    # Phase synch in GP and ST effected by delay from cortex
    for y in numpy.arange(2.5, 21., 2.5):

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


    return l


get()