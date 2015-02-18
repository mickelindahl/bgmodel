'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint


def get_solution():
    solution={}
    #Decreasing from 0.25 leads to ...
    #Increasing from 0.25 leads to ...
    d={'nest':{'M1_M1_gaba':{'weight':0.25},
               'M1_M2_gaba':{'weight':0.25},
               'M2_M1_gaba':{'weight':0.25},
               'M2_M2_gaba':{'weight':0.25}}}
    misc.dict_update(solution,d)    
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
    misc.dict_update(solution,d)    
    d={'node':{'EA':{'rate':330.0}}}
    misc.dict_update(solution,d)              
    
    # C1 and C2 have been set such that MSN D1 and MSN D2 fires at 
    # low firing rates (0.1-0,2 Hz). 
    d={'node':{'C1':{'rate':560.0},
               'C2':{'rate':700.}}}
    misc.dict_update(solution,d)              
           
    # Set such that GPe TI fires in accordance with slow wave sleep
    # in dopamine depleted rats.
    d={'node':{'EI':{'rate':1400.0}}} # Increase with 340, to get close to 24 Hz sw dopamine depleted rats TI
    misc.dict_update(solution,d)           
    
    # Decreasing from 0.8 leads to ...
    # Increasing from 0.8 leads to ... 
    d={'nest':{'GA_M1_gaba':{'weight':0.8}, 
               'GA_M2_gaba':{'weight':0.8}}}
    misc.dict_update(solution,d)            
     
    # Decreasing from 2 leads to ...
    # Increasing from 2 leads to ... 
    d={'nest':{'GA_FS_gaba':{'weight':2.}}}
    misc.dict_update(solution,d)           
    
    # Just assumed to be 12 ms    
    d={'nest':{'M1_low':{'GABAA_3_Tau_decay':12.},  
               'M2_low':{'GABAA_3_Tau_decay':12.},
               'FS_low':{'GABAA_2_Tau_decay':12.},     
               }}
    misc.dict_update(solution,d)           
    
    s_mul=[
           ['nest','M1_M1_gaba','weight'],
           ['nest','M1_M2_gaba','weight'],
           ['nest','M2_M1_gaba','weight'],
           ['nest','M2_M2_gaba','weight'],
           ['nest','ST_GA_ampa','weight'],
           ['nest','GA_GA_gaba','weight'], 
           ['nest','GI_GA_gaba','weight']
           ]
    s_equal=[
             ['node','C1', 'rate'],
             ['node','C2', 'rate'], 
             ['node','EI', 'rate'],
             ['node','EA', 'rate'], 
             ['nest','GA_M1_gaba','weight'],
             ['nest','GA_M2_gaba','weight'],
             ['nest','GA_FS_gaba','weight'], 
             ['nest','M1_low', 'GABAA_3_Tau_decay'],
             ['nest','M2_low', 'GABAA_3_Tau_decay'],
             ['nest','FS_low', 'GABAA_2_Tau_decay']
             ]
    
    return solution, s_mul, s_equal

def get_solution_slow_GP_striatum():
    
    solution, s_mul, s_equal=get_solution()
    # Decreasing from 0.8 leads to ...
    # Increasing from 0.8 leads to ... 
    d={'nest':{'GA_M1_gaba':{'weight':0.8/5}, 
               'GA_M2_gaba':{'weight':0.8/5}}}
    misc.dict_update(solution,d)            
     
    # Decreasing from 2 leads to ...
    # Increasing from 2 leads to ... 
    d={'nest':{'GA_FS_gaba':{'weight':2./5}}}
    misc.dict_update(solution,d)           
    
    # Just assumed to be 12 ms    
    d={'nest':{'M1_low':{'GABAA_3_Tau_decay':12.*5},  
               'M2_low':{'GABAA_3_Tau_decay':12.*5},
               'FS_low':{'GABAA_2_Tau_decay':12.*5},     
               }}
    misc.dict_update(solution,d)           
    
    return solution, s_mul, s_equal

def update(solution, d, keys):
    v=misc.dict_recursive_get(solution, keys)
    misc.dict_recursive_add(d, keys, v) 


def get():
        
    l=[]
    solution, s_mul, s_equal=get_solution()
    
    # Decrease/increase MSN to MSN 0.25-2.75 step=0.25 from base values. 
    for y in numpy.arange(0.5, 2.75, 0.25): 
        d={}
        for keys in s_mul:update(solution, d, keys) 
        
        d['nest']['M1_M1_gaba']['weight']*=y
        d['nest']['M1_M2_gaba']['weight']*=y
        d['nest']['M2_M1_gaba']['weight']*=y
        d['nest']['M2_M2_gaba']['weight']*=y
        l+=[pl(d, '*', **{'name':'mod_MS_MS_'+str(y)})]
        
        d={}
        for keys in s_equal: update(solution, d, keys) 
        l[-1]+=pl(d, '=', **{'name':''}) 
    
    # Decrease/increase GPe TA to MSN 0.25-2.75 step=0.25. 
    for y in numpy.arange(0.5, 2.75, 0.25):  
        d={}
        for keys in s_mul: update(solution, d, keys)
        
        l+=[pl(d, '*', **{'name':''})]
        
        d={}
        for keys in s_equal: update(solution, d, keys)
        
        d['nest']['GA_M1_gaba']['weight']*=y
        d['nest']['GA_M2_gaba']['weight']*=y
        l[-1]+=pl(d, '=', **{'name':'mod_GP_MS_'+str(y)}) 
        
    # Decrease/increase GPe TA to FSN 0.25-2.75 step=0.25. 
    for y in numpy.arange(0.5, 2.75, 0.25):  
        d={}
        for keys in s_mul: update(solution, d, keys) 

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        d['nest']['GA_FS_gaba']['weight']*=y
        l[-1]+=pl(d, '=', **{'name':'mod_GP_FS_'+str(y)}) 
        
    # Decrease/increase EI and EA on a grid 0.25-1.75 step=0.5
    v=numpy.arange(0.25, 2.,0.5)
    mod=[[x,y] for x in v for y in v]
    for m in mod:
        x,y=m
        
        d={}
        for keys in s_mul: update(solution, d, keys) 

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys)
        
        d['node']['EI']['rate']*=x
        d['node']['EA']['rate']*=y
        l[-1]+=pl(d, '=', **{'name':'mod_EI_EA_'+str(x)+'_'+str(y)}) 
              
    # To get reference points for how firing rate is changed locally by 
    # either changing ST-GP, GP-TA and EA. Will use this to create 
    # combinations of parameters for ST-GA,GP-GA and EA that maintains 
    # the firing rate of TA neurons around 12 Hz for sw in dopamine 
    # depleted rats. Perturbation 0.25-2.75 step=0.25.
    for y in numpy.arange(0.25, 3, 0.25): 
        d={}
        for keys in s_mul: update(solution, d, keys) 

        d['nest']['ST_GA_ampa']['weight']*=y
        l+=[pl(d, '*', **{'name':'mod_ST_GA_'+str(y)})]
          
        d={}
        for keys in s_equal: update(solution, d, keys)
        l[-1]+=pl(d, '=', **{'name':''}) 
  
    for y in numpy.arange(0.25, 3, 0.25): 
        d={}
        for keys in s_mul: update(solution, d, keys)
        d['nest']['GA_GA_gaba']['weight']*=y
        d['nest']['GI_GA_gaba']['weight']*=y
        l+=[pl(d, '*', **{'name':'mod_GP_GA_'+str(y)})]
          
        d={}
        for keys in s_equal: update(solution, d, keys)

        l[-1]+=pl(d, '=', **{'name':''})  

    return l

get()