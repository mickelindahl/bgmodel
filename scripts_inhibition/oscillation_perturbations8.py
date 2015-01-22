'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
pp=pprint.pprint


def get():
    
    solution={
              #Decreasing from 0.25 leads to ...
              #Increasing from 0.25 leads to ...
              'M1_M1_gaba':{'weight':0.25},
              'M1_M2_gaba':{'weight':0.25},
              'M2_M1_gaba':{'weight':0.25},
              'M2_M2_gaba':{'weight':0.25},
            
              # GA firing rate needs to be maintained, around 12 Hz for sw in 
              # dopamine depleted rats). When decreasing/increasing
              # ST-GA one need to either compensate by changing inhibition from 
              # GPe and/or EA external input (assume that synapses from TA and TI
              # are of equal strength. 
              
              # This script is first run where ST-GA GP-GA andEA indivudually are
              # perturbed to determine each relative influence to TA firing rate. 
              # From this data then combinations fo changes that should results in
              # 12 Hz TA firing rate are created.  
              'ST_GA_ampa':{'weight':0.25},
              'GA_GA_gaba':{'weight':0.25},
              'GI_GA_gaba':{'weight':0.25},
              'EA':{'rate':330.0},
              
              # C1 and C2 have been set such that MSN D1 and MSN D2 fires at 
              # low firing rates (0.1-0,2 Hz). 
              'C1':{'rate':560.0},
              'C2':{'rate':700.},
              
              # Set such that GPe TI fires in accordance with slow wave sleep
              # in dopamine depleted rats.
              'EI':{'rate':1400.0}, # Increase with 340, to get close to 24 Hz sw dopamine depleted rats TI
              
              # Decreasing from 0.8 leads to ...
              # Increasing from 0.8 leads to ... 
              'GA_M1_gaba':{'weight':0.8}, 
              'GA_M2_gaba':{'weight':0.8},
              
              # Decreasing from 2 leads to ...
              # Increasing from 2 leads to ... 
              'GA_FS_gaba':{'weight':2.},

              # Just assumed to be 12 ms
              'M1_low':{'GABAA_3_Tau_decay':12.},  
              'M2_low':{'GABAA_3_Tau_decay':12.},
              'FS_low':{'GABAA_2_Tau_decay':12.},     
              }
    
    l=[]
      
    s_mul=['M1_M1_gaba','M1_M2_gaba','M2_M1_gaba','M2_M2_gaba',
           'ST_GA_ampa', 'GA_GA_gaba', 'GI_GA_gaba']
    s_equal=['C1', 'C2', 'EI', 'EA', 'GA_M1_gaba','GA_M2_gaba',
             'GA_FS_gaba', 'M1_low', 'M2_low', 'FS_low']

    # Decrease/increase MSN to MSN 0.25-2.75 step=0.25. 
    for y in numpy.arange(0.5, 2.75, 0.25): 
        d={}
        for s in s_mul: d[s]=solution[s] 
        d['M1_M1_gaba']['weight']*=y
        d['M1_M2_gaba']['weight']*=y
        d['M2_M1_gaba']['weight']*=y
        d['M2_M2_gaba']['weight']*=y
        l+=[pl({'nest':d}, '*', **{'name':'mod_MS_MS_'+str(y)})]
        
        d={}
        for s in s_equal: d[s]=solution[s] 
        l[-1]+=pl({'node':d}, '=', **{'name':''}) 
    
    # Decrease/increase GPe TA to MSN 0.25-2.75 step=0.25. 
    for y in numpy.arange(0.5, 2.75, 0.25):  
        d={}
        for s in s_mul: d[s]=solution[s] 

        l+=[pl({'nest':d}, '*', **{'name':''})]
        
        d={}
        for s in s_equal: d[s]=solution[s] 
        d['GA_M1_gaba']['weight']*=y
        d['GA_M2_gaba']['weight']*=y
        l[-1]+=pl({'node':d}, '=', **{'name':'mod_GP_MS_'+str(y)}) 
        
    # Decrease/increase GPe TA to FSN 0.25-2.75 step=0.25. 
    for y in numpy.arange(0.5, 2.75, 0.25):  
        d={}
        for s in s_mul: d[s]=solution[s] 
        l+=[pl({'nest':d}, '*', **{'name':''})]
          
        d={}
        for s in s_equal: d[s]=solution[s] 
        d['GA_FS_gaba']['weight']*=y
        l[-1]+=pl({'node':d}, '=', **{'name':'mod_GP_FS_'+str(y)}) 
        
    # Decrease/increase EI and EA on a grid 0.25-1.75 step=0.5
    v=numpy.arange(0.25, 2.,0.5)
    mod=[[x,y] for x in v for y in v]
    for m in mod:
        x,y=m
        
        d={}
        for s in s_mul: d[s]=solution[s] 
        l+=[pl({'nest':d}, '*', **{'name':''})]
          
        d={}
        for s in s_equal: d[s]=solution[s] 
        d['EI']['rate']*=x
        d['EA']['rate']*=y
        l[-1]+=pl({'node':d}, '=', **{'name':'mod_EI_EA_'+str(y)}) 
              
    # To get reference points for how firing rate is changed locally by 
    # either changing ST-GP, GP-TA and EA. Will use this to create 
    # combinations of parameters for ST-GA,GP-GA and EA that maintains 
    # the firing rate of TA neurons around 12 Hz for sw in dopamine 
    # depleted rats. Perturbation 0.25-2.75 step=0.25.
    for y in numpy.arange(0.25, 3, 0.25): 
        d={}
        for s in s_mul: d[s]=solution[s] 
        d['ST_GA_ampa']['weight']*=y
        l+=[pl({'nest':d}, '*', **{'name':'mod_ST_GA_'+str(y)})]
          
        d={}
        for s in s_equal: d[s]=solution[s] 
        l[-1]+=pl({'node':d}, '=', **{'name':''}) 
  
    for y in numpy.arange(0.25, 3, 0.25): 
        d={}
        for s in s_mul: d[s]=solution[s] 
        d['GA_GA_gaba']['weight']*=y
        d['GI_GA_gaba']['weight']*=y
        l+=[pl({'nest':d}, '*', **{'name':'mod_GP_GA_'+str(y)})]
          
        d={}
        for s in s_equal: d[s]=solution[s] 

        l[-1]+=pl({'node':d}, '=', **{'name':''})  

    for y in numpy.arange(0.25, 3, 0.25): 
        d={}
        for s in s_mul: d[s]=solution[s] 
        l+=[pl({'nest':d}, '*', **{'name':''})]
          
        d={}
        for s in s_equal: d[s]=solution[s]
        d['EA']['rate']*=y 
        l[-1]+=pl({'node':d}, '=', **{'name':'mod_EA_'+str(y)})    
        
    return l

get()