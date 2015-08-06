'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint


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
    
    d={'node':{'EA':{'rate':330.0}}}
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
    
    # Decreasing from 0.8 leads to ...
    # Increasing from 0.8 leads to ... 
    d={'nest':{'GA_M1_gaba':{'weight':0.8}, 
               'GA_M2_gaba':{'weight':0.8}}}
    misc.dict_update(solution,{'equal':d})            
     
    # Decreasing from 2 leads to ...
    # Increasing from 2 leads to ... 
    d={'nest':{'GA_FS_gaba':{'weight':2.}}}
    misc.dict_update(solution,{'equal':d})           
    
    # Just assumed to be 12 ms    
    d={'nest':{'M1_low':{'GABAA_3_Tau_decay':12.},  
               'M2_low':{'GABAA_3_Tau_decay':12.},
               'FS_low':{'GABAA_2_Tau_decay':12.},     
               }}
    misc.dict_update(solution,{'equal':d})           

    
    return solution

def get_solution_slow():
    
    solution={'mul':{},
              'equal':{}}
    # Decreasing from 0.8 leads to ...
    # Increasing from 0.8 leads to ... 
    d={'nest':{'GA_M1_gaba':{'weight':0.8/5}, 
               'GA_M2_gaba':{'weight':0.8/5}}}
    misc.dict_update(solution,{'equal':d})            
     
    # Decreasing from 2 leads to ...
    # Increasing from 2 leads to ... 
    d={'nest':{'GA_FS_gaba':{'weight':2./5}}}
    misc.dict_update(solution,{'equal':d})           
    
    # Just assumed to be 12 ms    
    d={'nest':{'M1_low':{'GABAA_3_Tau_decay':12.*5},  
               'M2_low':{'GABAA_3_Tau_decay':12.*5},
               'FS_low':{'GABAA_2_Tau_decay':12.*5},     
               }}
    
    misc.dict_update(solution,{'equal':d})           
        

    return solution

def get_solution_ctx_dop_gpe_back():
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))


    solution={'mul':{},
              'equal':{}}
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
    
    # Decrease GP_TA rate by 0.7
#     misc.dict_update(solution,{'mul': {'node': {'EA':{'rate':0.7}}}})
    
    return solution


def get_dopMSGI_rates_beta():

    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

    factor_CSEI_rate=1.25
#     factor_CF_rate=1.

    solution={'mul':{},
              'equal':{}}
    
    d={'mul':{'node':{'CS':{'rate': factor_CSEI_rate}}}}
    misc.dict_update(solution,d)
    
#     d={'equal':{'node':{'EI':{'rate': 1400.*factor_CSEI_rate}}}}
#     misc.dict_update(solution,d)
    
#     d={'mul':{'node':{'CF':{'rate': factor_CF_rate}}}} 
#     misc.dict_update(solution,d) 

    d={'conn': {'GA_GA_gaba':{'fan_in0': 20}, 
                 'GI_GA_gaba':{'fan_in0': 10 }}}

    misc.dict_update(solution,{'equal':d})
    
    d={'equal':{'node':{'EI':{'rate':1400*factor_CSEI_rate}}}}
    misc.dict_update(solution,d)    


    d={'equal':{'node':{'EA':{'rate': 0.0}}}} 
    misc.dict_update(solution,d) 

    d={'equal':{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}}}
    misc.dict_update(solution,d)
    
    return solution

def get_dopMSGI_rates_beta2():

    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

    factor_CSEI_rate=1.05
#     factor_CF_rate=1.

    solution={'mul':{},
              'equal':{}}
    
    d={'mul':{'node':{'CS':{'rate': factor_CSEI_rate}}}}
    misc.dict_update(solution,d)
    
    d={'nest':{'ST_GA_ampa':{'weight':0.75*0.25}}}
    misc.dict_update(solution,{'mul':d})  

    d={'conn': {'GA_GA_gaba':{'fan_in0': 5}, 
                'GI_GA_gaba':{'fan_in0': 25 }}}

    misc.dict_update(solution,{'equal':d})
    
    d={'equal':{'node':{'EI':{'rate':1400*factor_CSEI_rate}}}}
    misc.dict_update(solution,d)    


    d={'equal':{'node':{'EA':{'rate': 600.0}}}} 
    misc.dict_update(solution,d) 

    d={'equal':{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(2)}}}}
    misc.dict_update(solution,d)
    
    return solution

def get_solution_slow_GP_striatum():
    args=[get_solution, get_solution_slow]
    return merge(*args)


def get_solution_slow_GP_striatum_2():
    args=[get_solution, get_solution_slow, get_solution_ctx_dop_gpe_back]
    return merge(*args)
 
def get_solution_final_beta():
    args=[get_solution, get_solution_slow, get_solution_ctx_dop_gpe_back,
          get_dopMSGI_rates_beta]
    return merge(*args) 

def get_solution_final_beta2():
    args=[get_solution, get_solution_slow, get_solution_ctx_dop_gpe_back,
          get_dopMSGI_rates_beta2]
    return merge(*args) 
        
def get_solution_2():
    args=[get_solution, get_solution_ctx_dop_gpe_back]
    return merge(*args)
  

def merge(*args):
    solution={}
#     s_mul=[]
#     s_equal=[]
    for f in args:
        s=f()
        misc.dict_update(solution,s) 
    
    return solution


def update(solution, d, keys):
    v=misc.dict_recursive_get(solution, keys)
    misc.dict_recursive_add(d, keys, v) 


def get():
        
    l=[]
    solution=get_solution()
    
    # Decrease/increase MSN to MSN 0.25-2.75 step=0.25 from base values. 
    for y in numpy.arange(0.5, 2.75, 0.25): 
        d={}
        misc.dict_update(d, solution['mul'])  
        
        d['nest']['M1_M1_gaba']['weight']*=y
        d['nest']['M1_M2_gaba']['weight']*=y
        d['nest']['M2_M1_gaba']['weight']*=y
        d['nest']['M2_M2_gaba']['weight']*=y
        l+=[pl(d, '*', **{'name':'mod_MS_MS_'+str(y)})]
        
        d={}
        misc.dict_update(d, solution['equal']) 
        l[-1]+=pl(d, '=', **{'name':''}) 
    
    # Decrease/increase GPe TA to MSN 0.25-2.75 step=0.25. 
    for y in numpy.arange(0.5, 2.75, 0.25):  
        d={}
        misc.dict_update(d, solution['mul'])  
        
        l+=[pl(d, '*', **{'name':''})]
        
        d={}
        misc.dict_update(d, solution['equal']) 
        
        d['nest']['GA_M1_gaba']['weight']*=y
        d['nest']['GA_M2_gaba']['weight']*=y
        l[-1]+=pl(d, '=', **{'name':'mod_GP_MS_'+str(y)}) 
        
    # Decrease/increase GPe TA to FSN 0.25-2.75 step=0.25. 
    for y in numpy.arange(0.5, 2.75, 0.25):  
        d={}
        misc.dict_update(d, solution['mul'])   

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        d['nest']['GA_FS_gaba']['weight']*=y
        l[-1]+=pl(d, '=', **{'name':'mod_GP_FS_'+str(y)}) 
        
    # Decrease/increase EI and EA on a grid 0.25-1.75 step=0.5
    v=numpy.arange(0.25, 2.,0.5)
    mod=[[x,y] for x in v for y in v]
    for m in mod:
        x,y=m
        
        d={}
        misc.dict_update(d, solution['mul'])  

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal'])
        
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
        misc.dict_update(d, solution['mul'])   

        d['nest']['ST_GA_ampa']['weight']*=y
        l+=[pl(d, '*', **{'name':'mod_ST_GA_'+str(y)})]
          
        d={}
        misc.dict_update(d, solution['equal'])
        l[-1]+=pl(d, '=', **{'name':''}) 
  
    for y in numpy.arange(0.25, 3, 0.25): 
        d={}
        misc.dict_update(d, solution['mul'])  
        d['nest']['GA_GA_gaba']['weight']*=y
        d['nest']['GI_GA_gaba']['weight']*=y
        l+=[pl(d, '*', **{'name':'mod_GP_GA_'+str(y)})]
          
        d={}
        misc.dict_update(d, solution['equal'])

        l[-1]+=pl(d, '=', **{'name':''})  

    return l
pp(get_solution_final_beta2())
get()