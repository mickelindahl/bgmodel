'''
Created on Aug 12, 2013

@author: lindahlm
'''
from copy import deepcopy
from toolbox import misc
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_slow_wave as Builder
from toolbox.parallel_excecution import loop

import simulate_slow_wave
import pprint
pp=pprint.pprint




def perturbations():
    sim_time=20000.0
    size=20000.0
    threads=4
    l=[]
    
  
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.25},
                    'GA_GA_gaba':{'weight':0.25},
                    'GI_GA_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.25-EA100'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EA':{'rate':100.0}}},
           '=',
            **{'name':''})   


    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.25},
                    'GA_GA_gaba':{'weight':0.25},
                    'GI_GA_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.25-EA200'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EA':{'rate':200.0}}},
           '=',
            **{'name':''})   



    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.3},
                    'GA_GA_gaba':{'weight':0.3},
                    'GI_GA_gaba':{'weight':0.5}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.3'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0}}},
           '=',
            **{'name':''}) 
     
    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.35},
                    'GA_GA_gaba':{'weight':0.35},
                    'GI_GA_gaba':{'weight':0.35}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.35'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0}}},
           '=',
            **{'name':''})         

    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.25},
                    'GA_GA_gaba':{'weight':0.25},
                    'GI_GA_gaba':{'weight':0.25}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.25-EAEI'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EA':{'rate':200.0},
                      'EI':{'rate':1060.0},
                      'EA':{'rate':330.0},}},
           '=',
            **{'name':''})   

    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.3},
                    'GA_GA_gaba':{'weight':0.3},
                    'GI_GA_gaba':{'weight':0.3}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.3-EAEI'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EI':{'rate':1060.0},
                      'EA':{'rate':330.0},}},
           '=',
            **{'name':''})

    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.35},
                    'GA_GA_gaba':{'weight':0.35},
                    'GI_GA_gaba':{'weight':0.35}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.35-EAEI'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EI':{'rate':1060.0},
                      'EA':{'rate':330.0},}},
           '=',
            **{'name':''})


    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.35},
                    'GA_GA_gaba':{'weight':0.35},
                    'GI_GA_gaba':{'weight':0.35}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.35-EAEI'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EA':{'rate':200.0},
                      'EI':{'rate':1060.0},
                      'EA':{'rate':330.0},}},
           '=',
            **{'name':''})


    l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                    'GA_M2_gaba':{'weight':0.25},
                    'M1_M1_gaba':{'weight':0.25},
                    'M1_M2_gaba':{'weight':0.25},
                    'M2_M1_gaba':{'weight':0.25},
                    'M2_M2_gaba':{'weight':0.25},
                    'ST_GA_ampa':{'weight':0.35},
                    'GA_GA_gaba':{'weight':0.35},
                    'GI_GA_gaba':{'weight':0.35}
                    }},
           '*',
            **{'name':'weightGPe-MSN*0.25and*0.25-minGA-ST0.35-EAEI'})]  
    l[-1]+=pl({
              'node':{'C1':{'rate':560.0},
                      'C2':{'rate':700.0},
                      'EA':{'rate':200.0},
                      'EI':{'rate':1060.0},
                      'EA':{'rate':330.0},}},
           '=',
            **{'name':''})

    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                          'threads':threads},
                  'netw':{'size':size}},
                  '=')

    ll=[]
    for amp in [0.85, 0.9, 0.95]: 
        d={'type':'oscillation', 
           'params':{'p_amplitude_mod':amp,
                     'freq': 1.}} 
        for i, _l in enumerate(l):
            _l=deepcopy(_l)
            dd={}
            for key in ['C1', 'C2', 'CF', 'CS']: 
                dd=misc.dict_update(dd, {'netw': {'input': {key:d} } })     
                      
            _l+=pl(dd,'=',**{'name':'amp_'+str(amp)})
        
            if i in [4, 5,6]:
                
                ll.append(_l)
        

    return ll


p_list=perturbations()
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

for j in range(0,3):
    for i, p in enumerate(p_list):
        from_disk=j

        fun=simulate_slow_wave.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
        fun(*[Builder, from_disk, p, script_name])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, script_name])


loop(args_list, path, 9)
        