'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
pp=pprint.pprint


def get():
    
    l=[]
      
    # Need to be at 0.4 otherwise there is to small coherence difference between control
    # and dopamine depletions for slow wave, and control becomes coherent.

    for y in [0.1, 0.2, 0.4, 0.6, 0.8]:
        l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25},
                        
#                         'GI_ST_gaba':{'weight':w},
#                         'ST_GI_ampa':{'weight':w}
                        
                        }},
               '*',
                **{'name':'op6-modGP_MS_'+str(y)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':''}) 
        
        l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':2.*y}, 
                           'GA_M2_gaba':{'weight':2.*y},
                           'GA_FS_gaba':{'weight':2.},
                           'M1_low':{'GABAA_3_Tau_decay':12.},  
                           'M2_low':{'GABAA_3_Tau_decay':12.},
                           'FS_low':{'GABAA_2_Tau_decay':12.},
                   
                   }},
          '=',
           **{'name':''})    

    #TA, TI input tuning to slow wave
    v=numpy.linspace(0.75,1.25,5)
    mod=[[x,y] for x in v for y in v]
    for m in mod:
        x,y=m
        l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25},
                        
#                         'GI_ST_gaba':{'weight':w},
#                         'ST_GI_ampa':{'weight':w}
                        
                        }},
               '*',
                **{'name':''})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0*x},
                          'EA':{'rate':330.0*y}}},
               '=',
                **{'name':'EI_'+str(x)+'EA_'+str(y)}) 
        
        l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':2.*0.4}, 
                           'GA_M2_gaba':{'weight':2.*0.4},
                           'GA_FS_gaba':{'weight':2.},
                           'M1_low':{'GABAA_3_Tau_decay':12.},  
                           'M2_low':{'GABAA_3_Tau_decay':12.},
                           'FS_low':{'GABAA_2_Tau_decay':12.},
                   
                   }},
          '=',
           **{'name':''})    
        
    #What happens when strength vetween STN at GPA TA is modified?
    for y in [0.1, 0.25, 0.5, 0.75, 1.]:
        l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        
                        'ST_GA_ampa':{'weight':y},
                        'GA_GA_gaba':{'weight':y},
                        'GI_GA_gaba':{'weight':y},
                        
#                         'GI_ST_gaba':{'weight':w},
#                         'ST_GI_ampa':{'weight':w}
                        
                        }},
               '*',
                **{'name':'op6-modGP_ST_'+str(y)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':''}) 
        
        l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':2.*.4}, 
                           'GA_M2_gaba':{'weight':2.*.4},
                           'GA_FS_gaba':{'weight':2.},
                           'M1_low':{'GABAA_3_Tau_decay':12.},  
                           'M2_low':{'GABAA_3_Tau_decay':12.},
                           'FS_low':{'GABAA_2_Tau_decay':12.},
                   
                   }},
          '=',
           **{'name':''})   


    #What happens when strength vetween MSN-MSN strehngth is modified?
    for y in [0.1, 0.25, 0.5, 0.75, 1.]:
        l+=[pl({'nest':{'M1_M1_gaba':{'weight':y},
                        'M1_M2_gaba':{'weight':y},
                        'M2_M1_gaba':{'weight':y},
                        'M2_M2_gaba':{'weight':y},
                        
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25},
                        
#                         'GI_ST_gaba':{'weight':w},
#                         'ST_GI_ampa':{'weight':w}
                        
                        }},
               '*',
                **{'name':'op6-modGP_MS_'+str(y)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':''}) 
        
        l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':2.*.4}, 
                           'GA_M2_gaba':{'weight':2.*.4},
                           'GA_FS_gaba':{'weight':2.},
                           'M1_low':{'GABAA_3_Tau_decay':12.},  
                           'M2_low':{'GABAA_3_Tau_decay':12.},
                           'FS_low':{'GABAA_2_Tau_decay':12.},
                   
                   }},
          '=',
           **{'name':''}) 
        
    return l

get()