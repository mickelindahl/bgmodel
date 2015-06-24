'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def STN_ampa_gaba_input_magnitude():
    l=[
       [20, 188, 0.08], # 20 Hz 
#        [20, 210, 0.08], # 20 Hz 
       [25, 290, 0.119], #25 Hz
       [30, 430, 0.18],      #30 Hz
#        [35, 540, 0.215],     #35 Hz,
#        [40, 702, 0.28],     #40 Hz
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
    
    misc.dict_update(solution,d) 
    
    return solution



def get():
    
    
    l=[]
    solution=get_solution()

    d={}
    misc.dict_update(d, solution['mul'])
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    misc.dict_update(d, solution['equal']) 
    l[-1]+=pl(d, '=', **{'name':'control_sim'}) 
        
    # Connectivity GPe-STN
    v=STN_ampa_gaba_input_magnitude()    
    agg=[]
    for _,a,b in v:
        agg+=[[a,b] for _ in range(1)]

    STN_beta_I=[]
    for f in [1.]:
        STN_beta_I.append(reversed(list(numpy.linspace(f*1.8, f*2.5, len(v)))))
    STN_beta_I=zip(*STN_beta_I)
    STN_beta_I=reduce(lambda x,y:x+y, STN_beta_I)
    
    for i in range(len(agg)):
        agg[i].append(STN_beta_I[i])

# 
#     M2_rate=[[1.0, 1.1, 1.2] for _ in range(len(agg))]
#     M2_rate=reduce(lambda x,y:x+y, M2_rate)
#     agg=[[e[:] for e in agg] for _ in range(3)]
#     agg=zip(*agg)
#     agg=reduce(lambda x,y:x[:]+y[:], agg)
#  
#     for i in range(len(agg)):
#         agg[i].append(M2_rate[i])

#     ST_GA_ampa_w=[[0.75,1.] for _ in range(len(agg))]
#     ST_GA_ampa_w=reduce(lambda x,y:x+y, ST_GA_ampa_w)
#     agg=[[e[:] for e in agg] for _ in range(2)]
#     agg=zip(*agg)
#     agg=reduce(lambda x,y:x[:]+y[:], agg)
# 
#     for i in range(len(agg)):
#         agg[i].append(ST_GA_ampa_w[i])

    
    EA_rate=[[200.,400., 600.] for _ in range(len(agg))]
    EA_rate=reduce(lambda x,y:x+y, EA_rate)
    agg=[[e[:] for e in agg] for _ in range(3)]
    agg=zip(*agg)
    agg=reduce(lambda x,y:x+y, agg)

    for i in range(len(agg)):
        agg[i].append(EA_rate[i])
    
        
    for a0, a1, a2, a4 in agg:

        d={}
        misc.dict_update(d, solution['mul'])   
        
 
#         dd={'node':{'M1':{'rate':a3}}}
#         dd={'nest':{'ST_GA_ampa':{'weight':a3*0.25}}}
#         misc.dict_update(d,dd) 
        l+=[pl(d, '*', **{'name':''})]
        
        d={}
        misc.dict_update(d, solution['equal'])  
        
        dd={'node':{'CS':{'rate': float(a0)},
                   'EA':{'rate':a4}},
           'nest':{'GI_ST_gaba':{'weight':a1 },
                   'ST':{'beta_I_AMPA_1': f_beta_rm(a2),
                         'beta_I_NMDA_1': f_beta_rm(a2)}}}
        
        misc.dict_update(d,dd)
        
        s='CS_{0}_GPST_{1}_STbI_{2}_EAr_{3}'.format(a0,str(a1)[0:4],str(a2)[0:4], int(a4))
        print s
        l[-1]+=pl(d, '=', **{'name':s})   
               
    return l


get()