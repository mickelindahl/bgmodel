from toolbox.default_params import Par
import nest
import pylab
import numpy
MODULE_PATH=  '/afs/nada.kth.se/home/w/u1yxbcfw/tools/NEST/dist/install-nest-2.2.2/lib/nest/ml_module'
nest.Install(MODULE_PATH) # Change ml_module to your module name

model= 'my_aeif_cond_exp'
n=nest.Create(model)
mm=nest.Create('multimeter')
sd=nest.Create("spike_recorder")
nest.SetStatus(mm, {'interval': 0.1, 'record_from': ['V_m']})
                  
rec={}    
rec[model] = nest.GetDefaults('my_aeif_cond_exp')['receptor_types'] 


par=Par()



tau_w    =333.0 # I-V relation, spike frequency adaptation
a_1      =  0.3    # I-V relation
a_2      =  0.0      # I-V relation
b        =  0.05    #0.1 #0.1#200./5.                                                     
C_m      = 60.0    # t_m/R_in
Delta_T  = 16.2                      
g_L      = 10.0
E_L      =-80.2                                                               
I_e      = -5.0
V_peak   = 15.0                                                                
V_reset  =-70.0    # I-V relation
V_a      =-70.0 # I-V relation
V_th     =-64.0                                                               
        
V_reset_slope1     = -10. # Slope u<0 
V_reset_slope2     = .0 #  Slope u>=0
V_reset_max_slope1 = -60. # Max v restet u<0  
V_reset_max_slope2 = V_reset # Max v restet u>=0  

#CTX-STN      
AMPA_1_Tau_decay = 4.0  # (Baufreton et al. 2005)
AMPA_1_E_rev     = 0.   # (Baufreton et al. 2009)
        
NMDA_1_Tau_decay = 160.   # n.d. estimated 1:2 AMPA:NMDA
NMDA_1_E_rev     =   0.   # n.d.; set as  E_ampa
NMDA_1_Vact      = -20.0
NMDA_1_Sact      =  16.0
beta_I_AMPA_1  = 0.4 # From Cortex
beta_I_NMDA_1  = 0.4 # From Cortex

#GPe-STN        
GABAA_1_Tau_decay =   8.   # (Baufreton et al. 2009)
GABAA_1_E_rev     = -84.0  # (Baufreton et al. 2009)
beta_I_GABAA_1 = 0.4 # From GPe I 
        
tata_dop = 0.8

# Params from Izhikevich regular spiking

params = { 'tau_w':tau_w, 'a_1' : a_1,'a_2' : a_2, 'b' : b, 'C_m' : C_m,
           'Delta_T' : Delta_T,  'g_L' : g_L, 'V_a' : V_a, 'V_peak' : V_peak, 
           'E_L' : E_L, 'V_th' : V_th, 'I_e' : I_e, 'V_reset':V_reset,
           'V_reset_slope1':V_reset_slope1,'V_reset_slope2':V_reset_slope2,'V_reset_max_slope1':V_reset_max_slope1,'V_reset_max_slope2':V_reset_max_slope2,
          'GABAA_1_E_rev':GABAA_1_E_rev, 'GABAA_1_Tau_decay':GABAA_1_Tau_decay,'beta_I_GABAA_1':beta_I_GABAA_1}
#            'beta_I_AMPA_1':beta_I_AMPA_1,'beta_I_NMDA_1':beta_I_NMDA_1,'AMPA_1_Tau_decay':AMPA_1_Tau_decay,'AMPA_1_E_rev':AMPA_1_E_rev,'NMDA_1_Tau_decay':NMDA_1_Tau_decay,'NMDA_1_E_rev':NMDA_1_E_rev,
#            'NMDA_1_Vact':NMDA_1_Vact,'NMDA_1_Sact':NMDA_1_Sact}

#template = 'tsodyks_synapse'
'''
receptor_type1 = rec['my_aeif_cond_exp'] [ 'AMPA_1' ] 
receptor_type2 = rec['my_aeif_cond_exp'] [ 'NMDA_1' ]

nest.CopyModel('static_synapse', 'CS_ST_ampa', {'weight':0.25,'delay':2.5})
nest.CopyModel('static_synapse', 'CS_ST_nmda', {'weight':0.00625,'delay':2.5})
'''
    
        
nest.CopyModel('static_synapse', 'GI_ST_gaba', {'weight':0.08,'delay':5.})

receptor_type = rec['my_aeif_cond_exp'] [ 'GABAA_1' ] 

     

spk=numpy.linspace(0,3000, 10*2+1)
sp=nest.Create('spike_generator', params={'spike_times':spk})
nest.SetStatus(n, params)  
nest.Connect(mm,n)


#nest.Connect(sp, n, params={'receptor_type':receptor_type2}, model='CS_ST_nmda')
#nest.Connect(sp, n, params={'receptor_type':receptor_type1}, model='CS_ST_ampa')
nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='GI_ST_gaba')

nest.Simulate(2000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.plot(spk,numpy.ones(len(spk))*-65, '|')
pylab.show()


