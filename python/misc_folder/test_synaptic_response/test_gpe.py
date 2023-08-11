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


        
# STN-GPe
AMPA_1_Tau_decay = 12.   # (Hanson & Dieter Jaeger 2002)
AMPA_1_E_rev     = 0.    # n.d.; same as CTX to STN
NMDA_1_Tau_decay = 100.  # n.d.; estimated
NMDA_1_E_rev     = 0.    # n.d.; same as CTX to STN
NMDA_1_Vact      = -20.0
NMDA_1_Sact      =  16.0
            
#EXT-GPe
AMPA_2_Tau_decay = 5.0
AMPA_2_E_rev     = 0.0
        
# GPe-GPe
GABAA_2_Tau_decay  = 5.  # (Sims et al. 2008)
GABAA_2_E_rev     = -65.  # n.d same as for MSN (Rav-Acha 2005)       
beta_E_L = 0.181
beta_V_a = 0.181
beta_I_AMPA_1  = 0.4 # From GPe A
beta_I_GABAA_2 = 0.8 # From GPe A
tata_dop = 0.0 #0.8 normal
        
#MSN D2-GPe
GABAA_1_E_rev     = -65.  # (Rav-Acha et al. 2005)
GABAA_1_Tau_decay = 6.     # (Shen et al. 2008)    

a_1       =  2.5    # I-V relation # I-V relation
a_2       =  a_1 
b       = 70.   # I-F relation
C_m     = 40.  # t_m/R_in
Delta_T =  1.7                      
g_L     =   1.
E_L     = -55.1  # v_t    = -56.4                                                               #
I_e     =  -6.0
tau_w   = 20.  # I-V relation, spike frequency adaptation
V_peak  =  15.  # Cooper and standford
V_reset = -60.  # I-V relation
V_th    = -54.7
V_a     = E_L
          

    
# Params from Izhikevich regular spiking


params = { 'tau_w':tau_w, 'a_1' : a_1,'a_2' : a_2, 'b' : b, 'C_m' : C_m,
           'Delta_T' : Delta_T,  'g_L' : g_L, 'V_a' : V_a, 'V_peak' : V_peak, 
           'E_L' : E_L, 'V_th' : V_th, 'I_e' : I_e, 'V_reset':V_reset,
           'GABAA_1_E_rev':GABAA_1_E_rev, 'GABAA_1_Tau_decay':GABAA_1_Tau_decay,#}
#           'GABAA_2_E_rev':GABAA_2_E_rev, 'GABAA_2_Tau_decay':GABAA_2_Tau_decay,'beta_E_L':beta_E_L,
#           'beta_V_a':beta_V_a,'beta_I_AMPA_1':beta_I_AMPA_1,'beta_I_GABAA_2':beta_I_GABAA_2,
#           'tata_dop':tata_dop} 
            'AMPA_1_Tau_decay':AMPA_1_Tau_decay,'AMPA_1_E_rev':AMPA_1_E_rev,'NMDA_1_Tau_decay':NMDA_1_Tau_decay,'NMDA_1_E_rev':NMDA_1_E_rev,'NMDA_1_Vact':NMDA_1_Vact,'NMDA_1_Sact':NMDA_1_Sact}

#template = 'tsodyks_synapse'
'''
receptor_type = rec['my_aeif_cond_exp'] [ 'GABAA_2' ] 


nest.CopyModel('static_synapse', 'GA_GA_gaba', {'weight':1.3,'delay':1.})
'''
'''
nest.CopyModel('tsodyks_synapse', 'M2_GI_gaba', {'weight':2./0.24,'U':0.24, 'tau_fac':13.0, 
                                                   'tau_rec':77.0, 'tau_psc':6.,'delay':7.})

receptor_type = rec['my_aeif_cond_exp'] [ 'GABAA_1' ] 
'''

nest.CopyModel('static_synapse', 'ST_GA_ampa', {'weight':0.35,'delay':5.0})
receptor_type = rec['my_aeif_cond_exp'] [ 'AMPA_1' ] 

     

spk=numpy.linspace(0,3000, 30*2+1)
sp=nest.Create('spike_generator', params={'spike_times':spk})
nest.SetStatus(n, params)  
nest.Connect(mm,n)

nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='ST_GA_ampa')
#nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='GA_GA_gaba')
#nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='M2_GI_gaba')

nest.Simulate(2000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.plot(spk,numpy.ones(len(spk))*-55, '|')
pylab.ylim([-75,-50])
pylab.show()


