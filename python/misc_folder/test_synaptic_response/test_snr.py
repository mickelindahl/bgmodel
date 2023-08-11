from toolbox.default_params import Par
import nest
import pylab
import numpy
MODULE_PATH=  '/afs/nada.kth.se/home/w/u1yxbcfw/tools/NEST/dist/install-nest-2.2.2/lib/nest/ml_module'
nest.Install(MODULE_PATH) # Change ml_module to your module name

par = Par()
print par['conn']['GA_GA_gaba']['fan_in0']
print par['conn']['GI_GA_gaba']['fan_in0']
print par['conn']['GA_GI_gaba']['fan_in0']
print par['conn']['GI_GI_gaba']['fan_in0']

model= 'my_aeif_cond_exp'
n=nest.Create(model)
mm=nest.Create('multimeter')
sd=nest.Create("spike_recorder")
nest.SetStatus(mm, {'interval': 0.1, 'record_from': ['V_m']})
                  
rec={}    
rec[model] = nest.GetDefaults('my_aeif_cond_exp')['receptor_types'] 

tau_w= 20.  # I-V relation, spike frequency adaptation
a_1 =  3.      # I-V relation
a_2= 3.       # I-V relation
b = 200.   # I-F relation
C_m =  80.    # t_m/R_in
Delta_T =  1.8                      
g_L     =   3.
E_L     = -55.8    #

I_e     = -50.0 
V_peak  =  20.                                                               # 
V_reset = -65.    # I-V relation
V_th    = -55.2    # 
V_a     = E_L     # I-V relation
        
#STN-SNr
AMPA_1_Tau_decay =  12.   # n.d.; set as for STN to GPE
AMPA_1_E_rev     =   0.   # n.d. same as CTX to STN
    
# EXT-SNr
AMPA_2_Tau_decay = 5.0
AMPA_2_E_rev     = 0.
        
# MSN D1-SNr
GABAA_1_E_rev     = -80.     # (Connelly et al. 2010)
GABAA_1_Tau_decay = 5.2      # (Connelly et al. 2010)
 
# GPe-SNr
GABAA_2_E_rev     = -72.     # (Connelly et al. 2010)
GABAA_2_Tau_decay = 2.1
    
# Params from Izhikevich regular spiking

params = { 'tau_w':tau_w, 'a_1' : a_1,'a_2' : a_2, 'b' : b, 'C_m' : C_m,
           'Delta_T' : Delta_T,  'g_L' : g_L, 'V_a' : V_a, 'V_peak' : V_peak, 
           'E_L' : E_L, 'V_th' : V_th, 'I_e' : I_e, 'V_reset':V_reset,
#           'GABAA_2_E_rev':GABAA_2_E_rev, 'GABAA_2_Tau_decay':GABAA_2_Tau_decay}
            'GABAA_1_E_rev':GABAA_1_E_rev,'GABAA_1_Tau_decay':GABAA_1_Tau_decay}
#            'AMPA_1_Tau_decay':AMPA_1_Tau_decay,'AMPA_1_E_rev':AMPA_1_E_rev}            



weight   = 76./0.196  #0.152*76., (Connelly et al. 2010)
delay    = 3.  
U        = 0.196 # GABAA plastic,                   
tau_fac  = 0.0
tau_rec  = 969.0
tau_psc  = 2.1    # (Connelly et al. 2010),
template = 'tsodyks_synapse'
'''   
receptor_type = rec['my_aeif_cond_exp'] [ 'GABAA_2' ] 

nest.CopyModel('tsodyks_synapse', 'GPE_SNR_gaba', {'weight':weight,'delay':delay, 'U':U, 'tau_fac':tau_fac, 
                                                   'tau_rec':tau_rec, 'tau_psc':tau_psc})
'''

nest.CopyModel('tsodyks_synapse', 'M1_SN_gaba', {'weight':2./0.0192,'U':0.0192, 'tau_fac':623., 
                                                   'tau_rec':559., 'tau_psc':5.2,'delay':7.3})

receptor_type = rec['my_aeif_cond_exp'] [ 'GABAA_1' ] 

'''
nest.CopyModel('tsodyks_synapse', 'ST_SN_ampa', {'weight':0.91*3.8/0.35,'U':0.35, 'tau_fac':0.0, 
                                                   'tau_rec':800.0, 'tau_psc':12.,'delay':4.6})
receptor_type = rec['my_aeif_cond_exp'] [ 'AMPA_1' ] 
'''
     

spk=numpy.linspace(0,3000, 10*2+1)
sp=nest.Create('spike_generator', params={'spike_times':spk})
nest.SetStatus(n, params)  
nest.Connect(mm,n)

nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='M1_SN_gaba')
#nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='GPE_SNR_gaba')
#nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='ST_SN_ampa')

nest.Simulate(3000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.plot(spk,numpy.ones(len(spk))*-50, '|')
pylab.show()


