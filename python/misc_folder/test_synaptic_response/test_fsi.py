import nest
import pylab
import numpy
MODULE_PATH=  '/afs/nada.kth.se/home/w/u1yxbcfw/tools/NEST/dist/install-nest-2.2.2/lib/nest/ml_module'
nest.Install(MODULE_PATH) # Change ml_module to your module name

model= 'izhik_cond_exp'
n=nest.Create(model)
mm=nest.Create('multimeter')
sd=nest.Create("spike_recorder")
nest.SetStatus(mm, {'interval': 0.1, 'record_from': ['V_m']})
                  
rec={}    
rec[model] = nest.GetDefaults('izhik_cond_exp')['receptor_types'] 
a      = 0.2    # (E.M. Izhikevich 2007)
b_1    = 0.0  # (E.M. Izhikevich 2007)
b_2    = 0.025  # (E.M. Izhikevich 2007)
c      = -60.   # (Tateno et al. 2004)
C_m    = 80.    # (Tateno et al. 2004)
d      = 0.     # (E.M. Izhikevich 2007)
E_L    = -70.   #*(1-0.8*0.1)   # (Tateno et al. 2004)
k      = 1.     # (E.M. Izhikevich 2007)
p_1    = 1.     # (E.M. Izhikevich 2007)
p_2    = 3.     # (E.M. Izhikevich 2007)
V_b    = -55.   # Humphries 2009
V_peak = 25.    # (E.M. Izhikevich 2007)
V_th   = -50.   # (Tateno et al. 2004)
    
    # CTX to FSN ampa
AMPA_1_Tau_decay = 12.   
AMPA_1_E_rev    =  0.   # n.d. set as for  CTX to MSN
         
        # From FSN
GABAA_1_E_rev    = -74.     # n.d.; set as for MSNs
GABAA_1_Tau_decay = 6.0
          
        # From GPe
GABAA_2_Tau_decay =   6.  # n.d. set as for FSN
GABAA_2_E_rev    = -74.  # n.d. set as for MSNs
          
beta_E_L = 0.078
tata_dop = 0.8
        
beta_I_GABAA_1 = 0.8 # From FSN
beta_I_GABAA_2 = 0.8 # From GPe A
    
        
# Params from Izhikevich regular spiking

params = { 'a' : a,'b_1' : b_1, 'b_2' : b_2, 'c':c, 'C_m' : C_m,'d':d,
           'V_b' : V_b, 'V_peak' : V_peak,'E_L' : E_L, 'V_th' : V_th, 'k' : k, 'p_1':p_1,'p_2':p_2,
#           'GABAA_1_E_rev':GABAA_1_E_rev, 'GABAA_1_Tau_decay':GABAA_1_Tau_decay,'beta_I_GABAA_1':beta_I_GABAA_1
            'beta_E_L':beta_E_L,'tata_dop':tata_dop,
#            'AMPA_1_Tau_decay':AMPA_1_Tau_decay,'AMPA_1_E_rev':AMPA_1_E_rev}
            'GABAA_2_E_rev':GABAA_2_E_rev, 'GABAA_2_Tau_decay':GABAA_2_Tau_decay,'beta_I_GABAA_2':beta_I_GABAA_2}

#template = 'tsodyks_synapse'
'''
receptor_type1 = rec['my_aeif_cond_exp'] [ 'AMPA_1' ] 
receptor_type2 = rec['my_aeif_cond_exp'] [ 'NMDA_1' ]

nest.CopyModel('static_synapse', 'CS_ST_ampa', {'weight':0.25,'delay':2.5})
nest.CopyModel('static_synapse', 'CS_ST_nmda', {'weight':0.00625,'delay':2.5})
'''
'''    
receptor_type = rec['izhik_cond_exp'] [ 'AMPA_1' ]
nest.CopyModel('static_synapse', 'CS_FS_ampa', {'weight':0.25,'delay':12.})
'''
receptor_type = rec['izhik_cond_exp'] [ 'GABAA_2' ]
nest.CopyModel('static_synapse', 'GA_FS_gaba', {'weight':1.,'delay':7.})


'''        
nest.CopyModel('tsodyks_synapse', 'FS_FS_gaba', {'weight':1./0.29,'U':0.29,'delay':1.7,'tau_fac':53.,'tau_rec':902.,'tau_psc':6.})

receptor_type = rec['izhik_cond_exp'] [ 'GABAA_1' ] 
'''
     

spk=numpy.linspace(0,3000, 20*2+1)
sp=nest.Create('spike_generator', params={'spike_times':spk})
nest.SetStatus(n, params)  
nest.Connect(mm,n)



nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='GA_FS_gaba')
#nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='FS_FS_gaba')

nest.Simulate(2000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.plot(spk,numpy.ones(len(spk))*-65, '|')
pylab.show()


