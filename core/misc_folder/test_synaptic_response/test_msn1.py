import nest
import pylab
import numpy
MODULE_PATH=  '/afs/nada.kth.se/home/w/u1yxbcfw/tools/NEST/dist/install-nest-2.2.2/lib/nest/ml_module'
nest.Install(MODULE_PATH) # Change ml_module to your module name

model= 'izhik_cond_exp'
n=nest.Create(model)
mm=nest.Create('multimeter')
sd=nest.Create('spike_detector')
nest.SetStatus(mm, {'interval': 0.1, 'record_from': ['V_m']})
                  
rec={}    
rec[model] = nest.GetDefaults('izhik_cond_exp')['receptor_types'] 
    
#CTX-D1    
AMPA_1_Tau_decay = 12.  # (Ellender 2011)
AMPA_1_E_rev     =  0.  # (Humphries, Wood, et al. 2009)
        
NMDA_1_Tau_decay = 160. # (Humphries, Wood, et al. 2009)
NMDA_1_E_rev     =  AMPA_1_E_rev    
NMDA_1_Vact      = -20.0
NMDA_1_Sact      =  16.0
        
        # From MSN
GABAA_2_Tau_decay = 12.  
GABAA_2_E_rev     = -74. # Koos 2004
    
        # From FSN
GABAA_1_E_rev     = -74. # Koos 2004
GABAA_1_Tau_decay = 8.
        
        # From GPE
GABAA_3_Tau_decay = 8.          
GABAA_3_E_rev     = -74. # n.d. set as for MSN and FSN
    
tata_dop = .8
        
a      =  0.01      # (E.M. Izhikevich 2007)
b_1    = -20.       # (E.M. Izhikevich 2007)
b_2    = -20.       # (E.M. Izhikevich 2007)
c      = -55.       # (Humphries, Lepora, et al. 2009)
C_m    =  15.2      # (Humphries, Lepora, et al. 2009) # C izh
d      =  66.9      # (Humphries, Lepora, et al. 2009)
E_L    = -81.85     # (Humphries, Lepora, et al. 2009) # v_r in izh
k      =   1.       # (E.M. Izhikevich 2007)
V_peak =  40.       # (E.M. Izhikevich 2007)
V_b    = E_L    # (E.M. Izhikevich 2007)
V_th   = -29.7      # (Humphries, Lepora, et al. 2009)
V_m    =  80.
I_e    =  260.        

d      =  66.9      # (E.M. Izhikevich 2007)
E_L    = -81.85     # (E.M. Izhikevich 2007)
        
beta_d        = 0.45
beta_E_L      = -0.0282 #Minus size it is plus in Humphrie 2009
beta_V_b      = beta_E_L 
beta_I_NMDA_1 = -1.04 #Minus size it is plus in Humphrie 2009
        



# Params from Izhikevich regular spiking

params = { 'a' : a,'b_2' : b_2, 'b_1' : b_1, 'c':c,'d':d, 'C_m' : C_m,
           'V_b' : V_b, 'V_peak' : V_peak,'k':k, 
           'E_L' : E_L, 'V_th' : V_th,'V_m':V_m,'beta_d':beta_d,'beta_E_L':beta_E_L,
           'beta_V_b':beta_V_b,'I_e':I_e,
#          'GABAA_1_E_rev':GABAA_1_E_rev, 'GABAA_1_Tau_decay':GABAA_1_Tau_decay}
#            'GABAA_2_Tau_decay':GABAA_2_Tau_decay,'GABAA_2_E_rev':GABAA_2_E_rev}
            'GABAA_3_Tau_decay':GABAA_3_Tau_decay,'GABAA_3_E_rev':GABAA_3_E_rev}
#           'beta_I_NMDA_1':beta_I_NMDA_1,'AMPA_1_Tau_decay':AMPA_1_Tau_decay,'AMPA_1_E_rev':AMPA_1_E_rev,'NMDA_1_Tau_decay':NMDA_1_Tau_decay,'NMDA_1_E_rev':NMDA_1_E_rev,
 #           'NMDA_1_Vact':NMDA_1_Vact,'NMDA_1_Sact':NMDA_1_Sact}

#template = 'tsodyks_synapse'

'''
receptor_type1 = rec['izhik_cond_exp'] [ 'AMPA_1' ] 
receptor_type2 = rec['izhik_cond_exp'] [ 'NMDA_1' ]

nest.CopyModel('static_synapse', 'C1_M1_ampa', {'weight':0.5,'delay':12.})
nest.CopyModel('static_synapse', 'C1_M1_nmda', {'weight':0.11,'delay':12.})
'''
'''
receptor_type = rec['izhik_cond_exp'] [ 'GABAA_2' ]
nest.CopyModel('static_synapse', 'M1_M1_gaba', {'weight':0.2,'delay':1.7})
'''
receptor_type = rec['izhik_cond_exp'] [ 'GABAA_3' ]
nest.CopyModel('static_synapse', 'GA_M1_gaba', {'weight':1.,'delay':1.7})


'''       
nest.CopyModel('tsodyks_synapse', 'FS_M1_gaba', {'weight':round(6./0.29,1),'delay':1.7,'U':0.29,'tau_fac':53.0,'tau_rec':902.0,'tau_psc':8.0})

receptor_type = rec['izhik_cond_exp'] [ 'GABAA_1' ] 
'''
     

spk=numpy.linspace(2000,5000,30*2+1)
sp=nest.Create('spike_generator', params={'spike_times':spk})
#bg=nest.Create('poisson_generator',1,{'rate':531.})
nest.SetStatus(n, params)  
nest.Connect(mm,n)



#nest.Connect(sp, n, params={'receptor_type':receptor_type1}, model='C1_M1_ampa')
#nest.Connect(sp, n, params={'receptor_type':receptor_type2}, model='C1_M1_nmda')
#nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='FS_M1_gaba')
#nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='M1_M1_gaba')
nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='GA_M1_gaba')
#nest.Connect(bg,n)
nest.Simulate(5000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.plot(spk,numpy.ones(len(spk))*-75, '|')
pylab.show()


