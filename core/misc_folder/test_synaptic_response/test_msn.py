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
    
        dic['nest']['MS']['a']      =  0.01      # (E.M. Izhikevich 2007)
        dic['nest']['MS']['b_1']    = -20.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['b_2']    = -20.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['c']      = -55.       # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['C_m']    =  15.2      # (Humphries, Lepora, et al. 2009) # C izh
        dic['nest']['MS']['d']      =  66.9      # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['E_L']    = -81.85     # (Humphries, Lepora, et al. 2009) # v_r in izh
        dic['nest']['MS']['k']      =   1.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_peak'] =  40.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_b']    = dic['nest']['MS']['E_L']    # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_th']   = -29.7      # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['V_m']    =  80.
    
        dic['nest']['MS']['AMPA_1_Tau_decay'] = 12.  # (Ellender 2011)
        dic['nest']['MS']['AMPA_1_E_rev']     =  0.  # (Humphries, Wood, et al. 2009)
        
        dic['nest']['MS']['NMDA_1_Tau_decay'] = 160. # (Humphries, Wood, et al. 2009)
        dic['nest']['MS']['NMDA_1_E_rev']     =  dic['nest']['MS']['AMPA_1_E_rev']    
        dic['nest']['MS']['NMDA_1_Vact']      = -20.0
        dic['nest']['MS']['NMDA_1_Sact']      =  16.0
        
        # From MSN
        dic['nest']['MS']['GABAA_2_Tau_decay'] = 12.  
        dic['nest']['MS']['GABAA_2_E_rev']     = -74. # Koos 2004
    
        # From FSN
        dic['nest']['MS']['GABAA_1_E_rev']     = -74. # Koos 2004
        dic['nest']['MS']['GABAA_1_Tau_decay'] = None
        
        # From GPE
        dic['nest']['MS']['GABAA_3_Tau_decay'] = 8.          
        dic['nest']['MS']['GABAA_3_E_rev']     = -74. # n.d. set as for MSN and FSN
    
        dic['nest']['MS']['tata_dop'] = None
        
        
        dic['nest']['M1']=deepcopy(dic['nest']['MS'])
        dic['nest']['M1']['d']      =  66.9      # (E.M. Izhikevich 2007)
        dic['nest']['M1']['E_L']    = -81.85     # (E.M. Izhikevich 2007)
        
        dic['nest']['M1']['beta_d']        = 0.45
        dic['nest']['M1']['beta_E_L']      = -0.0282 #Minus size it is plus in Humphrie 2009
        dic['nest']['M1']['beta_V_b']      = dic['nest']['M1']['beta_E_L'] 
        dic['nest']['M1']['beta_I_NMDA_1'] = -1.04 #Minus size it is plus in Humphrie 2009
        
        dic['nest']['M1_low']  = deepcopy(dic['nest']['M1'])
        dic['nest']['M1_high'] = deepcopy(dic['nest']['M1'])
        dic['nest']['M1_high']['GABAA_1_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M1_high']['GABAA_2_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M1_high']['GABAA_3_E_rev']  = -64.     # n.d. set asfor MSN and FSN

        
        dic['nest']['M2']   = deepcopy(dic['nest']['MS'])
        dic['nest']['M2']['d']    =  91.       # (E.M. Izhikevich 2007)
        dic['nest']['M2']['E_L']  = -80.       # (E.M. Izhikevich 2007)
        dic['nest']['M2']['V_b']  =  dic['nest']['M2']['E_L']
        dic['nest']['M2']['beta_I_AMPA_1'] = 0.26    
    
    
        dic['nest']['M2_low']  = deepcopy(dic['nest']['M2'])
        dic['nest']['M2_high'] = deepcopy(dic['nest']['M2'])
        dic['nest']['M2_high']['GABAA_1_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M2_high']['GABAA_2_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M2_high']['GABAA_3_E_rev']  = -64.     # n.d. set asfor MSN and FSN
    


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

     

spk=numpy.linspace(0,3000, 30*2+1)
sp=nest.Create('spike_generator', params={'spike_times':spk})
nest.SetStatus(n, params)  
nest.Connect(mm,n)



#nest.Connect(sp, n, params={'receptor_type':receptor_type1}, model='CS_ST_ampa')
nest.Connect(sp, n, params={'receptor_type':receptor_type}, model='GI_ST_gaba')

nest.Simulate(2000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.plot(spk,numpy.ones(len(spk))*-65, '|')
pylab.show()


