'''
Module:
lines

The module contains two functions models and network. In models neuron and 
synapse models are defined and in network populations (layers) and connections
are defined. 

Functions:
models   - define neuron and synapse models
network  - define layers and populations
'''

'''

Function:
models(Params_in)

Description: Parameters for neurons and synapse models are set and list with 
name of each model and corresponding parameters is returned. In NSET both 
neurons recording devices and synapses are considered as models. 

Inputs:
    Params_in 		- Parameter values to be changed 
		  
Returns:
    model_list 	- list with neuron, recording and synapse models name and 
                  corresponding parameters  
'''
def models( Params_in = {} ):
    #! Preparations
    #! ============
    #! Python imports
     
    ##! Imports 
    import nest # Has to be imported othervise not recognized.
    import numpy as np
    import matplotlib.pyplot as pl
    import pprint
    import time as ttime
    #current_path=os.getcwd()

    # First add parent directory to python path
    #parent_dir='/'.join(os.getcwd().split('/')[0:-1]) 
    import os

    
    #! Install nmda neuron model, need to do it twice due to some bug
    try:
        nest.Install('/usr/local/lib/nest/ml_module')
    except:
      pass
     

    #! ========================
    #! Configurable Parameters 
    #! ========================
    #! Here those parameters that taken to be configurable is defined. The 
    #! choice of configurable parameters is obviously arbitrary, and in 
    #! practice one would have far more configurable parameters. Only these 
    #! parameters should be change.Hard coded parameters should be moved up to 
    #! configurable parameters if they  are changed or a new model should be 
    #! created. 
    #!
    #! To have plastic and static synapses 10:1 in weight gives same
    #! conductance strenght.    
    
    CTX_MSN_g_ampa =  1.     # (Humphries, Wood, and Gurney 2009)
    CTX_MSN_g_nmda =  0.02   # (Humphries, Wood, and Gurney 2009)
    MSN_MSN_g_gaba =  0.75   # (Koos, Tepper, and Charles J Wilson 2004)
    FSN_MSN_g_gaba =  3.8    # (Koos, Tepper, and Charles J Wilson 2004)

    CTX_FSN_g_ampa = 1.#10.2 # (Humphries, Wood, and Gurney 2009)
    GPE_FSN_g_gaba = 35.     # gives 5 mV epsp , n.d. tuned to achieve realistic firing rates
    
    CTX_STN_g_ampa =  2.5/10.    # (Baufreton et al. 2005)
    CTX_STN_g_nmda =  0.05   # n.d.; same ratio ampa/nmda as MSN
    GPE_STN_g_gaba =   .1    # (Baufreton et al. 2009)

    MSN_GPE_g_gaba = 2.      # Half of  (Sims et al. 2008)
    STN_GPE_g_ampa = 0.33/2    # (Hanson & Dieter Jaeger 2002)
    STN_GPE_g_nmda = 0.03    # n.d; estimated
    GPE_GPE_g_gaba = 0.5    # Half (Sims et al. 2008)

    MSN_SNR_g_gaba = [2., 23.5] # Lower based on (Connelly et al. 2010) = [4.7, 24.], 50 Hz model = [5.8, 23.5]
    STN_SNR_g_ampa =  0.75      # (Shen and Johnson 2006)
    STN_SNR_g_nmda =  0.01       # n.d.; same ratio ampa/nmda as for MSN
    #GPE_SNR_g_gaba = [16., 70.  # (Connelly et al. 2010)
    GPE_SNR_g_gaba = [0.16*76., 76.]                                            # (Connelly et al. 2010)


    wfMSN_SNR=1
    wfGPE_SNR=1. # Change to 1 to get
    # s - static
    # p - plastic
    Params = {
             'CTX_MSN_ampa_s'     : CTX_MSN_g_ampa,                             # static
             'CTX_MSN_nmda_s'     : CTX_MSN_g_nmda,                             # static
             'MSN_MSN_gaba_s'     : MSN_MSN_g_gaba,                             # static, 0.75 ~ 0.2 mV at -80
             'FSN_MSN_gaba_p'     : round( FSN_MSN_g_gaba*1/0.29, 1) ,          # plastic, first spike 3.8 nS
             'FSN_MSN_gaba_s'     : FSN_MSN_g_gaba,                             # static, 3.8 ~ 1   mV at -80
             
             'CTX_FSN_ampa_s'     : CTX_FSN_g_ampa,                             # static
             'GPE_FSN_gaba_s'     : GPE_FSN_g_gaba,                             # static
			 
             'CTX_STN_ampa_s'     : CTX_STN_g_ampa,                             # static
             'CTX_STN_nmda_s'     : CTX_STN_g_nmda,                             # static
             'GPE_STN_gaba_s'     : GPE_STN_g_gaba,                             # static, 3 nS gives 5 mV IPSP as in Bevan 2002, 11 nS gives 11.3 mV
             'STN_STN_ampa_p'     : 0.75 * 10,                                  # plastic
             'STN_STN_ampa_s'     : 0.75 * 10,                                  # plastic
             'STN_STN_ampa_s'     : 0.75,                                       # static
             'STN_STN_nmda_s'     : 0.1,                                        # static
                         
             'MSN_GPE_gaba_p'     : MSN_GPE_g_gaba/0.24,                       # plastic, 101 gives ~ 11 nS at end of 10 spikes at 20 and 50 Hz
             'MSN_GPE_gaba_s_min' : MSN_GPE_g_gaba,                          # static, 1.4 nS gives 0.4 mV at -70
             'MSN_GPE_gaba_s_max' : MSN_GPE_g_gaba*1.9,                          # static, 10.1 nS gives 2.25 mV at -70

             'STN_GPE_ampa_p0'    : 0.35 * 10,                                  # plastic
             'STN_GPE_ampa_p1'    : 0.35 * 10,                                  # plastic
             'STN_GPE_ampa_p2'    : 0.35 * 10,                                  # plastic
             'STN_GPE_ampa_s'     : STN_GPE_g_ampa,                             # static, 0.6  5 mV at -70, 
             'STN_GPE_nmda_s'     : STN_GPE_g_nmda,                             # static
             'GPE_GPE_gaba_s'     : GPE_GPE_g_gaba,                          # static, 4 nS 2-3 mV at -75
             
             'MSN_SNR_gaba_p0'    : MSN_SNR_g_gaba[0]/0.0259*wfMSN_SNR,            # plastic Connelly fit 1
             'MSN_SNR_gaba_p1'    : MSN_SNR_g_gaba[0]/0.0192*wfMSN_SNR,            # plastic Connelly fit 2
             'MSN_SNR_gaba_p2'    : MSN_SNR_g_gaba[0]/0.0148*wfMSN_SNR,            # plastic Connelly fit 3
             'MSN_SNR_gaba_s_min' : MSN_SNR_g_gaba[0]*wfMSN_SNR,                          # static, 4.7 nS gives -1.4 mV at -70 
             'MSN_SNR_gaba_s_mid' : MSN_SNR_g_gaba[0]*wfMSN_SNR*3.8162/2.,
             'MSN_SNR_gaba_s_max' : MSN_SNR_g_gaba[0]*wfMSN_SNR*3.8162,                          # static, 24 nS gives -5 mV at -70
             'STN_SNR_ampa_p0'    : 3.0 * 10,                                   # plastic
   	         'STN_SNR_ampa_p1'    : 3.0 * 10,                                   # plastic
   		   	 'STN_SNR_ampa_p2'    : 3.0 * 10,                                   # plastic
             'STN_SNR_ampa_p3'    : 3.8*STN_SNR_g_ampa/0.35,                                   # plastic
   		     'STN_SNR_ampa_s'     : STN_SNR_g_ampa,                             # static, 1 approx 5 mV at -70, 
      		 'STN_SNR_nmda_s'     : STN_SNR_g_nmda,                             # static
             'GPE_SNR_gaba_s_ref' : GPE_SNR_g_gaba[0]*wfGPE_SNR,                 # static
             'GPE_SNR_gaba_s_max' : GPE_SNR_g_gaba[1]*wfGPE_SNR,                # static
             'GPE_SNR_gaba_p'     : GPE_SNR_g_gaba[1]*wfGPE_SNR/0.196,          # plastic, 4* MSN-SNR    
             } 

    #print 'Updating Params'                                                     
    Params.update(Params_in)                                                    # update the parameters with input parameter
        
    model_list = [] 
    #! =============        
    #! Input Models
    #! =============  
    #! MSN
    #! ===
    params={}
    model_list.append(('spike_generator', 'MSN_D1_spk_gen', params))
    model_list.append(('spike_generator', 'MSN_D2_spk_gen', params))
    model_list.append(('spike_generator', 'CTX_input_STN_spk_gen', params))
  
  
    #! =============        
    #! Neuron Models
    #! =============
    #!
    #! Entries in model list
    #! model_list = [(old model, new model, parameters),
    #! 				(old model, new model, parameters),
    #!               etc ...]
    #! Where old model can be NEST name a existing or a recently created model. 
    #! New model is the name of the for the new model to be registered and 
    #! parameters are the parameter to change for the model.
    
    #! The ``model_list`` mapp native NEST neuron models to the models used in 
    #! basal ganglia model. This is done in two step. First a shared set of 
    #! parameters are mapped to three intermediate models, ``'MSN_model'``, 
    #! ``'STN_GPE_SNR_model'`` and ``'input_model'``. Then models parameters 
    #! for each individual sub population model is mapped from this general 
    #! models with individualized parameters.


    #! MSN
    #! ===
    
    #! Firing rate and synaptic events
    #! -------------------------------
    #! Humphrie 2009, org- ref. Blackwell 2003 reported MSN recieves 800 
    #! synaptic events per second. Could not separate between. In their model 
    #! the assume 265 cortical cells that give 530 synaptical events per second. 
    
    #! Model
    #! -----
    
    #! - Rheobase around 320 pA
    
    a      =   0.01                                                             # (E.M. Izhikevich 2007)
    b      = -20.                                                               # (E.M. Izhikevich 2007)
    c      = -55.                                                               # (E.M. Izhikevich 2007)
    C      =  15.2                                                              # (Humphries, Lepora, et al. 2009)
    d      =  91.                                                               # (E.M. Izhikevich 2007)
    k      =   1.                                                               # (E.M. Izhikevich 2007)
    v_peak =  40.                                                               # (E.M. Izhikevich 2007)
    v_r    = -80.                                                               # (E.M. Izhikevich 2007)
    v_t    = -29.7                                                              # (Humphries, Lepora, et al. 2009)

    #! These paramters then gives MSN rheobase close to 230 which is more
    #! realistic than original izhikevich which is around 300 pA. Also with 
    #! these parameters shorter delay to second spike is in accordance with MSN
    #! experiment.
    
    CTX_MSN_tau_ampa =   6.                                                     # (Humphries, Wood, et al. 2009)
    CTX_MSN_tau_nmda = 160.                                                     # (Humphries, Wood, et al. 2009)
    CTX_MSN_E_ampa   =   0.                                                     # (Humphries, Wood, et al. 2009)
    CTX_MSN_E_nmda   =   0.                                                     # (Humphries, Wood, et al. 2009)

    MSN_MSN_tau_gaba =  12.4                                                    # (Koos et al. 2004)
    MSN_MSN_E_gaba   = -64.                                                     # (Bracci & Panzeri 2006)
    
    FSN_MSN_tau_gaba  =  11.4                                                    # (Koos et al. 2004)
    FSN_MSN_E_gaba    = -64.                                                     # (Bracci & Panzeri 2006)
    
    params = { 'a' :  a, 'b_1' : b, 'b_2' : b, 'c' : c, 'C_m' :  C, 'd' :  d,
               'k' : k, 'V_b' : v_r, 'V_peak' :  v_peak, 'V_r' : v_r,
             'V_t' : v_t, 'I_e' :  0., 'V_m' :-80.,
             'AMPA_Tau_decay'  : CTX_MSN_tau_ampa,
             'NMDA_Tau_decay'  : CTX_MSN_tau_nmda,
             'NMDA_Vact'       :-20.0,
             'NMDA_Sact'       :  16.0,
             'GABAA_1_Tau_decay' : FSN_MSN_tau_gaba,
             'GABAA_1_E_rev'     : FSN_MSN_E_gaba,
             'GABAA_2_Tau_decay' : MSN_MSN_tau_gaba,
             'GABAA_2_E_rev'     : MSN_MSN_E_gaba }                                          

    model_list.append(('izhik_cond_exp', 'MSN_izh', params))
    
    
    #! FSN
    #! ===
    
    #! Firing
    #! ------
    #! FSN recieve AMPA and GABA, NMDA is rare. Humphrie 2009 org ref Blackwell 
    #! Czybayko and Plentz 2003. Rheobase around 110 pA
    #! 61 nS e godtyckligt i Humphrie 2009, taked such that spike rates would fit.
    #! 140 synaptic input per second gives a rate at approx 40 - 80 Hz.

    # General input as poisson process FSN, Kimura 1990 2-7 hz type 1 
    #! (believed to be FSN) and 5-30 hz during walk task Chp8 HBBG2010, 
    #! sensetive to cortical stimulation

    #! Model
    #! -----
    a      = 0.2                                                                # (E.M. Izhikevich 2007)
    b      = 0.025                                                              # (E.M. Izhikevich 2007)
    c      = -60.                                                               # (Tateno et al. 2004)
    C      = 80.                                                                # (Tateno et al. 2004)
    d      = 0.                                                                 # (E.M. Izhikevich 2007)
    k      = 1.                                                                 # (E.M. Izhikevich 2007)
    p_1    = 1.                                                                 # (E.M. Izhikevich 2007)
    p_2    = 3.                                                                 # (E.M. Izhikevich 2007)
    v_b    = -55.                                                               # (E.M. Izhikevich 2007)
    v_peak = 25.                                                                # (E.M. Izhikevich 2007)
    v_r    = -70.                                                               # (Tateno et al. 2004)
    v_t    = -50.                                                               # (Tateno et al. 2004)

    CTX_FSN_tau_ampa = 6.                                                        # (Humphries, Wood, et al. 2009)
    CTX_FSN_E_ampa   = 0.                                                        # (Humphries, Wood, et al. 2009)
    
    GPE_FSN_tau_gaba  = 2.1                                                       # n.d. set as for GPE to SNR
    GPE_FSN_E_gaba    = -64.                                                      # n.d.; set as for MSNs
    
    params = { 'a' : a, 'b_1' : b, 'b_2' : b, 'c' : c, 'C_m' :  C, 'd' : d,
            'k' : k, 'p_2' :  p_2, 'V_b' : v_b, 'V_peak':  v_peak,
            'V_r' : v_r, 'V_t' : v_t, 'I_e' :  0., 'V_m'   :-60.,
            'AMPA_Tau_decay'    : CTX_FSN_tau_ampa,
            'GABAA_1_Tau_decay' : GPE_FSN_tau_gaba,
            'GABAA_1_E_rev'     : GPE_FSN_E_gaba, }             
    model_list.append(('izhik_cond_exp', 'FSN', params))

   
    #! STN
    #! ===

    #! Fring
    #! -----
    #! In rat brain slices STN neurons fire at a rate of 10.3 +-1.3 Hz. This 
    #! firing pattern was unaffected by applying GABAA blocker (antagnist) 
    #! picrotoxin. (Loucif 2005). For STN 75 with 1.5 current injection 
    #! autonomous firing rate at 11.2   
    
    #! Baufreton 2005 observed IPSP in some STN neuorns in slice suggesting that
    #! some connectivity between GPE and STN was present in slice. Then blocking 
    #! GABAA increased firing rate was obsertved. He argues that this shows that 
    #! a few potent GABAerigc axon can effect the firing of STN . 
    
    #! Model
    #! -----
    #! Exhibit rebound burst as in Bevan 2002. 2 parameters fit. On exibiting 
    #! short short rebound burst (75 percent) and one with long (25 percent).  
    #! No rheobase tonically active. 
    
    #! Several experiments show that the subthalamic neurons exhibits a rich 
    #! behavior with short and long rebound burst after hyperpolarization, 
    #! single spike mode, bursting spike mode, reverse spike frequency 
    #! adaptation and prolonged afterhyperpolarization after high frequency 
    #! stimulation. (Charles J Wilson et al. 2004; Paz et al. 2005; Baufreton 
    #! et al. 2005; Kass & Mintz 2006; M. A. Farries et al. 2010; Bevan, P J 
    #! Magill, et al. 2002; Bevan et al. 2000; Beurrier et al. 1999; Nakanishi 
    #! et al. 1987) It has been experimentally reported (Bevan, P J Magill, 
    #! et al. 2002; Bevan et al. 2000) that STN can be divided into two types 
    #! base on their response to hyperpolarizing current neurons. The most 
    #! common reported (~75%) give rise to short and strong rebound burst when 
    #! sufficiently hyperpolarized and the other type (~25%) give rice to a 
    #! prolonged burst. Here we have focus on modeling the first one. Our 
    #! neuron model was been tuned to I-V curves as in (Bevan 
    #! et al. 2000; Bevan & C J Wilson 1999; Nicholas E Hallworth et al. 2003) 
    #! and a I-F curve as in (Bevan & C J Wilson 1999; Nicholas
    #!  E Hallworth et al. 2003; Charles J Wilson et al. 2004) with potential 
    #! to firing at high frequencies and sigmoid I-V cuve. Neuron model also 
    #! show short rebound burst when hyperpolarized below -78 
    #! as seen in (Bevan et al. 2000) and current and frequency dependent 
    #! afterhyperpolarization when going back to spontaneous firing after 
    #! current stimulation.  
    
    CTX_STN_tau_ampa =   4.0                                                    # (Baufreton et al. 2005)
    CTX_STN_tau_nmda = 100.                                                     # n.d. estimated 
    CTX_STN_E_ampa   =   0.                                                     # (Baufreton et al. 2009)
    CTX_STN_E_nmda   =   0.                                                     # n.d.; set as  E_ampa
    
    GPE_STN_tau_gaba  =   7.8                                                    # (Baufreton et al. 2009)
    GPE_STN_E_gaba    = -84.0                                                    # (Baufreton et al. 2009)
    
    #! STN 75 percent
    #! ---------------   
    #! Adaptive exponential iaf 
    #! Alt 1 as Izh
    a      =   0.003 # I-V relation, spike frequency adaptation
    b_1    =  0.4    # I-V relation
    b_2    = 0.      # I-V relation
    c      = -72.    # I-V relation
    C      =  60.    # t_m/R_in
    d      =  0.1    #0.1 #0.1#200./5.                                                     
    gleak  =   5.
    v_peak =  15.                                                                
    v_b    =  -76.1  # I-V relation
    v_r    = -76.1                                                               
    v_t    = -64.                                                                

    s      =  12.                      
    I_e    = 0. 
    
    
    u_thr_V_reset_increase =0. 
    V_reset_slope=-10.
    V_reset_max_increase=10.;
    # Translation:
    # Mine   Brette
    # a      1/tau_w
    # b      a
    # c      V_reset
    # d      b
    # vr     E_leak
    # vt     V_th
    # s      Delta_T

    
    params = { 'a_1':b_1, 'a_2':b_2, 'b':d, 'C_m':C, 'Delta_T':s, 'E_L' : v_r, 'g_L' : gleak,
               'tau_w':1/a, 'V_a':v_b, 'V_th':v_t, 'V_reset':c, 'V_peak':v_peak, 
               'V_reset_max_increase':V_reset_max_increase, 
               'V_reset_slope':V_reset_slope, 
               'u_thr_V_reset_increase':u_thr_V_reset_increase,
               'I_e' : I_e,
               'AMPA_Tau_decay'   : CTX_STN_tau_ampa,
               'NMDA_Tau_decay'   : CTX_STN_tau_nmda,
               'NMDA_Vact'        :-20.0,
               'NMDA_Sact'        : 16.0,
               'GABAA_1_Tau_decay': GPE_STN_tau_gaba,
               'GABAA_1_E_rev'    : GPE_STN_E_gaba, } 
    
    model_list.append(('my_aeif_cond_exp', 'STN_75_aeif', params))
    
   
    #! STN 25 percent
    #! --------------
    params = params.copy()                                                      # Copy STN 75 parameters
    #params.update( { 'd_1' : 0.5, 'd_1_bound' : 0.0 } )
    model_list.append(('izhik_cond_exp' , 'STN_25' , params))


    #! GPE
    #! ===
    #! Firing
    #! ------
    #! Base activity in vivo of GPE has been reported to be between 8-15 hz 
    #! (Xue, Han, and Chen 2010; Chang et al. 2006) and in slice between 
    #! 8-15 hz (Rav-Acha et al. 2005; Bugaysen et al. 2010) 
    
    #! Model
    #! -----
    #! Several experimental studies have characterized GPE neurons and it 
    #! seems to be various types of neurons within GPE (H Kita and Kitai 
    #! 1991; Nambu and Llinas 1994; Cooper and I M Stanford 2000; Michael 
    #! A Farries, Meitzen, and Perkel 2005; Bugaysen et al. 2010)  A 
    #! resent modeling study has also suggested that the variation in GPE 
    #! population is due to a continuously change in the cellular 
    #! properties of the neurons (Gunay, Edgerton, and Dieter Jaeger 
    #! 2008). Our model were tuned to reproduce the I-F and I-V of 
    #! Type A (63 percent) of Cooper & Stanford (2000) and Type A and 
    #! Type C (67 percent) from (Bugaysen et al. 2010). Furthermore it 
    #! displays characteristic of repetitive firing neurons observed by 
    #! (H Kita and Kitai 1991),  type 2 neurons observed by (Nambu and 
    #! Llinas 1994) and the type  A observed by (Cooper and I M Stanford
    #! 2000) The neuron model shows prolonged afterhyperpolarization 
    #! after high frequency stimulation (Figure 2D), sags for strong 
    #! depolarizing current (Figure 2C), fast frequency adaptation and 
    #! can exhibit rebound firing after hyperpolarizing current pulse. 


    MSN_GPE_tau_gaba = 6.                                                      # (Sims et al. 2008)
    MSN_GPE_E_gaba   = -65.                                                     # (Rav-Acha et al. 2005)
    
    STN_GPE_tau_ampa = 12.                                                      # (Hanson & Dieter Jaeger 2002)
    STN_GPE_tau_nmda = 100.                                                     # n.d.; estimated
    STN_GPE_E_ampa   = 0.                                                       # n.d.; same as CTX to STN
    STN_GPE_E_nmda   = 0.                                                       # n.d.; same as CTX to STN
    
    GPE_GPE_tau_gaba  = 5.                                                     # (Sims et al. 2008)
    GPE_GPE_E_gaba    = -65.        
    #! GPE type A aeif Terman type
    #! Adaptive exponential iaf 
    #! Alt 1 as Izh
    a      = 0.05  # I-V relation, spike frequency adaptation
    b_1    = 2.    # I-V relation
    b_2    = 2.    # I-V relation
    c      = -55.  # I-V relation
    C      =  44.  # t_m/R_in
    d      = 70.   # I-F relation
    gleak  =   3.
    v_peak =  15.  # Cooper and standford
    v_b    =  -58.4 # I-V relation
    v_r    = -58.8  # v_t    = -56.4                                                               # 
    v_t    = -56.4
    s      =  4.8                      
    I_e    = 10.
    

    # Translation:
    # Mine   Brette
    # a      1/tau_w
    # b      a
    # c      V_reset
    # d      b
    # vr     E_leak
    # vt     V_th
    # s      Delta_T

    
    params = { 'a_1':b_1, 'a_2':b_2, 'b':d, 'C_m':C, 'Delta_T':s, 'E_L' : v_r, 'g_L' : gleak,
               'tau_w':1/a, 'V_a':v_b, 'V_th':v_t, 'V_reset':c, 'V_peak':v_peak, 
               'V_reset_max_increase':V_reset_max_increase, 
               'V_reset_slope':V_reset_slope, 
               'u_thr_V_reset_increase':u_thr_V_reset_increase,
               'I_e' : I_e,
               'AMPA_Tau_decay'    : STN_GPE_tau_ampa,
               'NMDA_Tau_decay'    : STN_GPE_tau_nmda,
               'NMDA_Vact'         : -20.0,
               'NMDA_Sact'         :  16.0,
               'GABAA_1_Tau_decay' : MSN_GPE_tau_gaba,
               'GABAA_1_E_rev'     : MSN_GPE_E_gaba,
               'GABAA_2_Tau_decay' : GPE_GPE_tau_gaba,
               'GABAA_2_E_rev'     : GPE_GPE_E_gaba, } 
    
    model_list.append(('my_aeif_cond_exp', 'GPE_aeif', params))
    
    #! SNR
    #! ===
    #! In SNR base activity in vivo has been reported to be between 7-9 Hz 
    #! (Chang et al. 2006)and in lice to be around 16 Hz (Chuhma et al. 2011). 
    
    
    #! Model
    #! -----
    #! Several studies suggest that the GABAergic non-dopaminergic neuron 
    #! in SNr exhibits less diversity than GPE neurons.(Richards, Shiroyama, 
    #! and Kitai 1997; Nakanishi et al. 1997; Rohrbacher, Ichinohe, and 
    #! Kitai 2000; Atherton and Bevan 2005; Lee and Tepper 2007) Our 
    #! model was tuned to reproduce I-V relation (se Figure 3A) of 
    #! (Rohrbacher, Ichinohe, and Kitai 2000; Atherton and Bevan 2005) 
    #! and the I-F relation (se Figure 3B) to (Rohrbacher, Ichinohe, and 
    #! Kitai 2000). The neuron model displays no sag to hyperpolarizing 
    #! current (Figure 3C ) and no rebound spike after hyperpolarizing 
    #! current injection. 

    
    
    MSN_SNR_tau_gaba =   5.2                                                    # (Connelly et al. 2010)
    MSN_SNR_E_gaba   = -80.                                                     # (Connelly et al. 2010)
    
    STN_SNR_tau_ampa =  12.                                                     # n.d.; set as for STN to GPE
    STN_SNR_tau_nmda = 100.                                                     # n.d.; estimated
    STN_SNR_E_ampa   =   0.                                                     # n.d. same as CTX to STN
    STN_SNR_E_nmda   =   0.                                                     # n.d. same as CTX to 
  
    GPE_SNR_tau_gaba  =   2.1                                                    # (Connelly et al. 2010)
    GPE_SNR_E_gaba    = -72.                                                     # (Connelly et al. 2010)
      
    
    #! Adaptive exponential iaf 
    #! Alt 1 as Izh
    a      =   0.05  # I-V relation, spike frequency adaptation
    b    =   3.      # I-V relation
    c      = -65.    # I-V relation
    C      =  80.    # t_m/R_in
    d      =  200.   # I-F relation
    gleak  =   5.
    v_peak =  20.                                                               # 
    v_r    = -57.    #
    v_t    = -55.    # 
    v_b    = -57.    # I-V relation
    s      =  3.5                      
    I_e    = 20.0 
      
    # Translation:
    # Mine   Brette
    # a      1/tau_w
    # b      a
    # c      V_reset
    # d      b
    # vr     E_leak
    # vt     V_th
    # s      Delta_T

    
    params = { 'a_1':b, 'a_2':b, 'b':d, 'C_m':C, 'Delta_T':s, 'E_L' : v_r, 'g_L' : gleak,
               'tau_w':1/a, 'V_a':v_b, 'V_th':v_t, 'V_reset':c, 'V_peak':v_peak, 
               'I_e' : I_e,
               'AMPA_Tau_decay'    : STN_SNR_tau_ampa,
               'NMDA_Tau_decay'    : STN_SNR_tau_nmda, # 100 ms standard approx, NMDA at -20 could shift to the left e.g. -40 
               'NMDA_Vact'         :-20.0,
               'NMDA_Sact'         : 16.0,
               'GABAA_1_Tau_decay' : MSN_SNR_tau_gaba,
               'GABAA_1_E_rev'     : MSN_SNR_E_gaba,
               'GABAA_2_Tau_decay' : GPE_SNR_tau_gaba,
               'GABAA_2_E_rev'     : GPE_SNR_E_gaba, }   
    
    model_list.append(('my_aeif_cond_exp', 'SNR_aeif', params))
    
  
       
    #! ========
    #! Synapses
    #! ========
    #! Are defined in the following order (axons origin):
    #! MSN, FSN, STN, GPE, SNR and cortex.
    
    for key, val in Params.iteritems():
        exec '%s = %s' % (key, str(val))# dictonary entries as variables
                                      
    rec = nest.GetDefaults('izhik_cond_exp')['receptor_types']                  # get receptor types
    
    
    #! General on Tsydoks short term plasticity model
    #! ===============================================
    #! u        - governes release probability. For a facilitating synapse if 
    #!            increased( 'MSN_SNR_gaba_p' , params = params ) then 
    #!            p1/p# decreases. 
    
    #! tau_fact - facilitation time constant. **For facilitating synapse** if 
    #!            **increased** then low frequency input will facilitate more, 
    #!            no major different for high frequency input max facilitation. 
    #!            If **decreased** then low frequency input will be facilitate 
    #!            less, but for high frequency input no change in facilitation 
    #!            max.
    
    #! tau_rec  - recovery time constant. **For facilitating synapse** if 
    #!            **increased** then high frequency input starts dip more and 
    #!            low frequency inputs starts to dip. If **decreased** high and 
    #!            low frequency starts to dip later. High frequency input 
    #!            facilitate to higher values. Low frequency input max does not
    #!            change and p1/p# increases
    
    
    #! MSN
    #! ===

    #! MSN to MSN
    #! ----------
    #! Weight should be 5 x less than FSN (Koos 2004). Henrike (2009) have data 
    #! 0.45 mV MSN to MSN clampted at -80 with E_rev_gaba at -59 (source 
    #! conversation with Henrike)  
    #!
    #! MSN    MSN    tau_gaba          12.4 ms    (Koos et al. 2004)    
    #! MSN    MSN    g_(peak-gaba)     0.75 nS    (Koos et al. 2004)    
    #! MSN    MSN    t_delay           1.7 ms     (Taverna et al. 2004)    
    #! MSN    MSN    E_gaba            -64 mV     (Bracci & Panzeri 2006)   


    #! GABA static
    params = {'delay' : 1.7, 'weight' : MSN_MSN_gaba_s ,
              'receptor_type' : rec[ 'GABAA_2' ] }                                   
    model_list.append(('static_synapse', 'MSN_MSN_gaba_s', params))
    
    
    #! MSN to GPE
    #! ----------
    #! MSN    GPE    tau_gaba        6.1 ms            (Sims et al. 2008)    
    #! MSN    GPE    g_(peak-gaba)   1.4  - 10.1 nS    (Sims et al. 2008)    
    #! MSN    GPE    t_delay         7 ms              (Park et al. 1982)    
    #! MSN    GPE    E_gaba          -65 mV            (Rav-Acha et al. 2005)  
    
    delay = 7.
    
    #! GABAA plastic, tuned to data from Sims (2008). Pair pulse ratio p1/p10
    #! equals 2. No data on recovery rate. Model parameters set such model 
    #! repdrouces same pair pulse ratio for ten spikes at 20 Hz and 50 Hz 
    #! as seen in experiments. U equals 0.1, high probability of release gives
    #!  p1/p8 = 2    
    '''
    params = { 'U'      : 0.05,                                                   # GABAA plastic                   
              'tau_fac' : 200.0,
              'tau_rec' : 150.0,
              'tau_psc' : 6.1,
              'delay'   : delay, 'weight' : MSN_GPE_gaba_p ,
              'receptor_type': rec[ 'GABAA_1' ] }                  
    '''    
    params = { 'U'      : 0.24,                                                   # GABAA plastic                   
              'tau_fac' : 13.0,
              'tau_rec' : 77.0,
              'tau_psc' : 7.0,
              'delay'   : delay, 'weight' : MSN_GPE_gaba_p ,
              'receptor_type': rec[ 'GABAA_1' ] }              
    model_list.append(('tsodyks_synapse' , 'MSN_GPE_gaba_p' , params))
 
 
    
    params = {'delay'        : delay, 'weight' : MSN_GPE_gaba_s_min ,           # GABAA static weak
              'receptor_type': rec[ 'GABAA_1' ] }                               
    model_list.append(('static_synapse' , 'MSN_GPE_gaba_s_min' , params))
    
    params = {'delay'        : delay, 'weight' : MSN_GPE_gaba_s_max ,           # GABAA static strong
              'receptor_type': rec[ 'GABAA_1' ] }                               
    model_list.append(('static_synapse' , 'MSN_GPE_gaba_s_max' , params))
    
 
 
    #! MSN to SNR
    #! ----------
    #! MSN    SNR   tau_gaba       5.2 ms                  (Connelly et al. 2010)
    #! MSN    SNR    g_gaba        4.7 - 24 nS [min max]   (Connelly et al. 2010)         
    #! MSN    SNR    delay_gaba    7.3 ms                  (Connelly et al. 2010)
    #! MSN    SNR    E_rev-gaba    -65 mV                  (Connelly et al. 2010)
    
    delay = 7.3
    
        
    #! GABAA plastic, tuned to Connelly (2010), three fits, first to
    #! burst 10, 50 100 Hz a 10 spikes, second to 100 Hz a 5 plus 
    #! recovery spike and last to both. 
    params = { 'U'      : 0.0259,
              'tau_rec' : 103.,
              'tau_fac' : 403., 
              'tau_psc' : 5.2,
              'delay'   : delay, 'weight' : MSN_SNR_gaba_p0, 
              'receptor_type': rec[ 'GABAA_1' ] }
    model_list.append(('tsodyks_synapse', 'MSN_SNR_gaba_p0', params)) 
    
    params = { 'U'      : 0.0192,
              'tau_fac' : 623., 
              'tau_rec' : 559., 
              'tau_psc' : 5.2,
              'delay'   : delay, 'weight' : MSN_SNR_gaba_p1,
              'receptor_type': rec[ 'GABAA_1' ] }
    model_list.append(('tsodyks_synapse', 'MSN_SNR_gaba_p1', params)) 
    
    params = { 'U'      : 0.0148,
              'tau_fac' : 1342., 
              'tau_rec' : 3133.,
              'tau_psc' : 5.2,
             'delay'   : delay, 'weight' : MSN_SNR_gaba_p2,
             'receptor_type': rec[ 'GABAA_1' ] }
    model_list.append(('tsodyks_synapse', 'MSN_SNR_gaba_p2', params)) 

    # GABAA static weak  
    params = {'delay'         :  delay, 'weight' : MSN_SNR_gaba_s_min ,          
              'receptor_type' : rec[ 'GABAA_1' ] } 
    model_list.append(('static_synapse', 'MSN_SNR_gaba_s_min', params))
    
    
    # GABAA static mid strength
    params = {'delay'         :  delay, 'weight' : MSN_SNR_gaba_s_mid ,         
              'receptor_type' : rec[ 'GABAA_1' ] } 
    model_list.append(('static_synapse', 'MSN_SNR_gaba_s_mid', params))
    
    # GABAA static strong
    params = {'delay'         :  delay, 'weight' : MSN_SNR_gaba_s_max ,         
              'receptor_type' : rec[ 'GABAA_1' ] } 
    model_list.append(('static_synapse', 'MSN_SNR_gaba_s_max', params))
 
 
    #! FSN
    #! ==
        	
    #! FSN to MSN
    #! ---------- 
    #! FSN    MSN    tau_gaba         11.4 ms   (Koos et al. 2004)    
    #! FSN    MSN    g_(peak-gaba)    3.8 nS    (Koos et al. 2004)    
    #! FSN    MSN    t_delay          1.7 ms    n.d.; set as for MSN to MSN    
    #! FSN    MSN    E_gaba           -64 mV    (Bracci & Panzeri 2006)    
    
    delay = 1.7
    
    params = { 'U'      : 0.29, # GABAA plastic
              'tau_fac' : 53.0,
              'tau_rec' : 902.0,
              'tau_psc' : 11.4,
              'weight'  : FSN_MSN_gaba_p, 'delay'   : delay,
              'receptor_type' : rec[ 'GABAA_1' ] }                              
    model_list.append(('tsodyks_synapse', 'FSN_MSN_gaba_p', params))

    params = {'delay'         :  delay, 'weight' : FSN_MSN_gaba_s , # GABAA static
              'receptor_type' : rec[ 'GABAA_1' ] }                              
    model_list.append(('static_synapse', 'FSN_MSN_gaba_s', params))
 
 
    #! STN
    #! ===
    
    #! STN to STN REMARK, not used
    #! ---------------------------
    #! AMPA plastic, At 10 Hz no facilitation and little effect, since Wilson 
    #! 2004 showed that glutamate antagonist did not effect firing frequency 
    #! in vivo
    params = { 'U'       : 0.01, # ampa_p
               'tau_fac' : 90.0,
               'tau_rec' : 1000.0,
               'tau_psc' : 4.0,
               'weight'  : STN_STN_ampa_p , 'delay'   : 1.0,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse' , 'STN_STN_ampa_p' , params))

    
    
    params = { 'delay'         : 2.0, 'weight' : STN_STN_ampa_s , # AMPA static
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('static_synapse' , 'STN_STN_ampa_s' , params))

    params = { 'delay'         : 2.0, 'weight' : STN_STN_nmda_s , # NMDA static
               'receptor_type' : rec[ 'NMDA' ] } 
    model_list.append(('static_synapse' , 'STN_STN_nmda_s' , params))


    #! STN to GPE
    #! -----------
    #! STN    GPE    tau_ampa           12 ms    (Hanson & Dieter Jaeger 2002)
    #! STN    GPE    tau_nmda           100 ms    n.d.; estimated    
    #! STN    GPE    g_(peak-ampa)      0.6 nS   (Hanson & Dieter Jaeger 2002)
    #! STN    GPE    g_(peak-nmda)      0.01 nS   n.d; estimated
    #! STN    GPE    t_delay            2.8 ms    (H Kita & Kitai 1991)
    #! STN    GPE    E_ampa,E_nmda      0 mV      n.d.; same as CTX to STN

    delay = 5.
    tau_ampa = 12.

    #! AMPA plastic 0, 1 and 2, parameter tau_fac and tau_rec from Hanson and 
    #! Jaeger 2002. Three types of short term plasticity with approximatly 
    #! equal number of recordes cells of each type.
    params = { 'U'       : 0.02, # AMPA plastic 0 
               'tau_fac' : 241.0,
               'tau_rec' : 491.0,
               'tau_psc' : tau_ampa,
               'delay'   :delay, 'weight'  : STN_GPE_ampa_p0,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse' , 'STN_GPE_ampa_p0', params))
    
    params = { 'U'       : 0.05, # AMPA plastic 1
               'tau_fac' : 345.0,
               'tau_rec' : 700.0,
               'tau_psc' : tau_ampa,
               'delay'   : delay, 'weight'  : STN_GPE_ampa_p1,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse' , 'STN_GPE_ampa_p1', params))

    params = { 'U'       : 0.3, # AMPA plastic 2   
               'tau_fac' : 148.0,
               'tau_rec' : 764.0,
               'tau_psc' : tau_ampa,
               'delay'   : delay, 'weight' : STN_GPE_ampa_p2 ,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse', 'STN_GPE_ampa_p2', params))
  
    params = { 'delay'         : delay, 'weight' : STN_GPE_ampa_s , # AMPA static         
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('static_synapse', 'STN_GPE_ampa_s', params))
 
    params = {'delay'         : delay, 'weight' : STN_GPE_nmda_s , # NMDA static   
              'receptor_type' : rec[ 'NMDA' ] } 
    model_list.append(('static_synapse' , 'STN_GPE_nmda_s' , params))

    
    #! STN to SNR
    #! -----------  
    #! STN    SNR    tau_ampa            12 ms      n.d.; set as for STN to GPE    
    #! STN    SNR    tau_nmda            100 ms     n.d.; estimated    
    #! STN    SNR    g_(peak-ampa)       0.5 nS    (Shen & Johnson 2006)    
    #! STN    SNR    g_(peak-nmda)       0.01 nS    n.d.; same ratio ampa/nmda as for MSN    
    #! STN    SNR    t_delay             4.6 ms    (Shen & Johnson 2006)    
    #! STN    SNR    E_(ampa,) E_nmda    0 mV       n.d. same as CTX to STN    

    delay = 4.6
    tau_ampa = 12. 
    
    #! AMPA plastic 0, 1 and 2, parameter tau_fac and tau_rec from Hanson and 
    #! Jaeger 2002. Three types of short term plasticity with approximatly 
    #! equal number of recordes cells of each type.
    params = { 'U'       : 0.02, # AMPA plastic 0 
               'tau_fac' : 241.0,
               'tau_rec' : 491.0,
               'tau_psc' : tau_ampa,
               'delay'   : delay, 'weight'  : STN_SNR_ampa_p0,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse' , 'STN_SNR_ampa_p0' , params))

    
    params = { 'U'       : 0.05, # AMPA plastic 1
               'tau_fac' : 345.0,
               'tau_rec' : 700.0,
               'tau_psc' : tau_ampa,
               'delay'   : delay, 'weight'  : STN_SNR_ampa_p1,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse' , 'STN_SNR_ampa_p1' , params))

    params = { 'U'       : 0.3, # AMPA plastic 2   
               'tau_fac' : 148.0,
               'tau_rec' : 764.0,
               'tau_psc' : tau_ampa,
               'delay'   : delay, 'weight' : STN_SNR_ampa_p2 ,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse', 'STN_SNR_ampa_p2', params))

    params = { 'U'       : 0.35, # AMPA plastic 2   
               'tau_fac' : 0.0,
               'tau_rec' : 800.0,
               'tau_psc' : tau_ampa,
               'delay'   : delay, 'weight' : STN_SNR_ampa_p3 ,
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('tsodyks_synapse', 'STN_SNR_ampa_p3', params))
     
    
    params = {'delay'         : delay, 'weight' : STN_SNR_ampa_s, # AMPA static
              'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('static_synapse' , 'STN_SNR_ampa_s' , params))
 
 
    
    params = {'delay'         : delay, 'weight' : STN_SNR_nmda_s, # NMDA static
              'receptor_type' : rec[ 'NMDA' ] } 
    model_list.append(('static_synapse' , 'STN_SNR_nmda_s' , params))
 

    #! GPE
    #! ====
    #! No plasticity seen in the GPE to GPE. (Sims 2008) 

    #! GPE to FSN
    #! -----------
    #! GPE    FSN    tau_gaba       2,1 ms    n.d. set as for GPE to SNR
    #! GPE    FSN    g_(peak-gaba)  -         n.d. tuned to achieve realistic firing rates
    #! GPE    FSN    t_delay        7 ms      n.d.; set as for MSN to GPE
    #! GPE    FSN    E_gaba        -64 mV     n.d.; set as for MSNs
   
    params = { 'delay'         : 5.0, 'weight' : GPE_FSN_gaba_s, # GABA static
               'receptor_type' : rec[ 'GABAA_1' ] } 
    model_list.append(('static_synapse' , 'GPE_FSN_gaba_s' , params))
    
    
    #! GPE to STN
    #! -----------
    #! GPE    STN    tau_gaba       7.8 ms   (Baufreton et al. 2009)
    #! GPE    STN    g_(peak-gaba)  11.3 nS  (Baufreton et al. 2009)
    #! GPE    STN    t_delay        4.8 ms   (Baufreton et al. 2009)
    #! GPE    STN    E_gaba        -84 mV    (Baufreton et al. 2009)

    params = { 'delay'         : 4.8, 'weight' : GPE_STN_gaba_s, # GABA static

               'receptor_type' : rec[ 'GABAA_1' ] } 
    model_list.append(('static_synapse' , 'GPE_STN_gaba_s' , params))

 
    #! GPE to GPE
    #! -----------
    #! GPE    GPE    tau_gaba         4.2 ms            (Sims et al. 2008)
    #! GPE    GPE    g_(peak-gaba)    1.8 - 2.5 ns      (Sims et al. 2008)    
    #! GPE    GPE    t_delay          1 ms               n.d.; estimated on proximity of cells
    #! GPE    GPE    E_gaba           -65 mV            (Rav-Acha et al. 2005)

    params = { 'delay'         : 1.0, 'weight' : GPE_GPE_gaba_s, # GABA static
               'receptor_type' : rec[ 'GABAA_2' ] }                             
    model_list.append(('static_synapse' , 'GPE_GPE_gaba_s' , params))
 
 
    #! GPE to SNR
    #! ----------- 
    #! GPE    SNR    tau_gaba       2.1 ms            (Connelly et al. 2010)
    #! GPE    SNR    g_gaba         16 - 70 nS        (Connelly et al. 2010)
    #! GPE    SNR    t_delay        3 ms              (Nakanishi et al. 1991)
    #! GPE    SNR    E_(rev-gaba)   -65 mV            (Connelly et al. 2010)

    delay = 3.
        
    #! GABAA plastic, tuned to Connelly (2010), parameters set such that 
    #! p1/p10 equals 0.4 at 10 Hz and recovery time equals 1760 ms. 
    params = { 'U'      : 0.196, # GABAA plastic,                   
              'tau_fac' : 0.0,
              'tau_psc' : 2.1,
              'tau_rec' : 969.0,
              'delay'   :delay, 'weight'  : GPE_SNR_gaba_p,
              'receptor_type' : rec[ 'GABAA_2' ] } 
      
    model_list.append(('tsodyks_synapse', 'GPE_SNR_gaba_p', params)) 
    
    params = { 'U'      : 0.196, # GABAA plastic,                   
              'tau_fac' : 0.0,
              'tau_psc' : 2.1,
              'tau_rec' : 969.0,
              'delay'   :delay, 'weight'  : GPE_SNR_gaba_p,
              'receptor_type' : rec[ 'GABAA_2' ],
              'm':10 } 
    model_list.append(('tsodyks_stocastic_synapse', 'GPE_SNR_gaba_p_stoc', params)) 
    
    params = { 'delay'         : delay, 'weight' :GPE_SNR_gaba_s_ref, # GABA static
               'receptor_type' : rec[ 'GABAA_2' ] } 
    model_list.append(('static_synapse', 'GPE_SNR_gaba_s_ref', params))
 
    params = { 'delay'         : delay, 'weight' :GPE_SNR_gaba_s_max, # GABA static
               'receptor_type' : rec[ 'GABAA_2' ] } 
    model_list.append(('static_synapse', 'GPE_SNR_gaba_s_max', params))
 
    #! Cortex
    #! ======
    #! Cortex connect to MSN (AMPA ans NMDA), FSN (AMPA) and STN (AMPA and NMDA).
    
    #! Cortex to MSN
    #! -------------
    #! CTX    MSN    tau_ampa        6 ms     (Humphries, Wood, et al. 2009)    
    #! CTX    MSN    tau_nmda        160 ms   (Humphries, Wood, et al. 2009)    
    #! CTX    MSN    g_(peak-ampa)   1 nS     (Humphries, Wood, et al. 2009)    
    #! CTX    MSN    g_(peak-nmda)   0.02 nS  (Humphries, Wood, et al. 2009)    
    #! CTX    MSN    t_delay         10 ms    (Swanson et al. 1996)    
    #! CTX    MSN    E_ampa,E_nmda   0 mV     (Humphries, Wood, et al. 2009)    

    delay = 10.

    params = { 'delay'         : delay, 'weight' : CTX_MSN_ampa_s, # AMPA
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('static_synapse' , 'CTX_MSN_ampa_s', params))
 
    params = { 'delay'         : delay, 'weight' : CTX_MSN_nmda_s, # NMDA
               'receptor_type' : rec[ 'NMDA' ] } 
    model_list.append(('static_synapse' , 'CTX_MSN_nmda_s' , params))
 
     
    #! Cortex to FSN
    #! ------------
    #! Cortex to FSN has only AMPA 
    #! CTX    FSN    tau_ampa        6 ms        (Humphries, Wood, et al. 2009)    
    #! CTX    FSN    g_(peak-ampa)   10.2 nS     (Humphries, Wood, et al. 2009)    
    #! CTX    FSN    t_delay         10 ms        n.d.; set as for CTX to MSN    
    #! CTX    FSN    E_ampa,E_nmda   0 mV        (Humphries, Wood, et al. 2009)    

    params = { 'delay'         : 10.0, 'weight' : CTX_FSN_ampa_s, # AMPA
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('static_synapse' , 'CTX_FSN_ampa_s' , params))
    
    
    #! Cortex to STN
    #! -------------
    #! Cortical delay 2.5 ms (Humphrie 2006, check
    #! org ref) 
    #! CTX    STN    tau_ampa         2.5 ms    (Baufreton et al. 2005)
    #! CTX    STN    tau_nmda         100 ms     n.d. estimated 
    #! CTX    STN    g_(peak-ampa)    3.7 nS    (Baufreton et al. 2005)
    #! CTX    STN    g_(peak-nmda)    0.07 nS    n.d.; same ratio ampa/nmda as MSN
    #! CTX    STN    E_ampa           0 mV      (Baufreton et al. 2009)
    #! CTX    STN    t_delay          2.5 ms    (Fujimoto & H Kita 1993)
    #! CTX    STN    E_nmda           0 mV        n.d.; set as  E_ampa

    delay = 2.5
    
    params = { 'delay'         : delay, 'weight' : CTX_STN_ampa_s, # AMPA 
               'receptor_type' : rec[ 'AMPA' ] } 
    model_list.append(('static_synapse' , 'CTX_STN_ampa_s' , params))

    params = { 'delay'         : delay, 'weight' : CTX_STN_nmda_s, # NMDA
               'receptor_type' : rec[ 'NMDA' ] } 
    model_list.append(('static_synapse' , 'CTX_STN_nmda_s' , params)) 
    
    model_dict={}
    for model in model_list:
        model_dict[model[1]]=model
     
    return model_list, model_dict
def network(model_dict, Params_in={}):
    
    #! Load pynest
    import nest
    import numpy
    import pprint
    import os
    import sys
    from numpy.random import uniform
    from scipy.optimize import fmin
    
    # Get directory where model and code resides 
    model_dir=   '/'.join(os.getcwd().split('/')[0:-1])    
    code_dir=  '/'.join(os.getcwd().split('/')[0:-2])  
    
    # Add model, code and current directories to python path
    sys.path.append(os.getcwd())  
    sys.path.append(model_dir)
    sys.path.append(code_dir+'/nest_toolbox') 
    
              
    # Imports dependent on adding code model and nest_toolbox path
    from model_params import models, network                                  
    from src import my_nest, misc
        
    #! Install nmda neuron model
    try:
    
     nest.Install('/home/mikael/activity-phd/projects/models10/dev/izhik/resources/ml_module/build-ver100725/ml_module')
    
    except:
      
      print 'izhik ml_module already installed'
        
    
    #! Configurable parameters
    #! =======================
    f=1.
    params = {'misc':{'n_actions': 5.},
              'neurons':{'MSN_D1':{'type':'MSN_D1_spk_gen',
                                   'n':25,
                                   'lesion':False},
                         'MSN_D2':{'type':'MSN_D2_spk_gen',
                                   'n':25,
                                   'lesion':False},
                         'STN':{'type':'STN_75_aeif',
                                   'n':50,
                                   'lesion':False},   
                         'GPE':{'type':'GPE_aeif',
                                   'n':100,
                                   'lesion':False}, 
                         'SNR':{'type':'SNR_aeif',
                                   'n':100,
                                   'lesion':False}},
            
            'conns':{'MSN_D1_SNR':{'lesion':False,
                                'n':25.0,
                                'syn':'MSN_SNR_gaba_p1',
                                'source':'MSN_D1',
                                'target':'SNR',
                                'type':'gaba'},   
                     'MSN_D2_GPE':{'lesion':False,
                                'n':25.0,
                                'syn':'MSN_GPE_gaba_p',
                                'source':'MSN_D2',
                                'target':'GPE',
                                'type':'gaba',
                                'lines':True}, 
                     'STN_GPE':{'lesion':False,
                                'n':30.0,
                                'syn':'STN_GPE_ampa_s',
                                'source':'STN',
                                'target':'GPE',
                                'type':'ampa'}, 
                     'GPE_GPE':{'lesion':False,
                                'n':30.0,
                                'syn':'GPE_GPE_gaba_s',
                                'source':'GPE',
                                'target':'GPE',
                                'type':'gaba'}, 
                     'GPE_STN':{'lesion':False,
                                'n':30.0,
                                'syn':'GPE_STN_gaba_s',
                                'source':'GPE',
                                'target':'STN',
                                'type':'gaba'}, 
                     'STN_SNR':{'lesion':False,
                                'n':30.0,
                                'syn':'STN_SNR_ampa_s',
                                'source':'STN',
                                'target':'SNR',
                                'type':'gaba'}, 
                     'GPE_SNR':{'lesion':False,
                                'n':30.0,
                                'syn':'GPE_SNR_gaba_p',
                                'source':'GPE',
                                'target':'SNR',
                                'type':'gaba'}}}
    
    #! Update the parameters with input parameter
    params=misc.dict_merge(Params_in, params)

    
    
    
    #! ======
    #! Layers
    #! ======
    #! Here model neuron populations are create and stored in a layer dictionary 
    #! ``layer_dic``. Layer properties are stored in a layer list later used for 
    #! plotting layer connection with connPlotter. One layer is defined for 
    #! MSN, STN and SNR 
    
    
    group_list = []
    group_dict = {}
    new_model_dict={}
    # Distribute neurons randomally
    
    pos, layer_list,connect_list,  dist,f, tau,k={},[],[],{},{},{},{}
    for source in params['neurons'].keys():
        
        source_type=params['neurons'][source]['type']
        n=params['neurons'][source]['n']
        print source, n
        #! ======
        #! Layers
        #! ======
        # Positions and distance of neurons in 2d
        pos[source]=([[ uniform(-0.5 ,0.5), uniform( -0.5 , 0.5 )] for j in xrange(n)])
        dist[source]=numpy.array([numpy.sqrt(x[0]**2+x[1]**2) for x in pos[source]])
                
        # Create layer list
        if not params['neurons'][source]['lesion']:
            
           setup={ 'model': source_type, 'n' : int(n), }
           group_list.append((source, source_type, setup))
           group_dict[source]=[source,source_type, setup]
    
    # ============
    # Connections
    # ============       
    # Create connection list
    connect_list=[]
    for conn in params['conns'].keys():            
            source=params['conns'][conn]['source']
            target=params['conns'][conn]['target']
            syn=params['conns'][conn]['syn']
            n=params['conns'][conn]['n']
                    
            model=model_dict[syn]
            setup={'n':n, 
                    'delays':{'uniform':{'min':0.5*model[2]['delay'],'max':1.5*model[2]['delay']}}, 
                    'weights':{'uniform':{'min':0.5*model[2]['weight'],'max':1.5*model[2]['weight']}},
                    'model':syn, 
                    'options':{'allow_autapses':False, 'allow_multapses':False}}
            
            
            connect_list.append((source, target, setup))
                   
    return group_list, group_dict, connect_list, params
    
       
