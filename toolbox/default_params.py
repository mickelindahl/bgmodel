'''
Module:
lines

Non dependable parameters are defined in konstructor of
the Par class. Dependable parameters are the initilized to
None. The class method update_dependable then updates all the 
dependable parameters.

Parameters are devided hierarchally into chategories:
simu - for simulation
netw - for network
conn - for connections
node - for a node in the networks
nest - parameters for available models and synapses 

simu:
Includes properties related to the simulation, like start and stop of simulation,
nest kernel values, number of threads.

netw:
This cathegory includes properties related to the network as a whole. 
For example size, dopamine level,    

conn:
Inclues parameters related to connectivity. Each connection parameter have a
name.

node:
Parameters related to neurons and inputs.

nest:
Parameters defining nest models, synapses, input and neurons

All parameters are stored in a single dictionary 'dic'. Example:
dic['netw']['size']=1000.0
dic['conn']['C1_M2_ampa']['syn']='C1_M2_ampa' - nest synapses
dic['node']['M1']['model']='M1' - copied nest model

with
dic['nest']['C1_M2_ampa']['weight']=1.0
then variations can be created
dic['nest']['C1_M2_ampa_s']['weight']=1.0*2 #a for ampa, s for static


Abrevation for readability:
C1 - Cortex input MSN D1 node
C2 - Cortex input MSN D2 node
CF - Cortex input FSN node
CS - Cortex input STN node
EA - External input GPe A node
EI - External input GPe I node
ES - External input SNr node
M1 - MSN D1 node
M2 - MSN D2 node
B1 - MSN D1 background node
B2 - MSN D2 background node
FS - FSN node
ST - STN node
GA - GPe type A node
GI - GPe type I node
SN - SNr node

Functions:
models   - define neuron and synapse models
network  - define layers and populations
'''

from copy import deepcopy
from toolbox import misc
from toolbox.network_connectivity import Units_input, Units_neuron
import nest # Has to after misc. 

MODULE_PATH=  '/afs/nada.kth.se/home/w/u1yxbcfw/tools/NEST/dist/install-nest-2.2.2/lib/nest/ml_module'

class Perturbation(object):   
     
    def __init__(self, keys, val, op):
        self.op=op
        self.keys=keys
        self.val=val
        
    def __repr__(self):
        return  '.'.join(self.keys)+self.op+str(self.val)  
     
     
class Pertubation_list(object):
    
    def __init__(self, iterator):
        self.list=[]
        self.applied=False
        if not isinstance(iterator[0], list):
            iterator=[iterator]
        
        for keys, val, op in iterator:
            self.list.append(Perturbation(keys.split('.'), val, op))

    def __repr__(self):
        return  'pl_obj:'+str(self.list)
    
    def __getitem__(self, val):
        return self.list[val]
    
    def update(self, dic, display=False):
        
        dic0=deepcopy(dic)
        for p in self.list:
            dic=misc.dict_recursive_update(dic, p.keys, p.val, p.op)
        
        dic0_red=misc.dict_reduce(dic0, {}, deliminator='.')
        dic_red=misc.dict_reduce(dic, {}, deliminator='.')
        s=''
        for key, val in dic0_red.iteritems():
            if key not in dic_red.keys():
                raise Exception('Key '+key +' gone missing')
            if val!=dic_red[key]:
                s+= key+':'+str(val) + ' -> ' + str(dic_red[key])+' '
        if display:
            print 'pertubations OK '+s 
        
        self.applied=True            
        return dic

    
class Par(object):
    
    def __init__(self, dic_rep={}, perturbations=None ):
        
        self.per=perturbations
        self._dic_con = {} #non dependable parameters
        self._dic_dep = {}
        self._dic_rep = dic_rep # parameters to change
        self.dic_set = False
        self.dic_con_set=dic_rep=={}
        self.dic_dep_set=False
        self.dic_rep_set=dic_rep=={}
        
        self.module_path=MODULE_PATH
       
        if not 'my_aeif_cond_exp' in nest.Models(): nest.Install( self.module_path)
        
        self.rec={}
        self.rec['my_aeif_cond_exp'] = nest.GetDefaults('my_aeif_cond_exp')['receptor_types']   # get receptor types
        self.rec['izhik_cond_exp']   = nest.GetDefaults('izhik_cond_exp')['receptor_types']     # get receptor types
    
        dic={}
        # ========================
        # Default netw parameters 
        # ========================
        
        dic['netw']={} 
        
        # @TODO defining input in par. Make network_construction use this information.
        dic['netw']['input']={'constant' :{'nodes':['C1', 'C2', 'CF', 'CS', 'EA', 'EI', 'ES']}}
        
        dic['netw']['size']=10000.0 
        dic['netw']['tata_dop']  = 0.8
        dic['netw']['tata_dop0'] = 0.8
    
        dic['netw']['prop_GPE_A'] = 0.2
        dic['netw']['prop_fan_in_GPE_A'] = 1/17. # 1/17.    
    
        dic['netw']['target_rate_GPE_A'] = 27.1
    
        dic['netw']['V_th_sigma']=1.0
        
        dic['netw']['sub_sampling']={'MS':1.0} 

        dic['netw']['n_nuclei']={'M1':2791000/2.,
                                 'M2':2791000/2.,
                                 'FS': 0.02*2791000, # 2 % if MSN population
                                 'ST': 13560.,
                                 'GP': 45960.,
                                 'SN': 26320.}
        
        
        
        '''
        n_nuclei={'M1':15000,
               'M2':15000,
               'FS': 0.02*30000, # 2 % if MSN population
               'ST': 100,
               'GP': 300/.8,
               'SN': 300}   
        '''
        
        prop=dic['netw']['n_nuclei'].copy()
        dic['netw']['prop']={}
        dic['netw']['n_nuclei_sub_sampling']={}
        for k in prop.keys(): 
            dic['netw']['prop'].update({k:None})  
            dic['netw']['n_nuclei_sub_sampling'].update({k:None})  
        
        
        
         
        # ========================
        # Default nest parameters 
        # ========================
        # Defining default parameters
        dic['nest']={}
        
        # CTX-FSN 
        dic['nest']['CF_FS_ampa']={}
        dic['nest']['CF_FS_ampa']['weight']   = 0.25    # n.d. set as for CTX to MSN   
        dic['nest']['CF_FS_ampa']['delay']    = 0.12    # n.d. set as for CTX to MSN   
        dic['nest']['CF_FS_ampa']['template'] = 'static_synapse'   
        dic['nest']['CF_FS_ampa']['receptor_type'] = self.rec['izhik_cond_exp'][ 'AMPA_1' ]     # n.d. set as for CTX to MSN   
    
    
        # FSN-FSN
        dic['nest']['FS_FS_gaba']={}
        dic['nest']['FS_FS_gaba']['weight']  = 1.    # five times weaker than FSN-MSN, Gittis 2010
        dic['nest']['FS_FS_gaba']['delay']   = 1.7/0.29    # n.d.same asfor FSN to MSN    
        dic['nest']['FS_FS_gaba']['U']       = 0.29
        dic['nest']['FS_FS_gaba']['tau_fac'] = 53.   
        dic['nest']['FS_FS_gaba']['tau_rec'] = 902.   
        dic['nest']['FS_FS_gaba']['tau_psc'] = 6.     #   Gittis 2010 have    
        dic['nest']['FS_FS_gaba']['template'] = 'tsodyks_synapse'                
        dic['nest']['FS_FS_gaba']['receptor_type'] = self.rec['izhik_cond_exp'][ 'GABAA_1' ]
        
        
        # GPe A-FSN
        dic['nest']['GA_FS_gaba']={}
        dic['nest']['GA_FS_gaba']['weight']   = 1.     # n.d. inbetween MSN and FSN GABAergic synapses
        dic['nest']['GA_FS_gaba']['delay']    = 7.  # n.d. same as MSN to GPE Park 1982
        dic['nest']['GA_FS_gaba']['template'] = 'static_synapse'
        dic['nest']['GA_FS_gaba']['receptor_type'] = self.rec['izhik_cond_exp'][ 'GABAA_2' ]
        
        
        # CTX-MSN D1
        dic['nest']['C1_M1_ampa']={}
        dic['nest']['C1_M1_ampa']['weight']   = .5     # constrained by Ellender 2011
        dic['nest']['C1_M1_ampa']['delay']    = 12.    # Mallet 2005
        dic['nest']['C1_M1_ampa']['template'] = 'static_synapse'
        dic['nest']['C1_M1_ampa']['receptor_type'] = self.rec['izhik_cond_exp'][ 'AMPA_1' ]
        
        dic['nest']['C1_M1_nmda'] = deepcopy(dic['nest']['C1_M1_ampa'])
        dic['nest']['C1_M1_nmda']['weight'] =  0.11   # (Humphries, Wood, and Gurney 2009)
        dic['nest']['C1_M1_nmda']['receptor_type'] = self.rec['izhik_cond_exp'][ 'NMDA_1' ]
        
        
        # CTX-MSN D2
        dic['nest']['C2_M2_ampa'] = deepcopy(dic['nest']['C1_M1_ampa'])
        dic['nest']['C2_M2_ampa']['weight'] =  .41     # constrained by Ellender 2011
        
        dic['nest']['C2_M2_nmda'] = deepcopy(dic['nest']['C1_M1_nmda'])
        dic['nest']['C2_M2_nmda']['weight'] =  0.019   # (Humphries, Wood, and Gurney 2009)
        
        
        # MSN-MSN    
        dic['nest']['M1_M1_gaba'] = {}
        dic['nest']['M1_M1_gaba']['weight']   =  0.2    # Koos 2004 %Taverna 2004
        dic['nest']['M1_M1_gaba']['delay']    =  1.7    # Taverna 2004          
        dic['nest']['M1_M1_gaba']['template'] = 'static_synapse'
        dic['nest']['M1_M1_gaba']['receptor_type'] = self.rec['izhik_cond_exp'][ 'GABAA_2' ]
        
        dic['nest']['M1_M2_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M2_M1_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M2_M2_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        
        
        # FSN-MSN
        dic['nest']['FS_M1_gaba']={}
        dic['nest']['FS_M1_gaba']['weight']  = round(6./0.29,1)     # Gittie #3.8    # (Koos, Tepper, and Charles J Wilson 2004)
        dic['nest']['FS_M1_gaba']['delay']   = 1.7    # Taverna 2004       
        dic['nest']['FS_M1_gaba']['U']       = 0.29     # GABAA plastic
        dic['nest']['FS_M1_gaba']['tau_fac'] = 53.0
        dic['nest']['FS_M1_gaba']['tau_rec'] = 902.0
        dic['nest']['FS_M1_gaba']['tau_psc'] = 8.0    # ?  Gittis 2010
        dic['nest']['FS_M1_gaba']['template'] = 'tsodyks_synapse' 
        dic['nest']['FS_M1_gaba']['receptor_type'] = self.rec['izhik_cond_exp'][ 'GABAA_1' ]   
        
        dic['nest']['FS_M2_gaba'] = deepcopy(dic['nest']['FS_M1_gaba'])
        
        
        # GPE-MSN    
        dic['nest']['GA_M1_gaba']={}
        dic['nest']['GA_M1_gaba']['weight']   = 1.  # n.d. inbetween MSN and FSN GABAergic synapses    
        dic['nest']['GA_M1_gaba']['delay']    = 1.7 
        dic['nest']['GA_M1_gaba']['template'] = 'static_synapse'  
        dic['nest']['GA_M1_gaba']['receptor_type'] = self.rec['izhik_cond_exp'][ 'GABAA_3' ]   
        
        dic['nest']['GA_M2_gaba'] = deepcopy(dic['nest']['GA_M1_gaba'])
       
            
        # CTX-STN
        dic['nest']['CS_ST_ampa']={}
        dic['nest']['CS_ST_ampa']['weight']   = 0.25
        dic['nest']['CS_ST_ampa']['delay']       = 2.5  # Fujimoto and Kita 1993
        dic['nest']['CS_ST_ampa']['template'] = 'static_synapse'  
        dic['nest']['CS_ST_ampa']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'AMPA_1' ]  
        
        dic['nest']['CS_ST_nmda'] = deepcopy(dic['nest']['CS_ST_ampa'])
        dic['nest']['CS_ST_nmda']['weight'] = 0.00625   # n.d.; same ratio ampa/nmda as MSN
        dic['nest']['CS_ST_nmda']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'NMDA_1' ]  
        
        
        # GPe I-STN  
        dic['nest']['GI_ST_gaba']={}
        dic['nest']['GI_ST_gaba']['weight'] = .08    # n.d.
        dic['nest']['GI_ST_gaba']['delay'] =  5.
        dic['nest']['GI_ST_gaba']['template'] = 'static_synapse' 
        dic['nest']['GI_ST_gaba']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'GABAA_1' ]  
        
        
        # EXT-GPe
        dic['nest']['EA_GA_ampa']={}
        dic['nest']['EA_GA_ampa']['weight']   = 0.167
        dic['nest']['EA_GA_ampa']['delay']    = 5.
        dic['nest']['EA_GA_ampa']['template'] = 'static_synapse'  
        dic['nest']['EA_GA_ampa']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'AMPA_2' ]  
        
        dic['nest']['EI_GI_ampa'] = deepcopy(dic['nest']['EA_GA_ampa'])
    
        
        # GPe-GPe
        dic['nest']['GA_GA_gaba']={}
        dic['nest']['GA_GA_gaba']['weight']   = 1.3    # constrained by (Sims et al. 2008)
        dic['nest']['GA_GA_gaba']['delay']    = 1.     #n.d. assumed due to approximity   
        dic['nest']['GA_GA_gaba']['template'] = 'static_synapse'  
        dic['nest']['GA_GA_gaba']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'GABAA_2' ]
        
        dic['nest']['GA_GI_gaba'] = deepcopy(dic['nest']['GA_GA_gaba'])
        dic['nest']['GI_GA_gaba'] = deepcopy(dic['nest']['GA_GA_gaba'])
        dic['nest']['GI_GI_gaba'] = deepcopy(dic['nest']['GA_GA_gaba']) 
    
         
        # MSN D2-GPe I 
        dic['nest']['M2_GI_gaba']={}
        dic['nest']['M2_GI_gaba']['weight']  = 2./0.24   # constrained by (Sims et al. 2008)
        dic['nest']['M2_GI_gaba']['delay']   = 7.       # Park 1982
        dic['nest']['M2_GI_gaba']['U']       = 0.24                                                   # GABAA plastic                   
        dic['nest']['M2_GI_gaba']['tau_fac'] = 13.0
        dic['nest']['M2_GI_gaba']['tau_rec'] = 77.0
        dic['nest']['M2_GI_gaba']['tau_psc'] = 6.    # (Shen et al. 2008)
        dic['nest']['M2_GI_gaba']['template'] = 'tsodyks_synapse' 
        dic['nest']['M2_GI_gaba']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'GABAA_1' ]         
     
        
        # STN-GPe
        dic['nest']['ST_GA_ampa']={}
        dic['nest']['ST_GA_ampa']['weight']   = 0.35     # constrained by (Hanson & Dieter Jaeger 2002)
        dic['nest']['ST_GA_ampa']['delay']    = 5.       # Ammari 2010
        dic['nest']['ST_GA_ampa']['template'] = 'static_synapse' 
        dic['nest']['ST_GA_ampa']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'AMPA_1' ]         
        
        dic['nest']['ST_GI_ampa'] = deepcopy(dic['nest']['ST_GA_ampa']) 
        
        
        # EXR-SNr
        dic['nest']['ES_SN_ampa']={}
        dic['nest']['ES_SN_ampa']['weight']   = 0.5
        dic['nest']['ES_SN_ampa']['delay']    = 5.0
        dic['nest']['ES_SN_ampa']['template'] = 'static_synapse'  
        dic['nest']['ES_SN_ampa']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'AMPA_2' ]  
     
        
        # MSN D1-SNr 
        dic['nest']['M1_SN_gaba']={}
        dic['nest']['M1_SN_gaba']['weight']   = 2./0.0192 # Lower based on (Connelly et al. 2010) = [4.7, 24.], 50 Hz model = [5.8, 23.5]
        dic['nest']['M1_SN_gaba']['delay']    = 7.3 
        dic['nest']['M1_SN_gaba']['U']        = 0.0192
        dic['nest']['M1_SN_gaba']['tau_fac']  = 623. 
        dic['nest']['M1_SN_gaba']['tau_rec']  = 559. 
        dic['nest']['M1_SN_gaba']['tau_psc']  = 5.2   
        dic['nest']['M1_SN_gaba']['template'] = 'tsodyks_synapse' 
        dic['nest']['M1_SN_gaba']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'GABAA_1' ] 
    
        
        # STN-SNr
        dic['nest']['ST_SN_ampa']={}
        dic['nest']['ST_SN_ampa']['weight']   = 0.91*3.8/0.35      # (Shen and Johnson 2006)
        dic['nest']['ST_SN_ampa']['delay']    = 4.6 
        dic['nest']['ST_SN_ampa']['U']        = 0.35 # AMPA plastic 2   
        dic['nest']['ST_SN_ampa']['tau_fac']  = 0.0
        dic['nest']['ST_SN_ampa']['tau_rec']  = 800.0
        dic['nest']['ST_SN_ampa']['tau_psc']  = 12.   # n.d.; set as for STN to GPE,
        dic['nest']['ST_SN_ampa']['template'] = 'tsodyks_synapse'
        dic['nest']['ST_SN_ampa']['receptor_type']= self.rec['my_aeif_cond_exp'] [ 'AMPA_1' ] 
     
        
        # GPe-SNr
        dic['nest']['GI_SN_gaba']={}
        dic['nest']['GI_SN_gaba']['weight']   = 76./0.196  #0.152*76., (Connelly et al. 2010)
        dic['nest']['GI_SN_gaba']['delay']    = 3.  
        dic['nest']['GI_SN_gaba']['U']        = 0.196 # GABAA plastic,                   
        dic['nest']['GI_SN_gaba']['tau_fac']  = 0.0
        dic['nest']['GI_SN_gaba']['tau_rec']  = 969.0
        dic['nest']['GI_SN_gaba']['tau_psc']  = 2.1    # (Connelly et al. 2010),
        dic['nest']['GI_SN_gaba']['template'] = 'tsodyks_synapse'   
        dic['nest']['GI_SN_gaba']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'GABAA_2' ] 
          
    
        # ============        
        # Input Models
        # ============ 
        
        dic['nest']['poisson_generator']={}
        dic['nest']['poisson_generator']['template']='poisson_generator'
        dic['nest']['poisson_generator']['rate']=0.0
        
        #CTX-MSN D1
        dic['nest']['C1']={}
        dic['nest']['C1']['template']='poisson_generator'
        dic['nest']['C1']['rate']=0.0
        
        #CTX-MSN D2
        dic['nest']['C2']={}
        dic['nest']['C2']['template']='poisson_generator'
        dic['nest']['C2']['rate']=0.0
    
        #CTX-FSN
        dic['nest']['CF']={}
        dic['nest']['CF']['template']='poisson_generator'
        dic['nest']['CF']['rate']=0.0
        
        #CTX-STN
        dic['nest']['CS']={}
        dic['nest']['CS']['template']='poisson_generator' 
        dic['nest']['CS']['rate']=0.0
      
        #EXT-GPe type A
        dic['nest']['EA']={}
        dic['nest']['EA']['template']='poisson_generator' 
        dic['nest']['EA']['rate']=0.0
        
        #EXT-GPe type I
        dic['nest']['EI']={}
        dic['nest']['EI']['template']='poisson_generator' 
        dic['nest']['EI']['rate']=0.0
    
        #EXT-SNr
        dic['nest']['ES']={}
        dic['nest']['ES']['template']='poisson_generator' 
        dic['nest']['ES']['rate']=0.0
    
    
        # =============        
        # Neuron Models
        # =============
    
        # MSN
        # ===
        dic['nest']['MS']={}    
        dic['nest']['MS']['template'] = 'izhik_cond_exp'
    
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
    
        
      
        # FSN
        # ===
        
        dic['nest']['FS']={}
        dic['nest']['FS']['template'] = 'izhik_cond_exp'
    
        dic['nest']['FS']['a']      = 0.2    # (E.M. Izhikevich 2007)
        dic['nest']['FS']['b_1']    = 0.0  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['b_2']    = 0.025  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['c']      = -60.   # (Tateno et al. 2004)
        dic['nest']['FS']['C_m']    = 80.    # (Tateno et al. 2004)
        dic['nest']['FS']['d']      = 0.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['E_L']    = -70.   #*(1-0.8*0.1)   # (Tateno et al. 2004)
        dic['nest']['FS']['k']      = 1.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['p_1']    = 1.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['p_2']    = 3.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['V_b']    = -55.   # Humphries 2009
        dic['nest']['FS']['V_peak'] = 25.    # (E.M. Izhikevich 2007)
        dic['nest']['FS']['V_th']   = -50.   # (Tateno et al. 2004)
    
        dic['nest']['FS']['AMPA_1_Tau_decay'] = 12.   # CTX to FSN ampa
        dic['nest']['FS']['AMPA_1_E_rev']    =  0.   # n.d. set as for  CTX to MSN
         
        # From FSN
        dic['nest']['FS']['GABAA_1_E_rev']    = -74.     # n.d.; set as for MSNs
        dic['nest']['FS']['GABAA_1_Tau_decay'] = None
          
        # From GPe
        dic['nest']['FS']['GABAA_2_Tau_decay'] =   6.  # n.d. set as for FSN
        dic['nest']['FS']['GABAA_2_E_rev']    = -74.  # n.d. set as for MSNs
          
        dic['nest']['FS']['beta_E_L'] = 0.078
        dic['nest']['FS']['tata_dop'] = None
        
        dic['nest']['FS']['beta_I_GABAA_1'] = 0.8 # From FSN
        dic['nest']['FS']['beta_I_GABAA_2'] = 0.8 # From GPe A
    
        
        dic['nest']['FS_low']  = deepcopy(dic['nest']['FS'])
        dic['nest']['FS_high'] = deepcopy(dic['nest']['FS'])
        dic['nest']['FS_high']['GABAA_1_E_rev']  = -64.     # n.d. set as for MSNs
        dic['nest']['FS_high']['GABAA_2_E_rev']  = -64.     # n.d. set as for MSNs
          

        # STN
        # ===
       
        dic['nest']['ST']={}
        dic['nest']['ST']['template'] = 'my_aeif_cond_exp'
        
        dic['nest']['ST']['tau_w']    =333.0 # I-V relation, spike frequency adaptation
        dic['nest']['ST']['a_1']      =  0.3    # I-V relation
        dic['nest']['ST']['a_2']      =  0.0      # I-V relation
        dic['nest']['ST']['b']        =  0.05    #0.1 #0.1#200./5.                                                     
        dic['nest']['ST']['C_m']      = 60.0    # t_m/R_in
        dic['nest']['ST']['Delta_T']  = 16.2                      
        dic['nest']['ST']['g_L']      = 10.0
        dic['nest']['ST']['E_L']      =-80.2                                                               
        dic['nest']['ST']['I_e']      =  6.0
        dic['nest']['ST']['V_peak']   = 15.0                                                                
        dic['nest']['ST']['V_reset']  =-70.0    # I-V relation
        dic['nest']['ST']['V_a']      =-70.0 # I-V relation
        dic['nest']['ST']['V_th']     =-64.0                                                               
        
        dic['nest']['ST']['V_reset_slope1']     = -10. # Slope u<0 
        dic['nest']['ST']['V_reset_slope2']     = .0 #  Slope u>=0
        dic['nest']['ST']['V_reset_max_slope1'] = -60. # Max v restet u<0  
        dic['nest']['ST']['V_reset_max_slope2'] = dic['nest']['ST']['V_reset'] # Max v restet u>=0  
       
        dic['nest']['ST']['AMPA_1_Tau_decay'] = 4.0  # (Baufreton et al. 2005)
        dic['nest']['ST']['AMPA_1_E_rev']     = 0.   # (Baufreton et al. 2009)
        
        dic['nest']['ST']['NMDA_1_Tau_decay'] = 160.   # n.d. estimated 1:2 AMPA:NMDA
        dic['nest']['ST']['NMDA_1_E_rev']     =   0.   # n.d.; set as  E_ampa
        dic['nest']['ST']['NMDA_1_Vact']      = -20.0
        dic['nest']['ST']['NMDA_1_Sact']      =  16.0
        
        dic['nest']['ST']['GABAA_1_Tau_decay'] =   8.   # (Baufreton et al. 2009)
        dic['nest']['ST']['GABAA_1_E_rev']     = -84.0  # (Baufreton et al. 2009)
    
        dic['nest']['ST']['beta_I_AMPA_1']  = 0.4 # From Cortex
        dic['nest']['ST']['beta_I_NMDA_1']  = 0.4 # From Cortex
        dic['nest']['ST']['beta_I_GABAA_1'] = 0.4 # From GPe I 
        
        dic['nest']['ST']['tata_dop'] = None
    
        
        # GPE
        # ===
    
        dic['nest']['GP']={}
        dic['nest']['GP']['template'] = 'my_aeif_cond_exp'
    
        dic['nest']['GP']['a_1']       =  2.5    # I-V relation # I-V relation
        dic['nest']['GP']['a_2']       =  dic['nest']['GP']['a_1'] 
        dic['nest']['GP']['b']       = 70.   # I-F relation
        dic['nest']['GP']['C_m']     = 40.  # t_m/R_in
        dic['nest']['GP']['Delta_T'] =  1.7                      
        dic['nest']['GP']['g_L']     =   1.
        dic['nest']['GP']['E_L']     = -55.1  # v_t    = -56.4                                                               #
        dic['nest']['GP']['I_e']     =  0.
        dic['nest']['GP']['tau_w']   = 20.  # I-V relation, spike frequency adaptation
        dic['nest']['GP']['V_peak']  =  15.  # Cooper and standford
        dic['nest']['GP']['V_reset'] = -60.  # I-V relation
        dic['nest']['GP']['V_th']    = -54.7
        dic['nest']['GP']['V_a']     = dic['nest']['GP']['E_L']
    
        
        # STN-GPe
        dic['nest']['GP']['AMPA_1_Tau_decay'] = 12.   # (Hanson & Dieter Jaeger 2002)
        dic['nest']['GP']['AMPA_1_E_rev']     = 0.    # n.d.; same as CTX to STN
        
        dic['nest']['GP']['NMDA_1_Tau_decay'] = 100.  # n.d.; estimated
        dic['nest']['GP']['NMDA_1_E_rev']     = 0.    # n.d.; same as CTX to STN
        dic['nest']['GP']['NMDA_1_Vact']      = -20.0
        dic['nest']['GP']['NMDA_1_Sact']      =  16.0
            
        #EXT-GPe
        dic['nest']['GP']['AMPA_2_Tau_decay'] = 5.0
        dic['nest']['GP']['AMPA_2_E_rev']     = 0.0
        
        # GPe-GPe
        dic['nest']['GP']['GABAA_2_Tau_decay']  = 5.  # (Sims et al. 2008)
        dic['nest']['GP']['GABAA_2_E_rev']     = -65.  # n.d same as for MSN (Rav-Acha 2005)       
        
    
        dic['nest']['GP']['beta_E_L'] = 0.181
        dic['nest']['GP']['beta_V_a'] = 0.181
        dic['nest']['GP']['beta_I_AMPA_1']  = 0.4 # From GPe A
        dic['nest']['GP']['beta_I_GABAA_2'] = 0.8 # From GPe A
    
        dic['nest']['GP']['tata_dop'] = None
        
        dic['nest']['GA']  = deepcopy(dic['nest']['GP'])
        dic['nest']['GI']  = deepcopy(dic['nest']['GP'])
        
        #MSN D2-GPe
        dic['nest']['GI']['GABAA_1_E_rev']     = -65.  # (Rav-Acha et al. 2005)
        dic['nest']['GI']['GABAA_1_Tau_decay'] = None     # (Shen et al. 2008)    

    
        # SNR
        # ===
     
        dic['nest']['SN']={}
        dic['nest']['SN']['template'] = 'my_aeif_cond_exp'
        
        dic['nest']['SN']['tau_w']   = 20.  # I-V relation, spike frequency adaptation
        dic['nest']['SN']['a_1']     =  3.      # I-V relation
        dic['nest']['SN']['a_2']     =  dic['nest']['SN']['a_1']      # I-V relation
        dic['nest']['SN']['b']       = 200.   # I-F relation
        dic['nest']['SN']['C_m']     =  80.    # t_m/R_in
        dic['nest']['SN']['Delta_T'] =  1.8                      
        dic['nest']['SN']['g_L']     =   3.
        dic['nest']['SN']['E_L']     = -55.8    #
        dic['nest']['SN']['I_e']     = 15.0 
        dic['nest']['SN']['V_peak']  =  20.                                                               # 
        dic['nest']['SN']['V_reset'] = -65.    # I-V relation
        dic['nest']['SN']['V_th']    = -55.2    # 
        dic['nest']['SN']['V_a']     = dic['nest']['SN']['E_L']     # I-V relation
        
        #STN-SNr
        dic['nest']['SN']['AMPA_1_Tau_decay'] =  12.   # n.d.; set as for STN to GPE
        dic['nest']['SN']['AMPA_1_E_rev']     =   0.   # n.d. same as CTX to STN
    
        # EXT-SNr
        dic['nest']['SN']['AMPA_2_Tau_decay'] = 5.0
        dic['nest']['SN']['AMPA_2_E_rev']     = 0.
        
        # MSN D1-SNr
        dic['nest']['SN']['GABAA_1_E_rev']     = -80.     # (Connelly et al. 2010)
        dic['nest']['SN']['GABAA_1_Tau_decay'] = None      # (Connelly et al. 2010)
 
        # GPe-SNr
        dic['nest']['SN']['GABAA_2_E_rev']     = -72.     # (Connelly et al. 2010)
        dic['nest']['SN']['GABAA_2_Tau_decay'] = None
    
        dic['nest']['SN']['beta_E_L'] = 0.0896
        dic['nest']['SN']['beta_V_a'] = 0.0896
     
        dic['nest']['SN']['tata_dop'] = None
        
        
        
        # ========================
        # Default node parameters 
        # ========================
        
        dic['node']={}
    
        
        # Model inputs
        inputs={'C1': {'model':'C1',  'rate':530.},
                'C2': {'model':'C2',  'rate':690.}, 
                'CF': {'model':'CF',  'rate':1010.},
                'CS': {'model':'CS',  'rate':160.},#295 
                'EA': {'model':'EA',  'rate':1130.},
                'EI': {'model':'EI',  'rate':1130.},
                'ES': {'model':'ES',  'rate':1800.}}#295 
        
        for key in inputs.keys():       
            inputs[key].update({'extent':[-0.5, 0.5],'edge_wrap':True, 'n':None,
                                'lesion':False, 'type':'input', 'n_sets':1,
                                'unit_class':Units_input })
                
        dic['node']=misc.dict_merge(dic['node'], inputs)
        
           
        network={'M1':{'model':'M1_low', 'n_sets':1, 'I_vitro':0.0, 'I_vivo':0.0,  'target_rate':0.1},
                 'M2':{'model':'M2_low', 'n_sets':1, 'I_vitro':0.0, 'I_vivo':0.0,  'target_rate':0.1},
                 'FS':{'model':'FS_low', 'n_sets':1, 'I_vitro':0.0, 'I_vivo':0.0,  'target_rate':20.0},
                 'ST':{'model':'ST',     'n_sets':1, 'I_vitro':6.0, 'I_vivo':6.0,  'target_rate':10.0},
                 'GA':{'model':'GA',     'n_sets':1, 'I_vitro':5.0, 'I_vivo':-3.6, 'target_rate':None}, #23, -8
                 'GI':{'model':'GI',     'n_sets':1, 'I_vitro':5.0, 'I_vivo':4.5,  'target_rate':None}, #51, 56
                 'SN':{'model':'SN',     'n_sets':1, 'I_vitro':15.0,'I_vivo':19.2, 'target_rate':30.0}}
        
        network['M1'].update({'target_rate_in_vitro':0.0})
        network['M2'].update({'target_rate_in_vitro':0.0})
        network['FS'].update({'target_rate_in_vitro':0.0})
        network['ST'].update({'target_rate_in_vitro':10.0})
        network['GA'].update({'target_rate_in_vitro':4.0})
        network['GI'].update({'target_rate_in_vitro':15.0})
        network['SN'].update({'target_rate_in_vitro':15.0})
        
    
        
        # Randomization of C_m and V_m
        for key in network.keys():     
            network[key].update({'type':'network', 'extent':[-0.5, 0.5],
                                 'edge_wrap':True, 'lesion':False,
                                 'unit_class': Units_neuron,  
                                 'prop':None, 
                                 'n':None,
                                 'randomization':{ 'C_m': {'gaussian':{'sigma':None, 'my':None}},
                                                   'V_th':{'gaussian':{'sigma':None, 'my':None, 
                                                                       'cut':True, 'cut_at':3.}},
                                                   'V_m': {'uniform': {'min': None,  'max':None }}
                                                   }})  
        dic['node']=misc.dict_merge(dic['node'], network)   
        
   
        # ========================
        # Default conn parameters 
        # ========================
        
        
        # Input
        conns={'C1_M1_ampa':{ 'syn':'C1_M1_ampa' },
               'C1_M1_nmda':{ 'syn':'C1_M1_nmda' },
               'C2_M2_ampa':{ 'syn':'C2_M2_ampa' },
               'C2_M2_nmda':{ 'syn':'C2_M2_nmda' },
               'CF_FS_ampa':{ 'syn':'CF_FS_ampa' },
               'CS_ST_ampa':{ 'syn':'CS_ST_ampa' },
               'CS_ST_nmda':{ 'syn':'CS_ST_nmda' },
               'EA_GA_ampa':{ 'syn':'EA_GA_ampa' },
               'EI_GI_ampa':{ 'syn':'EI_GI_ampa' },
               'ES_SN_ampa':{ 'syn':'ES_SN_ampa' }}
        for key in conns.keys():
            conns[key].update({'fan_in0':1,  'rule':'1-1' })
        
        dic['conn']={}
        # Number of incomming connections from a nucleus to another nucleus.
        dic['conn']['M1_SN_gaba']={'fan_in0': 500}
        dic['conn']['M2_GI_gaba']={'fan_in0': 500}
        dic['conn']['M1_M1_gaba']={'fan_in0': int(round(2800*0.1/2.0))}
        dic['conn']['M1_M2_gaba']={'fan_in0': dic['conn']['M1_M1_gaba']['fan_in0']}
        dic['conn']['M2_M1_gaba']={'fan_in0': dic['conn']['M1_M1_gaba']['fan_in0']}
        dic['conn']['M2_M2_gaba']={'fan_in0': dic['conn']['M1_M1_gaba']['fan_in0']}
        dic['conn']['FS_M1_gaba']={'fan_in0': 10}
        dic['conn']['FS_M2_gaba']={'fan_in0': dic['conn']['FS_M1_gaba']['fan_in0'] }
        dic['conn']['FS_FS_gaba']={'fan_in0': 10}
        dic['conn']['ST_GA_ampa']={'fan_in0': 30}
        dic['conn']['ST_GI_ampa']={'fan_in0': 30}
        dic['conn']['ST_SN_ampa']={'fan_in0': 30}
        dic['conn']['GA_FS_gaba']={'fan_in0': 10}
        dic['conn']['GA_M1_gaba']={'fan_in0': 10}
        dic['conn']['GA_M2_gaba']={'fan_in0': 10}
        dic['conn']['GA_GA_gaba']={'fan_in0': None}
        dic['conn']['GA_GI_gaba']={'fan_in0': None}
        dic['conn']['GI_GA_gaba']={'fan_in0': None}
        dic['conn']['GI_GI_gaba']={'fan_in0': None}
        dic['conn']['GI_ST_gaba']={'fan_in0': 30}
        dic['conn']['GI_SN_gaba']={'fan_in0': 32}
    
        # Network        
        conns.update(
               {'M1_SN_gaba':{ 'syn':'M1_SN_gaba', 'rule':'set-set' },
                'M2_GI_gaba':{ 'syn':'M2_GI_gaba', 'rule':'set-set' },
                       
                'M1_M1_gaba':{ 'syn':'M1_M1_gaba', 'rule':'set-not_set' },
                'M1_M2_gaba':{ 'syn':'M1_M2_gaba', 'rule':'set-not_set' },                     
                'M2_M1_gaba':{ 'syn':'M2_M1_gaba', 'rule':'set-not_set' },
                'M2_M2_gaba':{ 'syn':'M2_M2_gaba', 'rule':'set-not_set' },                     
               
                'FS_M1_gaba':{ 'syn':'FS_M1_gaba', 'rule':'all' },
                'FS_M2_gaba':{ 'syn':'FS_M2_gaba', 'rule':'all', 'beta_fan_in':0.8},                       
                'FS_FS_gaba':{ 'syn':'FS_FS_gaba', 'rule':'all' },
              
                'ST_GA_ampa':{ 'syn':'ST_GA_ampa', 'rule':'all' },
                'ST_GI_ampa':{ 'syn':'ST_GI_ampa', 'rule':'all' },
                'ST_SN_ampa':{ 'syn':'ST_SN_ampa', 'rule':'all' },
               
                'GA_FS_gaba':{ 'syn':'GA_FS_gaba', 'rule':'all' },
                'GA_M1_gaba':{ 'syn':'GA_M1_gaba', 'rule':'all' },
                'GA_M2_gaba':{ 'syn':'GA_M2_gaba', 'rule':'all' },
                'GA_GA_gaba':{ 'syn':'GA_GA_gaba', 'rule':'all' }, 
                'GA_GI_gaba':{ 'syn':'GA_GI_gaba', 'rule':'all' },
                'GI_GI_gaba':{ 'syn':'GI_GI_gaba', 'rule':'all' },
                'GI_GA_gaba':{ 'syn':'GI_GA_gaba', 'rule':'all' },
                
                'GI_ST_gaba':{ 'syn':'GI_ST_gaba', 'rule':'all' },
                'GI_SN_gaba':{ 'syn':'GI_SN_gaba', 'rule':'all' }})
        
        
        #conns={'MSN_D1_bg-MSN_D1':{'fan_in':fan_in_D1_bg_DX,'syn':'MSN_SNR_gaba_s_min', 'sets':1,  'rule':'all'}}
        # Add extend to conn
        for k in sorted(conns.keys()): 
            source=k.split('_')[0]
            if k=='FS_M2_gaba':
                conns[k]['beta_fan_in']=0.8
            else:
                conns[k]['beta_fan_in']=0.0
            conns[k]['data_path_learned_conn']='' #No     
            # Cortical input do not randomally change CTX input
            if dic['node'][source]['type']=='input':
                delay_setup  = {'type':'constant', 'params':None}
                weight_setup = {'type':'constant', 'params':None}
            
            else:
                pr=0.5
                delay_setup =  {'type':'uniform', 'params':{'min':None,  'max':None}}
                weight_setup = {'type':'uniform', 'params':{'min':None, 'max':None}}
            
            conns[k].update({'lesion':False, 'connection_type':'divergent',
                             'delay_val':None,
                             'delay_setup':delay_setup, 
                             'weight_val':None,
                             'weight_setup':weight_setup,
                             'mask':None,
                             'fan_in':None})
            
            pre, post=k.split('_')[0:2]
            if pre[0:2] in ['M1', 'M2'] and post[0:2] in ['M1', 'M2']:
                mask=None
            elif pre[0:2] =='FS' and post[0:2] in ['M1', 'M2', 'FS']:
                mask=None
            else:
                mask=[-0.5, 0.5]    
            
    
            conns[k].update({'mask':mask})  
        
        dic['conn']=misc.dict_merge(dic['conn'], conns)            
                               
        self._dic_con=dic
        
    def __repr__(self):
        return  self.__class__.__name__
    
    @property
    def dic(self):
        
        if self.dic_set and (not self.per or self.per.applied):
            pass        
        else:
            # Test if dic and dic dependable are correct. That is
            # the values in dic_dep should be None values in dic
            dic=deepcopy(self.dic_con)
            dic_dep=deepcopy(self.dic_dep)
        
            dic=misc.dict_merge(dic, dic_dep)
            if not self.per==None:
                dic=self.per.update(dic)
        

            self._dic=dic
            self.dic_set=True
        return self._dic   
   
    @property 
    def dic_con(self): 
        
        if self.dic_con_set:
            return self._dic_con

        #self.dic_print_change('constant', self.dic_rep, self._dic_con)       
        self._dic_con = misc.dict_merge(self._dic_con, self.dic_rep) 
        self.dic_con_set=True
        return self._dic_con  
        
    @property 
    def dic_dep(self): 
        
        if self.dic_dep_set:
            return self._dic_dep
        
        #self.dic_print_change('dependable', self._dic_rep, self.get_dependable_par())          
        self._dic_dep=misc.dict_merge(self.get_dependable_par(), self.dic_rep)   
        self.dic_dep_set=True
        return self._dic_dep
    
    @property
    def dic_rep(self):
        self.dic_set=False
        self.dic_con_set=False
        self.dic_dep_set=False
        return self._dic_rep
    
    @dic_rep.setter
    def dic_rep(self, value):
        self.dic_set=False
        self.dic_con_set=False
        self.dic_dep_set=False
        self._dic_rep=value
    
    def dic_print_change(self, s, d1, d2):
        
        d1_reduced=misc.dict_reduce(d1, {}, deliminator='.')
        d2_reduced=misc.dict_reduce(d2, {}, deliminator='.')
                
        for key, val in d1_reduced.iteritems():
            if key in d2_reduced.keys():
                val_old=d2_reduced[key]
                if val_old!=val and val_old!=None:
                    print 'Change '+s +' '+ key +' '+ str(val_old)+'->'+str(val)
    
    def __getitem__(self, key):
        return self.dic[key]

    def __setitem__(self, key, val):
        self.dic[key] = val        


    
    def get_conns_keys(self):
        keys=[]
        for key in self.dic_con['conn'].keys(): 
                keys.append(key)
        return keys

    def get_nest_setup(self, model):
        p=deepcopy(self.dic['nest'][model])
        from_model=p['template']
        del p['template']
        return {model:[from_model, model, p]}
    
    def get_node_network_keys(self):
        keys=[]
        for key in self.dic_con['node'].keys():
            if self.dic_con['node'][key]['type']=='network':
                keys.append(key)
        return keys
    
    def get_node_input_keys(self):
        keys=[]
        for key in self.dic_con['node'].keys():
            if self.dic_con['node'][key]['type']=='input':
                keys.append(key)
        return keys
    
    def get_nest_neurons_keys(self):
        k=[]
        for key, val in self._dic_con['nest'].items():
            val=deepcopy(val)
            if val['template'] in self.rec.keys():
                k.append(key)
            
        return k
    
    
    def get_dependable_par(self):
        dc=self._dic_con
        
        ddep={}
        dra=misc.dict_recursive_add
        # ===========================
        # Dependables nest parameters 
        # ===========================    
        ddep['netw']={}
        ddep['netw']['n_nuclei_sub_sampling']={}
        for k in dc['netw']['n_nuclei']:
            p=1
            if k in dc['netw']['sub_sampling'].keys():
                p= dc['netw']['sub_sampling'][k]
            ddep['netw']['n_nuclei_sub_sampling'][k]=dc['netw']['n_nuclei'][k]/p
            
        n_nuclei=ddep['netw']['n_nuclei_sub_sampling'].copy()
        
        ddep['netw']['prop']={}
        for k in n_nuclei.keys(): 
            ddep['netw']['prop'].update({k:n_nuclei[k]/sum(n_nuclei.values())}) 
        
        for model in self.get_nest_neurons_keys():
 
            dra(ddep, ['nest', model,'tata_dop'], dc['netw']['tata_dop']-dc['netw']['tata_dop0'])       
        
            
        # Exp decay for plastic synapses
        for model in ['MS', 'M1','M1_low','M1_high', 'M2', 'M2_low','M2_high']: 
            dra(ddep, ['nest',model,'GABAA_1_Tau_decay'], dc['nest']['FS_M1_gaba']['tau_psc'])  
        for model in ['FS', 'FS_low', 'FS_high']:
            dra(ddep, ['nest',model,'GABAA_1_Tau_decay'], dc['nest']['FS_FS_gaba']['tau_psc']) # 6','     # Gittis 2010
        dra(ddep, ['nest','GI','GABAA_1_Tau_decay'], dc['nest']['M2_GI_gaba']['tau_psc'])        # (Shen et al. 2008)    
        dra(ddep, ['nest','SN','GABAA_1_Tau_decay'], dc['nest']['M1_SN_gaba']['tau_psc'])        # (Connelly et al. 2010)
        dra(ddep, ['nest','SN','GABAA_2_Tau_decay'], dc['nest']['GI_SN_gaba']['tau_psc'])        # (Connelly et al. 2010)
        
        
        # ===========================
        # Dependables node parameters 
        # ===========================    
        
        for key in self.get_node_network_keys():     
            
            
            C_m=dc['nest'][dc['node'][key]['model']]['C_m']
            V_th=dc['nest'][dc['node'][key]['model']]['V_th']
            
            
            dra(ddep,['node', key ,'randomization' ,'C_m' ,'gaussian','my'],C_m)
            dra(ddep,['node', key ,'randomization' ,'C_m' ,'gaussian','sigma'],0.1*C_m)
            dra(ddep,['node', key ,'randomization' ,'V_th','gaussian','my'], dc['netw']['V_th_sigma'])
            dra(ddep,['node', key ,'randomization' ,'V_th','gaussian','sigma'],V_th)
            dra(ddep,['node', key ,'randomization' ,'V_m' ,'uniform' ,'min'],V_th-20)
            dra(ddep,['node', key ,'randomization' ,'V_m' ,'uniform' ,'max'],V_th)            

        
        # Model inputs
        
        # Create and adjust dependable ratio
        for key in dc['netw']['prop'].keys():
            
            if key=='GP':
                p1=ddep['netw']['prop']['GP']*dc['netw']['prop_GPE_A']
                p2=ddep['netw']['prop']['GP']*(1-dc['netw']['prop_GPE_A'])
                dra(ddep,['node','GA','prop'],p1 )
                dra(ddep,['node','GI','prop'],p2 )
            else:
                dra(ddep, ['node', key, 'prop'], ddep['netw']['prop'][key]) 
        
        # Population sizes and proportions
        for key in self.get_node_network_keys(): 
                 
            n=int(ddep['node'][key]['prop']*dc['netw']['size'])
            dra(ddep, ['node', key ,'n'], n)                 
            
        for k in sorted(self.get_conns_keys()):
            source, target = k.split('_')[0:2]
            if self._dic_con['node'][source]['type']=='input':
                dra(ddep, ['node', source, 'n'], ddep['node'][target]['n'])


        # ===========================
        # Dependables conn parameters 
        # ===========================
        
        GPE_A_target_rate=dc['netw']['target_rate_GPE_A']
        GPE_pop_target_rate=30.
        GPE_I_target_rate=(GPE_pop_target_rate-dc['netw']['prop_GPE_A']*GPE_A_target_rate)/(1-dc['netw']['prop_GPE_A'])
                
        dra(ddep,['node','GA','target_rate'], GPE_A_target_rate) #23, -8
        dra(ddep,['node','GI','target_rate'], GPE_I_target_rate) #51, 56
  
        GA_GA_fan_in=int(round(30*dc['netw']['prop_fan_in_GPE_A']))
        dra(ddep,['conn','GA_GA_gaba','fan_in0'], GA_GA_fan_in)
        dra(ddep,['conn','GA_GI_gaba','fan_in0'], GA_GA_fan_in)
        dra(ddep,['conn','GI_GA_gaba','fan_in0'], 30-GA_GA_fan_in)
        dra(ddep,['conn','GI_GI_gaba','fan_in0'], 30-GA_GA_fan_in)
        

        
        for k in sorted(self.get_conns_keys()): 
            
            syn=self._dic_con['conn'][k]['syn']
            nest_params=self._dic_con['nest'][syn]
            
            source=syn.split('_')[0]
             
            # Cortical input do not randomally change CTX input
            delay_setup={}
            weight_setup={}
            if self._dic_con['conn'][k]['delay_setup']['type']=='constant':
                delay_setup  = {'params':nest_params['delay']}
            if self._dic_con['conn'][k]['weight_setup']['type'] in ['constant', 'learned']:
                weight_setup = {'params':nest_params['weight']}
                
            if self._dic_con['conn'][k]['delay_setup']['type']=='uniform':
                pr=0.5
                delay_setup =  {'params':{'min':(1-pr)*nest_params['delay'],  
                                          'max':(1+pr)*nest_params['delay']}}
                
            
            if self._dic_con['conn'][k]['weight_setup']['type']=='uniform':
                pr=0.5
                weight_setup = {'params':{'min':(1-pr)*nest_params['weight'], 
                                          'max':(1+pr)*nest_params['weight']}}
                
                
            dra(ddep,['conn', k,'delay_val'],    nest_params['delay'])
            dra(ddep,['conn', k,'delay_setup'],  delay_setup) 
            dra(ddep,['conn', k,'weight_val'],   nest_params['weight'])
            dra(ddep,['conn', k,'weight_setup'], weight_setup)
            
            pre, post = k.split('_')[0:2]
            
            n_MSN=ddep['node']['M1']['n']+ddep['node']['M2']['n']
            n_MSN*=dc['netw']['sub_sampling']['MS']
            # Set mask, set distance specific mask for MSN and FSN
            e=False
            if pre[0:2] in ['M1', 'M2'] and post[0:2] in ['M1', 'M2']:
                e=min(2800./(n_MSN)/2, 0.5)
            elif pre[0:2] in ['FS', 'F1', 'F2'] and post[0:2] in ['M1', 'M2', 'F1', 'F2', 'FS']:
                e=min(560./(n_MSN)/2, 0.5)
            if e:
                dra(ddep,['conn', k,'mask'], [ -e, e])   
            
            if k in ddep['conn'].keys() and 'fan_in0' in ddep['conn'][k].keys():
                fan_in0= ddep['conn'][k]['fan_in0']
            else:
                fan_in0= dc['conn'][k]['fan_in0']
            dra(ddep, ['conn',k,'fan_in'],fan_in0*(1-(dc['netw']['tata_dop']-dc['netw']['tata_dop0'])*dc['conn'][k]['beta_fan_in']))
        
        return ddep

    
        
    def update(self, dic):   
        self.dic=misc.dict_merge(self.dic, dic)
        self.update_dependable_par()

    
    def overwrite_dependables(self, dic_in):
        
        # Model inputs
        # Population sizes
        for key in ['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']:
            if misc.dict_recursive_get(dic_in, ['node',key,'n'])!=None:
                self.dic['node'][key]['n']=misc.dict_recursive_get(dic_in, ['node',key,'n'])


class Par_slow_wave(Par):
    
    # @TODO defining input in par
    def __init__(self, dic_rep={}, perturbations=None ):
        super( Par_slow_wave, self ).__init__( dic_rep, perturbations )       
        self.dic['netw']['input']={'oscillations':{'cycles':10.0,
                                                   'p_amplitude_mod':0.9,
                                                   'freq': 1.}}
class Par_bcpnn(Par):
    
    def __init__(self, dic_rep={}, perturbations=None ):
        super( Par_bcpnn, self ).__init__( dic_rep, perturbations )     
        
        dic={}
        
        dic['netw']={}
        
        dic['netw']['sub_sampling']={'M1':50.0*4,'M2':50.0*4, 'CO':300.0}
        
        dic['netw']['n_states']=10
        dic['netw']['n_actions']=5
        
        dic['netw']['n_nuclei']={'CO':17000000}
        dic['netw']['n_nuclei_sub_sampling']={'CO':None}
        dic['netw']['prop']={'CO':None}
        
        dic['netw']['input']={'constant' : {'nodes':['EA', 'EI', 'ES']}}
        dic['netw']['input'].update({'bcpnn': {'nodes':['EC'], 'time':100.0, 'n_set_pre':10, 'p_amplitude':3, 'n_MSN_stim':20}})
        # ========================
        # Default nest parameters 
        # ========================
        # Defining default parameters
        
        # EXT-CTX 
        dic['nest']={}
        dic['nest']['EC_CO_ampa']={}
        dic['nest']['EC_CO_ampa']['weight']   = 0.25    # n.d. set as for CTX to MSN   
        dic['nest']['EC_CO_ampa']['delay']    = 12.    # n.d. set as for CTX to MSN   
        dic['nest']['EC_CO_ampa']['template'] = 'static_synapse'   
        dic['nest']['EC_CO_ampa']['receptor_type'] = self.rec['izhik_cond_exp'][ 'AMPA_1' ]     # n.d. set as for CTX to MSN   
    
        
        # CTX-FSN 
        dic['nest']['CO_FS_ampa']={}
        dic['nest']['CO_FS_ampa']['weight']   = None    # n.d. set as for CTX to MSN   
        dic['nest']['CO_FS_ampa']['delay']    = 12.    # n.d. set as for CTX to MSN   
        dic['nest']['CO_FS_ampa']['template'] = 'static_synapse'   
        dic['nest']['CO_FS_ampa']['receptor_type'] = self.rec['izhik_cond_exp'][ 'AMPA_1' ]     # n.d. set as for CTX to MSN   
        
        
        # CTX-MSN D1
        dic['nest']['CO_M1_ampa']={}
        dic['nest']['CO_M1_ampa']['weight']   = None     # constrained by Ellender 2011
        dic['nest']['CO_M1_ampa']['delay']    = 12.    # Mallet 2005
        dic['nest']['CO_M1_ampa']['template'] = 'static_synapse'
        dic['nest']['CO_M1_ampa']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'AMPA_1' ]
        
        dic['nest']['CO_M1_nmda'] = deepcopy(dic['nest']['CO_M1_ampa'])
        dic['nest']['CO_M1_nmda']['weight'] =  None   # (Humphries, Wood, and Gurney 2009)
        dic['nest']['CO_M1_nmda']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'NMDA_1' ]
        
        
        # CTX-MSN D2
        dic['nest']['CO_M2_ampa'] = deepcopy(dic['nest']['CO_M1_ampa'])
        dic['nest']['CO_M2_ampa']['weight'] =  None     # constrained by Ellender 2011
        
        dic['nest']['CO_M2_nmda'] = deepcopy(dic['nest']['CO_M1_nmda'])
        dic['nest']['CO_M2_nmda']['weight'] =  None  # (Humphries, Wood, and Gurney 2009) 
    

        # CTX-STN
        dic['nest']['CO_ST_ampa']={}
        dic['nest']['CO_ST_ampa']['weight']   = None
        dic['nest']['CO_ST_ampa']['delay']       = 2.5  # Fujimoto and Kita 1993
        dic['nest']['CO_ST_ampa']['template'] = 'static_synapse'  
        dic['nest']['CO_ST_ampa']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'AMPA_1' ]  
        
        dic['nest']['CO_ST_nmda'] = deepcopy(dic['nest']['CO_ST_ampa'])
        dic['nest']['CO_ST_nmda']['weight'] = None   # n.d.; same ratio ampa/nmda as MSN
        dic['nest']['CO_ST_nmda']['receptor_type'] = self.rec['my_aeif_cond_exp'] [ 'NMDA_1' ]  
        
        # ============        
        # Input Models
        # ============ 
        
        
        #EXT-CTX
        dic['nest']['EC']={}
        dic['nest']['EC']['template']='poisson_generator' 
        dic['nest']['EC']['rate']=0.0
        
        
        # =============        
        # Neuron Models
        # =============
    
        # CTX
        # ===
        dic['nest']['CO']={}    
        dic['nest']['CO']['template'] = 'izhik_cond_exp'
    
        dic['nest']['CO']['a']      =  0.03      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['b_1']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['CO']['b_2']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['CO']['c']      = -50.       # (E.M. Izhikevich 2007)
        dic['nest']['CO']['C_m']    =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['d']      =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['E_L']    = -60.0      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['k']      =   0.7      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_peak'] =  35.       # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_b']    = dic['nest']['CO']['E_L']    # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_th']   = -40.        # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_m']    =  -50.    
    
        dic['nest']['CO']['AMPA_1_Tau_decay'] = 12.  # Same as MSN
        dic['nest']['CO']['AMPA_1_E_rev']     =  0.  # Same as MSN
        
        dic['nest']['CO']['NMDA_1_Tau_decay'] = 160.# Same as MSN
        dic['nest']['CO']['NMDA_1_E_rev']     =  dic['nest']['CO']['AMPA_1_E_rev']    
        dic['nest']['CO']['NMDA_1_Vact']      = -20.0
        dic['nest']['CO']['NMDA_1_Sact']      =  16.0
        
        dic['nest']['CO']['tata_dop']      =  None
        
        
        # ========================
        # Default node parameters 
        # ========================
   

        dic['node']={}
        inputs={'EC': {'model':'EC',  'rate':300., 'n_sets':1}} 
        
        

        for key in inputs.keys():       
            inputs[key].update({'extent':[-0.5, 0.5],'edge_wrap':True, 'n':None,
                                'lesion':False, 'type':'input',
                                'unit_class':Units_input })
        dic['node']=misc.dict_merge(dic['node'], inputs) 

        network={'CO':{'model':'CO', 'I_vitro':0.0, 'I_vivo':0.0,  'target_rate':0.5, 'target_rate_in_vitro':0.0}}
        for key in network.keys():     
            network[key].update({'type':'network', 'extent':[-0.5, 0.5],
                                 'edge_wrap':True, 'lesion':False,
                                 'unit_class': Units_neuron,  
                                 'prop':None, 
                                 'n':None,
                                 'n_sets':None,
                                 'randomization':{ 'C_m': {'gaussian':{'sigma':None, 'my':None}},
                                                   'V_th':{'gaussian':{'sigma':None, 'my':None, 
                                                                       'cut':True, 'cut_at':3.}},
                                                   'V_m': {'uniform': {'min': None,  'max':None }}
                                                   }})  
                
        network.update({'M1':{'n_sets':None},
                        'M2':{'n_sets':None},
                        'GI':{'n_sets':None},
                        'SN':{'n_sets':None}})  
        
        dic['node']=misc.dict_merge(dic['node'], network) 
                 
        # ========================
        # Default conn parameters 
        # ========================
        

        conns={'EC_CO_ampa':{ 'syn':'EC_CO_ampa' }}
        for key in conns.keys():
            conns[key].update({'fan_in0':1,  'rule':'1-1' })
        dic['conn']={}    
        dic['conn']['CO_M1_ampa']={'fan_in0': 20}   
        dic['conn']['CO_M1_nmda']={'fan_in0': 20}   
        dic['conn']['CO_M2_ampa']={'fan_in0': 20}
        dic['conn']['CO_M2_nmda']={'fan_in0': 20}   
        dic['conn']['CO_FS_ampa']={'fan_in0': 20}  
        dic['conn']['CO_ST_ampa']={'fan_in0': 20}
        dic['conn']['CO_ST_nmda']={'fan_in0': 20} 
        
        
        conns.update({'CO_M1_ampa':{ 'syn':'CO_M1_ampa', 'rule':'set-all_to_all'},
                      'CO_M1_nmda':{ 'syn':'CO_M1_nmda', 'rule':'set-all_to_all' },
                      'CO_M2_ampa':{ 'syn':'CO_M2_ampa', 'rule':'set-all_to_all' },
                      'CO_M2_nmda':{ 'syn':'CO_M2_nmda', 'rule':'set-all_to_all' },
                      'CO_FS_ampa':{ 'syn':'CO_FS_ampa', 'rule':'all' },
                      'CO_ST_ampa':{ 'syn':'CO_ST_ampa', 'rule':'all' },
                      'CO_ST_nmda':{ 'syn':'CO_ST_nmda', 'rule':'all' },
                      'GI_SN_gaba':{ 'syn':'GI_SN_gaba', 'rule':'set-set'}})  
        

        # Add extend to conn
        data_path_learned_conn={'M1':'~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h0/CO_M1.pkl'}
        data_path_learned_conn['M2']='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h0/CO_M2.pkl'
        for k in sorted(conns.keys()): 


            source, target=k.split('_')[0:2]
            if source=='CO' and target in ['M1', 'M2']:
                conns[k]['data_path_learned_conn']=data_path_learned_conn[target]
            else:    
                conns[k]['data_path_learned_conn']='' #No    
            conns[k]['beta_fan_in']=0.0 
            # Cortical input do not randomally change CTX input
            if k[0] in ['E']:
                delay_setup  = {'type':'constant', 'params':None}
                weight_setup = {'type':'constant', 'params':None}

            else:
                delay_setup =  {'type':'uniform', 'params':{'min':None,  'max':None}}
                if k in ['CO_M1_ampa', 'CO_M1_nmda','CO_M2_ampa','CO_M2_nmda']:
                    weight_setup =  {'type':'learned', 'params':None}
                else:
                    weight_setup = {'type':'uniform', 'params':{'min':None, 'max':None}}
            
            conns[k].update({'lesion':False, 'connection_type':'divergent',
                             'delay_val':None,
                             'delay_setup':delay_setup, 
                             'weight_val':None,
                             'weight_setup':weight_setup,
                             'mask':[-0.5, 0.5],
                             'fan_in':None})
    
       
            
        dic['conn']=misc.dict_merge(dic['conn'], conns)  

        # Remove models
        
        dic=misc.dict_merge(self._dic_con, dic)
        conn_rem=['C1_M1_ampa', 'C1_M1_nmda',
                  'C2_M2_ampa', 'C2_M2_nmda',
                  'CF_FS_ampa', 'CS_ST_ampa',
                  'CS_ST_nmda']
        for model in conn_rem: del dic['conn'][model]
        nest_models_rem=['CF_FS_ampa',
                         'C1_M1_ampa',
                         'C1_M1_nmda',
                         'C2_M2_ampa',
                         'C2_M2_nmda',
                         'CS_ST_ampa',
                         'CS_ST_nmda']
        
    
        for model in nest_models_rem: del dic['nest'][model]
        node_rem=['CF', 'C1', 'C2', 'CS']
        for model in node_rem: del dic['node'][model] 
        
        self._dic_con=dic

    def get_dependable_par(self):
        
        dc=self._dic_con
        dra=misc.dict_recursive_add
        
        ddep={}
        mr_ctx=2.0
        f_M1=1.5
        f_M2=3.0
        if 'CO_FS_ampa' in dc['conn'].keys():
            dra(ddep,['nest', 'CO_FS_ampa','weight'], 0.25*1010.0/dc['conn']['CO_FS_ampa']['fan_in0']/mr_ctx*2.)
        dra(ddep,['nest', 'CO_M1_ampa','weight'], 0.5*530.0/dc['conn']['CO_M1_ampa']['fan_in0']/mr_ctx*0.8*f_M1)
        dra(ddep,['nest', 'CO_M1_nmda','weight'], 0.019*530.0/dc['conn']['CO_M1_nmda']['fan_in0']/mr_ctx*0.8*f_M1)
        dra(ddep,['nest', 'CO_M2_ampa','weight'], 0.41*690.0/dc['conn']['CO_M2_ampa']['fan_in0']/mr_ctx*.7*0.8*f_M2)
        dra(ddep,['nest', 'CO_M2_nmda','weight'], 0.11*690.0/dc['conn']['CO_M2_nmda']['fan_in0']/mr_ctx*.7*0.8*f_M2)        
        dra(ddep,['nest', 'CO_ST_ampa','weight'], 0.25*160.0/dc['conn']['CO_ST_ampa']['fan_in0']/mr_ctx*1.1)
        dra(ddep,['nest', 'CO_ST_nmda','weight'], 0.00625*160.0/dc['conn']['CO_ST_nmda']['fan_in0']/mr_ctx*1.1)  
        
        dra(ddep,['node', 'CO','n_sets'], dc['netw']['n_states'])    
        dra(ddep,['node', 'M1','n_sets'], dc['netw']['n_actions'])  
        dra(ddep,['node', 'M2','n_sets'], dc['netw']['n_actions'])  
        dra(ddep,['node', 'GI','n_sets'], dc['netw']['n_actions'])         
        dra(ddep,['node', 'SN','n_sets'], dc['netw']['n_actions'])
        
        dic_con=deepcopy(self._dic_con)
        self._dic_con=misc.dict_merge(self._dic_con, ddep)
        ddep2=Par.get_dependable_par(self)
        self._dic_con=dic_con
        ddep=misc.dict_merge(ddep2, ddep)
        return ddep

class Par_bcpnn_h1(Par_bcpnn):
    
    def __init__(self, dic_rep={}, perturbations=None ):
        super( Par_bcpnn_h1, self ).__init__( dic_rep, perturbations )    
        dic=self._dic_con
        
        dic['netw']['n_nuclei'].update({'F1':dic['netw']['n_nuclei']['FS']/2})
        dic['netw']['n_nuclei'].update({'F2':dic['netw']['n_nuclei']['FS']/2})
        
        dic['netw']['prop'].update({'F1':dic['netw']['n_nuclei']['FS']/2})
        dic['netw']['prop'].update({'F2':dic['netw']['n_nuclei']['FS']/2})

        for k in ['F1','F2']: 
            dic['netw']['prop'].update({k:None})  
            dic['netw']['n_nuclei_sub_sampling'].update({k:None})  
        
        dic['node']['F1'] = deepcopy(dic['node']['FS'])
        dic['node']['F1']['n_sets']=None

        dic['node']['F2'] = deepcopy(dic['node']['FS'])
        dic['node']['F2']['n_sets']=None


        
        dic['conn']['CO_F1_ampa'] = deepcopy(dic['conn']['CO_FS_ampa'])
        dic['conn']['CO_F1_ampa'].update( {'rule':'set-all_to_all'})
        
        dic['conn']['CO_F2_ampa'] = deepcopy(dic['conn']['CO_FS_ampa'])
        dic['conn']['CO_F2_ampa'].update( {'rule':'set-all_to_all'})

        dic['conn']['F1_M1_gaba'] = deepcopy(dic['conn']['FS_M1_gaba'])
        dic['conn']['F2_M2_gaba'] = deepcopy(dic['conn']['FS_M2_gaba'])        
        dic['conn']['GA_F1_gaba'] = deepcopy(dic['conn']['GA_FS_gaba'])
        dic['conn']['GA_F2_gaba'] = deepcopy(dic['conn']['GA_FS_gaba'])
        dic['conn']['F1_F1_gaba'] = deepcopy(dic['conn']['FS_FS_gaba'])
        dic['conn']['F1_F2_gaba'] = deepcopy(dic['conn']['FS_FS_gaba'])
        dic['conn']['F2_F1_gaba'] = deepcopy(dic['conn']['FS_FS_gaba'])
        dic['conn']['F2_F2_gaba'] = deepcopy(dic['conn']['FS_FS_gaba'])
        
        data_path_learned_conn={'CO_M1_ampa':'~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M1.pkl'}
        data_path_learned_conn['CO_M1_nmda']='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M1.pkl'
        data_path_learned_conn['CO_M2_ampa']='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M2.pkl'
        data_path_learned_conn['CO_M2_nmda']='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M2.pkl'
        data_path_learned_conn['CO_F1_ampa']='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_F1.pkl'
        data_path_learned_conn['CO_F2_ampa']='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_F2.pkl'

        for node in data_path_learned_conn:
            dic['conn'][node]['data_path_learned_conn']=data_path_learned_conn[node]
        
        dic['conn']['CO_F1_ampa']['weight_setup'].update({'type':'learned', 'params':None})
        dic['conn']['CO_F2_ampa']['weight_setup'].update({'type':'learned', 'params':None})
        
        node_rem=['FS']
        for model in node_rem: del dic['node'][model]
  
        del dic['netw']['n_nuclei']['FS']
        del dic['netw']['prop']['FS']
        del dic['netw']['n_nuclei_sub_sampling']['FS']
        
        rem=['CO_FS_ampa','FS_M1_gaba', 'FS_M2_gaba', 'GA_FS_gaba', 'FS_FS_gaba']
        for model in rem:  
            del dic['conn'][model]  
            
    def get_dependable_par(self):
         
        dc=self._dic_con
        dra=misc.dict_recursive_add
        
        ddep={}
        

        
        
        dra(ddep,['node', 'F1','n_sets'], dc['netw']['n_actions'])  
        dra(ddep,['node', 'F2','n_sets'], dc['netw']['n_actions']) 
     
        mr_ctx=2
        dra(ddep,['nest', 'CO_FS_ampa','weight'], 0.25*1010.0/dc['conn']['CO_F1_ampa']['fan_in0']/mr_ctx*2.)
        
        dic_con=deepcopy(self._dic_con)
        self._dic_con=misc.dict_merge(self._dic_con, ddep)
        ddep2=Par_bcpnn.get_dependable_par(self)
        self._dic_con=dic_con
        ddep=misc.dict_merge(ddep2, ddep)
        

        
        return ddep
import unittest

class TestPar(unittest.TestCase):
    
    def setUp(self):
        import my_nest
        self.MyLoadModels= my_nest.MyLoadModels
        self.par = Par({})
        self.par_test_par_rep= Par( {'netw': {'size': self.par.dic['netw']['size']*2}},  None)

    def test_dic_integrity(self):
        '''
        Make sure that all None values in dic_con are 
        in dic_dep
        '''
        
        d=self.par.dic_dep
        d1=misc.dict_reduce(d, {}, deliminator='.')
        d2=misc.dict_reduce(self.par.dic_con, {}, deliminator='.')
        keys=d2.keys()
        for key in keys:
            if d2[key]!=None:
                del d2[key]
        self.assertListEqual(sorted(d1.keys()), sorted(d2.keys()))    
    
    def test_input_par_added(self):
        '''
        Test that par is updated with par_rep for both constant
        and dependable parameters.  
        '''
        dic_rep={'netw':{'size':20000.0}, # constant value    
                 'node':{'GA':{'target_rate':0.0}}} # dependable value
        
        self.par.dic_rep.update(dic_rep)
        dic_post=self.par.dic
        dic_post_dep=self.par.dic_dep
        dic_post_con=self.par.dic_con
        dic_netw=self.par['netw'] #Test that get item works
        dic_node=self.par['node'] 
        
        l1=[dic_rep['netw']['size'],  dic_rep['node']['GA']['target_rate']]*3
        l2=[dic_post['netw']['size'], dic_post['node']['GA']['target_rate'],
            dic_post_con['netw']['size'], dic_post_dep['node']['GA']['target_rate'],
            dic_netw['size'], dic_node['GA']['target_rate'],]
        self.assertListEqual(l1, l2)
    
    def test_pertubations(self):
        l=[]
        l+=[ Pertubation_list(['netw.size', 0.5 , '*'])]
        l+=[ Pertubation_list(['node.GA.target_rate', 0.5, '*'])]
        
        l1=[]
        l2=[]
        for pl in l:
            for p in pl:
                l1.append(misc.dict_recursive_get(self.par.dic, p.keys)*p.val)
            self.par.per=pl
            for p in pl:
                l2.append(misc.dict_recursive_get(self.par.dic, p.keys))
 
        self.assertListEqual(l1, l2)    
    
    def test_change_con_effect_dep(self):
        
        dic=deepcopy(self.par.dic)
        dic_rep={'netw':{'size':dic['netw']['size']*2}} # constant value    
        v1=int(dic['node']['M1']['prop']*(dic['netw']['size']*2))
        self.par.dic_rep=dic_rep
        v2=self.par.dic['node']['M1']['n']
        self.assertEqual(v1,v2)
        

    def test_change_con_effect_dep_initiation(self):
        
        dic=deepcopy(self.par_test_par_rep.dic)
        dic_rep={'netw':{'size':dic['netw']['size']*2}} # constant value    
        v1=int(dic['node']['M1']['prop']*(dic['netw']['size']*2))
        self.par.dic_rep=dic_rep
        v2=self.par.dic['node']['M1']['n']
        self.assertEqual(v1,v2)
        
    def test_none_value_left(self):
        
        dic_reduced=misc.dict_reduce(self.par.dic, {},  deliminator='.')
        d1={}
        d2={}
        for key, val in dic_reduced.items():
            if val==None:
                d1[key]=val
        self.assertDictEqual(d1, d2)
            
    def test_nest_params_exist(self):
        
        d_nest=self.par.dic['nest']

        
        d1={}
        d2={}
        for dn in d_nest.values():
                template=dn['template']
                del dn['template']
                df=nest.GetDefaults(template)
                for key, val in dn.items():
                    if key not in df.keys():
                        d1[key]=val
        self.assertDictEqual(d1, d2)
   
    def test_proportion_sum_to_one(self):
        s=0.0
        for name in self.par.get_node_network_keys():
            s+=self.par.dic['node'][name]['prop']
        self.assertAlmostEqual(s, 1.0,7)
    
    def test_conn_keys_integrity(self):
        keys=[]
        for val in self.par.dic['conn'].values():
            keys.extend(val.keys())
        unique_keys=set(keys)
        counts=[]
        for key in unique_keys:
            counts.append(keys.count(key))
    
    def test_node_keys_integrity(self):
        keys1=[]
        keys2=[]
        for key, val in self.par.dic['node'].items():
            if val['type']=='input': 
                keys1.extend(val.keys())
            if val['type']=='network': 
                keys2.extend(val.keys())
                
        unique_keys1=set(keys1)
        unique_keys2=set(keys2)
        counts1=[]
        counts2=[]
        for key in unique_keys1: 
            counts1.append(keys1.count(key))            
        for key in unique_keys2: 
            counts2.append(keys2.count(key))    
          
        self.assertEqual(counts1[0], float(sum(counts1))/len(counts1))
        self.assertEqual(counts2[0], float(sum(counts2))/len(counts2))
        
    def test_1_1_rule_integrity(self):   
        for key in self.par.dic['conn'].keys():
            source, target=key.split('_')[0:2]
            if self.par.dic['conn'][key]['rule']=='1-1' :
                self.assertEqual(self.par.dic['node'][source]['n'], self.par.dic['node'][target]['n'])
            
    def test_model_copy(self):
        for model in self.par['nest'].keys():
            params=self.par.get_nest_setup(model)
            self.MyLoadModels( params, [model] )
        for val in self.par['node'].values():    
            model=val['model']        
            params=self.par.get_nest_setup(model)
            self.MyLoadModels( params, [model] )
        for val in self.par['conn'].values():    
            model=val['syn']        
            params=self.par.get_nest_setup(model)
            self.MyLoadModels( params, [model] )
        
     
class TestPar_bcpnn(TestPar):     
    def setUp(self):
        import my_nest
        self.par = Par_bcpnn({})
        self.MyLoadModels= my_nest.MyLoadModels
        self.par_test_par_rep= Par_bcpnn( {'netw': {'size': 20000.0}})
        
class TestPar_bcpnn_h1(TestPar):     
    def setUp(self):
        import my_nest
        self.par = Par_bcpnn_h1({})
        self.MyLoadModels= my_nest.MyLoadModels
        self.par_test_par_rep= Par_bcpnn( {'netw': {'size': 20000.0}})        
        
    
if __name__ == '__main__':
    unittest.main()    