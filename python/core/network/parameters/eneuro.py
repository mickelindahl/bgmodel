# Create by Mikael Lindahl on 4/10/17.

from copy import deepcopy
from core import misc
from core import directories as dr
from core.network.default_params import Par_base, \
    Par_base_mixin, \
    GetNest, \
    GetNetw, \
    GetNode, \
    DepNetw, \
    DepNode

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

class EneuroParBase( object ):
    def _get_par_constant(self):
        dic = {}

        # ========================
        # Default simu parameters
        # ========================

        dic['simu'] = {}
        dic['simu']['do_reset'] = False
        dic['simu']['mm_params'] = {'interval': 0.5,
                                    'to_file': True,
                                    'to_memory': False,
                                    'record_from': ['V_m']}

        dp = dr.HOME_DATA + '/'
        dco = dp + 'conn/'
        dcl = dp + self.__class__.__name__ + '/'
        df = dp + 'fig/'
        dn = dp + self.__class__.__name__ + '/nest/'

        dic['simu']['path_data'] = dp
        dic['simu']['path_conn'] = dco
        dic['simu']['path_class'] = dcl
        dic['simu']['path_figure'] = df
        dic['simu']['path_nest'] = dn
        dic['simu']['print_time'] = True
        dic['simu']['save_conn'] = {'active': True, 'overwrite': False}
        dic['simu']['sd_params'] = {'to_file': False, 'to_memory': True}
        dic['simu']['sim_time'] = 2000.0
        dic['simu']['sim_stop'] = 2000.0
        dic['simu']['stop_rec'] = 2000.0
        dic['simu']['start_rec'] = 1000.0
        dic['simu']['local_num_threads'] = 1
        #         dic['simu']['threads_local']=1

        # ========================
        # Default netw parameters
        # ========================

        dic['netw'] = {}

        dic['netw']['attr_popu'] = self._get_attr_popu()
        dic['netw']['attr_surf'] = self._get_attr_surf()

        dic['netw']['fan_in_distribution'] = 'constant'
        dic['netw']['FF_curve'] = {'input': 'C1',
                                   'output': 'M1'}
        dic['netw']['GP_fan_in'] = 30
        dic['netw']['GP_rate'] = 30.
        dic['netw']['GP_fan_in_prop_GA'] = 1 / 17.
        dic['netw']['GA_prop'] = 0.25 # 0.2
        dic['netw']['GI_prop'] = 0.675 # 0.72  # <= 0.9*0.8
        dic['netw']['GF_prop'] = 0.075 # 0.08  # <= 0.1*0.8 10 % of TI cells project to striatum
        #         dic['netw']['GN_prop']=0.9

        dic['netw']['MS_prop'] = 0.475
        dic['netw']['FS_prop'] = 0.02

        d = {'type': 'constant', 'params': {}}
        dic['netw']['input'] = {}
        for key in ['C1', 'C2', 'CF', 'CS', 'EA',
                    'EF',
                    'EI', 'ES']:
            dic['netw']['input'][key] = deepcopy(d)

        dic['netw']['n_actions'] = 1

        dic['netw']['n_nuclei'] = {'M1': 2791000 * GetNetw('MS_prop'),
                                   'M2': 2791000 * GetNetw('MS_prop'),
                                   'FS': 2791000 * GetNetw('FS_prop'),  # 2 % if MSN population
                                   'ST': 13560.,
                                   'GF': 45960. * GetNetw('GF_prop'),
                                   'GI': 45960. * GetNetw('GI_prop'),
                                   'GA': 45960. * GetNetw('GA_prop'),
                                   'SN': 26320.}

        '''
        n_nuclei={'M1':15000,
               'M2':15000,
               'FS': 0.02*30000, # 2 % if MSN population
               'ST': 100,
               'GP': 300/.8,
               'SN': 300}
        '''
        dic['netw']['optimization'] = {'f': ['M1'],
                                       'x': ['node.C1.rate'],
                                       'x0': [700.0]}

        dic['netw']['rand_nodes'] = {'C_m': True, 'V_th': True, 'V_m': True,
                                     'I_e': False  # need to be false so that I_vivo can be modified
                                     }

        dic['netw']['size'] = 10000.0
        dic['netw']['sub_sampling'] = {'M1': 1.0, 'M2': 1.0}
        dic['netw']['tata_dop'] = 0.8
        dic['netw']['tata_dop0'] = 0.8
        dic['netw']['V_th_sigma'] = 1.0

        # ========================
        # Default nest parameters
        # ========================
        # Defining default parameters
        dic['nest'] = {}

        # CTX-FSN
        dic['nest']['CF_FS_ampa'] = {}
        dic['nest']['CF_FS_ampa']['weight'] = 0.25  # n.d. set as for CTX to MSN
        dic['nest']['CF_FS_ampa']['delay'] = 2.5 # 12.0  # n.d. set as for CTX to MSN
        dic['nest']['CF_FS_ampa']['type_id'] = 'static_synapse'
        dic['nest']['CF_FS_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']  # n.d. set as for CTX to MSN

        # FSN-FSN
        dic['nest']['FS_FS_gaba'] = {}
        dic['nest']['FS_FS_gaba']['weight'] = 1. / 0.29  # five times weaker than FSN-MSN, Gittis 2010
        dic['nest']['FS_FS_gaba']['delay'] = 1.7  # n.d.same asfor FSN to MSN
        dic['nest']['FS_FS_gaba']['U'] = 0.29
        dic['nest']['FS_FS_gaba']['tau_fac'] = 53.
        dic['nest']['FS_FS_gaba']['tau_rec'] = 902.
        dic['nest']['FS_FS_gaba']['tau_psc'] = 6.  # Gittis 2010 have
        dic['nest']['FS_FS_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['FS_FS_gaba']['receptor_type'] = self.rec['izh']['GABAA_1']

        # GPE-FSN
        dic['nest']['GA_FS_gaba'] = {}
        dic['nest']['GA_FS_gaba']['weight'] = 2. / 0.29 * (17. / 66.)  # n.d. inbetween MSN and FSN GABAergic synapses
        dic['nest']['GA_FS_gaba']['delay'] = 7.  # n.d. same as MSN to GPE Park 1982
        dic['nest']['GA_FS_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['GA_FS_gaba']['receptor_type'] = self.rec['izh']['GABAA_2']

        dic['nest']['GA_FS_gaba']['U'] = 0.29
        dic['nest']['GA_FS_gaba']['tau_fac'] = 53.
        dic['nest']['GA_FS_gaba']['tau_rec'] = 902.
        dic['nest']['GA_FS_gaba']['tau_psc'] = 66.

        dic['nest']['GI_FS_gaba'] = {}
        dic['nest']['GI_FS_gaba']['weight'] = 1. / 5  # n.d. inbetween MSN and FSN GABAergic synapses
        dic['nest']['GI_FS_gaba']['delay'] = 7.  # n.d. same as MSN to GPE Park 1982
        dic['nest']['GI_FS_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GI_FS_gaba']['receptor_type'] = self.rec['izh']['GABAA_2']

        dic['nest']['GF_FS_gaba'] = {}
        dic['nest']['GF_FS_gaba']['weight'] = 2. / 0.29  # n.d. inbetween MSN and FSN GABAergic synapses
        dic['nest']['GF_FS_gaba']['delay'] = 7.  # n.d. same as MSN to GPE Park 1982
        dic['nest']['GF_FS_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['GF_FS_gaba']['receptor_type'] = self.rec['izh']['GABAA_3']

        dic['nest']['GF_FS_gaba']['U'] = 0.29
        dic['nest']['GF_FS_gaba']['tau_fac'] = 53.
        dic['nest']['GF_FS_gaba']['tau_rec'] = 902.
        dic['nest']['GF_FS_gaba']['tau_psc'] = 17.  # Gittis 2010 have

        # CTX-MSN D1
        dic['nest']['C1_M1_ampa'] = {}
        dic['nest']['C1_M1_ampa']['weight'] = .5  # constrained by Ellender 2011
        dic['nest']['C1_M1_ampa']['delay'] = 2.5 # 12.  # Mallet 2005
        dic['nest']['C1_M1_ampa']['type_id'] = 'static_synapse'
        dic['nest']['C1_M1_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']

        dic['nest']['C1_M1_nmda'] = deepcopy(dic['nest']['C1_M1_ampa'])
        dic['nest']['C1_M1_nmda']['weight'] = 0.11  # (Humphries, Wood, and Gurney 2009)
        dic['nest']['C1_M1_nmda']['receptor_type'] = self.rec['izh']['NMDA_1']

        # CTX-MSN D2
        dic['nest']['C2_M2_ampa'] = deepcopy(dic['nest']['C1_M1_ampa'])
        dic['nest']['C2_M2_ampa']['weight'] = .41  # constrained by Ellender 2011

        dic['nest']['C2_M2_nmda'] = deepcopy(dic['nest']['C1_M1_nmda'])
        dic['nest']['C2_M2_nmda']['weight'] = 0.019  # (Humphries, Wood, and Gurney 2009)

        # MSN-MSN
        dic['nest']['M1_M1_gaba'] = {}
        dic['nest']['M1_M1_gaba']['weight'] = 0.15 #0.6  # Taverna 2008
        dic['nest']['M1_M1_gaba']['delay'] = 1.7  # Taverna 2004
        dic['nest']['M1_M1_gaba']['type_id'] = 'static_synapse'
        dic['nest']['M1_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_2']

        dic['nest']['M1_M2_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M1_M2_gaba']['weight'] = 0.375 #1.5  # Taverna 2008

        dic['nest']['M2_M1_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M2_M1_gaba']['weight'] = 0.45 # 1.8  # Taverna 2008

        dic['nest']['M2_M2_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M2_M2_gaba']['weight'] = 0.35 # 1.4  # Taverna 2008

        # FSN-MSN
        dic['nest']['FS_M1_gaba'] = {}
        dic['nest']['FS_M1_gaba']['weight'] = round(6. / 0.29,
                                                    1)  # Gittie #3.8    # (Koos, Tepper, and Charles J Wilson 2004)
        dic['nest']['FS_M1_gaba']['delay'] = 1.7  # Taverna 2004
        dic['nest']['FS_M1_gaba']['U'] = 0.29  # GABAA plastic
        dic['nest']['FS_M1_gaba']['tau_fac'] = 53.0
        dic['nest']['FS_M1_gaba']['tau_rec'] = 902.0
        dic['nest']['FS_M1_gaba']['tau_psc'] = 8.0  # ?  Gittis 2010
        dic['nest']['FS_M1_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['FS_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_1']

        dic['nest']['FS_M2_gaba'] = deepcopy(dic['nest']['FS_M1_gaba'])

        # FSN-MSN static
        dic['nest']['FS_M1_gaba_s'] = {}
        dic['nest']['FS_M1_gaba_s']['weight'] = 6. * 0.3
        dic['nest']['FS_M1_gaba_s']['delay'] = 1.7  # Taverna 2004
        dic['nest']['FS_M1_gaba_s']['type_id'] = 'static_synapse'
        dic['nest']['FS_M1_gaba_s']['receptor_type'] = self.rec['izh']['GABAA_1']

        dic['nest']['FS_M2_gaba_s'] = deepcopy(dic['nest']['FS_M1_gaba_s'])

        # GPE-MSN
        dic['nest']['GA_M1_gaba'] = {}
        dic['nest']['GA_M1_gaba']['weight'] = 0.04 # 1. / 5  # Glajch 2013
        dic['nest']['GA_M1_gaba']['delay'] = 1.7
        dic['nest']['GA_M1_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GA_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_3']

        dic['nest']['GA_M2_gaba'] = deepcopy(dic['nest']['GA_M1_gaba'])
        dic['nest']['GA_M2_gaba']['weight'] = 0.08 #1. * 2 / 5  ## Glajch 2013

        dic['nest']['GI_M1_gaba'] = {}
        dic['nest']['GI_M1_gaba']['weight'] = 0.2 # 1. / 5  # Glajch 2013
        dic['nest']['GI_M1_gaba']['delay'] = 1.7
        dic['nest']['GI_M1_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GI_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_3']

        dic['nest']['GI_M2_gaba'] = deepcopy(dic['nest']['GA_M1_gaba'])
        dic['nest']['GI_M2_gaba']['weight'] = 1. * 2 / 5  ## Glajch 2013

        dic['nest']['GF_M1_gaba'] = {}
        dic['nest']['GF_M1_gaba']['weight'] = 0.2 #1. / 5  # Glajch 2013
        dic['nest']['GF_M1_gaba']['delay'] = 1.7
        dic['nest']['GF_M1_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GF_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_3']

        dic['nest']['GF_M2_gaba'] = deepcopy(dic['nest']['GA_M1_gaba'])
        dic['nest']['GF_M2_gaba']['weight'] = 1. * 2 / 5  ## Glajch 2013

        # CTX-STN
        dic['nest']['CS_ST_ampa'] = {}
        dic['nest']['CS_ST_ampa']['weight'] = 0.25
        dic['nest']['CS_ST_ampa']['delay'] = 2.5  # Fujimoto and Kita 1993
        dic['nest']['CS_ST_ampa']['type_id'] = 'static_synapse'
        dic['nest']['CS_ST_ampa']['receptor_type'] = self.rec['aeif']['AMPA_1']

        dic['nest']['CS_ST_nmda'] = deepcopy(dic['nest']['CS_ST_ampa'])
        dic['nest']['CS_ST_nmda']['weight'] = 0.00625  # n.d.; same ratio ampa/nmda as MSN
        dic['nest']['CS_ST_nmda']['receptor_type'] = self.rec['aeif']['NMDA_1']

        # GPe I-STN
        dic['nest']['GI_ST_gaba'] = {}
        dic['nest']['GI_ST_gaba']['weight'] = .08  # n.d.
        dic['nest']['GI_ST_gaba']['delay'] = 1. #5.
        dic['nest']['GI_ST_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GI_ST_gaba']['receptor_type'] = self.rec['aeif']['GABAA_1']

        dic['nest']['GF_ST_gaba'] = deepcopy(dic['nest']['GI_ST_gaba'])

        # STN-STN
        dic['nest']['ST_ST_ampa'] = {}
        dic['nest']['ST_ST_ampa']['weight'] = 0.0  # constrained by (Hanson & Dieter Jaeger 2002)
        dic['nest']['ST_ST_ampa']['delay'] = 1.  # Ammari 2010
        dic['nest']['ST_ST_ampa']['type_id'] = 'static_synapse'
        dic['nest']['ST_ST_ampa']['receptor_type'] = self.rec['aeif']['AMPA_1']

        dic['nest']['ST_ST_ampa'] = deepcopy(dic['nest']['ST_ST_ampa'])

        # EXT-GPe
        dic['nest']['EA_GA_ampa'] = {}
        dic['nest']['EA_GA_ampa']['weight'] = 0.167
        dic['nest']['EA_GA_ampa']['delay'] = 5.
        dic['nest']['EA_GA_ampa']['type_id'] = 'static_synapse'
        dic['nest']['EA_GA_ampa']['receptor_type'] = self.rec['aeif']['AMPA_2']

        dic['nest']['EI_GI_ampa'] = deepcopy(dic['nest']['EA_GA_ampa'])
        dic['nest']['EF_GF_ampa'] = deepcopy(dic['nest']['EA_GA_ampa'])

        # GPe-GPe
        dic['nest']['GA_GA_gaba'] = {}
        dic['nest']['GA_GA_gaba']['weight'] = 0.325 # 1.3  # constrained by (Sims et al. 2008)
        dic['nest']['GA_GA_gaba']['delay'] = 1.  # n.d. assumed due to approximity
        dic['nest']['GA_GA_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GA_GA_gaba']['receptor_type'] = self.rec['aeif']['GABAA_2']

        dic['nest']['GI_GI_gaba'] = deepcopy(dic['nest']['GA_GA_gaba'])
        dic['nest']['GI_GI_gaba']['weight'] = 1.3  # constrained by (Sims et al. 2008)

        dic['nest']['GA_GI_gaba'] = deepcopy(dic['nest']['GI_GI_gaba'])
        dic['nest']['GI_GA_gaba'] = deepcopy(dic['nest']['GA_GA_gaba'])

        dic['nest']['GF_GA_gaba'] = deepcopy(dic['nest']['GA_GA_gaba'])
        dic['nest']['GF_GI_gaba'] = deepcopy(dic['nest']['GI_GI_gaba'])
        dic['nest']['GF_GF_gaba'] = deepcopy(dic['nest']['GI_GI_gaba'])
        dic['nest']['GI_GF_gaba'] = deepcopy(dic['nest']['GI_GI_gaba'])
        dic['nest']['GA_GF_gaba'] = deepcopy(dic['nest']['GI_GI_gaba'])
        #

        # MSN D2-GPe I
        dic['nest']['M2_GI_gaba'] = {}
        dic['nest']['M2_GI_gaba']['weight'] = 2. / 0.24  # constrained by (Sims et al. 2008)
        dic['nest']['M2_GI_gaba']['delay'] = 7.  # Park 1982
        dic['nest']['M2_GI_gaba']['U'] = 0.24  # GABAA plastic
        dic['nest']['M2_GI_gaba']['tau_fac'] = 13.0
        dic['nest']['M2_GI_gaba']['tau_rec'] = 77.0
        dic['nest']['M2_GI_gaba']['tau_psc'] = 6.  # (Shen et al. 2008)
        dic['nest']['M2_GI_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['M2_GI_gaba']['receptor_type'] = self.rec['aeif']['GABAA_1']

        dic['nest']['M2_GF_gaba'] = deepcopy(dic['nest']['M2_GI_gaba'])

        # MSN D2-GPe A
        dic['nest']['M2_GA_gaba'] = {}
        dic['nest']['M2_GA_gaba']['weight'] = 2. / 0.24  # constrained by (Sims et al. 2008)
        dic['nest']['M2_GA_gaba']['delay'] = 7.  # Park 1982
        dic['nest']['M2_GA_gaba']['U'] = 0.24  # GABAA plastic
        dic['nest']['M2_GA_gaba']['tau_fac'] = 13.0
        dic['nest']['M2_GA_gaba']['tau_rec'] = 77.0
        dic['nest']['M2_GA_gaba']['tau_psc'] = 6.  # (Shen et al. 2008)
        dic['nest']['M2_GA_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['M2_GA_gaba']['receptor_type'] = self.rec['aeif']['GABAA_1']

        # STN-GPe
        dic['nest']['ST_GA_ampa'] = {}
        dic['nest']['ST_GA_ampa']['weight'] = 0.105 # 0.35  # constrained by (Hanson & Dieter Jaeger 2002)
        dic['nest']['ST_GA_ampa']['delay'] = 2.0 #5.  # Ammari 2010
        dic['nest']['ST_GA_ampa']['type_id'] = 'static_synapse'
        dic['nest']['ST_GA_ampa']['receptor_type'] = self.rec['aeif']['AMPA_1']

        dic['nest']['ST_GI_ampa'] = deepcopy(dic['nest']['ST_GA_ampa'])
        dic['nest']['ST_GI_ampa']['weight'] = 0.35  # constrained by (Hanson & Dieter Jaeger 2002)

        dic['nest']['ST_GF_ampa'] = deepcopy(dic['nest']['ST_GI_ampa'])

        # EXR-SNr
        dic['nest']['ES_SN_ampa'] = {}
        dic['nest']['ES_SN_ampa']['weight'] = 0.5
        dic['nest']['ES_SN_ampa']['delay'] = 5.0
        dic['nest']['ES_SN_ampa']['type_id'] = 'static_synapse'
        dic['nest']['ES_SN_ampa']['receptor_type'] = self.rec['aeif']['AMPA_2']

        # MSN D1-SNr
        dic['nest']['M1_SN_gaba'] = {}
        dic['nest']['M1_SN_gaba'][
            'weight'] = 2. / 0.0192  # Lower based on (Connelly et al. 2010) = [4.7, 24.], 50 Hz model = [5.8, 23.5]
        dic['nest']['M1_SN_gaba']['delay'] = 7.3
        dic['nest']['M1_SN_gaba']['U'] = 0.0192
        dic['nest']['M1_SN_gaba']['tau_fac'] = 623.
        dic['nest']['M1_SN_gaba']['tau_rec'] = 559.
        dic['nest']['M1_SN_gaba']['tau_psc'] = 5.2
        dic['nest']['M1_SN_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['M1_SN_gaba']['receptor_type'] = self.rec['aeif']['GABAA_1']

        # STN-SNr
        dic['nest']['ST_SN_ampa'] = {}
        dic['nest']['ST_SN_ampa']['weight'] = 0.91 * 3.8 / 0.35  # (Shen and Johnson 2006)
        dic['nest']['ST_SN_ampa']['delay'] = 4.6
        dic['nest']['ST_SN_ampa']['U'] = 0.35  # AMPA plastic 2
        dic['nest']['ST_SN_ampa']['tau_fac'] = 0.0
        dic['nest']['ST_SN_ampa']['tau_rec'] = 800.0
        dic['nest']['ST_SN_ampa']['tau_psc'] = 12.  # n.d.; set as for STN to GPE,
        dic['nest']['ST_SN_ampa']['type_id'] = 'tsodyks_synapse'
        dic['nest']['ST_SN_ampa']['receptor_type'] = self.rec['aeif']['AMPA_1']

        # GPe TI-SNr
        dic['nest']['GI_SN_gaba'] = {}
        dic['nest']['GI_SN_gaba']['weight'] = 76. / 0.196  # 0.152*76., (Connelly et al. 2010)
        dic['nest']['GI_SN_gaba']['delay'] = 3.
        dic['nest']['GI_SN_gaba']['U'] = 0.196  # GABAA plastic,
        dic['nest']['GI_SN_gaba']['tau_fac'] = 0.0
        dic['nest']['GI_SN_gaba']['tau_rec'] = 969.0
        dic['nest']['GI_SN_gaba']['tau_psc'] = 2.1  # (Connelly et al. 2010),
        dic['nest']['GI_SN_gaba']['type_id'] = 'tsodyks_synapse'
        dic['nest']['GI_SN_gaba']['receptor_type'] = self.rec['aeif']['GABAA_2']

        dic['nest']['GF_SN_gaba'] = deepcopy(dic['nest']['GI_SN_gaba'])

        # ============
        # Input Models
        # ============

        dic['nest']['poisson_generator'] = {}
        dic['nest']['poisson_generator']['type_id'] = 'poisson_generator'
        dic['nest']['poisson_generator']['rate'] = 0.0

        # CTX-MSN D1
        dic['nest']['C1'] = {}
        dic['nest']['C1']['type_id'] = 'poisson_generator'
        dic['nest']['C1']['rate'] = 0.0

        # CTX-MSN D2
        dic['nest']['C2'] = {}
        dic['nest']['C2']['type_id'] = 'poisson_generator'
        dic['nest']['C2']['rate'] = 0.0

        # CTX-FSN
        dic['nest']['CF'] = {}
        dic['nest']['CF']['type_id'] = 'poisson_generator'
        dic['nest']['CF']['rate'] = 0.0

        # CTX-STN
        dic['nest']['CS'] = {}
        dic['nest']['CS']['type_id'] = 'poisson_generator'
        dic['nest']['CS']['rate'] = 0.0

        # EXT-GPe type A
        dic['nest']['EA'] = {}
        dic['nest']['EA']['type_id'] = 'poisson_generator'
        dic['nest']['EA']['rate'] = 0.0

        # EXT-GPe type I
        dic['nest']['EI'] = {}
        dic['nest']['EI']['type_id'] = 'poisson_generator'
        dic['nest']['EI']['rate'] = 0.0

        dic['nest']['EF'] = {}
        dic['nest']['EF']['type_id'] = 'poisson_generator'
        dic['nest']['EF']['rate'] = 0.0

        # EXT-SNr
        dic['nest']['ES'] = {}
        dic['nest']['ES']['type_id'] = 'poisson_generator'
        dic['nest']['ES']['rate'] = 0.0

        # =============
        # Neuron Models
        # =============

        # MSN
        # ===
        dic['nest']['MS'] = {}
        dic['nest']['MS']['type_id'] = 'izhik_cond_exp'

        dic['nest']['MS']['a'] = 0.01  # (E.M. Izhikevich 2007)
        dic['nest']['MS']['b_1'] = -20.  # (E.M. Izhikevich 2007)
        dic['nest']['MS']['b_2'] = -20.  # (E.M. Izhikevich 2007)
        dic['nest']['MS']['c'] = -55.  # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['C_m'] = 15.2  # (Humphries, Lepora, et al. 2009) # C izh
        dic['nest']['MS']['d'] = 66.9  # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['E_L'] = -81.85  # (Humphries, Lepora, et al. 2009) # v_r in izh
        dic['nest']['MS']['I_e'] = 0.
        dic['nest']['MS']['k'] = 1.  # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_peak'] = 40.  # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_b'] = dic['nest']['MS']['E_L']  # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_th'] = -29.7  # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['V_m'] = 80.

        # CTX_MSN
        dic['nest']['MS']['AMPA_1_Tau_decay'] = 12.  # (Ellender 2011)
        dic['nest']['MS']['AMPA_1_E_rev'] = 0.  # (Humphries, Wood, et al. 2009)

        dic['nest']['MS']['NMDA_1_Tau_decay'] = 160.  # (Humphries, Wood, et al. 2009)
        dic['nest']['MS']['NMDA_1_E_rev'] = dic['nest']['MS']['AMPA_1_E_rev']
        dic['nest']['MS']['NMDA_1_Vact'] = -20.0
        dic['nest']['MS']['NMDA_1_Sact'] = 16.0

        # From MSN
        dic['nest']['MS']['GABAA_2_Tau_decay'] = 12.
        dic['nest']['MS']['GABAA_2_E_rev'] = -74.  # Koos 2004
        dic['nest']['MS']['beta_I_GABAA_2'] = 0.56  # -0.625 #Dopamine leads to weakening of MSN synspase

        # From FSN
        dic['nest']['MS']['GABAA_1_E_rev'] = -74.  # Koos 2004
        dic['nest']['MS']['GABAA_1_Tau_decay'] = GetNest('FS_M1_gaba', 'tau_psc')

        # From GPE
        dic['nest']['MS']['GABAA_3_Tau_decay'] = 87. # 12 * 5.
        dic['nest']['MS']['GABAA_3_E_rev'] = -74.  # n.d. set as for MSN and FSN
        dic['nest']['MS']['beta_I_GABAA_3'] = 0.0  # -0.625 #Dopamine leads to weakening of MSN synspase

        dic['nest']['MS']['tata_dop'] = DepNetw('calc_tata_dop')

        dic['nest']['M1'] = deepcopy(dic['nest']['MS'])
        dic['nest']['M1']['d'] = 66.9  # (E.M. Izhikevich 2007)
        dic['nest']['M1']['E_L'] = -81.85  # (E.M. Izhikevich 2007)

        dic['nest']['M1']['beta_d'] = 0.45
        dic['nest']['M1']['beta_E_L'] = -0.0282  # Minus size it is plus in Humphrie 2009
        dic['nest']['M1']['beta_V_b'] = dic['nest']['M1']['beta_E_L']
        dic['nest']['M1']['beta_I_NMDA_1'] = 1.04  # -1.04 #Minus size it is plus in Humphrie 2009

        dic['nest']['M1']['GABAA_3_Tau_decay'] = 87.

        dic['nest']['M1_low'] = deepcopy(dic['nest']['M1'])
        dic['nest']['M1_low']['beta_I_GABAA_3'] = f_beta_rm(2.6)
        dic['nest']['M1_low']['beta_I_GABAA_2'] = f_beta_rm(0.25)  #

        dic['nest']['M1_high'] = deepcopy(dic['nest']['M1'])
        dic['nest']['M1_high']['GABAA_1_E_rev'] = -64.  # (Bracci & Panzeri 2006)
        dic['nest']['M1_high']['GABAA_2_E_rev'] = -64.  # (Bracci & Panzeri 2006)
        dic['nest']['M1_high']['GABAA_3_E_rev'] = -64.  # n.d. set asfor MSN and FSN



        dic['nest']['M2'] = deepcopy(dic['nest']['MS'])
        dic['nest']['M2']['d'] = 91.  # (E.M. Izhikevich 2007)
        dic['nest']['M2']['E_L'] = -80.  # (E.M. Izhikevich 2007)
        dic['nest']['M2']['V_b'] = dic['nest']['M2']['E_L']
        dic['nest']['M2']['beta_I_AMPA_1'] = -0.26  # 0.26
        dic['nest']['M2']['GABAA_3_Tau_decay'] = 87.

        dic['nest']['M2_low'] = deepcopy(dic['nest']['M2'])
        dic['nest']['M2_low']['beta_I_GABAA_3'] = f_beta_rm(2.5)
        dic['nest']['M2_low']['beta_I_GABAA_2'] = f_beta_rm(0.25)  #
        dic['nest']['M2_low']['GABAA_3_Tau_decay'] = 76.

        dic['nest']['M2_high'] = deepcopy(dic['nest']['M2'])
        dic['nest']['M2_high']['GABAA_1_E_rev'] = -64.  # (Bracci & Panzeri 2006)
        dic['nest']['M2_high']['GABAA_2_E_rev'] = -64.  # (Bracci & Panzeri 2006)
        dic['nest']['M2_high']['GABAA_3_E_rev'] = -64.  # n.d. set asfor MSN and FSN

        # FSN
        # ===
        dic['nest']['FS'] = {}
        dic['nest']['FS']['type_id'] = 'izhik_cond_exp'

        dic['nest']['FS']['a'] = 0.2  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['b_1'] = 0.0  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['b_2'] = 0.025  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['c'] = -60.  # (Tateno et al. 2004)
        dic['nest']['FS']['C_m'] = 80.  # (Tateno et al. 2004)
        dic['nest']['FS']['d'] = 0.  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['E_L'] = -70.  # *(1-0.8*0.1)   # (Tateno et al. 2004)
        dic['nest']['FS']['I_e'] = 0.
        dic['nest']['FS']['k'] = 1.  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['p_1'] = 1.  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['p_2'] = 3.  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['V_b'] = -55.  # Humphries 2009
        dic['nest']['FS']['V_peak'] = 25.  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['V_th'] = -50.  # (Tateno et al. 2004)

        # CTX-FSN
        dic['nest']['FS']['AMPA_1_Tau_decay'] = 12.  # CTX to FSN ampa
        dic['nest']['FS']['AMPA_1_E_rev'] = 0.  # n.d. set as for  CTX to MSN

        # From FSN
        dic['nest']['FS']['GABAA_1_E_rev'] = -74.  # n.d.; set as for MSNs
        dic['nest']['FS']['GABAA_1_Tau_decay'] = GetNest('FS_FS_gaba', 'tau_psc')

        # From GPe TA
        dic['nest']['FS']['GABAA_2_Tau_decay'] = 66.
        dic['nest']['FS']['GABAA_2_E_rev'] = -74.  # n.d. set as for MSNs

        # From GPe TI (10 % TF)
        dic['nest']['FS']['GABAA_3_Tau_decay'] = 17.
        dic['nest']['FS']['GABAA_3_E_rev'] = -74.  # n.d. set as for MSNs

        dic['nest']['FS']['beta_E_L'] = 0.078
        dic['nest']['FS']['tata_dop'] = DepNetw('calc_tata_dop')

        dic['nest']['FS']['beta_I_GABAA_1'] = -0.83  # 0.8 # From FSN
        dic['nest']['FS']['beta_I_GABAA_2'] = 0.0  # -0.83 #0.8 # From GPe A
        dic['nest']['FS']['beta_I_GABAA_3'] = 0.0  # -0.83 #0.8 # From GPe A

        dic['nest']['FS_low'] = deepcopy(dic['nest']['FS'])
        dic['nest']['FS_low']['beta_I_GABAA_2'] = f_beta_rm(1.6)

        dic['nest']['FS_high'] = deepcopy(dic['nest']['FS'])
        dic['nest']['FS_high']['GABAA_1_E_rev'] = -64.  # n.d. set as for MSNs
        dic['nest']['FS_high']['GABAA_2_E_rev'] = -64.  # n.d. set as for MSNs

        # STN
        # ===

        dic['nest']['ST'] = {}
        dic['nest']['ST']['type_id'] = 'my_aeif_cond_exp'

        dic['nest']['ST']['tau_w'] = 333.0  # I-V relation, spike frequency adaptation
        dic['nest']['ST']['a_1'] = 0.3  # I-V relation
        dic['nest']['ST']['a_2'] = 0.0  # I-V relation
        dic['nest']['ST']['b'] = 0.05  # 0.1 #0.1#200./5.
        dic['nest']['ST']['C_m'] = 60.0  # t_m/R_in
        dic['nest']['ST']['Delta_T'] = 16.2
        dic['nest']['ST']['g_L'] = 10.0
        dic['nest']['ST']['E_L'] = -80.2
        dic['nest']['ST']['I_e'] = 6.0
        dic['nest']['ST']['V_peak'] = 15.0
        dic['nest']['ST']['V_reset'] = -70.0  # I-V relation
        dic['nest']['ST']['V_a'] = -70.0  # I-V relation
        dic['nest']['ST']['V_th'] = -64.0

        dic['nest']['ST']['V_reset_slope1'] = -10.  # Slope u<0
        dic['nest']['ST']['V_reset_slope2'] = .0  # Slope u>=0
        dic['nest']['ST']['V_reset_max_slope1'] = -60.  # Max v restet u<0
        dic['nest']['ST']['V_reset_max_slope2'] = dic['nest']['ST']['V_reset']  # Max v restet u>=0

        # CTX-STN
        dic['nest']['ST']['AMPA_1_Tau_decay'] = 4.0  # (Baufreton et al. 2005)
        dic['nest']['ST']['AMPA_1_E_rev'] = 0.  # (Baufreton et al. 2009)

        dic['nest']['ST']['NMDA_1_Tau_decay'] = 160.  # n.d. estimated 1:2 AMPA:NMDA
        dic['nest']['ST']['NMDA_1_E_rev'] = 0.  # n.d.; set as  E_ampa
        dic['nest']['ST']['NMDA_1_Vact'] = -20.0
        dic['nest']['ST']['NMDA_1_Sact'] = 16.0

        # GPE-STN
        dic['nest']['ST']['GABAA_1_Tau_decay'] = 8.  # (Baufreton et al. 2009)
        dic['nest']['ST']['GABAA_1_E_rev'] = -84.0  # (Baufreton et al. 2009)

        dic['nest']['ST']['beta_I_AMPA_1'] = f_beta_rm(2.5) #-0.45  # 0.4 # From Cortex
        dic['nest']['ST']['beta_I_NMDA_1'] = f_beta_rm(2.5) #-0.45  # 0.4 # From Cortex
        dic['nest']['ST']['beta_I_GABAA_1'] = -0.24  # 0.4 # From GPe I

        dic['nest']['ST']['tata_dop'] = DepNetw('calc_tata_dop')

        # GPE
        # ===

        dic['nest']['GP'] = {}
        dic['nest']['GP']['type_id'] = 'my_aeif_cond_exp'

        dic['nest']['GP']['a_1'] = 2.5  # I-V relation # I-V relation
        dic['nest']['GP']['a_2'] = dic['nest']['GP']['a_1']
        dic['nest']['GP']['b'] = 70.  # I-F relation
        dic['nest']['GP']['C_m'] = 40.  # t_m/R_in
        dic['nest']['GP']['Delta_T'] = 1.7
        dic['nest']['GP']['g_L'] = 1.
        dic['nest']['GP'][
            'E_L'] = -55.1  # v_t    = -56.4                                                               #
        dic['nest']['GP']['I_e'] = 0.
        dic['nest']['GP']['tau_w'] = 20.  # I-V relation, spike frequency adaptation
        dic['nest']['GP']['V_peak'] = 15.  # Cooper and standford
        dic['nest']['GP']['V_reset'] = -60.  # I-V relation
        dic['nest']['GP']['V_th'] = -54.7
        dic['nest']['GP']['V_a'] = dic['nest']['GP']['E_L']

        # STN-GPe
        dic['nest']['GP']['AMPA_1_Tau_decay'] = 12.  # (Hanson & Dieter Jaeger 2002)
        dic['nest']['GP']['AMPA_1_E_rev'] = 0.  # n.d.; same as CTX to STN

        dic['nest']['GP']['NMDA_1_Tau_decay'] = 100.  # n.d.; estimated
        dic['nest']['GP']['NMDA_1_E_rev'] = 0.  # n.d.; same as CTX to STN
        dic['nest']['GP']['NMDA_1_Vact'] = -20.0
        dic['nest']['GP']['NMDA_1_Sact'] = 16.0

        # EXT-GPe
        dic['nest']['GP']['AMPA_2_Tau_decay'] = 5.0
        dic['nest']['GP']['AMPA_2_E_rev'] = 0.0

        # GPe-GPe
        dic['nest']['GP']['GABAA_2_Tau_decay'] = 5.  # (Sims et al. 2008)
        dic['nest']['GP']['GABAA_2_E_rev'] = -65.  # n.d same as for MSN (Rav-Acha 2005)

        dic['nest']['GP']['beta_E_L'] = 0.181
        dic['nest']['GP']['beta_V_a'] = 0.181
        dic['nest']['GP']['beta_I_AMPA_1'] = -0.45  # 0.4 # From STN
        dic['nest']['GP']['beta_I_GABAA_1'] = 0.0  # 0.8 # From From MSNs
        dic['nest']['GP']['beta_I_GABAA_2'] = -0.83  # 0.8 # From GPe A

        dic['nest']['GP']['tata_dop'] = DepNetw('calc_tata_dop')

        dic['nest']['GA'] = deepcopy(dic['nest']['GP'])
        dic['nest']['GA']['C_m'] = 60.
        dic['nest']['GA']['Delta_T'] = 2.55
        dic['nest']['GA']['b'] = 105.

        #         dic['nest']['GA']['b'] = dic['nest']['GA']['b'] *1.5 # I-F relation
        #         dic['nest']['GA']['C_m']=dic['nest']['GA']['C_m']*1.5
        # #         dic['nest']['GA']['a_1']  = 0.5
        #         dic['nest']['GA']['Delta_T'] = dic['nest']['GA']['Delta_T']*1.5 # 1.7*2
        dic['nest']['GI'] = deepcopy(dic['nest']['GP'])
        dic['nest']['GI']['beta_I_GABAA_1']=f_beta_rm(2)

        # MSN D2-GPe
        dic['nest']['GI']['GABAA_1_E_rev'] = -65.  # (Rav-Acha et al. 2005)
        dic['nest']['GI']['GABAA_1_Tau_decay'] = GetNest('M2_GI_gaba', 'tau_psc')  # (Shen et al. 2008)

        dic['nest']['GF'] = deepcopy(dic['nest']['GI'])
        dic['nest']['GF']['beta_I_GABAA_1'] = f_beta_rm(2)

        # SNR
        # ===

        dic['nest']['SN'] = {}
        dic['nest']['SN']['type_id'] = 'my_aeif_cond_exp'

        dic['nest']['SN']['tau_w'] = 20.  # I-V relation, spike frequency adaptation
        dic['nest']['SN']['a_1'] = 3.  # I-V relation
        dic['nest']['SN']['a_2'] = dic['nest']['SN']['a_1']  # I-V relation
        dic['nest']['SN']['b'] = 200.  # I-F relation
        dic['nest']['SN']['C_m'] = 80.  # t_m/R_in
        dic['nest']['SN']['Delta_T'] = 1.8
        dic['nest']['SN']['g_L'] = 3.
        dic['nest']['SN']['E_L'] = -55.8  #
        dic['nest']['SN']['I_e'] = 15.0
        dic['nest']['SN']['V_peak'] = 20.  #
        dic['nest']['SN']['V_reset'] = -65.  # I-V relation
        dic['nest']['SN']['V_th'] = -55.2  #
        dic['nest']['SN']['V_a'] = dic['nest']['SN']['E_L']  # I-V relation

        # STN-SNr
        dic['nest']['SN']['AMPA_1_Tau_decay'] = 12.  # n.d.; set as for STN to GPE
        dic['nest']['SN']['AMPA_1_E_rev'] = 0.  # n.d. same as CTX to STN

        # EXT-SNr
        dic['nest']['SN']['AMPA_2_Tau_decay'] = 5.0
        dic['nest']['SN']['AMPA_2_E_rev'] = 0.

        # MSN D1-SNr
        dic['nest']['SN']['GABAA_1_E_rev'] = -80.  # (Connelly et al. 2010)
        dic['nest']['SN']['GABAA_1_Tau_decay'] = GetNest('M1_SN_gaba', 'tau_psc')  # (Connelly et al. 2010)
        dic['nest']['SN']['beta_I_GABAA_1'] = 0.56  # 0.8 # From MSN D1

        # GPe-SNr
        dic['nest']['SN']['GABAA_2_E_rev'] = -72.  # (Connelly et al. 2010)
        dic['nest']['SN']['GABAA_2_Tau_decay'] = GetNest('GI_SN_gaba', 'tau_psc')

        dic['nest']['SN']['beta_E_L'] = 0.0896
        dic['nest']['SN']['beta_V_a'] = 0.0896

        dic['nest']['SN']['tata_dop'] = DepNetw('calc_tata_dop')

        # ========================
        # Default node parameters
        # ========================

        dic['node'] = {}

        # Model inputs
        inputs = {'C1': {'target': 'M1', 'rate': 560.},  # 530.-20.0},
                  'C2': {'target': 'M2', 'rate': 740.},  # 690.-20.0},
                  'CF': {'target': 'FS', 'rate': 807.5},  # 624.},
                  'CS': {'target': 'ST', 'rate': 250.0},  # 160.},#295
                  'EA': {'target': 'GA', 'rate': 300.},
                  'EI': {'target': 'GI', 'rate': 1430.0},  # 1130.},
                  'EF': {'target': 'GF', 'rate': 1430.0},  # 1130.},
                  'ES': {'target': 'SN', 'rate': 2000.}}  # 295

        for key, val in inputs.items():
            d = self._get_defaults_node_input(key, val['target'])
            inputs[key] = misc.dict_update(d, inputs[key])

        dic['node'] = misc.dict_merge(dic['node'], inputs)

        network = {'M1': {'model': 'M1_low', 'I_vitro': 0.0, 'I_vivo': 0.0,},
                   'M2': {'model': 'M2_low', 'I_vitro': 0.0, 'I_vivo': 0.0,},
                   'FS': {'model': 'FS_low', 'I_vitro': 0.0, 'I_vivo': 0.0,},
                   'ST': {'model': 'ST', 'I_vitro': 6.0, 'I_vivo': 6.0,},
                   'GA': {'model': 'GA', 'I_vitro': 1.0, 'I_vivo': -3.6,},  # 8 Hz, -8
                   'GI': {'model': 'GI', 'I_vitro': 12.0, 'I_vivo': 4.5,},  # 18 Hz, 56
                   'GF': {'model': 'GF', 'I_vitro': 12.0, 'I_vivo': 4.5,},  # 51, 56
                   'SN': {'model': 'SN', 'I_vitro': 15.0, 'I_vivo': 19.2,}}

        GA_prop = GetNetw('GA_prop')
        GP_tr = GetNetw('GP_rate')
        GA_tr = GetNode('GA', 'rate')
        network['M1'].update({'rate': 0.1, 'rate_in_vitro': 0.0})
        network['M2'].update({'rate': 0.1, 'rate_in_vitro': 0.0})
        network['FS'].update({'rate': 15.0, 'rate_in_vitro': 0.0})
        network['ST'].update({'rate': 10.0, 'rate_in_vitro': 10.0})
        network['GA'].update({'rate': 5.0, 'rate_in_vitro': 4.0})
        network['GI'].update({'rate': (GP_tr - GA_prop * GA_tr) / (1 - GA_prop),
                              'rate_in_vitro': 15.0})
        network['GF'].update({'rate': (GP_tr - GA_prop * GA_tr) / (1 - GA_prop),
                              'rate_in_vitro': 15.0})
        network['SN'].update({'rate': 30., 'rate_in_vitro': 15.0})

        # Randomization of C_m and V_m
        for key in network.keys():
            # model=GetNode(key, 'model')
            d = self._get_defaults_node_network(key)

            network[key] = misc.dict_update(d, network[key])

        d = {'M1': {'n_sets': GetNetw('n_actions')},
             'M2': {'n_sets': GetNetw('n_actions')},
             'GI': {'n_sets': GetNetw('n_actions')},
             'GF': {'n_sets': GetNetw('n_actions')},
             'SN': {'n_sets': GetNetw('n_actions')}}
        network = misc.dict_update(network, d)

        dic['node'] = misc.dict_update(dic['node'], network)

        # ========================
        # Default conn parameters
        # ========================


        # Input
        conns = {'C1_M1_ampa': {},
                 'C1_M1_nmda': {},
                 'C2_M2_ampa': {},
                 'C2_M2_nmda': {},
                 'CF_FS_ampa': {},
                 'CS_ST_ampa': {},
                 'CS_ST_nmda': {},
                 'EA_GA_ampa': {},
                 'EI_GI_ampa': {},
                 'EF_GF_ampa': {},
                 'ES_SN_ampa': {}}

        for k in conns.keys():
            conns[k].update({'fan_in0': 1, 'rule': '1-1'})

        dic['conn'] = {}
        # Number of incomming connections from a nucleus to another nucleus.

        # conns.update(d)
        # Network
        GP_fi = GetNetw('GP_fan_in')
        GA_pfi = GetNetw('GP_fan_in_prop_GA')

        M1_M1 = int(round(2800 * 0.13)) / GetNetw('sub_sampling', 'M1')
        M1_M2 = int(round(2800 * 0.03)) / GetNetw('sub_sampling', 'M1')
        M2_M1 = int(round(2800 * 0.14)) / GetNetw('sub_sampling', 'M2')
        M2_M2 = int(round(2800 * 0.18)) / GetNetw('sub_sampling', 'M2')

        FS_M1 = int(round(60 * 0.27))
        FS_M2 = int(round(60 * 0.18))

        GA_XX = 5.  # GP_fi*GA_pfi
        GI_GX = 22  # 25*GetNetw('GI_prop')/(1-GetNetw('GA_prop')) #GP_fi*(1-GA_pfi)*GetNetw('GI_prop')/(1-GetNetw('GA_prop'))
        GF_GX = 3  # 25-GI_GX #GP_fi*(1-GA_pfi)*GetNetw('GF_prop')/(1-GetNetw('GA_prop'))

        GI_ST = 30 * GetNetw('GI_prop') / (1 - GetNetw('GA_prop'))
        GF_ST = 30 * GetNetw('GF_prop') / (1 - GetNetw('GA_prop'))

        GI_SN = 32 * GetNetw('GI_prop') / (1 - GetNetw('GA_prop'))
        GF_SN = 32 * GetNetw('GF_prop') / (1 - GetNetw('GA_prop'))

        M1_SN = 500 * 1 / GetNetw('sub_sampling', 'M1')
        M2_GX = 500 * 1 / GetNetw('sub_sampling', 'M2')

        d = {'M1_SN_gaba': {'fan_in0': M1_SN, 'rule': 'set-set'},
             'M2_GF_gaba': {'fan_in0': M2_GX, 'rule': 'set-set'},
             'M2_GI_gaba': {'fan_in0': M2_GX, 'rule': 'set-set'},
             'M2_GA_gaba': {'fan_in0': M2_GX, 'rule': 'all-all'},

             'M1_M1_gaba': {'fan_in0': M1_M1, 'rule': 'all-all'},
             'M1_M2_gaba': {'fan_in0': M1_M2, 'rule': 'all-all'},
             'M2_M1_gaba': {'fan_in0': M2_M1, 'rule': 'all-all'},
             'M2_M2_gaba': {'fan_in0': M2_M2, 'rule': 'all-all'},

             'FS_M1_gaba': {'fan_in0': FS_M1, 'rule': 'all-all'},
             'FS_M2_gaba': {'fan_in0': FS_M2, 'rule': 'all-all'},
             'FS_FS_gaba': {'fan_in0': 9, 'rule': 'all-all'},

             'ST_GA_ampa': {'fan_in0': 30, 'rule': 'all-all'},
             'ST_GF_ampa': {'fan_in0': 30, 'rule': 'all-all'},
             'ST_GI_ampa': {'fan_in0': 30, 'rule': 'all-all'},
             'ST_SN_ampa': {'fan_in0': 30, 'rule': 'all-all'},
             'ST_ST_ampa': {'fan_in0': 10, 'rule': 'all-all'},

             'GA_FS_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GA_M1_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GA_M2_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GA_GA_gaba': {'fan_in0': GA_XX, 'rule': 'all-all'},
             'GA_GI_gaba': {'fan_in0': GA_XX, 'rule': 'all-all'},
             'GA_GF_gaba': {'fan_in0': GA_XX, 'rule': 'all-all'},

             'GI_FS_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GI_M1_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GI_M2_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GI_GA_gaba': {'fan_in0': GI_GX, 'rule': 'all-all'},
             'GI_GI_gaba': {'fan_in0': GI_GX, 'rule': 'all-all'},
             'GI_ST_gaba': {'fan_in0': GI_ST, 'rule': 'all-all'},
             'GI_SN_gaba': {'fan_in0': GI_SN, 'rule': 'all-all'},
             'GI_GF_gaba': {'fan_in0': GI_GX, 'rule': 'all-all'},

             'GF_FS_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GF_M1_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GF_M2_gaba': {'fan_in0': 10, 'rule': 'all-all'},
             'GF_GA_gaba': {'fan_in0': GF_GX, 'rule': 'all-all'},
             'GF_GI_gaba': {'fan_in0': GF_GX, 'rule': 'all-all'},
             'GF_ST_gaba': {'fan_in0': GF_ST, 'rule': 'all-all'},
             'GF_SN_gaba': {'fan_in0': GF_SN, 'rule': 'all-all'},
             'GF_GF_gaba': {'fan_in0': GF_GX, 'rule': 'all-all'},

             }

        conns.update(d)
        #         pp(d)

        # Add extend to conn
        for k in sorted(conns.keys()):
            source = k.split('_')[0]

            # Cortical input do not randomally change CTX input

            if dic['node'][source]['type'] == 'input':
                kwargs = {'conn_type': 'constant'}
            if dic['node'][source]['type'] == 'network':
                kwargs = {'conn_type': 'uniform'}

            # Get Defaults
            d = self._get_defaults_conn(k, k, **kwargs)

            # Set beta for FS_M2
            if k == 'FS_M2_gaba':
                d['beta_fan_in'] = -0.9  # 0.8
            # Set beta for MS_MS
            if k in ['M1_M1_gaba', 'M1_M2_gaba', 'M2_M1_gaba', 'M2_M2_gaba']:
                d['beta_fan_in'] = 0.882352941176 #0.56

            e = None
            if k in ['M1_M1_gaba', 'M1_M2_gaba', 'M2_M1_gaba',
                     'M2_M2_gaba']:  # pre[0:2] in ['M1', 'M2'] and post[0:2] in ['M1', 'M2']:
                e = 2800. / (DepNode('M1', 'calc_n') + DepNode('M2', 'calc_n'))  # None
                e = min(e, 0.5)
            elif k in ['FS_M1', 'FS_M2', 'FS_FS']:  # pre[0:2] =='FS' and post[0:2] in ['M1', 'M2', 'FS']:
                e = 560. / (DepNode('M1', 'calc_n') + DepNode('M2', 'calc_n'))
                e = min(e, 0.5)
            if e != None:
                d['mask'] = [-e, e]

                # Default they are leasioned
            if k in [
                'M2_GA_gaba',
                'GI_FS_gaba', 'GI_M1_gaba', 'GI_M2_gaba',
                # 'GF_FS_gaba',
                'GF_M1_gaba', 'GF_M2_gaba'
            ]:
                d['lesion'] = True
            conns[k] = misc.dict_update(d, conns[k])

        dic['conn'] = misc.dict_update(dic['conn'], conns)

        return dic


class EneuroPar(Par_base, EneuroParBase, Par_base_mixin):
    pass

