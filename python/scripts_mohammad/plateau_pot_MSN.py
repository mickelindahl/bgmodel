# Create by Mohammad Mohagheghi on 05/03/18.

from core.network import structure
from core.network import engine
from core import my_nest, data_to_disk
from core.network.parameters.eneuro import EneuroPar

from core.network.parameters.eneuro_activation import EneuroActivationPar
from core.network.parameters.eneuro_activation_beta import EneuroActivationBetaPar
from core.network.parameters.eneuro_sw import EneuroSwPar

from core.network.parameters.eneuro_ramp import EneuroRampPar
# from core.network.default_params import Beta
from scripts_inhibition.base_oscillation import add_GI, add_GPe
import pprint
import json
import os
import sys

import plot
import randomized_params_plot
import mean_firing_rates
import mean_firing_rates_plot
import list_parameters

import scipy.io as sio
import numpy
import nest
import nest.raster_plot as raster
import nest.voltage_trace as voltmeter
import matplotlib.pylab as plt

pp = pprint.pprint


def save_node_random_params(pops, path):

    d = {}

    for node in ['FS', 'GI',  'GF', 'GA', 'M1', 'M2', 'ST', 'SN']:

        d[node] = {}

        for param in ['V_th', 'V_m', 'C_m', 'E_L']:

            d[node][param] = [s[param] for s in my_nest.GetStatus(pops[node].ids)]


        d[node]['V_th-E_L']= [a-b for a,b in zip(d[node]['V_th'], d[node]['E_L'])]


    if not os.path.isdir(os.path.dirname(path)):
        data_to_disk.mkdir(os.path.dirname(path))

    json.dump( d, open(path, 'w'))

def build(par):
    # ******
    # Build
    # ******
    surfs, pops = structure.build(par.get_nest(),
                                  par.get_surf(),
                                  par.get_popu())

    return surfs, pops


def connect(par, surfs, pops):
    args = [pops, surfs, par.get_nest(), par.get_conn(), True]

    structure.connect(*args)


def postprocessing(pops):
    d = {}

    signal_type = 'spike_signal'

    d_signals = pops.get(signal_type)

    for name, signal in d_signals.items():
        engine.fill_duds_node(d, name, signal_type, signal)

    add_GI(d)
    add_GPe(d)

    return d


def test_plateau_pot(mode):
    my_nest.ResetKernel()

    # Get parameters
    # par = EneuroBetaPar(other=EneuroPar())

    if mode in ['activation-control',  'activation-dopamine-depleted']:

        if mode == 'activation-control':

            par = EneuroActivationPar(other=EneuroPar())
            dop = 0.8

        elif mode == 'activation-dopamine-depleted':
            par = EneuroActivationBetaPar(other=EneuroPar())
            dop = 0.0

    elif mode in ['slow-wave-control', 'slow-wave-dopamine-depleted']:

        par = EneuroSwPar(other=EneuroPar())

        if mode == 'slow-wave-control':
            dop = 0.8
        elif mode == 'slow-wave-dopamine-depleted':
            dop = 0.0

    params_dic = par.dic
    cat = 'conn'
    connection = 'C1_M1_ampa'
    key = 'lesion'
    params_dic[cat][connection][key] = True
    connection = 'C2_M2_ampa'
    params_dic[cat][connection][key] = True

    connection = 'C1_M1_nmda'
    params_dic[cat][connection][key] = True
    connection = 'C2_M2_nmda'
    params_dic[cat][connection][key] = True

    # Modifying the firing rate
    params_dic['node']['C1']['rate'] = 50000.0
    params_dic['node']['C2']['rate'] = 50000.0

    stim_dc_start = 1000.0
    stim_dc_stop = 1500.0

    stim_spktr_start = 0.0
    stim_spktr_stop = 0.0

    spiketimes = numpy.array(500.)
    spikeweights = numpy.array(1.0)

    # times = numpy.arange(510.,600.,100.)
    # spiketimes = numpy.append(spiketimes,times)
    spikeweights = numpy.append(spikeweights,1.0*numpy.ones(times.size))



    dc_amp = 0.0

    # Current to MSN D1

    DC1 = nest.Create('dc_generator', 1, params = {'amplitude': dc_amp,
                                                   'start': stim_dc_start,
                                                   'stop': stim_dc_stop})

    # Spikes to MSN D1

    SG1 = nest.Create('spike_generator',1,params = {'spike_times':spiketimes,
                                                    'spike_weights':spikeweights})

    # Generating cortical inputs to MSN D1

    # C1_n = params_dic['node']['C1']['n']
    C1_n = 1
    C1_model = params_dic['nest']['C1']['type_id']
    C1_rate = params_dic['node']['C1']['rate']
    C1_start = 1.0
    # C1_stop = params_dic['node']['C1']['spike_setup'][0]['t_stop']

    C1 = nest.Create(C1_model, C1_n, params = {'rate':C1_rate,
                                             'start':stim_spktr_start,
                                             'stop':stim_spktr_stop})

    # Generating cortical inputs to MSN D2

    # C1_n = params_dic['node']['C1']['n']
    C2_n = 1
    C2_model = params_dic['nest']['C2']['type_id']
    C2_rate = params_dic['node']['C2']['rate']
    C2_start = 1.0
    # C2_stop = params_dic['node']['C2']['spike_setup'][0]['t_stop']

    C2 = nest.Create(C2_model, C2_n, params = {'rate':C2_rate,
                                             'start':stim_spktr_start,
                                             'stop':stim_spktr_stop})

    # Generating cortical inputs to MSN D1 (NMDA)

    # C1_n = params_dic['node']['C1']['n']
    C1_n = 1
    C1_model = params_dic['nest']['C1']['type_id']
    C1_rate = params_dic['node']['C1']['rate']
    C1_start = 1.0
    C1_stop = params_dic['node']['C1']['spike_setup'][0]['t_stop']

    C1N = nest.Create(C1_model, C1_n, params = {'rate':C1_rate,
                                             'start':stim_spktr_start,
                                             'stop':stim_spktr_stop})

    # Generating cortical inputs to MSN D2 (NMDA)

    # C1_n = params_dic['node']['C1']['n']
    C2_n = 1
    C2_model = params_dic['nest']['C2']['type_id']
    C2_rate = params_dic['node']['C2']['rate']
    C2_start = 1.0
    C2_stop = params_dic['node']['C2']['spike_setup'][0]['t_stop']

    C2N = nest.Create(C2_model, C2_n, params = {'rate':C2_rate,
                                             'start':stim_spktr_start,
                                             'stop':stim_spktr_stop})

    # M1_n = params_dic['node']['M1']['n']
    M1_n = 1
    M1_model = params_dic['nest']['M1']['type_id']
    M1_params = params_dic['nest']['M1']
    M1_params.pop('type_id')
    M1_params['I_e'] = params_dic['node']['M1']['nest_params']['I_e']


    nest.CopyModel(M1_model, 'MSND1', params = M1_params)

    M1 = nest.Create('MSND1',M1_n)

    # M2_n = params_dic['node']['M2']['n']
    M2_n = 1
    M2_model = params_dic['nest']['M2']['type_id']
    M2_params = params_dic['nest']['M2']
    M2_params.pop('type_id')
    M2_params['I_e'] = params_dic['node']['M2']['nest_params']['I_e']

    nest.CopyModel(M2_model, 'MSND2', params = M2_params)

    M2 = nest.Create('MSND2', M2_n)


    conn_name = 'C1_M1_ampa'
    syndic = params_dic['nest'][conn_name]
    synmodel = syndic.pop('type_id')
    syn = params_dic['conn'][conn_name]['syn']
    isconn = params_dic['conn'][conn_name]['lesion']

    conn_het_dic = params_dic['conn'][conn_name]

    if not isconn:
        nest.CopyModel(synmodel, syn, syndic)
        syndic['model'] = syn
        syndic.pop('receptor_type')
        nest.Connect(C1, M1, syn_spec=syndic)

    # Connecting cortical input to MSN D1 (NMDA)

    conn_name = 'C1_M1_nmda'
    syndic = params_dic['nest'][conn_name]
    synmodel = syndic.pop('type_id')
    syn = params_dic['conn'][conn_name]['syn']
    isconn = params_dic['conn'][conn_name]['lesion']

    conn_het_dic = params_dic['conn'][conn_name]

    if not isconn:
        nest.CopyModel(synmodel, syn, syndic)
        syndic['model'] = syn
        syndic.pop('receptor_type')
        nest.Connect(C1N, M1, syn_spec=syndic)

    # Connecting cortical input to MSN D2 (AMPA)

    conn_name = 'C2_M2_ampa'
    syndic = params_dic['nest'][conn_name]
    synmodel = syndic.pop('type_id')
    syn = params_dic['conn'][conn_name]['syn']
    isconn = params_dic['conn'][conn_name]['lesion']

    conn_het_dic = params_dic['conn'][conn_name]

    if not isconn:
        nest.CopyModel(synmodel, syn, syndic)
        syndic['model'] = syn
        syndic.pop('receptor_type')
        nest.Connect(C2N, M2, syn_spec=syndic)

    # Connecting cortical input to MSN D2 (NMDA)

    conn_name = 'C2_M2_nmda'
    syndic = params_dic['nest'][conn_name]
    synmodel = syndic.pop('type_id')
    syn = params_dic['conn'][conn_name]['syn']
    isconn = params_dic['conn'][conn_name]['lesion']

    conn_het_dic = params_dic['conn'][conn_name]

    if not isconn:
        nest.CopyModel(synmodel, syn, syndic)
        syndic['model'] = syn
        syndic.pop('receptor_type')
        nest.Connect(C2, M2, syn_spec=syndic)

    # Connecting DC current generator to MSNs

    nest.Connect(DC1, M1, syn_spec = {'receptor_type':7})
    nest.Connect(DC1, M2, syn_spec = {'receptor_type':7})

    # Connecting Spike generator to MSNs
    nest.Connect(SG1, M1, syn_spec = {'receptor_type':1})
    nest.Connect(SG1, M2, syn_spec = {'receptor_type':1})



    mul_meter = nest.Create('multimeter')
    nest.SetStatus(mul_meter, {'to_file': True, 'to_memory': True, 'record_from': ['V_m'], 'withtime': True})
    spk_det = nest.Create('spike_detector')
    nest.SetStatus(spk_det,  {'to_file': True, 'to_memory': True})

    nest.Connect(M1,spk_det)
    nest.Connect(M2,spk_det)

    nest.Connect(mul_meter,M1)
    nest.Connect(mul_meter,M2)

    dir_name = 'plateau-pot'

    base = os.path.join(os.getenv('BGMODEL_HOME_DATA'), 'example/eneuro', mode, dir_name)

    if not os.path.exists(base):
        os.makedirs(base)


    nest.SetKernelStatus({'data_path':base, 'overwrite_files': True, 'print_time': True})

    my_nest.Simulate(2000.0)

    # raster.from_device(spk_det)
    voltmeter.from_device(mul_meter)
    plt.legend(['MSN D1','MSN D2'])
    plt.savefig('/Users/Mohammad/Documents/PhD/Journal-Clubs-Seminars/Meeting-Jeanette/180320/nest-ampa-plateau-pot.pdf',format='pdf')
    # plt.show()

def GA_M1_IPSC(mode):
    my_nest.ResetKernel()

    # Get parameters
    # par = EneuroBetaPar(other=EneuroPar())

    if mode in ['activation-control',  'activation-dopamine-depleted']:

        if mode == 'activation-control':

            par = EneuroActivationPar(other=EneuroPar())
            dop = 0.8

        elif mode == 'activation-dopamine-depleted':
            par = EneuroActivationBetaPar(other=EneuroPar())
            dop = 0.0

    elif mode in ['slow-wave-control', 'slow-wave-dopamine-depleted']:

        par = EneuroSwPar(other=EneuroPar())

        if mode == 'slow-wave-control':
            dop = 0.8
        elif mode == 'slow-wave-dopamine-depleted':
            dop = 0.0

    params_dic = par.dic
    cat = 'conn'
    connection = 'C1_M1_ampa'
    key = 'lesion'
    params_dic[cat][connection][key] = False
    connection = 'C2_M2_ampa'
    params_dic[cat][connection][key] = False

    connection = 'C1_M1_nmda'
    params_dic[cat][connection][key] = False
    connection = 'C2_M2_nmda'
    params_dic[cat][connection][key] = False

    # Modifying the firing rate
    params_dic['node']['C1']['rate'] = 50000.0
    params_dic['node']['C2']['rate'] = 50000.0

    stim_dc_start = 0.0
    stim_dc_stop = 4000.0

    stim_spktr_start = 0.0
    stim_spktr_stop = 0.0

    spiketimes = numpy.array([2500.])
    spikeweights = numpy.array([10.])

    # times = numpy.arange(2510.,2600.,100.)
    # spiketimes = numpy.append(spiketimes,times)
    # spikeweights = numpy.append(spikeweights,2.5*numpy.ones(times.size))



    dc_amp = 270.0

    # Current to MSN D1

    DC1 = nest.Create('dc_generator', 1, params = {'amplitude': dc_amp,
                                                   'start': stim_dc_start,
                                                   'stop': stim_dc_stop})

    # Spikes to MSN D1

    SG1 = nest.Create('spike_generator',1,params = {'spike_times':spiketimes,
                                                    'spike_weights':spikeweights})

    SG2 = nest.Create('spike_generator',1,params = {'spike_times':spiketimes,
                                                    'spike_weights':spikeweights})


    # M1_n = params_dic['node']['M1']['n']
    M1_n = 1
    M1_model = params_dic['nest']['M1']['type_id']
    M1_params = params_dic['nest']['M1']
    M1_params.pop('type_id')
    M1_params['I_e'] = params_dic['node']['M1']['nest_params']['I_e']


    nest.CopyModel(M1_model, 'MSND1', params = M1_params)

    M1 = nest.Create('MSND1',M1_n)

    # M2_n = params_dic['node']['M2']['n']
    M2_n = 1
    M2_model = params_dic['nest']['M2']['type_id']
    M2_params = params_dic['nest']['M2']
    M2_params.pop('type_id')
    M2_params['I_e'] = params_dic['node']['M2']['nest_params']['I_e']

    nest.CopyModel(M2_model, 'MSND2', params = M2_params)

    M2 = nest.Create('MSND2', M2_n)

    # Connecting GPA input to MSN D1 (GABA)

    conn_name = 'GA_M1_gaba'
    syndic = params_dic['nest'][conn_name]
    synmodel = syndic.pop('type_id')
    syn = params_dic['conn'][conn_name]['syn']
    isconn = params_dic['conn'][conn_name]['lesion']

    conn_het_dic = params_dic['conn'][conn_name]

    if not isconn:
        nest.CopyModel(synmodel, syn, syndic)
        syndic['model'] = syn
        syndic.pop('receptor_type')
        nest.Connect(SG1, M1, syn_spec=syndic)

    # Connecting GPA input to MSN D2 (GABA)

    conn_name = 'GA_M2_gaba'
    syndic = params_dic['nest'][conn_name]
    synmodel = syndic.pop('type_id')
    syn = params_dic['conn'][conn_name]['syn']
    isconn = params_dic['conn'][conn_name]['lesion']

    conn_het_dic = params_dic['conn'][conn_name]

    if not isconn:
        nest.CopyModel(synmodel, syn, syndic)
        syndic['model'] = syn
        syndic.pop('receptor_type')
        nest.Connect(SG2, M2, syn_spec=syndic)

    # Connecting DC current generator to MSNs

    nest.Connect(DC1, M1, syn_spec = {'receptor_type':7})
    nest.Connect(DC1, M2, syn_spec = {'receptor_type':7})

    # # Connecting Spike generator to MSNs
    # nest.Connect(SG1, M1, syn_spec = {'receptor_type':1})
    # nest.Connect(SG1, M2, syn_spec = {'receptor_type':1})



    mul_meter = nest.Create('multimeter')
    nest.SetStatus(mul_meter, {'to_file': True, 'to_memory': True, 'record_from': ['V_m'], 'withtime': True})
    spk_det = nest.Create('spike_detector')
    nest.SetStatus(spk_det,  {'to_file': True, 'to_memory': True})

    nest.Connect(M1,spk_det)
    nest.Connect(M2,spk_det)

    nest.Connect(mul_meter,M1)
    nest.Connect(mul_meter,M2)

    dir_name = 'plateau-pot'

    base = os.path.join(os.getenv('BGMODEL_HOME_DATA'), 'example/eneuro', mode, dir_name)

    if not os.path.exists(base):
        os.makedirs(base)


    nest.SetKernelStatus({'data_path':base, 'overwrite_files': True, 'print_time': True})

    my_nest.Simulate(4000.0)

    # raster.from_device(spk_det)
    voltmeter.from_device(mul_meter)
    plt.legend(['MSN D1','MSN D2'])
    dir_name = 'IPSC-M1-M2'
    base = os.path.join(os.getenv('BGMODEL_HOME_DATA'), 'example', dir_name)

    if not os.path.isdir(base):
        os.mkdir(base)
    plt.show()
    # plt.savefig(base+ '/nest-IPSC_M2.pdf',format='pdf')

def find(key, dictionary):
    for k, v in dictionary.iteritems():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result

# main()
if __name__ == '__main__':

    modes = ['activation-control']

    for mode in modes:
        # test_plateau_pot(mode)
        GA_M1_IPSC(mode)
