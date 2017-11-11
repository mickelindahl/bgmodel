# Create by Mikael Lindahl on 4/12/17.

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


def main(mode, size):
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

    base = os.path.join(os.getenv('BGMODEL_HOME'), 'results/example/eneuro', str(size), mode)

    # Configure simulation parameters
    par.set({
        'simu': {
            'local_num_threads': 4,
            'path_data': base+'/data',
            'path_figure': base+'/fig',
            'path_nest': base+'/nest/',  # trailing slash important
            'stop_rec': 1000000.,
            'sim_stop': 1000000.,
            'print_time': True,
            'sd_params': {
                'to_file': True,
                'to_memory': False
            },

        },
        'netw': {
            'tata_dop':dop,
            'size': size
        },
        # 'node': {
        #     'C1': {'rate': 546.},
        #     'C2': {'rate': 722.},
        #     'CF': {'rate': 787.},
        #     'CS': {'rate': 250.},
        #     'ES': {'rate': 1530.}
        # },
    })

    par.nest_set_kernel_status()

    # Save parametesr
    list_parameters.main(base, par)

    # Clear nest data directory
    par.nest_clear_data_path({'display': True})

    # Show kernel status
    pp(my_nest.GetKernelStatus())

    # Create news populations and connections structures
    surfs, pops = build(par)

    stim_pars = {'stim_start':4000.0,
                 'h_rate':600.0,
                 'l_rate':0.0,
                 'duration':1000.0,
                 'res':10.0}
    stim_chg_pars = {'value':700.0,
                     'res':100.0,
                     'waittime':2000.0}

    #Example getting C1 nest ids
    # >> pops['C1'].ids
    # Extracting nodes which are going to get the modulatory input
    stim_spec = {'C1':0.0,'C2':0.0,'CF':0.0}
    for node_name in ['C1', 'C2', 'CF']:
        modpop_ids = extra_modulation(pops, 0.3, node_name)
        [stimmod_id,stim_time] = modulatory_stim(stim_pars,stim_chg_pars)
        nest.Connect(stimmod_id,modpop_ids)
        stim_spec[node_name] = stim_time
        stim_spec[node_name].update({'stim_subpop':modpop_ids})

    sio.savemat(base+'/stimspec.mat',stim_spec)
    save_node_random_params(pops,base+'/randomized-params.json')

    # print(pops)

    # Connect populations accordingly to connections structure
    connect(par, surfs, pops)
    #
    # # Simulate
    my_nest.Simulate(max(stim_spec['C1']['start_times'])+5000.0)

    #
    # # Create spike signals
    d = postprocessing(pops)
    #
    # # Save
    sd = data_to_disk.Storage_dic.load(par.dic['simu']['path_data'], ['Net_0'])
    sd.save_dic({'Net_0': d}, **{'use_hash': False})

def extra_modulation(pops,subpop_ratio,node_name):
    node_ids = pops[node_name].ids
    #spkdet = nest.Create('spike_detector')
    #nest.Connect(node_ids,spkdet)
    perm_ids = numpy.random.permutation(node_ids)
    #subpop_ratio = 0.3
    subpop_num = numpy.int(subpop_ratio*len(node_ids))
    subpop_ids = perm_ids[range(0,subpop_num)]
    subpop_ids = subpop_ids.tolist()
    return subpop_ids

def modulatory_stim(stim_params,chg_stim_param):
    mod_inp = nest.Create('poisson_generator_dynamic',1)
    all_rates = []
    all_start_times = []
    #h_rate = 200.0
    #l_rate = 0.0
    #slope = 2.       # Hz/ms
    #res = 10.        # ms
    #stim_start = 1000.0
    stim_start = stim_params['stim_start']
    h_rate = stim_params['h_rate']
    l_rate = stim_params['l_rate']
    #slope = stim_params['slope']
    stim_dur = stim_params['duration']
    res = stim_params['res']
    # rate_step = slope*res
    stim_stop = stim_start + stim_dur + res
    timevec = numpy.arange(stim_start,stim_stop,res)
    timveclen = timevec.size
    ratevec = numpy.linspace(l_rate,h_rate,timveclen)

    chg_stim_val = chg_stim_param['value']
    chg_stim_res = chg_stim_param['res']
    waittime = chg_stim_param['waittime']
    all_start_times.append(stim_start)
    all_rates.append(h_rate)
    h_rate = h_rate + chg_stim_res

    while h_rate <= chg_stim_val:
        all_rates.append(h_rate)
        ratevec = numpy.append(ratevec,0.0)
        timevec = numpy.append(timevec,timevec[-1]+res)

        ratevec = numpy.append(ratevec,numpy.linspace(l_rate,h_rate,timveclen))
        stim_start = stim_stop + waittime
        stim_stop = stim_start + stim_dur + res
        timevec = numpy.append(timevec,numpy.arange(stim_start,stim_stop,res))
        h_rate = h_rate + chg_stim_res
        all_start_times.append(stim_start)

    nest.SetStatus(mod_inp,{'rates': ratevec.round(0),'timings': timevec,
                            'start': timevec[0], 'stop': stim_stop})
    stim_vecs = {'rates':all_rates,
                 'start_times':all_start_times}
    return mod_inp,stim_vecs


# main()
if __name__ == '__main__':

    #size = sys.argv[1] if len(sys.argv)>1 else 3000
    size = 10000
    modes = ['activation-control']

#    modes = [
#        'activation-control',
#        'activation-dopamine-depleted',
#        'slow-wave-control',
#        'slow-wave-dopamine-depleted'
#    ]
    for mode in modes:
        main(mode, size)

        plot.main(mode, size)

        randomized_params_plot.main(mode, size)
        mean_firing_rates.main(mode, size)
        mean_firing_rates_plot.main(mode, size)
