# Create by Mikael Lindahl on 4/12/17.

from core.network import structure
from core.network import engine
from core import my_nest, data_to_disk
# from core.network.parameters.eneuro import EneuroPar
from core.network.parameters.eneuro_NoDynSyn import EneuroPar as Eneuropar_NDS
from core.network.parameters.eneuro import EneuroPar

# from core.network.parameters.eneuro_activation_NoDynSyn import EneuroActivationPar
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


def main(mode, size, trnum, threads_num, les_src,les_trg,chg_gpastr,total_num_trs,stat_syn):
    my_nest.ResetKernel()

    # Get parameters
    # par = EneuroBetaPar(other=EneuroPar())

    if mode in ['activation-control',  'activation-dopamine-depleted']:

        if mode == 'activation-control':
            if stat_syn:
                par = EneuroActivationPar(other=Eneuropar_NDS())
                print 'All synapses are static!'
            else:
                par = EneuroActivationPar(other=EneuroPar())
                print 'Default synaptic configuration!'
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


    stim_pars = {'STRramp':{'stim_start':4000.0,
                             'h_rate':400.0,
                             'l_rate':0.0,
                             'duration':140.0,
                             'res':10.0,
                             'do':False,
                             'w-form':'ramp',
                             'stim_target':['C1','C2','CF'],
                             'target_name':'STR',
                             'stim_spec':{'C1':0.0,'C2':0.0,'CF':0.0},
                             'stim_ratio':{'C1':0.3,'C2':0.3,'CF':0.3}},
                 'STNstop':{'stim_start':4000.0,
                             'h_rate':1000.0,
                             'l_rate':0.0,
                             'duration':10.0,
                             'res':10.0,
                             'do':False,
                             'w-form':'pulse',
                             'stim_target':['CS'],
                             'target_name':'STN',
                             'stim_spec':{'CS':0.0},
                             'stim_ratio':{'CS':1.0}},
                 'GPAstop':{'stim_start':4000.0,
                             'h_rate':1000.0,
                             'l_rate':0.0,
                             'duration':40.0,
                             'res':10.0,
                             'do':False,
                             'w-form':'pulse',
                             'stim_target':['EA'],
                             'target_name':'GPA',
                             'stim_spec':{'EA':0.0},
                             'stim_ratio':{'EA':1.0}}}
    stim_chg_pars = {'STRramp':{'value':500.0,
                                 'res':100.0,
                                 'waittime':2000.0},
                     'STNstop':{'value':2000.0,
                                 'res':500.0,
                                 'waittime':2000.0},
                     'GPAstop':{'value':2000.0,
                                 'res':500.0,
                                 'waittime':2000.0},
                     'Reltime':{'STNstop':{'l_val':0.0,
                                       'h_val':20.0,
                                       'res':10.0,
                                       'ref':'STRramp'},
                                'GPAstop':{'l_val':0.0,
                                       'h_val':20.0,
                                       'res':10.0,
                                       'ref':'STRramp'}}}

    pp(stim_pars)
    pp(stim_chg_pars)

    # Changing GPA to STR connection weight leading to prominent response in STR

    binalg = False

    if chg_gpastr:
        div_var = trnum/total_num_trs
        trnum = ((trnum-1)%total_num_trs) + 1
        if binalg:
            weight_coef_base = 0.5

            # residual_temp = div_var%2
            if (div_var%2) == 0:
                weight_coef = weight_coef_base + weight_coef_base/(2**(div_var/2))
            else:
                weight_coef = weight_coef_base - weight_coef_base/(2**(div_var+1)/2)
        else:
            weight_coef_base = 0.2
            weight_coef_inc_rate = 0.05

            weight_coef = weight_coef_base + weight_coef_inc_rate * div_var

        par.set({'nest':{'GA_M1_gaba':{'weight':weight_coef}}})
        par.set({'nest':{'GA_M2_gaba':{'weight':weight_coef*2.0}}})
    else:
        weight_coef = par.dic['nest']['GA_M1_gaba']['weight']
        div_var = 0

    # stim_pars_STN = {'stim_start':4000.0,
    #                  'h_rate':200.0,
    #                  'l_rate':0.0,
    #                  'duration':10.0,
    #                  'res':10.0,
    #                  'do':True}
    # stim_chg_pars_STN = {'value':210.0,
    #                  'res':10.0,
    #                  'waittime':2000.0}

    '''
        The naming of the result directory has the following order:
        1st number: stimulation duration
        2nd number: maximum rate increase
        3rd number: maximum change of rate increase
        4th number: steps of increase
    '''

    dir_name = 'GPASTR-Wmod'+str(int(weight_coef*100))+'-' + str(div_var) + '-'

    for keys in stim_pars:
        if stim_pars[keys]['do']:
            dir_name = dir_name + \
                       stim_pars[keys]['target_name']+'-'+\
                       str(stim_pars[keys]['duration'])+ '-'+\
                       str(stim_pars[keys]['h_rate'])+ '-'+\
                       str(stim_chg_pars[keys]['value'])+ '-'+\
                       str(stim_chg_pars[keys]['res'])+ '-'
            last_stimpars = keys
    dir_name = dir_name + 'stst-tr'+ str(trnum)

    base = os.path.join(os.getenv('BGMODEL_HOME_DATA'), 'example/eneuro', str(size), mode, dir_name)

    # if stim_pars['STRramp']['do'] and stim_pars['STNstop']['do']:
    #     base = os.path.join(os.getenv('BGMODEL_HOME_DATA'), 'example/eneuro', str(size), mode,
    #                         'mutlistim-STN-dur'+
    #                         str(stim_pars['STNstop']['duration'])+ '-'+
    #                         str(stim_pars['STNstop']['h_rate'])+ '-'+
    #                         str(stim_chg_pars['STNstop']['value'])+ '-'+
    #                         str(stim_chg_pars['STNstop']['res'])+ '-'+
    #                         'tr'+ str(trnum))
    # elif stim_pars['STNstop']['do']:
    #     base = os.path.join(os.getenv('BGMODEL_HOME_DATA'), 'example/eneuro', str(size), mode,
    #                         'STN-dur'+
    #                         str(stim_pars['STNstop']['duration'])+ '-'+
    #                         str(stim_pars['STNstop']['h_rate'])+ '-'+
    #                         str(stim_chg_pars['STNstop']['value'])+ '-'+
    #                         str(stim_chg_pars['STNstop']['res'])+ '-'+
    #                         'tr'+ str(trnum))
    # else:
    #     base = os.path.join(os.getenv('BGMODEL_HOME_DATA'), 'example/eneuro', str(size), mode,
    #                         'STR-dur'+
    #                         str(stim_pars['STRramp']['duration'])+ '-'+
    #                         str(stim_pars['STRramp']['h_rate'])+ '-'+
    #                         str(stim_chg_pars['STRramp']['value'])+ '-'+
    #                         str(stim_chg_pars['STRramp']['res'])+ '-'+
    #                         'tr'+ str(trnum))

    rand_conn = False

    if rand_conn:
        pathconn = par.get()['simu']['path_conn']+ str(trnum)+ '/'
    else:
        pathconn = par.get()['simu']['path_conn']

    if len(les_src) > 0:
        print 'Lesion will be applied to source(s): ',les_src,' projecting to ',les_trg
        lesion(par,les_src,les_trg)
    else:
        print 'No lesion!'



    # Configure simulation parameters
    par.set({
        'simu': {
            'local_num_threads': threads_num,
            'path_data': base+'/data',
            'path_figure': base+'/fig',
            'path_nest': base+'/nest/',  # trailing slash important
            'path_conn': pathconn,
            'stop_rec': 10000000.,
            'sim_stop': 10000000.,
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
        # 'conn':{'GA_M1_gaba':{'weight':{'params':{'max':1.0,'min':0.6}}},   # Default: min = 0.02; max = 0.06
        #         'GA_M2_gaba':{'weight':{'params':{'max':2.0,'min':1.2}}}}, # Default: min = 0.04; max = 0.12
    })

    mod_GPASTR_weights = {'GA_M1':par.dic['conn']['GA_M1_gaba']['weight']['params'],
                          'GA_M2':par.dic['conn']['GA_M2_gaba']['weight']['params']}

    par.nest_set_kernel_status()

    # Save parametesr
    list_parameters.main(base, par)

    # Clear nest data directory
    par.nest_clear_data_path({'display': True})

    # Show kernel status
    pp(my_nest.GetKernelStatus())

    sim_res = nest.GetKernelStatus('resolution')

    if trnum != 1:
        def_rng_seed = nest.GetKernelStatus('rng_seeds')
        l_def_rng_seed = len(def_rng_seed)
        b_lim = (trnum - 1)*l_def_rng_seed
        u_lim = (trnum - 1)*2*l_def_rng_seed + 1
        nest.GetKernelStatus({'rng_seeds':range(b_lim,u_lim,1)})


    # Create news populations and connections structures
    surfs, pops = build(par)

    #Example getting C1 nest ids
    # >> pops['C1'].ids
    # Extracting nodes which are going to get the modulatory input
    # STR
    stim_combine = []
    for key in stim_pars.keys():
        stim_combine.append(stim_pars[key]['do'])

    ind_comb = -1
    comb_dic = {}

    if sum(stim_combine) == len(stim_combine):
        ratescomb = []
        for stim_type in stim_pars.keys():
            ind_comb = ind_comb + 1
            # l_rate = stim_pars[stim_type]['l_rate']
            h_rate = stim_pars[stim_type]['h_rate']
            res = stim_chg_pars[stim_type]['res']
            max_h_rate = stim_chg_pars[stim_type]['value']
            ratescomb.append(numpy.arange(h_rate,max_h_rate+res,res))
            comb_dic.update({stim_type:ind_comb})
        reltimecomb = []
        if stim_chg_pars.has_key('Reltime'):
            for rel_type in stim_chg_pars['Reltime'].keys():
                ind_comb = ind_comb + 1
                h_value = stim_chg_pars['Reltime'][rel_type]['h_val']
                l_value = stim_chg_pars['Reltime'][rel_type]['l_val']
                res_val = stim_chg_pars['Reltime'][rel_type]['res']
                reltimecomb.append(numpy.arange(l_value,h_value+res_val,res_val))
                if comb_dic.has_key('reltime'):
                    comb_dic['reltime'].update({rel_type:ind_comb})
                else:
                    comb_dic.update({'reltime':{rel_type:ind_comb}})
            comb = numpy.array(numpy.meshgrid(ratescomb[0],ratescomb[1],ratescomb[2],reltimecomb[0],reltimecomb[1]))
        comb_resh = comb.reshape(comb.shape[0],numpy.prod(comb.shape[1:]))
        stim_time = modulatory_multiplestim(comb_resh,stim_pars,stim_chg_pars,comb_dic)
        stim_spec = {'C1':0.0,'C2':0.0,'CF':0.0,'CS':0.0,'EA':0.0}
        for stim_type in stim_time.keys():
            for node_name in stim_pars[stim_type]['stim_target']:
                [modpop_ids,allpop_ids] = extra_modulation(pops, stim_pars[stim_type]['stim_ratio'][node_name], node_name)
                nest.Connect(stim_time[stim_type]['stim_pois_id'], modpop_ids)
                # stim_spec[node_name] = stim_time
                stim_spec[node_name] = {'stim_subpop':modpop_ids,
                                        'allpop':allpop_ids,
                                        'stim_id':stim_time[stim_type]['stim_pois_id']}
        stim_spec.update(stim_time)
        sio.savemat(base+'/stimspec.mat',stim_spec)

    elif sum(stim_combine) > 1 and sum(stim_combine) != len(stim_combine):
        ratescomb = []
        for stim_type in stim_pars.keys():
            if stim_pars[stim_type]['do']:
                ind_comb = ind_comb + 1
                # l_rate = stim_pars[stim_type]['l_rate']
                h_rate = stim_pars[stim_type]['h_rate']
                res = stim_chg_pars[stim_type]['res']
                max_h_rate = stim_chg_pars[stim_type]['value']
                ratescomb.append(numpy.arange(h_rate,max_h_rate+res,res))
                comb_dic.update({stim_type:ind_comb})
            else:
                stim_pars.pop(stim_type)
                stim_chg_pars.pop(stim_type)
                stim_chg_pars['Reltime'].pop(stim_type)
        reltimecomb = []
        if stim_chg_pars.has_key('Reltime'):
            for rel_type in stim_chg_pars['Reltime'].keys():
                if stim_pars[rel_type]['do']:
                    ind_comb = ind_comb + 1
                    h_value = stim_chg_pars['Reltime'][rel_type]['h_val']
                    l_value = stim_chg_pars['Reltime'][rel_type]['l_val']
                    res_val = stim_chg_pars['Reltime'][rel_type]['res']
                    reltimecomb.append(numpy.arange(l_value,h_value+res_val,res_val))
                    if comb_dic.has_key('reltime'):
                        comb_dic['reltime'].update({rel_type:ind_comb})
                    else:
                        comb_dic.update({'reltime':{rel_type:ind_comb}})
            comb = numpy.array(numpy.meshgrid(ratescomb[0],ratescomb[1],reltimecomb[0]))
        comb_resh = comb.reshape(comb.shape[0],numpy.prod(comb.shape[1:]))
        stim_time = modulatory_multiplestim(comb_resh,stim_pars,stim_chg_pars,comb_dic)
        stim_spec = {'C1':0.0,'C2':0.0,'CF':0.0,'CS':0.0}
        for stim_type in stim_time.keys():
            for node_name in stim_pars[stim_type]['stim_target']:
                [modpop_ids,allpop_ids] = extra_modulation(pops, stim_pars[stim_type]['stim_ratio'][node_name], node_name)
                nest.Connect(stim_time[stim_type]['stim_pois_id'], modpop_ids)
                # stim_spec[node_name] = stim_time
                stim_spec[node_name] = {'stim_subpop':modpop_ids,
                                        'allpop':allpop_ids,
                                        'stim_id':stim_time[stim_type]['stim_pois_id']}
        stim_spec.update(stim_time)
        sio.savemat(base+'/stimspec.mat',stim_spec)

    elif sum(stim_combine) > 0:
        dic_keys = numpy.array(stim_pars.keys())
        whichkey = dic_keys[numpy.array(stim_combine)][0]

        stim_spec = stim_pars[whichkey]['stim_spec']
        for node_name in stim_pars[whichkey]['stim_target']:
            [modpop_ids,allpop_ids] = extra_modulation(pops, stim_pars[whichkey]['stim_ratio'][node_name], node_name)
            [stimmod_id,stim_time] = modulatory_stim(stim_pars[whichkey],stim_chg_pars[whichkey])
            nest.Connect(stimmod_id,modpop_ids)
            # stim_spec[node_name] = stim_time
            stim_spec[node_name] = {'stim_subpop':modpop_ids,
                                    'allpop':allpop_ids,
                                    'stim_id':stimmod_id}
        stim_spec.update({whichkey:stim_time})

        sio.savemat(base+'/stimspec.mat',stim_spec)
    sio.savemat(base+'/modifiedweights.mat',mod_GPASTR_weights)
    # # STN
    # if stim_pars_STN['do']:
    #     stim_spec = {'CS':0.0}
    #     for node_name in ['CS']:
    #         [modpop_ids,allpop_ids] = extra_modulation(pops, 1.0, node_name)
    #         [stimmod_id,stim_time] = modulatory_stim(stim_pars_STN,stim_chg_pars_STN)
    #         nest.Connect(stimmod_id,modpop_ids)
    #         stim_spec[node_name] = stim_time
    #         stim_spec[node_name].update({'stim_subpop':modpop_ids,
    #                                     'allpop':allpop_ids})
    #
    #     sio.savemat(base+'/stimspec.mat',stim_spec)

    save_node_random_params(pops,base+'/randomized-params.json')
    nest.SetKernelStatus({'print_time':True})

    # print(pops)

    # Connect populations accordingly to connections structure
    connect(par, surfs, pops)

    # Sampling paramteres

    sim_time = 6000.0
    start_wr = 0.0
    end_wr = sim_time
    res_wr = 10.0

    # Connecting voltmeter to the targets.

    trgs = ['M1', 'M2', 'FS', 'SN']
    voltmeter = voltmeter_connect(pops, trgs, 1.0)

    # # Simulate

    print 'Simulation\'s just started ...'
    if sum(stim_combine) == 0:
        # sim_time = max(stim_spec['STRramp']['start_times'])+5000.0
        # sim_time = 6000.0
        # start_wr = 0.0
        # end_wr = sim_time
        # res_wr = 10.0   #end_wr
        # source_wr = ['FS','M1','M2','GI']
        # target_wr = ['SN']
        weight_dic = {'FS':{'M1':{'s':[], 't':[], 'w':[]},
                            'M2':{'s':[], 't':[], 'w':[]},
                            'FS':{'s':[], 't':[], 'w':[]}},
                      'GI':{'SN':{'s':[], 't':[], 'w':[]}},
                      'M1':{'SN':{'s':[], 't':[], 'w':[]}},
                      'M2':{'SN':{'s':[], 't':[], 'w':[]}},
                      'ST':{'SN':{'s':[], 't':[], 'w':[]}},
                      'GA':{'FS':{'s':[], 't':[], 'w':[]}},
                      'GI':{'FS':{'s':[], 't':[], 'w':[]}},
                      'time':[]}

        volt_dic = {'M1':{'voltage':[], 'ids':[]},
                    'M2':{'voltage':[], 'ids':[]},
                    'SN':{'voltage':[], 'ids':[]},
                    'FS':{'voltage':[], 'ids':[]}}

        while start_wr <= end_wr:

            my_nest.Simulate(res_wr)
            start_wr = start_wr + res_wr
            weight_dic['time'].append(start_wr)

            for sour_key in weight_dic.keys():
                if sour_key != 'time':
                    for tar_key in weight_dic[sour_key].keys():
                        conns = nest.GetConnections(pops[sour_key].ids,pops[tar_key].ids)
                        weights = nest.GetStatus(conns,'weight')
                        weight_dic[sour_key][tar_key]['w'].append(weights)
                        s_ids = [conns[i][0] for i in xrange(len(conns))]
                        weight_dic[sour_key][tar_key]['s'].append(s_ids)
                        t_ids = [conns[i][1] for i in xrange(len(conns))]
                        weight_dic[sour_key][tar_key]['t'].append(t_ids)

            # for trg in trgs:
            #     times = nest.GetStatus(voltmeter)[0]['events']['times']
            #     idx = times == max(times)
            #     volt_dic[trg]['voltage'].append(nest.GetStatus(voltmeter)[0]['events']['V_m'][idx])
            #     volt_dic[trg]['ids'].append(nest.GetStatus(voltmeter)[0]['events']['senders'][idx])

            #my_nest.Simulate(max(stim_spec['STRramp']['start_times'])+5000.0)
        sio.savemat(base+'/weights.mat', weight_dic)
        sio.savemat(base+'/voltages.mat', volt_dic)
    else:
        my_nest.Simulate(10000.0)

    print 'Simulation is now finished!\n'

    print 'Contatenating .gdf files to a .mat file for each nucleus ...'

    sys_var = os.system('matlab -nodisplay -r \'data_concat_save_as_mat '+base+'/nest/ '+str(sim_res)+'; exit;\'')
    if sys_var == 0:
        print 'gdf files are now in .mat files!'
        os.system('mv '+base+'/nest/mat_data '+base)
        # os.system('rm -rf '+base+'/nest')
    else:
        print 'Error! No .mat file is produced!'

    #
    # # Create spike signals
    # d = postprocessing(pops)
    #
    # # Save
    # sd = data_to_disk.Storage_dic.load(par.dic['simu']['path_data'], ['Net_0'])
    # sd.save_dic({'Net_0': d}, **{'use_hash': False})

def voltmeter_connect(pops, targets, interval):
    vol_id = nest.Create('voltmeter', 1, {'to_file':True, 'to_memory':False, 'interval':interval})
    for tr in targets:
        nest.Connect(vol_id, pops[tr].ids)

    return vol_id

def extra_modulation(pops,subpop_ratio,node_name):
    node_ids = pops[node_name].ids
    #spkdet = nest.Create('spike_detector')
    #nest.Connect(node_ids,spkdet)
    perm_ids = numpy.random.permutation(node_ids)
    #subpop_ratio = 0.3
    subpop_num = numpy.int(subpop_ratio*len(node_ids))
    subpop_ids = perm_ids[range(0,subpop_num)]
    subpop_ids = subpop_ids.tolist()
    return subpop_ids,node_ids

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

        if stim_params['w-form'] == 'ramp':
            ratevec = numpy.append(ratevec,numpy.linspace(l_rate,h_rate,timveclen))
        elif stim_params['w-form'] == 'pulse':
            ratevec_tmp = []
            ratevec_tmp.append(l_rate)
            ratevec_tmp = numpy.append(ratevec_tmp,h_rate*numpy.ones(timveclen - 1))
            ratevec = numpy.append(ratevec,ratevec_tmp)

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


def modulatory_multiplestim(all_rates,stim_params,chg_stim_param,comb_dic):
    num_pois_gen = len(stim_params.keys())
    mod_inp = nest.Create('poisson_generator_dynamic',num_pois_gen)
    ind = -1
    ind_dic = {}
    stim_vecs = {}
    refchgkey = list(find('ref',chg_stim_param))
    u_refchgkey = numpy.unique(refchgkey)[0]
    if type(u_refchgkey) is not list:
        ordered_keys = [u_refchgkey]
    for keys in stim_params.keys():
        ind = ind + 1
        ind_dic.update({keys:ind})
        if not keys in u_refchgkey:
            ordered_keys.append(keys)


    for keys in ordered_keys:
        ind = ind_dic[keys]
        timevec = numpy.array(0.0)
        ratevec = numpy.array(0.0)
        waittime = chg_stim_param[keys]['waittime']
        stim_start = stim_params[keys]['stim_start']
        # all_rates = []
        all_start_times = []
        all_stop_times = []
        for rate_ind, rateval in enumerate(all_rates[ind]):
            #h_rate = 200.0
            #l_rate = 0.0
            #slope = 2.       # Hz/ms
            #res = 10.        # ms
            #stim_start = 1000.0
            # stim_start = stim_params[keys]['stim_start']
            h_rate = rateval
            l_rate = stim_params[keys]['l_rate']
            #slope = stim_params['slope']
            stim_dur = stim_params[keys]['duration']
            res = stim_params[keys]['res']
            # rate_step = slope*res
            if keys in refchgkey:
                stim_stop = stim_start + stim_dur + res
            else:
                stim_start = prev_var_stops[rate_ind] + all_rates[comb_dic['reltime'][keys]][rate_ind]
                stim_stop = stim_start + stim_dur + res
            single_timvec = numpy.arange(stim_start,stim_stop,res)
            timevec = numpy.append(timevec,single_timvec)
            timveclen = single_timvec.size
            # ratevec = numpy.append(ratevec,numpy.linspace(l_rate,h_rate,timveclen))

            if stim_params[keys]['w-form'] == 'ramp':
                ratevec = numpy.append(ratevec,numpy.linspace(l_rate,h_rate,timveclen))
            elif stim_params[keys]['w-form'] == 'pulse':
                ratevec_tmp = []
                ratevec_tmp.append(l_rate)
                ratevec_tmp = numpy.append(ratevec_tmp,h_rate*numpy.ones(timveclen - 1))
                ratevec = numpy.append(ratevec,ratevec_tmp)

            # Establishing the pause between the stimulations
            ratevec = numpy.append(ratevec,0.0)
            timevec = numpy.append(timevec,timevec[-1]+res)

            # chg_stim_val = chg_stim_param[keys]['value']
            # chg_stim_res = chg_stim_param[keys]['res']
            # waittime = chg_stim_param[keys]['waittime']
            all_start_times.append(stim_start)
            all_stop_times.append(stim_stop)
            stim_start = stim_stop + waittime
            # all_rates.append(h_rate)
            # h_rate = h_rate + chg_stim_res

            # while h_rate <= chg_stim_val:
            #     all_rates.append(h_rate)
            #     ratevec = numpy.append(ratevec,0.0)
            #     timevec = numpy.append(timevec,timevec[-1]+res)

                # ratevec = numpy.append(ratevec,numpy.linspace(l_rate,h_rate,timveclen))
                # stim_start = stim_stop + waittime
                # stim_stop = stim_start + stim_dur + res
                # timevec = numpy.append(timevec,numpy.arange(stim_start,stim_stop,res))
                # h_rate = h_rate + chg_stim_res
                # all_start_times.append(stim_start)
        if keys in refchgkey:
            prev_var_stops = all_stop_times

        nest.SetStatus((mod_inp[ind],),{'rates': ratevec.round(0),'timings': timevec,
                                'start': timevec[0], 'stop': stim_stop})
        stim_vecs.update({keys:{'rates':all_rates,
                                'start_times':all_start_times,
                                'stop_times':all_stop_times,
                                'stim_pois_id':(mod_inp[ind],)}})
    return stim_vecs

def lesion(params,source,target):
    connlist = params.get()['conn']
    connkeys = connlist.keys()
    for keys in connkeys:
        str_temp = keys.split('_')
        if str_temp[0] in source and str_temp[1] in target:
            params.set({'conn':{keys:{'lesion':True}}})
    return params

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

    #size = sys.argv[1] if len(sys.argv)>1 else 3000
    if len(sys.argv) > 1:
        numtrs = int(sys.argv[1])
        size = int(sys.argv[2])
        loc_num_th = int(sys.argv[3])
    else:
        numtrs = 40
        size = 10000
        loc_num_th = 1
        lesion_source = []
        lesion_target = []
        tot_num_trs = 10
        chg_GPASTR = False

    if len(sys.argv) > 4:
        les_s_tmp = sys.argv[4]
        lesion_source = les_s_tmp.split(',')
        les_t_tmp = sys.argv[5]
        lesion_target = les_t_tmp.split(',')
    else:
        lesion_source = []
        lesion_target = []

    if len(sys.argv) > 6:                       # GPA to STR connections get strengthen so that they can affect ramping
        tot_num_trs = int(sys.argv[6])          # response in MSN D1 and D2
        chg_GPASTR = True
    else:
        tot_num_trs = 10
        chg_GPASTR = False



    modes = ['activation-control']
    all_static_syns = True

#    modes = [
#        'activation-control',
#        'activation-dopamine-depleted',
#        'slow-wave-control',
#        'slow-wave-dopamine-depleted'
#    ]
    for mode in modes:
        main(mode, size, numtrs, loc_num_th, lesion_source, lesion_target,chg_GPASTR,tot_num_trs,all_static_syns)

        plot.main(mode, size)

        randomized_params_plot.main(mode, size)
        mean_firing_rates.main(mode, size)
        mean_firing_rates_plot.main(mode, size)
