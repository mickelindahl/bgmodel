'''
Created on May 10, 2014

@author: mikael
'''

from copy import deepcopy
from toolbox import misc
from toolbox.network.manager import save, load, compute, run
from toolbox.network import default_params
from toolbox.network.default_params import Perturbation_list as pl
from toolbox import my_socket
from toolbox.data_to_disk import Storage_dic

from inhibition_gather_results import process

import numpy
import toolbox.plot_settings as ps
import pprint
from toolbox import data_to_disk
pp=pprint.pprint



def cmp_psd(d_pds, models, dd):
    for key1 in dd.keys():
        for model in models:
            psd=dd[key1][model]['firing_rate'].get_psd(**d_pds)
            dd[key1][model]['psd'] = psd


def get_conn_matricies(net, models, attr):
    d={}
    for model in models:
        soruce, target, _=model.split('_')
        d[model]={attr:net.get_conn_matrix(soruce, target, model)}
    return d

def get_file_name(script_name,  par=None):
    if not par:
        par=default_params.Inhibition()
    path=par.get_path_data()
    file_name = path + script_name
#     file_name = home + '/results/papers/inhibition/network/' + script_name
    return file_name



def get_file_name_figs(script_name, par=None):
    if not par:
        par=default_params.Inhibition()
    path=par.get_path_figure()
    file_name = path + script_name

#     file_name = path +'/fig/'+ script_name

#     file_name_figs = home + '/results/papers/inhibition/network/fig/' + script_name
    return file_name


def get_args_list(*args, **kwargs):
    
    builder=kwargs.get('Builder')
    do_obj=kwargs.get('do_obj') 
    file_name=kwargs.get('file_name')    
    module=kwargs.get('module')
   
    p_list, get_setup_args_and_kwargs=args
    args_list=[]
    for j in range(0, 3):
        for i, p in enumerate(p_list):
            if j<2: nets_list=[[nets] for nets in kwargs.get('nets')]
            else: nets_list=[kwargs.get('nets')]
        
            for nets in nets_list:
                
                script_name='{}/script_{:0>4}_{}'.format(file_name, i, p.name)     
                
                a, k=get_setup_args_and_kwargs(i, **kwargs)
                
                k.update({'nets_to_run':nets})
                
                setup = module.Setup(*a, **k)
                
                obj = module.Main(**{'builder':builder, 
                                     'from_disk':j, 
                                     'perturbation_list':p, 
                                     'script_name':script_name, 
                                     'setup':setup})
                
                if do_obj:
                    obj.do()
                    
                args_list.append(obj)
    return args_list               
                
def get_args_list_oscillation(p_list, **kwargs):
    

    def get_setup_args_and_kwargs(_, **kwargs):
        freq_oscillation=kwargs.get('freq_oscillation')
        local_num_threads=kwargs.get('local_num_threads')
        args=[1000.0 / freq_oscillation, 
               local_num_threads,]
        
        return args, {}
    
    args=[p_list, get_setup_args_and_kwargs]
    return get_args_list(*args, **kwargs)
    
#     args_list=[]
#     for j in range(0, 3):
#         for i, p in enumerate(p_list):
#             if j<2: nets_list=[[nets] for nets in kwargs.get('nets')]
#             else: nets_list=[kwargs.get('nets')]
#         
#             for nets in nets_list:
# 
#                 script_name='{}/script_{:0>4}_{}'.format(file_name, i, p.name)
# #                 script_name = (file_name + '/script_' + str(i) 
# #                                + '_' + p.name)
#                 
#                 
#                 
#                 setup = module.Setup(1000.0 / freq_oscillation, 
#                                      local_num_threads,
#                                      **d)
#                 
#                 obj = module.Main(**{'builder':builder, 
#                                      'from_disk':j, 
#                                      'perturbation_list':p, 
#                                      'script_name':script_name, 
#                                      'setup':setup})
#                 
#                 if do_obj:
#                     obj.do()
#                     
#                 args_list.append(obj)
#     return args_list

def get_args_list_inhibition(p_list, **kwargs):
    

    def get_setup_args_and_kwargs(_, **kwargs):

        local_num_threads=kwargs.get('local_num_threads')
        lower=kwargs.get(('lower'))
        res=kwargs.get('resolution')
        rep=kwargs.get('repetitions')
        upper=kwargs.get('upper')
                
        
        kwargs={'local_num_threads':local_num_threads,
                  'resolution':res,
                  'repetition':rep,
                  'lower':lower,
                  'upper':upper}
        return [], kwargs
    
    args=[p_list, get_setup_args_and_kwargs]
    return get_args_list(*args, **kwargs)
 

def get_args_list_MSN_cluster_compete(p_list, **kwargs):
    

    def get_setup_args_and_kwargs(_, **kwargs):

        local_num_threads=kwargs.get('local_num_threads')
        rep=kwargs.get('repetitions')
                
        
        kwargs={'local_num_threads':local_num_threads,
                  'repetition':rep}
        return [], kwargs
    
    args=[p_list, get_setup_args_and_kwargs]
    return get_args_list(*args, **kwargs)

def get_args_list_Go_NoGo_compete(p_list, **kwargs):
    
    def get_setup_args_and_kwargs(i_p_list, **kwargs):
        duration=kwargs.get('duration')
        labels=kwargs.get('labels',['Only D1', 
                                    'D1,D2',
                                    'MSN lesioned (D1, D2)',
                                    'FSN lesioned (D1, D2)',
                                    'GPe TA lesioned (D1,D2)'])
        laptime=kwargs.get('laptime')
        l_mean_rate_slices=kwargs.get('l_mean_rate_slices')
        local_num_threads=kwargs.get('local_num_threads')
        other_scenario=kwargs.get('other_scenario', False)
        props_conn=kwargs.get('proportion_connected', 1.)
        res=kwargs.get('res')
        rep=kwargs.get('rep')
        time_bin=kwargs.get('time_bin')

        if type(props_conn)==list:
            pc=props_conn[i_p_list]
        else:
            pc=props_conn

        kwargs={'duration':duration,
                'l_mean_rate_slices':l_mean_rate_slices,
                'labels':labels,
                'laptime':laptime,
                'local_num_threads':local_num_threads,
                'other_scenario':other_scenario,
                'proportion_connected':pc,
                'resolution':res,
                'repetition':rep,
                'time_bin':time_bin}
        return [], kwargs
    
    args=[p_list, get_setup_args_and_kwargs]
    return get_args_list(*args, **kwargs)

# def get_args_list_Go_NoGo_compete(p_list, **kwargs):
#     
#     builder=kwargs.get('Builder')
#     do_obj=kwargs.get('do_obj') 
# #     do_runs=kwargs.get('do_runs')
#     duration=kwargs.get('duration')
#     file_name=kwargs.get('file_name')
#     from_disk_0=kwargs.get('from_disk_0')
#     labels=kwargs.get('labels',['Only D1', 
#                                 'D1,D2',
#                                 'MSN lesioned (D1, D2)',
#                                 'FSN lesioned (D1, D2)',
#                                 'GPe TA lesioned (D1,D2)'])
#     laptime=kwargs.get('laptime')
#     l_mean_rate_slices=kwargs.get('l_mean_rate_slices')
#     local_num_threads=kwargs.get('local_num_threads')
#     module=kwargs.get('module')
#     other_scenario=kwargs.get('other_scenario', False)
#     props_conn=kwargs.get('proportion_connected', 1.)
#     res=kwargs.get('res')
#     rep=kwargs.get('rep')
#     time_bin=kwargs.get('time_bin')
#     
#     
#     if type(props_conn) != list:
#         props_conn=[props_conn]*len(p_list)
#     
#     args_list=[]
#     for j in range(0, 3):
#         for i, p in enumerate(p_list):
#             
#             if j<2: nets_list=[[nets] for nets in kwargs.get('nets')]
#             else: nets_list=[kwargs.get('nets')]
#         
#             for nets in nets_list:
# 
#                 script_name='{}/script_{:0>4}_{}'.format(file_name, 
#                                                              i, 
#                                                              p.name)
# 
#                 d={'duration':duration,
#                     'l_mean_rate_slices':l_mean_rate_slices,
#                     'labels':labels,
#                     'laptime':laptime,
#                     'local_num_threads':local_num_threads,
#                     'nets_to_run':nets,
#                     'other_scenario':other_scenario,
#                     'proportion_connected':props_conn[i],
#                     'resolution':res,
#                     'repetition':rep,
#                     'time_bin':time_bin}
#                 setup = module.Setup(**d)
#                
#                 d={'builder':builder, 
#                    'from_disk':j, 
#                    'perturbation_list':p, 
#                    'script_name':script_name, 
#                    'setup':setup}
#                 
#                 obj = module.Main(**d)
#                 
#                 if do_obj:
#                     obj.do()
#                          
#                 args_list.append(obj)
#                         
#     return args_list
    
def get_kwargs_list_indv_nets(n_pert, kwargs):
    do_runs=kwargs.get('do_runs')
    from_disk_0=kwargs.get('from_disk_0')
    
    kwargs_list=[]
    index=-1
    for j in range(0, 3):
        for i in range(n_pert):
            
            if j<2: nets_list=[[nets] for nets in kwargs.get('nets')]
            else: nets_list=[kwargs.get('nets')]
        
            do_nets=kwargs.get('do_nets')
                    
            for nets in nets_list:
                index+=1
            
                if (i not in do_runs) and do_runs:
                    kwargs['active']=False
                elif j < from_disk_0:
                    kwargs['active']=False
                elif do_nets and j<2:
                    kwargs['active']=True
                    for net in nets:
                        if net in do_nets:
                            continue
                        kwargs['active']=False
                else:
                    kwargs['active']=True
              
                kwargs['hours']=kwargs['l_hours'][j]
                kwargs['minutes']=kwargs['l_minutes'][j]
                kwargs['seconds']=kwargs['l_seconds'][j]
                kwargs['index']=index
                kwargs_list.append(kwargs.copy())
    
    return kwargs_list

def get_kwargs_list(n_pert, kwargs):
    do_runs=kwargs.get('do_runs')
    from_disk_0=kwargs.get('from_disk_0')
    
    kwargs_list=[]
    index=-1
    for j in range(0, 3):
        for i in range(n_pert):
            index+=1
            if (i not in do_runs) and do_runs:
                kwargs['active']=False
            elif j < from_disk_0:
                kwargs['active']=False
            else:
                kwargs['active']=True
            
            kwargs['hours']=kwargs['l_hours'][j]
            kwargs['minutes']=kwargs['l_seconds'][j]
            kwargs['seconds']=kwargs['l_minutes'][j]
            kwargs['index']=index
            kwargs_list.append(kwargs.copy())
    
    return kwargs_list


     
def get_path_logs(from_milner_on_supermicro, file_name):
    _bool = my_socket.determine_host() == 'supermicro'
    if from_milner_on_supermicro and _bool:
        path_results = (default_params.HOME_DATA_BASE 
                        + 'milner/' 
                        + file_name 
                        + '/')
    else:
        path_results = (default_params.HOME_DATA 
                        + file_name 
                        + '/')
    return path_results

def get_path_nest(script_name, keys, par=None):
    if not par:
        par=default_params.Inhibition()
    path=par.get_path_data()
    file_name = path +script_name+ '/'+'_'.join(keys)+'/nest/'
#     file_name = home + '/results/papers/inhibition/network/' + script_name
    data_to_disk.mkdir(file_name)

    return file_name

def get_path_rate_runs(simulation_name):
    if my_socket.determine_computer() == 'milner':
        path_rate_runs = default_params.HOME_DATA + simulation_name
    else:
        path_rate_runs = default_params.HOME_DATA + simulation_name
    return path_rate_runs

def get_threads_postprocessing(t_shared, t_mpi, shared):
    if shared:
        threads = t_shared
    else:
        threads = t_mpi
    return threads

def get_type_of_run(shared=False): 
    if my_socket.determine_computer()=='milner':
        type_of_run='mpi_milner'
    else: 
        if not shared:
            type_of_run='mpi_supermicro'
        else:
            type_of_run='shared_memory'
    return type_of_run
    
def main_loop_conn(from_disk, attr, models, sets, nets, kwargs_dic, sd):
    d = {}
    from_disks = [from_disk] * len(nets.keys())
    for net, fd in zip(nets.values(), from_disks):
        if fd == 0:
            net.do_connect()
            dd=get_conn_matricies(net, models, attr)
            save(sd, dd)
        elif fd == 1:
            pass   
        elif fd == 2:
            filt = [net.get_name()] + sets + models + attr
            dd = load(sd, *filt)
            
    return from_disks, d

def main_loop(from_disk, attr, models, sets, nets, kwargs_dic, sd_list, **kwargs):
    
    run_method=kwargs.get('run',run)
    compute_method=kwargs.get('compute',compute)
    
    d = {}
    from_disks = [from_disk] * len(nets.keys())
    
    if type(sd_list)==list:
        iterator=[nets.values(), from_disks, sd_list]
    else:
        iterator=[nets.values(), from_disks]
    
    for vals in zip(*iterator):
        if type(sd_list)==list:
            net, fd, sd=vals
        else:
            net, fd=vals
            
        if fd == 0:
            dd = run_method(net)
            save(sd, dd)
        
        elif fd == 1:
            filt = [net.get_name()] + models + ['spike_signal']
            dd = load(sd, *filt)
            dd = compute_method(dd, models, attr, **kwargs_dic)
            save(sd, dd)
        
        elif fd == 2:
            filt =([net.get_name()] + sets + models + attr 
                   + kwargs.get('attrs_load',[]))
            dd = load(sd, *filt)
        
        d = misc.dict_update(d, dd)
    
    return from_disks, d


def par_process_and_thread(**kwargs):
    
    cores_milner=kwargs.get('cores_milner',40)
    cores_superm=kwargs.get('cores_superm',20)
    local_threads_milner=kwargs.get('local_threads_milner',10)
    local_threads_superm=kwargs.get('local_threads_superm',5)
    
    # core have to be multiple of 40 for milner
    host = my_socket.determine_computer() 

    
    if host == 'milner':
        local_threads=local_threads_milner
        
    
        d={
           'cores_hosting_OpenMP_threads':40/local_threads,
           'local_num_threads':local_threads, 
           'memory_per_node':int(819*local_threads),
           'num-mpi-task':cores_milner/local_threads,
           'num-of-nodes':cores_milner/40,
           'num-mpi-tasks-per-node':40/local_threads,
           'num-threads-per-mpi-process':local_threads,
           } 
        
    elif host == 'supermicro':
        local_threads=local_threads_superm
        d={
           'num-mpi-task':cores_superm/local_threads,
           'local_num_threads':local_threads, 
           'num-threads-per-mpi-process':local_threads,
           }
        
    return d


def pert_add_go_nogo_ss(**kwargs):

    l=kwargs.get('perturbation_list')


    p_sizes=kwargs.get('p_sizes')
    p_subsamp=kwargs.get('p_subsamp')
    max_size=kwargs.get('max_size')
    local_num_threads=kwargs.get('local_num_threads')
    do_not_record=kwargs.get('do_not_record',[])
    to_memory=kwargs.get('to_memory',False)
    to_file=kwargs.get('to_file',True)
    
    ll=[]

    p_sizes=[p/max(p_sizes) for p in p_sizes]
    for ss, p_size in zip(p_subsamp, p_sizes): 
        
        for i, _l in enumerate(l):
            _l=deepcopy(_l)
            per=pl({'netw':{'size':int(p_size*max_size), 
                            'sub_sampling':{'M1':ss,
                                            'M2':ss},}},
                      '=', 
                      **{'name':'ss-'+str(ss)})
            _l+=per
    
            _l+=pl({'simu':{'local_num_threads':local_num_threads,
                            'do_reset':True,
                            'sd_params':{'to_file':to_file, 
                                         'to_memory':to_memory}
                            }},'=')
            
            if do_not_record:
                for model in do_not_record:
                    _l+=pl({'node':{model:{'sd':{'active':False}}}}
                           ,'=',
#                            **{'name':'exclude-'+'_'.join(do_not_record)}
                            )
            ll.append(_l)

    return ll

def pert_add(p_list, **kwargs):
    
    op=kwargs.get('op_pert_add')
    if not isinstance(op, list):
        op=op.get() 
    
    out = []
    for l in op:
        l_copy = deepcopy(l)
        for ll in p_list:
            ll_copy = deepcopy(ll)
            ll_copy += l_copy
            out.append(ll_copy)
    return out

def pert_add_oscillations(**kwargs):
    
    amp_base=kwargs.get('amp_base')     
    freqs=kwargs.get('freqs')
    freq_oscillation=kwargs.get('freq_oscillation') 
    local_num_threads=kwargs.get('local_num_threads')
    path_rate_runs=kwargs.get('path_rate_runs')
    perturbation_list=kwargs.get('perturbation_list')
    sim_time=kwargs.get('sim_time')
    size=kwargs.get('size')
    
    l=perturbation_list
    for i in range(len(l)):
        l[i] += pl({'simu':{'do_reset':True,
                            'sd_params':{'to_file':True, 'to_memory':False},
                            'sim_time':sim_time, 
                            'sim_stop':sim_time,
                            'local_num_threads':local_num_threads}, 
                'netw':{'size':size}}, 
            '=')
    
    damp = process(path_rate_runs, freqs)
    for key in sorted(damp.keys()):
        val = damp[key]
        print numpy.round(val, 2), key
    
    ll = []
    for j, _ in enumerate(freqs):
        for i, _l in enumerate(l):

            amp = [numpy.round(damp[_l.name][j], 2), amp_base[j]]
            d = {'type':'oscillation2', 
                    'params':{'p_amplitude_mod':amp[0], 
                              'p_amplitude0':amp[1], 
                              'freq':freq_oscillation}}
            
            _l = deepcopy(_l)
            dd = {}
            for key in ['C1', 'C2', 'CF', 'CS']:
                dd = misc.dict_update(dd, {'netw':{'input':{key:d}}})
            
            _l += pl(dd, '=', **{'name':'amp_{0}-{1}'.format(*amp)})
            ll.append(_l)
    
    return ll


def pert_add_inhibition(**kwargs):
    

    l=kwargs.get('perturbation_list')
    local_num_threads=kwargs.get('local_num_threads')
    rep=kwargs.get('repetitions')
    res=kwargs.get('resolution')
    sim_time=rep*res*1000.0
    size=kwargs.get('size')
    
    for i in range(len(l)):
        l[i]+=pl({'simu':{'sim_time':sim_time,
                          'sim_stop':sim_time,
                           'local_num_threads':local_num_threads},
                  'netw':{'size':size}},
                  '=')
        
    return l

def pert_add_MSN_cluster_compete(**kwargs):
    
    l=kwargs.get('perturbation_list')
    local_num_threads=kwargs.get('local_num_threads')

    for i in range(len(l)):
        l[i]+=pl({'simu':{'local_num_threads':local_num_threads}},
                  '=')
        
    return l

def pert_set_data_path_to_milner_on_supermicro(l, set_it):
    if (my_socket.determine_host()=='milner') or (not set_it):
        return l
    
    dp=default_params.HOME_DATA_BASE+'/milner/'
    df=default_params.HOME_DATA_BASE+'milner_supermicro/fig/'
    for i in range(len(l)):
        l[i] += pl({'simu':{'path_data':dp, 
                            'path_figure':df}}, 
            '=')
    
    return l

def show_plot(axs, name, d, models=['M1','M2','FS', 'GA', 'GI','ST', 'SN'], **k):
    dd={}
    by_sets=k.pop('by_sets', False)
    

    for keys, val in misc.dict_iter(d):
        
        if keys[-1]!=name:
            continue
        if by_sets and keys[0][0:3]!='set':
            continue
        
        first_keys=keys[:-2]
        if type(first_keys)==str:
            first_keys=[first_keys]
        
        new_keys=['_'.join(first_keys)]+keys[-2:]
        
        dd=misc.dict_recursive_add(dd, new_keys, val)
        
    d=dd
    
#     if k.get('fig_and_axes', False):
        
#     else:
#         fig, axs=ps.get_figure(n_rows=len(models), n_cols=1, w=1000.0, h=800.0, 
#                            fontsize=k.get('fontsize',10))  
    labels=k.pop('labels', sorted(d.keys()))
#     colors=misc.make_N_colors('Paired', max(len(labels), 6))
    colors=k.pop('colors',misc.make_N_colors('jet', max(len(labels), 1)))
    linestyles=['-']*len(labels)
    
    j=0
    nets=k.get('nets', sorted(d.keys()))
    for key in nets:
        v=d[key]
#         axs[0].set_title(k)
        for i, model in enumerate(models):
            kk={'label':model+' '+labels[j],
                'linestyle':linestyles[j],
                'color':colors[j]}
            if 'win' in k.keys():
                kk['win']=k['win']
            if 't_start' in k.keys():
                kk['t_start']=k['t_start']
            if 't_stop' in k.keys():
                kk['t_stop']=k['t_stop']
            
            if model in v.keys():
                v[model][name].plot(ax=axs[i], **kk)
        j+=1    
    
    for ax in axs:
        ax.legend()
    
#     return fig, axs

def save_figures(figs, script_name, dpi=None):
    file_name_figs=get_file_name_figs(script_name)
    sd_figs = Storage_dic.load(file_name_figs)
    sd_figs.save_figs(figs, format='png', dpi=dpi)
    sd_figs.save_figs(figs, format='svg', in_folder='svg')

def show_coherence(d, models, **k):
    fig, axs=show_plot('mean_coherence',d, models)
    for ax in axs:
        
        ax.set_xlim(k.get('xlim', [0,50]))
    return fig
def show_fr(d, models, axs,  **k):
    
    show_plot(axs, 'firing_rate',d, models, **k)
#     return fig

def show_fr_sets(d, models, **k):
    fig, _ =show_plot('firing_rate',d, models, **k)
    return fig

def show_mr(d, models, axs, **k):
    show_plot(axs, 'mean_rate_slices',d, models, **k)
    
    if k.get('relative', False):
        r_to1, r_to2=k.get('relative_to') #index
        for ax in axs:
            print ax.lines
            y_upp=ax.lines[r_to1].get_ydata()
            y_low=ax.lines[r_to2].get_ydata()
            y=y_upp-y_low
            
            for i, line in enumerate(ax.lines):
                
#                 ax.lines[i]._y/=y
                line.set_ydata(1-(line.get_ydata()-y_low)/y)
#                 line.set_ydata(1-line.get_ydata())#               
#   import copy
#                 ax.lines[i]._y=copy.deepcopy(1-line._y)
#                 ax.lines[i]._y=copy.deepcopy(1-line._y)
#                 ax.lines[i]._y=copy.deepcopy(1-line._y)            
#                 print ax.lines[i]._y


#     import pylab
#     pylab.show()

    
    if k.get('delete', False):
        j=0
        for i in k.get('delete'):
            del ax.lines[i-j]
            j+=1 
            
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)       

    if k.get('y_lim', False):
        for ax in axs:
            ax.set_ylim(k.get('y_lim'))

    if k.get('x_lim', False):
        for ax in axs:
            ax.set_xlim(k.get('x_lim'))
#     return fig

def show_mr_diff(d, models, axs, **k):
    show_plot(axs, 'mean_rate_diff',d, models, **k)
    
    for ax in axs:
        ax.set_xlabel('Active MSNs (%)')
        
    for ax in axs:
        ax.set_ylabel('Firing rate (spike/s)')
#     return fig

def show_hr(d, models, **k):
    fig, _ =show_hist('mean_rates',d, models, **k)
    return fig

def show_hist(name, d, models=['M1','M2','FS', 'GA', 'GI','ST', 'SN'], **k):

    fig, axs=ps.get_figure(n_rows=len(models), n_cols=1, w=1000.0, h=800.0, fontsize=10)   
    labels=k.pop('labels', sorted(d.keys()))

    colors=misc.make_N_colors('jet', len(labels))
    linestyles=['solid']*len(labels)
    linewidth=[2.0]*len(labels)
    j=0
    
    for key in sorted(d.keys()):
        v=d[key]
#         axs[0].serunt_title(k)
        
        for i, model in enumerate(models):
            if 'spike_stastistic' in v[model]:
                st=v[model]['spike_statistic']
                st.rates={'mean':round(st.rates['mean'],2),
                          'std':round(st.rates['std'],2),
                          'CV':round(st.rates['CV'],2)}
                s=str(st.rates)
            else:
                s=''
#             print s 
#             print labels[j]
            k.update({'label':(model+' '+labels[j]+' ' +s),
                     'histtype':'step',
                     'linestyle':linestyles[j],
                     'color':colors[j],
                     'linewidth':linewidth[j]})
#             print k
            h=v[model][name].hist(ax=axs[i],**k) 
            
            ylim=list(axs[i].get_ylim())
            ylim[0]=0.0
            axs[i].set_ylim(ylim)
            axs[i].legend_box_to_line()
        j+=1 
#     import pylab
#     pylab.show()
    return fig, axs

def show_phase_diff(d, models, **k):
    fig, _ =show_hist('phase_diff',d, models, **k)
    return fig

def show_psd(d, models):
    fig, axs=show_plot('psd',d, models)   
    for ax in axs:
        ax.set_xlim([0,50])
    return fig


import unittest
class TestModuleFunctions(unittest.TestCase):
    def setUp(self):
        def run(net):
            return {net:{'unittest_1':'spike_signa'}}
        
        def compute(d, *args, **kwargs):
            d={}
            for net in d.keys():
                return {net:{'unittest_1':'spike_signa'}}
            
            
    def test_main_loop(self):
        from toolbox import data_to_disk
        data_to_disk.Storage_dic
        
#         main_loop(from_disk, attr, models, sets, nets, kwargs_dic, sd_list)
        pass

if __name__ == '__main__':
    d={

       TestModuleFunctions:[
#                            'test_create',
#                            'test_2_simulate_show',
                           ],

       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite) 
