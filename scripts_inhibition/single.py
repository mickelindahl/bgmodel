'''
Created on 21 mar 2014

@author: mikael
'''
from copy import deepcopy

from toolbox import pylab, misc
from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.manager import compute, save, load
from toolbox.network.optimization import Fmin

from os.path import expanduser
home = expanduser("~")

import pprint
pp=pprint.pprint

def _ud(k,d):
    k=deepcopy(k)
    k['kwargs_builder'].update(d)
    return k
            

def beautify_hist(ax, colors, linestyles, linewidth):
    for p, c, ls, ln in zip(ax.patches, colors, linestyles, linewidth):
        pylab.setp(p, edgecolor=c, linestyle=ls, linewidth=ln)
    
    ylim = list(ax.get_ylim())
    ylim[0] = 0.0
    ax.set_ylim(ylim)
    ax.legend_box_to_line()

def beautify(axs):
    colors=['b', 'b', 'g','g',]
    linestyles=['-','--','-','--']  
    linewidth=[2,2,2,2]  
    linestyles2=['solid','dashed','solid','dashed'] 
    
    for i in [0,2]:
        for c, ls, l,lw in zip(colors, linestyles, axs[i].get_lines(),
                            linewidth):
            pylab.setp(l,color=c, linestyle=ls, linewidth=lw)
            pylab.setp(axs[i].collections, color=c)
        axs[i].legend(loc='upper left')
  
    for c, ls, i, lw in zip(colors, linestyles, range(len(axs[1].collections)),
                        linewidth):
        pylab.setp(axs[1].get_lines()[i*10:(i+1)*10],color=c, linestyle=ls,
                   linewidth=lw)
        pylab.setp(axs[1].collections, color=c)
    
    axs[1].legend(loc='upper left')

    beautify_hist(axs[3], colors, linestyles2, linewidth)


def get_if_argv(val, index, argv, data_type):
    if len(argv) > index:
        return data_type(argv[index])
    else:
        return val


def get_kwargs_builder():
    return {'print_time':False, 
            'rand_nodes':{'C_m':False, 'V_th':False, 'V_m':False},
            'save_conn':{'overwrite':True},
            'sim_stop': 5000.0,
            'sim_time': 5000.0, 
            'size':9.0, 
            'start_rec':1000.0,
            'threads':1}    
    
def _get_networks(Builder, **kwargs):
    info, nets, _=manager.get_networks(Builder, 
                         kwargs['kwargs_builder'], 
                         kwargs['kwargs_engine'])
    
    return info, nets   


 

def get_setup_IV(Builder, k):
    a, b = _get_networks(Builder, **_ud(k, {'lesion':True, 
                'mm':True, 
               'rand_nodes':{'C_m':False, 
                            'V_th':False, 
                            'V_m':False},
                'sim_time':k.get('IV_time', 5000.), 
                'size':k.get('IV_size', 9)}))
    return a, b


def get_setup_IF(Builder, k):
    a, b = _get_networks(Builder, **_ud(k, {'lesion':True, 
                'rand_nodes':{'C_m':False, 
                              'V_th':False, 
                              'V_m':False},
                'sim_time':k.get('IF_time', 5000.), 
                'size':k.get('IF_size', 9)}))
    return a, b


def get_setup_FF(Builder, k):
    a, b = _get_networks(Builder, **_ud(k, {'lesion':False, 
                'size':k.get('FF_size', 50), 
                'sim_time':k.get('FF_time', 5000.), 
                'threads':k.get('threads', 4)}))
    return a, b


def get_setup_opt_rate(Builder, k):
    a, b = _get_networks(Builder, **_ud(k, {'lesion':False, 
                'size':k.get('FF_size', 50), 
                'sim_time':k.get('opt_rate_time', 10000.), 
                'sim_stop':k.get('opt_rate_time', 10000.), 
                'threads':k.get('threads', 4)}))
    return a, b


def get_setup_hist(Builder, k):
    a, b = _get_networks(Builder, **_ud(k, {'lesion':False, 
                'size':k.get('hist_size', 200), 
                'sim_time':k.get('hist_time', 10000.), 
                'sim_stop':k.get('hist_time', 10000.), 
                'threads':k.get('threads', 4)}))
    return a, b

def get_setup(Builder, **k):
    '''
    kwargs can contain 
    'kwargs_builder' which is parameters set in the builder
    'kwargs_engine' which is parameres for network
    and  the forexample
    'FF_size' which is a short cut to setting the size of FF simulation
    population. This value is updated in the kwargs_builder by update
    function
    
    '''
    k.update({'IV_time':5000.0,
               'IV_size':9.0,       
               'IF_time':5000.0,
               'IF_size':9.0,
               'FF_time':5000.0,
               'FF_size':50.0,
               'opt_rate_time':10000.0,
               'opt_rate_size':50.0,
               'hist_time':10000.0,
               'hist_size':50.0,
               'threads':16})
    
    d={}
    dinfo={}
    
    dinfo['IV'],d['IV']=get_setup_IV(Builder, k)
    dinfo['IF'],d['IF']=get_setup_IF(Builder, k)
    dinfo['FF'],d['FF']=get_setup_FF(Builder, k)  
    dinfo['opt_rate'],d['opt_rate']=get_setup_opt_rate(Builder, k)
    dinfo['hist'],d['hist']=get_setup_hist(Builder, k) 
    dinfo['fig'],d['fig']='figure',{}
     
    return dinfo, d

def get_file_names(suffix, data_names):
    file_name0=main_path()+suffix
    file_names=[]
    for s in data_names:
        if s=='fig':
            file_names.append(main_path()+'fig_'+suffix) 
        else:
            file_names.append(file_name0+'/'+s)
    pp(file_names)
    return file_names

def get_storages(suffix, data_names, dinfo):
    file_names=get_file_names(suffix, data_names) 
    d= {}
    for fn, dn in zip(file_names, data_names):
        d[dn]=Storage_dic.load(fn)
        d[dn].add_info(dinfo[dn])
        d[dn].garbage_collect()

    return d

def main_path():
    return home+ '/results/papers/inhibition/single/'

def _optimize(flag, net, storage_dic, d, from_disk, **kwargs):

    model = net.get_single_unit() 
    inp = net.get_single_unit_input()  
    attr='fmin'
    f=[model]
    if flag=='opt_rate':
        x=['node.'+inp+'.rate']
    if flag=='opt_curr':
        x=['node.' + model + '.nest_params.I_e']
    
    x0=kwargs.get('x0',900.)
    
    opt={'netw':{'optimization':{'f':f,
                                'x':x,
                                'x0':x0}}}    
    net.par.update_dic_rep(opt)
    
    kwargs_fmin={'model':net,
                 'call_get_x0':'get_x0',
                 'call_get_error':'sim_optimization', 
                 'verbose':True}
    
    if not from_disk: 
        f=Fmin(net.name, **kwargs_fmin)
        dd={net.get_name():{model:{attr:f.fmin()}}}    
        print dd 
        save(storage_dic, dd)
    elif from_disk:
        filt = [net.get_name()] + [model] + [attr]
        dd = load(storage_dic, *filt) 

    dd=reduce_levels(dd,  [model] + [attr])

    d = misc.dict_update(d, dd)
     
    return d 

def optimize(flag, dn, from_disks, ds, **k):
    d={} 
    nets=dn[flag]
    storage_dic=ds[flag]
    
    for net, from_disk in zip(nets, from_disks):
        d = _optimize(flag, net, storage_dic, d, from_disk, **k)
    return {flag:d}    


def reduce_levels(dd, levels):
    for level in levels:     
        dd = misc.dict_remove_level(dd, level)
    return dd

def _run(storage_dic, d, net, from_disk, attr, **kwargs):
    model = net.get_single_unit()
    kwargs_dic={attr:kwargs}
    if not from_disk:

        dd = {net.get_name(): net.simulation_loop()}
        dd = compute(dd, [model], [attr], **kwargs_dic)
        save(storage_dic, dd)
    elif from_disk:
#         filt = [net.get_name()] + [model] + ['spike_signal']
#         dd = load(storage_dic, *filt)
#         dd = compute(dd, [model], [attr], **kwargs_dic)
#         save(storage_dic, dd)
        filt = [net.get_name()] + [model] + [attr]
        dd = load(storage_dic, *filt)
        
    dd=reduce_levels(dd,  [model] + [attr])
    d = misc.dict_update(d, dd)
    return d
     
       
def run(flag, dn, from_disks, ds, attr, **kwargs_dic):
    d={}
    nets=dn[flag]
    storage_dic=ds[flag]
    for net, from_disk in zip(nets, from_disks):
        d=_run(storage_dic, d, net, from_disk, attr, **kwargs_dic)
    return {flag:d}

def _run_XX(flag, storage_dic, stim, d, net, from_disk):
    model = net.get_single_unit()
    inp = net.get_single_unit_input()
    
    if flag=='IV':
        attr='IV_curve'
        stim_name='node.' + model + '.nest_params.I_e'
        call=getattr(net, 'sim_IV_curve') 
        
    if flag=='IF':
        attr='IF_curve'
        stim_name='node.' + model + '.nest_params.I_e'
        call=getattr(net, 'sim_IF_curve')     
 
    if flag=='FF':
        attr='FF_curve'
        stim_name='node.'+inp+'.rate'
        call=getattr(net, 'sim_FF_curve')             
    
    if not from_disk:
        k = {'stim':stim, 
             'stim_name':stim_name, 
             'stim_time':net.get_sim_time(), 
             'model':model} 
         
        dd = {net.get_name(): call(**k)}
        dd = compute(dd, [model], [attr])
        save(storage_dic, dd)
    elif from_disk:
        filt = [net.get_name()] + [model] + [attr]
        dd = load(storage_dic, *filt)
    
    pp(dd)
    dd=reduce_levels(dd,  [model] + [attr])
    d = misc.dict_update(d, dd)
    return d


def run_XX(flag, dn, from_disks, ds, dstim):
    
    d={} 
    nets=dn[flag]
    storage_dic=ds[flag]
    stim=dstim[flag]
    for net, from_disk in zip(nets, from_disks):
        d = _run_XX(flag, storage_dic, stim, d, net, from_disk)
                
    return {flag:d}

   
def set_optimization_val(data, nets):
    for net in nets:
        inp = net.get_single_unit_input()
        x='node.'+inp+'.rate' #kwargs.get('x', ['node.CFp.rate'])
        dic=misc.dict_recursive_add({}, x.split('.'), data.xopt[-1][0])
        net.update_dic_rep(dic)
        
        
def show(dstim, d, axs, names):
    for model, ax in zip(['IV', 'IF', 'FF'], axs):
        for key, name in zip(sorted(d[model].keys()), names):
            data=d[model][key]
            
            data.plot(ax, x=dstim[model], **{'label':name})
    
    for key in d['opt_rate'].keys():
        data=d['opt_rate'][key]
        data.plot(axs[3], names[0])
    
    for key, name in zip(sorted(d['hist'].keys()), names):
        data=d['hist'][key]
        data.hist(axs[3], **{'label':name})
    
    beautify(axs)
    

def show_opt_hist(d, axs, name):
    colors=misc.make_N_colors('jet',len(d['opt_rate'].keys()))
    i=0
    for key in sorted(d['opt_rate'].keys()):
        data=d['opt_rate'][key]
        data.plot(axs[0], name, **{'color':colors[i]})
        i+=1
    for key in sorted(d['hist'].keys()):
        data=d['hist'][key]
        data.hist(axs[0])
    
    linestyles=['solid']*len(colors)
    linewidth=[2]*len(colors)  
    beautify_hist(axs[0], colors, linestyles, linewidth)   

