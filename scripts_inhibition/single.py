'''
Created on 21 mar 2014

@author: mikael
'''
from copy import deepcopy
from toolbox.network.default_params import Single_unit, Inhibition
from toolbox.network.construction import Network
from toolbox.network.optimization import Fmin
from toolbox.network.handling_single_units import Activity_model_dic
from toolbox import misc, data_to_disk
from toolbox import plot_settings as pl
import pylab
import numpy

def beautify(axs):
    colors=['b', 'b', 'g','g',]
    linestyles=['-','--','-','--']  
    linestyles2=['solid','dashed','solid','dashed'] 
    
    for i in [0,2]:
        for c, ls, l in zip(colors, linestyles, axs[i].get_lines()):
            pylab.setp(l,color=c, linestyle=ls)
            pylab.setp(axs[i].collections, color=c)
        axs[i].legend(loc='upper left')
    
    
    for c, ls, i in zip(colors, linestyles, range(len(axs[1].collections))):
        pylab.setp(axs[1].get_lines()[i*10:(i+1)*10],color=c, linestyle=ls)
        pylab.setp(axs[1].collections, color=c)
    
    axs[1].legend(loc='upper left')
    
    for p, c, ls in zip(axs[3].patches, colors, linestyles2):    
        pylab.setp(p, edgecolor=c, linestyle=ls) 
    ylim=list(axs[3].get_ylim())
    ylim[0]=0.0
    axs[3].set_ylim(ylim)
          
def build_general(**kwargs):
    d={'simu':{
               'mm_params':{'to_file':False, 'to_memory':True},
               'print_time':False,
               'save_conn':False,
               'sd_params':{'to_file':False, 'to_memory':True},
               'sim_stop':kwargs.get('sim_stop', 41000.0),
               'sim_time':kwargs.get('sim_time', 41000.0),
               'start_rec':kwargs.get('start_rec', 1000.0),
               'stop_rec':kwargs.get('stop_rec',numpy.inf),
               
               'threads':kwargs.get('threads', 1),
               },
        'netw':{'rand_nodes':{'C_m':False, 'V_th':False, 'V_m':False},
                }
}
    return d

def create_list_dop(su):
    l = [{'netw':{'tata_dop':0.8}, 'node':{su:{'model':su}}}]
    l += [{'netw':{'tata_dop':0.0}, 
            'node':{su:{'model':su}}}]

    return l

def create_list_dop_high_low(su):
    l = [{'netw':{'tata_dop':0.8}, 'node':{su:{'model':su + '_low'}}}]
    l += [{'netw':{'tata_dop':0.0}, 
            'node':{su:{'model':su + '_low'}}}]
    l += [{'netw':{'tata_dop':0.8}, 
            'node':{su:{'model':su + '_high'}}}]
    l += [{'netw':{'tata_dop':0.0}, 
            'node':{su:{'model':su + '_high'}}}]
    return l

def creat_dic_specific(kwargs, su, inputs):
    d = {'netw':{'size':kwargs.get('size', 9), 
            'single_unit':su}, 
        'node':{su:{'mm':{'active':kwargs.get('mm', False)}, 
                'sd':{'active':True}, 
                'n_sets':1}}}
    for inp in inputs:
        d['node'][inp] = {'lesion':kwargs.get('lesion', False)}
    
    return d

def create_nets(l, d, names):
    nets = []
    for name, e in zip(names, l):
        dd = misc.dict_update(e, d)
        par = Single_unit(**{'dic_rep':dd, 'other':Inhibition()})
        net = Network(name, **{'verbose':True, 'par':par})
        nets.append(net)
    
    return nets
        
def do(method, nets, loads, **kwargs):
    module= __import__(__name__)
    call=getattr(module, method)
    if type(loads)!=list: 
        loads=[loads for _ in nets]
    
    l=[]
    for net, load in zip(nets, loads):
        l.append(call(net, load, **kwargs))
    return l

def evaluate(obj, method, load, **k):
    fileName=obj.get_path_data()+obj.get_name()+'_'+method
    if load:
        return data_to_disk.pickle_load(fileName)
    else:
        call=getattr(obj, method)
        duds=call(**k)
        data_to_disk.pickle_save(duds, fileName)
        return duds

def save_dud(*args):
    for a in args:
        data_to_disk.pickle_save(a, a.get_file_name())
        
    

def optimize(net, load,  ax=None, opt=None, **kwargs):
    if ax==None:
        ax=pylab.subplot(111)
    f=kwargs.get('f', ['FS'])
    x=kwargs.get('x', ['node.CFp.rate'])
    x0=kwargs.get('x0', 'CFp')
    opt={'netw':{'optimization':{'f':f,
                                'x':x,
                                'x0':x0}}}    
    net.par.update_dic_rep(opt)
    
    kwargs={'model':net,
            'call_get_x0':'get_x0',
            'call_get_error':'sim_optimization', 
            'verbose':True}
    
    f=Fmin(net.name, **kwargs)

    data=evaluate(f, 'fmin', load,  **{})

    p,x,y=ax.texts, 0.02,0.9
    if len(p):
        x=p[-1]._x
        y=p[-1]._y-0.1
        
        
               
    p=ax.text( x, y, net.get_name()+':'+str(data['xopt'][-1][0])+' Hz', 
               transform=ax.transAxes, 
        fontsize=pylab.rcParams['font.size']-2)
    
    return data 

def plot_IV_curve(net, load, ax=None, **kwargs):
    if ax==None:
        ax=pylab.subplot(111)
    curr=kwargs.get('curr',range(-200,300,100))
    node=kwargs.get('node', 'FS')
    k={'stim':curr,
       'stim_name':'node.'+node+'.nest_params.I_e',
       'stim_time':kwargs.get('sim_time',net.get_sim_time()),
       'model':node}

    dud=evaluate(net, 'sim_IV_curve', load, **k)['voltage_signal'] 
    dud[node].plot_IV_curve(ax=ax, x=curr, **{'label':net.name, 
                                              'marker':'o',
                                              'linewidth':3.0}) 


def plot_IF_curve(net, load, ax=None, **kwargs):
    if ax==None:
        ax=pylab.subplot(111)
    curr=kwargs.get('curr',range(0,500,100))
    node=kwargs.get('node', 'FS')
    k={'stim':curr,
       'stim_name':'node.'+node+'.nest_params.I_e',
       'stim_time':kwargs.get('sim_time', net.get_sim_time()),
       'model':node}

    dud=evaluate(net, 'sim_IF_curve', load, **k)['spike_signal'] 
    dud[node].plot_IF_curve(ax=ax, x=curr, **{'label':net.name, 
                                              'linewidth':3.0}) 

def plot_FF_curve(net, load, ax=None, **kwargs):
    if ax==None:
        ax=pylab.subplot(111)
    rate=kwargs.get('rate',range(0,500,100))
    node=kwargs.get('node', 'FS')
    input=kwargs.get('input', 'CFp')
    k={'stim':rate,
       'stim_name':'node.'+input+'.rate',
       'stim_time':kwargs.get('sim_time', net.get_sim_time()),
       'model':node}

    dud=evaluate(net, 'sim_FF_curve', load, **k)['spike_signal'] 
    dud[node].plot_FF_curve(ax=ax, x=rate, **{'label':net.name, 
                                              'linewidth':3.0}) 

def plot_hist_isis(net, load, ax=None, **kwargs):
    node=kwargs.get('node', 'FS')
    dud=evaluate(net, 'simulation_loop', load, **{})['spike_signal']
    dud[node].plot_hist_isis(ax=ax, **{'label':net.name, 'histtype':'step'})

def plot_hist_rates(net, load, ax=None, **kwargs):
    node=kwargs.get('node', 'FS')
    dud=evaluate(net, 'simulation_loop', load, **{})['spike_signal']
    st=dud[node].get_spike_stats()
    st['rates']={'mean':round(st['rates']['mean'],2),
                'std':round(st['rates']['std'],2),
                'CV':round(st['rates']['CV'],2)}
    dud[node].plot_hist_rates(ax=ax, t_start=net.get_start_rec(),
                              t_stop=net.get_sim_stop(), 
                              **{'label':net.name+' '+str(st['rates']), 'histtype':'step',
                                 'bins':20})

def sim(net, load, ax=None, **kwargs):
    dud=evaluate(net, 'simulation_loop', load, **{})['spike_signal']
    return dud

# def plot_firing_rate(dud, load, ax=None, **kwargs):
#     node=kwargs.get('node', 'FS')
#     dud[node].plot_firing_rate(ax=ax, t_start=net.get_start_rec(),
#                               t_stop=net.get_sim_stop(), 
#                               **{'label':node})    
#     
# def plot_firing_rates(dud, load, ax=None, **kwargs):
#     nodes=kwargs['nodes']
#     for _ax, models in zip(ax, nodes):
#         for name in models:
#             kwargs['node']=name
#             plot_firing_rate(dud, load, ax=_ax, **kwargs)
        
        
    
def set_optimization_val(datas, nets, **kwargs):
    for d, n in zip(datas, nets):
        x=kwargs.get('x', ['node.CFp.rate'])
        dic=misc.dict_recursive_add({}, x[0].split('.'), d['xopt'][-1][0])
        n.update_dic_rep(dic)
