'''
Created on Jul 4, 2013

@author: lindahlm
'''

from copy import deepcopy
from toolbox.network.default_params import Single_unit, Inhibition
from toolbox.network.construction import Network
from toolbox.network.handling_single_units import Activity_model_dic
from toolbox import misc, data_to_disk
from toolbox import plot_settings as pl
import pylab
import numpy

import pprint
pp=pprint.pprint


def beautify(axs):
    colors=['b', 'b', 'g','g',]
    linestyles=['-','--','-','--']  
    
    for i in [0,2]:
        for c, ls, l in zip(colors, linestyles, axs[i].get_lines()):
            pylab.setp(l,color=c, linestyle=ls)
            pylab.setp(axs[i].collections, color=c)
        axs[i].legend()
    
    
    for c, ls, i in zip(colors, linestyles, range(4)):
        pylab.setp(axs[1].get_lines()[i*10:(i+1)*10],color=c, linestyle=ls)
        pylab.setp(axs[1].collections, color=c)
    axs[1].legend()
        

def build_general(**kwargs):
    d={'simu':{
               'mm_params':{'to_file':False, 'to_memory':True},
               'print_time':False,
               'sd_params':{'to_file':False, 'to_memory':True},
               'sim_stop':kwargs.get('sim_stop', 1000.0),
               'sim_time':kwargs.get('sim_time', 1000.0),
               'start_rec':0.0,
               'stop_rec':kwargs.get('sim_stop', 1000.0),
               'threads':kwargs.get('threads', 1),
               },}
    return d

def build_specific(**kwargs):
    su=kwargs.get('single_unit', 'FS')
    inputs=kwargs.get('inputs',['FSp', 'GAp', 'CFp'])   
    
    d={'netw':{
               'size':kwargs.get('size',9),
               'single_unit':su,
            },
       'node':{su:{'mm':{'active':kwargs.get('mm', False)},
                   'sd':{'active':True},
                   'n_sets':1}}}
       

    for inp in inputs:
        d['node'][inp]={'lesion':kwargs.get('lesion', False)}
  
    return d

def build_cases(**kwargs):
    l=[{'netw':{'tata_dop':0.8},
        'node':{'FS':{'model':'FS_low'}}}]
    l+=[{'netw':{'tata_dop':0.0},
        'node':{'FS':{'model':'FS_low'}}}]
    l+=[{'netw':{'tata_dop':0.8},
        'node':{'FS':{'model':'FS_high'}}}]
    l+=[{'netw':{'tata_dop':0.0},
             'node':{'FS':{'model':'FS_high'}}}]
    
    d=build_general(**kwargs)
    d=misc.dict_update(d, build_specific(**kwargs))
    
    names=['$FSN_{+d}^{l}$',
           '$FSN_{-d}^{l}$',
           '$FSN_{+d}^{h}$',
           '$FSN_{-d}^{h}$']
    
    nets=[]
    for name, e in zip(names, l):
        dd=misc.dict_update(e, d)
        par=Single_unit(**{'dic_rep':dd, 'other':Inhibition()})
        net=Network(name, **{'par':par}) 
        nets.append(net)
        
    return nets

def get_simulation(obj, method, load, **k):
    fileName=obj.get_path_data()+obj.get_name()+'_'+method
    if load:
        return data_to_disk.pickle_load(fileName)
    else:
        call=getattr(obj, method)
        d=call(**k)
        data_to_disk.pickle_save(d, fileName)
        return d
        
def plots(method, nets, loads, **kwargs):
    module= __import__(__name__)
    call=getattr(module, method)
    if type(loads)!=list: loads=[loads for _ in nets]
    
    for net, load in zip(nets, loads):
        call(net, load, **kwargs)


def plot_IV_curve(net, load, ax=None, **kwargs):
    if ax==None:
        ax=pylab.subplot(111)
    curr=kwargs.get('curr',range(-200,300,100))
    node=kwargs.get('node', 'FS')
    k={'stim':curr,
       'stim_name':'node.'+node+'.nest_params.I_e',
       'stim_time':kwargs.get('sim_time',1000.0),
       'model':node}

    dud=get_simulation(net, 'sim_IV_curve', load, **k)['voltage_signal'] 
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
       'stim_time':kwargs.get('sim_time',1000.0),
       'model':node}

    dud=get_simulation(net, 'sim_IF_curve', load, **k)['spike_signal'] 
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
       'stim_time':kwargs.get('sim_time',1000.0),
       'model':node}

    dud=get_simulation(net, 'sim_FF_curve', load, **k)['spike_signal'] 
    dud[node].plot_FF_curve(ax=ax, x=rate, **{'label':net.name, 
                                              'linewidth':3.0}) 
           
def main():    
    IV=build_cases(**{'lesion':True, 'mm':True})
    IF=build_cases(**{'lesion':True})
    FF=build_cases(**{'lesion':False})
    
    curr_IV=range(-200,300,100)
    curr_IF=range(0,500,100)
    rate_FF=range(100,1500,100)
    _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)     
    
    plots('plot_IV_curve', IV, 1, **{'ax':axs[0],'curr':curr_IV, 'node':'FS'})
    plots('plot_IF_curve', IF, 1, **{'ax':axs[1],'curr':curr_IF, 'node':'FS'})
    plots('plot_FF_curve', FF, 1, **{'ax':axs[2],'rate':rate_FF, 'node':'FS',
                                     'input':'CFp'})    
    beautify(axs)
    pylab.show()
    
if __name__ == "__main__":
    main()     
    
    
