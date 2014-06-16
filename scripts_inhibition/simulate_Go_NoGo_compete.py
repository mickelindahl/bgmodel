'''
Created on 25 mar 2014

@author: mikael
'''

import numpy
import pylab
import toolbox.plot_settings as ps
from simulate import cmp_mean_rates_intervals
from toolbox import misc

from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.manager import compute, run, save, load
from toolbox.network.manager import Builder_Go_NoGo_compete as Builder

import pprint
pp=pprint.pprint

def get_kwargs_builder(rep, res):
    return {'print_time':False,
            'resolution':res, 
            'repetitions':rep,
            'sim_time':1000.0, # need to sum to the durationof silence + burst in burst3 setup
            'sim_stop':1000.*res*res*rep, 
            'size':750.0, 
            'start_rec':0.0,  
            'stop_rec':1000.*res*res*rep,
            'sub_sampling':10,
            'threads':4,}   
    
def get_kwargs_engine():
    return {'verbose':True}

def get_networks(*args):
    info, nets, builder=manager.get_networks(Builder,
                                             get_kwargs_builder(*args),
                                             get_kwargs_engine())
    
    intervals=builder.dic['intervals']
    rep=builder.dic['repetitions']
    
    return info, nets, intervals, rep

def show_fr(d):
    _, axs=ps.get_figure(n_rows=7, n_cols=1, w=1000.0, h=800.0, fontsize=10)  
    for model, i in [['M1',0], ['M2', 1], ['GI',2], ['SN',3]]:
        d[model]['firing_rate'].plot(ax=axs[i],  **{'label':model})
        

def classify(x, y, threshold):
    if (x < threshold) and (y < threshold):
        return 0
    if (x >= threshold):
        return 1
    if (y >= threshold):
        return 2
    else:
        return 3
    
def process_data(data, threshold=5):
    
    outs=[]
    for d in data:
        x,y=d[2][0]['y'], d[2][1]['y'] 
        outs.append(numpy.abs(x-y))
    return outs

def plot(data):
    _, axs=ps.get_figure(n_rows=1, n_cols=5, w=1200.0, h=600.0, fontsize=16)   

    for d in data:
        i=0
        for v in d.transpose():
            axs[i].hist(v,**{'histtype':'step', 
                             'bins':20})
            ylim=list(axs[i].get_ylim())
            ylim[0]=0.0
            axs[i].set_ylim(ylim)
            i+=1
  



def get_kwargs_dic():
    {'firing_rate':{'time_bin':5}}
    

def main(builder=Builder,
         from_disk=0,
         perturbation_list=None,
         rep=2,
         res=2,
         script_name=__file__.split('/')[-1][0:-3],):
    
    
    from os.path import expanduser
    home = expanduser("~")  
    file_name=(home+ '/results/papers/inhibition/network/'
               +__file__.split('/')[-1][0:-3])
    
    models=['M1', 'M2', 'SN']
    
    info, nets, intervals, x = get_networks(rep,res)

    sd=Storage_dic.load(file_name)
    sd.add_info(info)
    sd.garbage_collect()
    
    d={}
    from_disks=[from_disk]*2
    for net, fd in zip(nets.values(), from_disks):
        if fd==0:
            dd = run(net)       
            save(sd, dd)
        elif fd==1:
            filt=[net.get_name()]+models+[
                                          'spike_signal',
                                          ]
            dd=load(file_name, *filt)
            kwargs_dic=get_kwargs_dic()
            dd=compute(dd, models,  ['firing_rate'], **kwargs_dic )  

            dd=cmp_mean_rates_intervals(dd,intervals[1], x, rep)
            save(sd, dd)
        elif fd==2:
            filt=[net.get_name()]+models+['firing_rate',
                                          'mean_rates_intervals',
                                         ]
            dd=load(file_name, *filt)
        d=misc.dict_update(d, dd)
            
    
    
    show_fr(d['net_0'])
    data=misc.dict_remove_level(d['net_0'], 'mean_rates_intervals')
    print data
    if 1:
        data=process_data(data)
    print data
    plot(data)
    
    pylab.show()
 
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()
    
    