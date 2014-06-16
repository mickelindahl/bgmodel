'''
Created on May 14, 2014

@author: mikael
'''

import numpy
import os
from simulate import show_fr, show_mr_diff,cmp_mean_rates_intervals
from toolbox import misc, pylab
from toolbox.my_signals import Data_generic
from toolbox.network import manager
from toolbox.network.manager import (add_perturbations, compute, 
                                     run, save, load, get_storage)

from toolbox.network.manager import Builder_MSN_cluster_compete as Builder

import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')



def cmp_mean_rate_diff(d, models, parings, x):
    for model in models:
        y=[]
        y_std=[]
        for net0, net1 in parings:
            d0=d[net0]
            d1=d[net1]
   
            v0=d0[model]['mean_rates_intervals']
            v1=d1[model]['mean_rates_intervals']
            
            y.append(numpy.mean(numpy.abs(v0.y_raw_data-v1.y_raw_data)))
            y_std.append(numpy.std(numpy.abs(v0.y_raw_data-v1.y_raw_data)))
        
        dd={'y':numpy.array(y),
           'y_std':numpy.array(y_std),
           'x':numpy.array(x)}
        obj=Data_generic(**dd)
        
        d=misc.dict_recursive_add(d, ['Difference',
                                  model,
                                 'mean_rate_diff'], obj)
    pp(d)
    return d    
        
      
        
def get_kwargs_builder():
    return {'print_time':False, 
            'threads':10, 
            'save_conn':{'active':False,'overwrite':False},
            'sim_time':5000.0, 
            'sim_stop':5000.0, 
            'size':3000.0, 
            'start_rec':0.0, 
            'sub_sampling':1}

def get_kwargs_engine():
    return {'verbose':True}

def get_networks():
    info, nets, builder=manager.get_networks(Builder,
                                             get_kwargs_builder(),
                                             get_kwargs_engine())
    
    intervals=builder.dic['intervals']
    rep=builder.dic['repetitions']
    x=builder.dic['percents']
    
    return info, nets, intervals, rep, x

    
def main(from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3]):
    
    
    from os.path import expanduser
    home = expanduser("~")

    
    file_name=(home+ '/results/papers/inhibition/network/'+script_name)
    
    #models=['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN']
    
    models=['M1', 'M2']
    
    info, nets, intervals, rep, x = get_networks()
    add_perturbations(perturbation_list, nets)
    print info
#     pp(nets[1].par.dic['node']['M1'])
#     pp(nets[0].par['node']['C1'])
#     pp(nets[5].par['node']['C1'])
    
    sd = get_storage(file_name, info)#, nets=['Net_0'])
    sd.garbage_collect()
    
    d={}
    from_disks=[0]*10
    for net, mode in zip(nets.values(), from_disks):
        pp(net)
        if mode==0:
            dd = run(net)    
            save(sd, dd)
            print sd
#         elif mode==1:
            filt=[net.get_name()]+models+['spike_signal']
             
            dd=load(sd, *filt)
            
            dd=compute(dd, models,  ['firing_rate'], 
                        **{'firing_rate':{'time_bin':1, 'set':0}} )  
            dd=cmp_mean_rates_intervals(dd, intervals, [1]*rep, rep, **{'set':0})
            pp(dd)
            save(sd, dd)
        elif mode==2:
            filt=[net.get_name()]+models+[
                                          'spike_signal',
                                          'firing_rate',
                                          'mean_rates_intervals'
                                          ]
            dd=load(sd, *filt)
        d=misc.dict_update(d, dd)
    d=cmp_mean_rate_diff(d, models, [['Net_0', 'Net_5'],
                                    ['Net_1', 'Net_6'],
                                    ['Net_2', 'Net_7'],
                                    ['Net_3', 'Net_8'],
                                    ['Net_4', 'Net_9']], x)
#     if numpy.all(numpy.array(from_disks) =2):                     
    figs=[]                      
    labels=['Unspec active {}%'.format(int(100/v)) for v in [5, 10, 20, 40, 80]]
    labels+=['Spec cluster size {}%'.format(int(100/v)) for v in [5, 10, 20, 40, 80]]
    print labels
    figs.append(show_fr(d, models, **{'labels':labels}))
    figs.append(show_mr_diff(d, models, **{'labels':['']}))
#         figs.append(show_mr(d, models, **{'labels':labels}))
     
    sd.save_figs(figs)
     
#     show_hr(d, models)

    if DISPLAY: pylab.show() 
    
    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

