'''
Created on May 14, 2014

@author: mikael
'''
import numpy
import os
from network import show_fr, show_mr, cmp_mean_rates_intervals
from toolbox import misc, pylab
from toolbox.network import manager
from toolbox.network.manager import (add_perturbations, compute, 
                                     run, save, load, get_storage)

from toolbox.network.manager import Builder_MSN_cluster_compete as Builder

import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')

def get_kwargs_builder():
    return {'print_time':True, 
            'threads':20, 
            'save_conn':{'overwrite':False},
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
    
    return info, nets, intervals, rep

    
def main(from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3]):
    
    
    from os.path import expanduser
    home = expanduser("~")

    
    file_name=(home+ '/results/papers/inhibition/network/'+script_name)
    
    #models=['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN']
    
    models=['M1', 'M2', 'FS']
    
    info, nets, intervals, rep = get_networks()
    add_perturbations(perturbation_list, nets)
    
    pp(nets[0].par['node']['C1'])
    pp(nets[5].par['node']['C1'])
    
    sd = get_storage(file_name, info)
    sd.garbage_collect()
    
    d={}
    from_disks=[2]*10
    for net, mode in zip(nets, from_disks):
        if mode==0:
            dd = run(net)    
            save(sd, dd)
            print sd
#         elif mode==1:
            filt=[net.get_name()]+models+['spike_signal']
            
            dd=load(sd, *filt)
            
            dd=compute(dd, models,  ['firing_rate'], 
                        **{'firing_rate':{'time_bin':1}} )  
#             dd=cmp_mean_rates_intervals(dd, intervals, amplitudes, rep)
#             pp(dd)
            save(sd, dd)
        elif mode==2:
            filt=[net.get_name()]+models+['spike_signal',
                                          'firing_rate'
                                         ]
            dd=load(sd, *filt)
        d=misc.dict_update(d, dd)
    
    if numpy.all(numpy.array(from_disks)==2):                     
        figs=[]                      
        labels=['Unspec active {}%'.format(int(100/v)) for v in [5, 10, 20, 40, 80]]
        labels+=['Spec cluster size {}%'.format(int(100/v)) for v in [5, 10, 20, 40, 80]]
        print labels
        figs.append(show_fr(d, models, **{'labels':labels}))
#         figs.append(show_mr(d, models, **{'labels':labels}))
        
        sd.save_figs(figs)
        
#     show_hr(d, models)

    if DISPLAY: pylab.show() 
    
    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

