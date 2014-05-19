'''
Created on Jun 27, 2013

@author: lindahlm
'''
import os
import pylab

from scripts_inhibition.network import show_fr, show_hr
from toolbox import misc
from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.manager import compute, run, save, load
from manager import Builder_network as Builder
import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')

def get_kwargs_builder():
    return {'print_time':False, 
            'threads':12, 
            'save_conn':{'overwrite':False},
            'sim_time':5000.0, 
            'sim_stop':5000.0, 
            'size':3000.0, 
            'start_rec':0.0, 
            'sub_sampling':1}

def get_kwargs_engine():
    return {'verbose':True}

def get_networks():
    return manager.get_networks(Builder, 
                                get_kwargs_builder(), 
                                get_kwargs_engine())

def main():
    k=get_kwargs_builder()
    
    from os.path import expanduser
    home = expanduser("~")

    attr=[ 'firing_rate', 
           'mean_rates', 
           'spike_statistic']  
    
    kwargs_dic={'mean_rates': {'t_start':k['start_rec']},
                'spike_statistic': {'t_start':k['start_rec']},}
    file_name=(home+ '/results/papers/inhibition/network/jyotika'
               +__file__.split('/')[-1][0:-3])
    
    models=['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN']
    
    info, nets, _ = get_networks()

    sd=Storage_dic.load(file_name)
    sd.add_info(info)
    sd.garbage_collect()
    
    d={}
    for net, from_disk in zip(nets, [1]*2):
        if not from_disk:
            dd = run(net)  
            dd = compute(dd, models,  attr, **kwargs_dic)      
            save(sd, dd)
        elif from_disk:
            filt=[net.get_name()]+models+attr
            dd=load(sd, *filt)
        d=misc.dict_update(d, dd)
                                         
    figs=[]

    figs.append(show_fr(d, models, **{'labels':['Normal', 'Double STN-GPe TI']}))
    figs.append(show_hr(d, models, **{'labels':['Normal', 'Double STN-GPe TI']}))
    
    sd.save_figs(figs)
    
    if DISPLAY: pylab.show()     

    pylab.show()
 
    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

   


    

    
