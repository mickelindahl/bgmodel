'''
Created on Mar 19, 2014

@author: lindahlm
'''

import os
import single
import core.plot_settings as pl

from single import (get_storages, optimize, run, 
                    set_optimization_val, show_opt_hist)
from core import pylab
from core.network.manager import Builder_single_GA_GI as Builder

import pprint
pp=pprint.pprint

DISPLAY=os.environ.get('DISPLAY')

def get_kwargs_builder(**k_in):
    k=single.get_kwargs_builder()
    k.update({'inputs': ['GIp', 'GAp', 'EIp', 'M2p', 'STp'],
              'rand_nodes':{'C_m':k_in.get('rand_nodes'), 
                            'V_th':k_in.get('rand_nodes'), 
                            'V_m':k_in.get('rand_nodes')},
              'single_unit':'GI',
              'single_unit_input':'EIp',
              'start_rec':1000.0,
              'TA_rates':[7.5, 10.0, 15.0, 20.0, 25.0]})

    return k

def get_kwargs_engine():
    return {'verbose':True}

def get_setup(**k_in):
    
    k={'kwargs_builder':get_kwargs_builder(**k_in),
       'kwargs_engine':get_kwargs_engine()}
    
    dinfo, d={}, {}
    dinfo['opt_rate'],d['opt_rate']=single.get_setup_opt_rate(Builder, k)
    dinfo['hist'],d['hist']=single.get_setup_hist(Builder, k) 
    dinfo['fig'],d['fig']='figure',{}
    return dinfo, d
 
def main(rand_nodes=False, 
         script_name= __file__.split('/')[-1][0:-3], 
         from_disk=1):   
      
    k=get_kwargs_builder()
    
    dinfo, dn = get_setup(**{'rand_nodes':rand_nodes})
    

    dinfo, dn = get_setup()
    ds = get_storages(script_name, dn.keys(), dinfo)
  
    d={}
    d.update(optimize('opt_rate', dn, [from_disk]*5, ds, **{ 'x0':200.0}))   
    
    for key in sorted(dn['hist'].keys()): 
        net=dn['hist'][key]
        set_optimization_val(d['opt_rate'][net.get_name()], [net]) 
    d.update(run('hist', dn, [from_disk]*5, ds, 'mean_rates',
                           **{'t_start':k['start_rec']}))                   


    fig, axs=pl.get_figure(n_rows=1, n_cols=1, w=1000.0, h=800.0, fontsize=16) 

    show_opt_hist(d, axs, '$GPe_{+d}^{TI}$')
    ds['fig'].save_fig(fig)
    
    if DISPLAY: pylab.show()   

if __name__ == "__main__":
    main() 

