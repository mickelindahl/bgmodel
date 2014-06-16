'''
Created on Mar 19, 2014

@author: lindahlm

'''

import os
import single
import toolbox.plot_settings as pl

from single import (get_storages, optimize, run, 
                    set_optimization_val, show_opt_hist)
from toolbox import pylab, misc
from toolbox.network.manager import Builder_single_M2_weights as Builder

import pprint
pp=pprint.pprint

DISPLAY=os.environ.get('DISPLAY')
NAMES=['$M1_{+d}^{l}$',
       '$M1_{-d}^{l}$',
       '$M1_{+d}^{h}$',
       '$M1_{-d}^{h}$']



def get_kwargs_builder(**k_in):
    k=single.get_kwargs_builder()
    k.update({'inputs':['FSp', 'GAp', 'C2p', 'M1p', 'M2p'],
              'rand_nodes':{'C_m':k_in.get('rand_nodes'), 
                            'V_th':k_in.get('rand_nodes'), 
                            'V_m':k_in.get('rand_nodes')},
              'single_unit':'M2',
              'single_unit_input':'C2p',
              'start_rec':1000.0,
              'conductance_scale':[0.2,0.25,0.3]})

    return k

def get_kwargs_engine():
    return {'verbose':False}

def get_setup(**k_in):
    
    k={'kwargs_builder':get_kwargs_builder(**k_in),
       'kwargs_engine':get_kwargs_engine(),
         'threads':16,}
    
    dinfo, d={}, {}
    dinfo['opt_rate'],d['opt_rate']=single.get_setup_opt_rate(Builder, k)
    dinfo['hist'],d['hist']=single.get_setup_hist(Builder, k) 
    dinfo['fig'],d['fig']='figure',{}
    return dinfo, d

def main(rand_nodes=False, 
         script_name= __file__.split('/')[-1][0:-3], 
         from_disk=0):   
    
    k=get_kwargs_builder()    
    dinfo, dn = get_setup(**{'rand_nodes':rand_nodes})
    pp(dn['opt_rate']['Net_0'].par['simu'])
    ds = get_storages(script_name, dn.keys(), dinfo)
  
    d={}
    d.update(optimize('opt_rate', dn, [from_disk]*5, ds, **{ 'x0':800.0}))   
    
    for key in sorted(dn['hist'].keys()): 
        net=dn['hist'][key]
        set_optimization_val(d['opt_rate'][net.get_name()], [net]) 
    d.update(run('hist', dn, [from_disk]*5, ds, 'mean_rates',
                           **{'t_start':k['start_rec']}))                   


    fig, axs=pl.get_figure(n_rows=1, n_cols=1, w=1000.0, h=800.0, fontsize=16) 

    show_opt_hist(d, axs, '$MSN_{+d}^{l}$')
    ds['fig'].save_fig(fig)
    
    if DISPLAY: pylab.show()  

with misc.Std_to_files(not DISPLAY,__file__.split('/')[-1][0:-3] ):

    if __name__ == "__main__":
        main()     
      