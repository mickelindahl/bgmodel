'''
Created on Mar 19, 2014

@author: lindahlm
'''

import os
import single
import core.plot_settings as pl

from single import (get_storages, optimize, run, run_XX, 
                    set_optimization_val, show)
from core import pylab
from core.network.manager import Builder_single_rest as Builder

import pprint
pp=pprint.pprint


NAMES=['$GPe_{+d}^{TA}$',
       '$GPe_{-d}^{TA}$']

def get_kwargs_builder(**k_in):
    k=single.get_kwargs_builder()
    k.update({'inputs': ['GIp', 'GAp', 'EAp', 'STp'],
              'rand_nodes':{'C_m':k_in.get('rand_nodes'), 
                            'V_th':k_in.get('rand_nodes'), 
                            'V_m':k_in.get('rand_nodes')},
              'single_unit':'GA',
              'single_unit_input':'EAp'})

    return k

def get_kwargs_engine():
    return {'verbose':True}

def get_setup(**k_in):
    
    k={'kwargs_builder':get_kwargs_builder(**k_in),
       'kwargs_engine':get_kwargs_engine()}
    
    return single.get_setup(Builder,**k)

def modify(dn):
    dn['opt_rate']=[dn['opt_rate'][0]]
    return dn
       
def main(rand_nodes=False, 
         script_name= __file__.split('/')[-1][0:-3], 
         from_disk=0):   
      
    k=get_kwargs_builder()
    
    dinfo, dn = get_setup(**{'rand_nodes':rand_nodes})
    dinfo, dn = get_setup()
    dn=modify(dn)
    ds = get_storages(script_name, dn.keys(), dinfo)

    dstim={}
    dstim ['IV']=map(float, range(-300,300,100)) #curr
    dstim ['IF']=map(float, range(0,500,100)) #curr
    dstim ['FF']=map(float, range(0,1500,100)) #rate
  
    d={}
    d.update(run_XX('IV', dn, [from_disk]*4, ds, dstim))
    d.update(run_XX('IF', dn, [from_disk]*4, ds, dstim))
    d.update(run_XX('FF', dn, [from_disk]*4, ds, dstim))   
    d.update(optimize('opt_rate', dn, [from_disk]*1, ds, **{ 'x0':200.0}))   
    set_optimization_val(d['opt_rate']['Net_0'], dn['hist']) 
    d.update(run('hist', dn, [from_disk]*2, ds, 'mean_rates', 
                 **{'t_start':k['start_rec']}))                 


    fig, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16) 
    show(dstim, d, axs, NAMES)
    
    ds['fig'].save_fig(fig)
    
    if not os.environ.get('DISPLAY'): pylab.show()

if __name__ == "__main__":
    main() 

