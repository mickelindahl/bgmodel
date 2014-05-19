'''
Created on Jul 4, 2013

@author: lindahlm
'''
import os
import single
import toolbox.plot_settings as pl

from single import (get_storages, optimize, run, run_XX, 
                    set_optimization_val, show)
from toolbox import pylab
from toolbox.network.manager import Builder_single_FS as Builder

import pprint
pp=pprint.pprint

DISPLAY=os.environ.get('DISPLAY')
NAMES=['$FS_{+d}^{l}$',
       '$FS_{-d}^{l}$',
       '$FS_{+d}^{h}$',
       '$FS_{-d}^{h}$']

def get_kwargs_builder(**k_in):
    k=single.get_kwargs_builder()
    k.update({'inputs':['FSp', 'GAp', 'CFp'],
              'rand_nodes':{'C_m':k_in.get('rand_nodes'), 
                            'V_th':k_in.get('rand_nodes'), 
                            'V_m':k_in.get('rand_nodes')},
              'single_unit':'FS',
              'single_unit_input':'CFp'})

    return k

def get_kwargs_engine():
    return {'verbose':True}

def get_setup(**k_in):
    
    k={'kwargs_builder':get_kwargs_builder(**k_in),
       'kwargs_engine':get_kwargs_engine()}
    
    return single.get_setup(Builder,**k)

def modify(dn):
    dn['opt_rate']=[dn['opt_rate'][0]]
    dn['hist']=dn['hist'][0:2]
    return dn

def main(rand_nodes=False, 
         script_name= __file__.split('/')[-1][0:-3], 
         from_disk=1):   
      
    k=get_kwargs_builder()
    
    dinfo, dn = get_setup(**{'rand_nodes':rand_nodes})

    pp(dinfo)
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
    d.update(optimize('opt_rate', dn, [from_disk]*1, ds, **{ 'x0':900.0}))   
    set_optimization_val(d['opt_rate']['Net_0'], dn['hist']) 
    d.update(run('hist', dn, [from_disk]*2, ds, 'mean_rates', 
                 **{'t_start':k['start_rec']}))                   


    fig, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16) 

    show(dstim, d, axs, NAMES)
    ds['fig'].save_fig(fig)
    
    if not DISPLAY: pylab.show()  
    

if __name__ == "__main__":
    main()     
    
    
