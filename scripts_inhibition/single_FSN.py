'''
Created on Jul 4, 2013

@author: lindahlm
'''

import single
import toolbox.plot_settings as pl

from single import (get_storages, optimize, run, run_XX, 
                    set_optimization_val, show)
from toolbox.network.manager import Builder_single_FS as Builder

import pprint
pp=pprint.pprint

NAMES=['$FS_{+d}^{l}$',
       '$FS_{-d}^{l}$',
       '$FS_{+d}^{h}$',
       '$FS_{-d}^{h}$']

def get_kwargs_builder():
    k=single.get_kwargs_builder()
    k.update({'inputs':['FSp', 'GAp', 'CFp'],
              'single_unit':'FS',
              'single_unit_input':'CFp'})

    return k

def get_kwargs_engine():
    return {'verbose':True}

def get_setup():
    
    k={'kwargs_builder':get_kwargs_builder(),
       'kwargs_engine':get_kwargs_engine(),
       'IV_time':5000.0,
       'IV_size':9.0,       
       'IF_time':5000.0,
       'IF_size':9.0,
       'FF_time':5000.0,
       'FF_size':50.0,
       'opt_rate_time':10000.0,
       'opt_rate_size':50.0,
       'hist_time':10000.0,
       'hist_size':50.0,
       }
    
    return single.get_setup(Builder,**k)

def modify(dn):
    dn['opt_rate']=[dn['opt_rate'][0]]
    dn['hist']=dn['hist'][0:2]
    return dn

def main():   
    
    dinfo, dn = get_setup()
    pp(dinfo)
    dn=modify(dn)
    ds = get_storages(__file__.split('/')[-1][0:-3], dn.keys(), dinfo)

    dstim={}
    dstim ['IV']=map(float, range(-300,300,100)) #curr
    dstim ['IF']=map(float, range(0,500,100)) #curr
    dstim ['FF']=map(float, range(0,1500,100)) #rate
  
    d={}
    d.update(run_XX('IV', dn, [1]*4, ds, dstim))
    d.update(run_XX('IF', dn, [1]*4, ds, dstim))
    d.update(run_XX('FF', dn, [1]*4, ds, dstim))   
    d.update(optimize('opt_rate', dn, [1]*1, ds, **{ 'x0':900.0}))   
    set_optimization_val(d['opt_rate']['Net_0'], dn['hist']) 
    d.update(run('hist', dn, [1]*2, ds, 'mean_rates'))                   


    _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16) 

    show(dstim, d, axs, NAMES)
    

if __name__ == "__main__":
    main()     
    
    
