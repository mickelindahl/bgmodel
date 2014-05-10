'''
Created on Mar 19, 2014

@author: lindahlm
'''

import single
import toolbox.plot_settings as pl

from single import (get_storages, optimize, run, run_XX, 
                    set_optimization_val, show)
from toolbox.network.manager import Builder_single_rest as Builder

import pprint
pp=pprint.pprint

NAMES=['$GPe_{+d}^{TA}$',
       '$GPe_{-d}^{TA}$']

def get_kwargs_builder():
    k=single.get_kwargs_builder()
    k.update({'inputs': ['GIp', 'GAp', 'EAp', 'STp'],
              'single_unit':'GA',
              'single_unit_input':'EAp'})

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
    return dn
    
def main():   
    
    k=get_kwargs_builder()
    
    dinfo, dn = get_setup()
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
    d.update(optimize('opt_rate', dn, [1]*1, ds, **{ 'x0':200.0}))   
    set_optimization_val(d['opt_rate']['Net_0'], dn['hist']) 
    d.update(run('hist', dn, [1]*2, ds, 'mean_rates', 
                 {'t_start':k['start_rec']}))                 


    _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16) 

    show(dstim, d, axs, NAMES)
    

if __name__ == "__main__":
    main() 

# from single import (beautify, build_general, creat_dic_specific, create_nets,  
#                     create_list_dop, do, set_optimization_val)
# import pylab
# import toolbox.plot_settings as pl
# from toolbox import misc
# 
# 
# import pprint
# pp=pprint.pprint
# 
# def build_cases(**kwargs):
#     su=kwargs.get('single_unit', 'GA')
#     
#     
#     inputs=kwargs.get('inputs',['GIp', 'GAp', 'EAp', 'STp']) 
#     d=build_general(**kwargs)
#     d=misc.dict_update(d, creat_dic_specific(kwargs, su, inputs))
#     
#     l = create_list_dop(su)
#     
# 
#     l[1]=misc.dict_update(l[1], {'node':{'STp':{'rate':15.0}}})
#     
#     names=['$GPe_{+d}^{TA}$',
#            '$GPe_{-d}^{TA}$',
#            ]
#     
#     nets = create_nets(l, d, names)
#     
#     return nets
# 
# def main():   
#     node='GA' 
#     IV=build_cases(**{'lesion':True, 'mm':True})
#     IF=build_cases(**{'lesion':True})
#     FF=build_cases(**{'lesion':False, 'size':50, 'threads':4})
#     opt=build_cases(**{'lesion':False, 'size':50,  'threads':4})
#     hist=build_cases(**{'lesion':False, 'size':200,  'threads':4})
# 
#     curr_IV=range(-200,300,100)
#     curr_IF=range(0,500,100)
#     rate_FF=range(100,1500,100)
#     _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)     
#     
#     do('plot_IV_curve', IV, 0, **{'ax':axs[0],'curr':curr_IV, 'node':node})
#     do('plot_IF_curve', IF, 0, **{'ax':axs[1],'curr':curr_IF, 'node':node})
#     do('plot_FF_curve', FF, 0, **{'ax':axs[2],'rate':rate_FF, 'node':node,
#                                      'input':'EAp', 'sim_time':5000.0})    
# 
#     d=do('optimize', [opt[0]], 0, **{'ax':axs[3], 'x0':200.0,'f':[node],
#                                    'x':['node.EAp.rate']})
#     set_optimization_val(d, [hist[0]], **{'x':['node.EAp.rate'], 'node':node})
#     do('plot_hist_rates', [hist[0]], 1, **{'ax':axs[3], 'node':node})
# 
# 
#     beautify(axs)
#     pylab.show()
#     
# if __name__ == "__main__":
#     main()  