'''
Created on Mar 19, 2014

@author: lindahlm
'''

from single import (beautify, build_general, creat_dic_specific, create_nets,  
                    create_list_dop, do, set_optimization_val)
import pylab
import toolbox.plot_settings as pl
from toolbox import misc


import pprint
pp=pprint.pprint

def build_cases(**kwargs):
    su=kwargs.get('single_unit', 'GA')
    
    
    inputs=kwargs.get('inputs',['GIp', 'GAp', 'EAp', 'STp']) 
    d=build_general(**kwargs)
    d=misc.dict_update(d, creat_dic_specific(kwargs, su, inputs))
    
    l = create_list_dop(su)
    

    l[1]=misc.dict_update(l[1], {'node':{'STp':{'rate':15.0}}})
    
    names=['$GPe_{+d}^{TA}$',
           '$GPe_{-d}^{TA}$',
           ]
    
    nets = create_nets(l, d, names)
    
    return nets

def main():   
    node='GA' 
    IV=build_cases(**{'lesion':True, 'mm':True})
    IF=build_cases(**{'lesion':True})
    FF=build_cases(**{'lesion':False})
    opt=build_cases(**{'lesion':False, 'sim_stop':20000.0, 'sim_time':20000.0})
    hist=build_cases(**{'lesion':False, 'size':200})

    curr_IV=range(-200,300,100)
    curr_IF=range(0,500,100)
    rate_FF=range(100,1500,100)
    _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)     
    
    do('plot_IV_curve', IV, 0, **{'ax':axs[0],'curr':curr_IV, 'node':node})
    do('plot_IF_curve', IF, 0, **{'ax':axs[1],'curr':curr_IF, 'node':node})
    do('plot_FF_curve', FF, 0, **{'ax':axs[2],'rate':rate_FF, 'node':node,
                                     'input':'EAp', 'sim_time':5000.0})    

    d=do('optimize', opt, 0, **{'ax':axs[3], 'x0':200.0,'f':[node],
                                   'x':['node.EAp.rate']})
    set_optimization_val(d, hist, **{'x':['node.EAp.rate'], 'node':node})
    do('plot_hist_rates', hist, 0, **{'ax':axs[3], 'node':node})


    beautify(axs)
    pylab.show()
    
if __name__ == "__main__":
    main()  