'''
Created on Mar 19, 2014

@author: lindahlm
'''

from single import (creat_dic_specific, create_nets, build_general, 
                        do, beautify, create_list_dop_high_low,
                        set_optimization_val)
import pylab
import toolbox.plot_settings as pl
from toolbox import misc


def build_cases(**kwargs):
    su=kwargs.get('single_unit', 'M2')
    l = create_list_dop_high_low(su)
    
    inputs=kwargs.get('inputs',['FSp', 'GAp', 'C2p', 'M1p', 'M2p']) 
    d=build_general(**kwargs)
    d=misc.dict_update(d, creat_dic_specific(kwargs, su, inputs))
    
    names=['$D2_{+d}^{l}$',
           '$D2_{-d}^{l}$',
           '$D2_{+d}^{h}$',
           '$D2_{-d}^{h}$']
    
    nets = create_nets(l, d, names)
    
    return nets

def main():   
    node='M2' 
    IV=build_cases(**{'lesion':True, 'mm':True})
    IF=build_cases(**{'lesion':True})
    FF=build_cases(**{'lesion':False, 'size':50, 'threads':4})
    opt=build_cases(**{'lesion':False, 'size':50,  'threads':4})
    hist=build_cases(**{'lesion':False, 'size':200,  'threads':4})

        
    curr_IV=range(-200,300,100)
    curr_IF=range(0,500,100)
    rate_FF=range(100,1500,100)
    _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)     
    
    do('plot_IV_curve', IV, 0, **{'ax':axs[0],'curr':curr_IV, 'node':node})
    do('plot_IF_curve', IF, 0, **{'ax':axs[1],'curr':curr_IF, 'node':node})
    do('plot_FF_curve', FF, 0, **{'ax':axs[2],'rate':rate_FF, 'node':node,
                                     'input':'C2p'})    
    d=do('optimize', opt, 0, **{'ax':axs[3], 'x0':700.0,'f':[node],
                                   'x':['node.C2p.rate']})

    set_optimization_val(d, hist, **{'x':['node.C2p.rate'], 'node':node})
    do('plot_hist_rates', hist, 0, **{'ax':axs[3], 'node':node})


    beautify(axs)
    pylab.show()
    
if __name__ == "__main__":
    main()  