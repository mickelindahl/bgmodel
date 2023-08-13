'''
Created on Mar 19, 2014

@author: lindahlm
'''

import os
import single
import python.core.plot_settings as pl

from single import (get_storages, optimize, run, run_XX,
                    set_optimization_val, show)
from python.core import pylab
from python.core.network.manager import Builder_single_rest_dop as Builder

import pprint

pp = pprint.pprint

NAMES = ['$GPe_{+d}^{TA}$',
         #        '$GPe_{-d}^{TA}$'
         ]


def get_kwargs_builder(**k_in):
    k = single.get_kwargs_builder()
    k.update({'inputs': ['GIp', 'GAp', 'EAp', 'STp'],
              'rand_nodes': {'C_m': k_in.get('rand_nodes'),
                             'V_th': k_in.get('rand_nodes'),
                             'V_m': k_in.get('rand_nodes')},
              'single_unit': 'GA',
              'single_unit_input': 'EAp'})

    return k


def get_kwargs_engine():
    return {'verbose': True}


def get_setup(**k_in):
    k = {'kwargs_builder': get_kwargs_builder(**k_in),
         'kwargs_engine': get_kwargs_engine()}

    return single.get_setup(Builder, **k)


def modify(dn):
    dn['opt_rate'] = [dn['opt_rate'][0]]
    return dn


def main(rand_nodes=False,
         script_name=__file__.split('/')[-1][0:-3],
         from_disk=0):
    k = get_kwargs_builder()

    dinfo, dn = get_setup(**{'rand_nodes': rand_nodes})
    dinfo, dn = get_setup()
    #     dn=modify(dn)
    ds = get_storages(script_name, dn.keys(), dinfo)

    dstim = {}
    #     dstim ['IV']=map(float, range(-300,300,100)) #curr
    dstim['IF'] = list(map(float, range(0, 150, 50)))  # curr
    #     dstim ['FF']=map(float, range(0,1500,100)) #rate

    d = {}
    #     d.update(run_XX('IV', dn, [from_disk]*4, ds, dstim))
    d.update(run_XX('IF', dn, [from_disk] * 4, ds, dstim))
    #     d.update(run_XX('FF', dn, [from_disk]*4, ds, dstim))
    #     d.update(optimize('opt_rate', dn, [from_disk]*1, ds, **{ 'x0':200.0}))
    #     set_optimization_val(d['opt_rate']['Net_0'], dn['hist'])
    #     d.update(run('hist', dn, [from_disk]*2, ds, 'mean_rates',
    #                  **{'t_start':k['start_rec']}))

    fig, axs = pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)

    show(dstim, d, axs, NAMES, **{'models': ['IF']})
    #     ds['fig'].save_fig(fig)
    ds['fig'].save_figs([fig], format='png', dpi=200)

    if not os.environ.get('DISPLAY'): pylab.show()


if __name__ == "__main__":
    main()

# C_m 40
# {'rates': array([  1.00000000e-03,   3.20391194e+01,   5.37567401e+01]),
#  'slope': 0.53755740114774175,
#  'x': [0.0, 50.0, 100.0]}

# C_m 30
# {'rates': array([  2.15895459,  35.54907551,  59.69797634]),
#  'slope': 0.57539021756374609,
#  'x': [0.0, 50.0, 100.0]}

# C_m 60  0.85
# {'rates': array([  1.00000000e-03,   2.66800297e+01,   4.59894666e+01]),
#  'slope': 0.45988466641317932,
#  'x': [0.0, 50.0, 100.0]}

# C_m 80
# {'rates': array([  1.00000000e-03,   2.34767748e+01,   4.14972493e+01]),
#  'slope': 0.41496249293195553,
#  'x': [0.0, 50.0, 100.0]}

# b 140 0.67
# {'rates': array([  1.00000000e-03,   2.44274200e+01,   3.61669561e+01]),
#  'slope': 0.36165956138068439,
#  'x': [0.0, 50.0, 100.0]}

# Delta_T 3.4 0.85
# {'rates': array([  1.00000000e-03,   2.73841570e+01,   4.64733077e+01]),
#  'slope': 0.46472307680952407,
#  'x': [0.0, 50.0, 100.0]}
