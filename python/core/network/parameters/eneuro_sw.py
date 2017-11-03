# Create by Mikael Lindahl on 4/12/17.

from core import misc
from core.network.default_params import Par_base, \
    Par_base_mixin
import copy
import pprint
pp=pprint.pprint

class EneuroSwParBase(object):
    def _get_par_constant(self):
        dic_other = self.other.get_par_constant()

        # self._dic_con['node']['M1']['n']


        dic = {
            'simu': {
                'print_time': False,
                'sim_stop': 20000.0,
                'sim_time': 20000.0,
                'local_num_threads': 4,
                'do_reset':True

            },
            'netw': {
                'input': {},
                'size': 5000.
            },
            'nest': {},
            'node': {
                'CS':{'rate':170.},
                'EF':{'rate':620.},
                'EI':{'rate':620.},
                'EA': {'rate': 200.}
            }}

        d = {'type': 'oscillation2',
             'params': {'p_amplitude_upp': 0.09,
                        'p_amplitude_down': -0.09,
                        'p_amplitude0': .8,
                        'freq': 1.,
                        'freq_min': None,
                        'freq_max': None,
                        'period': 'constant'}}

        for key in ['C1', 'C2', 'CF', 'CS']:
            dic['netw']['input'][key] = copy.deepcopy(d)

            if (key == 'CS'):
                dic['netw']['input'][key]['params']['p_amplitude0'] = 1.

            new_name = key + 'd'
            dic['nest'][new_name]  = {'type_id': 'poisson_generator_dynamic',
                                     'rates': [0.],
                                     'timings': [1.]}

            if not dic['node'].get(key):
                dic['node'][key]={}

            dic['node'][key].update({'model': new_name})

        dic = misc.dict_update(dic_other, dic)
        return dic


class EneuroSwPar(Par_base, EneuroSwParBase, Par_base_mixin):
    pass
