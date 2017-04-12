# Create by Mikael Lindahl on 4/12/17.

from core import misc
from core.network.default_params import Par_base, \
    Par_base_mixin


class EneuroBetaParBase(object):
    def _get_par_constant(self):
        dic_other = self.other.get_par_constant()

        # self._dic_con['node']['M1']['n']


        dic = {'netw': {'input': {}},
               'nest': {},
               'node': {}}

        d = {'type': 'oscillation2',
             'params': {'p_amplitude_upp': 0.1,
                        'p_amplitude_down': -0.1,
                        'p_amplitude0': 1.0,
                        'freq': 20.,
                        'freq_min': None,
                        'freq_max': None,
                        'period': 'constant'}}

        for key in ['C1', 'C2', 'CF', 'CS']:
            dic['netw']['input'][key] = d
            new_name = key + 'd'
            dic['nest'][new_name] = {'type_id': 'poisson_generator_dynamic',
                                     'rates': [0.],
                                     'timings': [1.]}
            dic['node'][key] = {'model': new_name}

        dic = misc.dict_update(dic_other, dic)
        return dic


class EneuroBetaPar(Par_base, EneuroBetaParBase, Par_base_mixin):
    pass