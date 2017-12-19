# Create by Mikael Lindahl on 4/12/17.

from core import misc
from core.network.default_params import Par_base, \
    Par_base_mixin


class EneuroRampParBase(object):
    def _get_par_constant(self):
        dic_other = self.other.get_par_constant()

        # self._dic_con['node']['M1']['n']

        dic = {'netw': {'input': {}},
               'nest': {},
               'node': {}}

        d = {'type': 'ramp',
             'params': {'step': 50.,
                        'slope': .2,
                        'ramp_dur': 10000.,
                        'ramp_start': 2000.}}

        for key in ['MC1', 'MC2', 'MCF']:
            dic['netw']['input'][key] = d
            new_name = key + 'd'
            dic['nest'][new_name] = {'type_id': 'poisson_generator_dynamic',
                                     'rates': [0.],
                                     'timings': [1.]}
            dic['node'][key] = {'model': new_name}

        dic = misc.dict_update(dic_other, dic)
        return dic


class EneuroRampPar(Par_base, EneuroRampParBase, Par_base_mixin):
    pass
