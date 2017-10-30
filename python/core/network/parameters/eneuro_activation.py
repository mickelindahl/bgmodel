# Create by Mikael Lindahl on 4/12/17.

from core import misc
from core.network.default_params import Par_base, \
    Par_base_mixin
import copy


class EneuroActivationParBase(object):
    def _get_par_constant(self):
        dic_other = self.other.get_par_constant()

        # self._dic_con['node']['M1']['n']
        factor=0.975

        dic = {'node': {

            'C1': {'rate':dic_other['node']['C1']['rate'] * factor},
            'C2': {'rate':dic_other['node']['C2']['rate'] * factor},
            'CF': {'rate':dic_other['node']['CF']['rate'] * factor}

        }}

        dic = misc.dict_update(dic_other, dic)
        return dic


class EneuroActivationPar(Par_base, EneuroActivationParBase, Par_base_mixin):
    pass
