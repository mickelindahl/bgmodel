# Create by Mikael Lindahl on 4/10/17.

from core import misc
from core.network.parameters.eneuro import EneuroPar
from core.network.default_params import Inhibition, \
    print_dic_comparison
from scripts_inhibition.eNeuro_fig_01_and_02_pert import  get
import unittest

class TestEneuro( unittest.TestCase ):
    def test_show_par(self):
        dic=misc.dict_reduce(EneuroPar().dic, {})

        for key in sorted(list(dic.keys())):
            print key, dic[key]

    def test_diff_eNeuro_perturbations_sw(self):
        perturbations=get()[0]
        d1 = Inhibition(perturbations=perturbations).dic
        d2 = EneuroPar().dic

        print_dic_comparison(d1, d2, flag='values', names=['pert added:', '   no pert:'])

        EneuroPar().dic_print_change('', d1, d2)

if __name__ == '__main__':


    d = { TestEneuro:[
        # 'test_show_par',
        # 'test_diff_eNeuro_perturbations_sw'
    ]}

    suite = unittest.TestSuite()
    for test_class, val in d.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=1).run(suite)