# Create by Mikael Lindahl on 4/10/17.

from core import misc
from core.network.parameters.eneuro import EneuroPar
from core.network.parameters.eneuro_activation2 import EneuroActivation2Par
from core.network.parameters.eneuro_activation_beta import EneuroActivationBetaPar
from core.network.parameters.eneuro_sw import EneuroSwPar

from core.network.default_params import Inhibition, \
    print_dic_comparison
from scripts_inhibition.eNeuro_fig_01_and_02_pert import  get
import unittest
import pickle

class TestEneuro( unittest.TestCase ):
    def test_show_par(self):
        dic = misc.dict_reduce(EneuroPar().dic, {})

        for key in sorted(list(dic.keys())):
            print key, dic[key]

    def test_diff_eNeuro_activation_control(self):
        perturbations=get()[0]
        # d1 = Inhibition(perturbations=perturbations).dic
        d1 = pickle.load(open('data/activation-control.pkl'))
        d2 = EneuroActivation2Par(other=EneuroPar()).dic

        # print_dic_comparison(d1, d2, flag='values', names=['pert added:', '   no pert:'])

        EneuroPar().dic_print_change('', d1, d2)

    def test_diff_eNeuro_activation_dp(self):
        perturbations = get()[0]
        # d1 = Inhibition(perturbations=perturbations).dic
        d1 = pickle.load(open('data/activation-dopamine-depleted.pkl'))
        # d2 = EneuroActivationBetaPar(other=EneuroPar()).dic
        par = EneuroActivationBetaPar(other=EneuroPar())
        par.set({'netw': {'tata_dop': 0.0}})

        d2 = par.dic

        # print_dic_comparison(d1, d2, flag='values', names=['pert added:', '   no pert:'])

        EneuroPar().dic_print_change('', d1, d2)

    def test_diff_eNeuro_sw_control(self):
        perturbations=get()[0]
        # d1 = Inhibition(perturbations=perturbations).dic
        d1 = pickle.load(open('data/slow-wave-control.pkl'))
        d2 = EneuroSwPar(other=EneuroPar()).dic

        # print_dic_comparison(d1, d2, flag='values', names=['pert added:', '   no pert:'])

        EneuroPar().dic_print_change('', d1, d2)

    def test_diff_eNeuro_sw_dp(self):
        perturbations=get()[0]
        # d1 = Inhibition(perturbations=perturbations).dic
        d1 = pickle.load(open('data/slow-wave-dopamine-depleted.pkl'))
        par = EneuroSwPar(other=EneuroPar())
        par.set({'netw': {'tata_dop': 0.0}})

        d2 = par.dic

        # print_dic_comparison(d1, d2, flag='values', names=['pert added:', '   no pert:'])

        EneuroPar().dic_print_change('', d1, d2)


class Test( unittest.TestCase ):
    def test_example(self):
        print 'test'

if __name__ == '__main__':


    d = { TestEneuro:[
        # 'test_show_par',
        # 'test_diff_eNeuro_activation_control'
        'test_diff_eNeuro_activation_dp',
        # 'test_diff_eNeuro_sw_control',
        # 'test_diff_eNeuro_sw_dp'
    ],
    Test:[
        # 'test_example'
    ]}

    suite = unittest.TestSuite()
    for test_class, val in d.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=1).run(suite)
