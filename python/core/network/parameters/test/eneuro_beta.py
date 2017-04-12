# Create by Mikael Lindahl on 4/10/17.

from core.network.parameters.eneuro import EneuroPar
from core.network.parameters.eneuro_beta import EneuroBetaPar
from core.network.default_params import print_dic_comparison
import unittest


class TestEneuroBeta( unittest.TestCase ):

    def test_usage(self):
        d1 = EneuroPar().dic
        d2 = EneuroBetaPar(other=EneuroPar()).dic

        # print_dic_comparison(d1, d2, flag='values', names=['pert added:', '   no pert:'])

        EneuroPar().dic_print_change('',d1,d2, flag='2 rows')

if __name__ == '__main__':


    d = { TestEneuroBeta:[
        'test_usage'
    ]}

    suite = unittest.TestSuite()
    for test_class, val in d.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=1).run(suite)