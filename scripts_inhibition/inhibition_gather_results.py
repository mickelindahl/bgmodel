'''
Created on Jun 30, 2014

@author: mikael
'''
import numpy
import pylab
import os

from scipy.interpolate import interp1d
from toolbox import misc
from toolbox.data_to_disk import Storage_dic


import pprint
pp=pprint.pprint




def get_amlitudes(freqs, data):
    
    out={}
    for key, val in data.items():
        out[key]=[]
        for freq in freqs:
            if freq<=val.x[-1] and freq>=val.x[0]:
                out[key].append(val(freq)-1)
            else:
                out[key].append(numpy.NAN)
                print val.x[-1]
    return out

def gather(path): 
    fs=os.listdir(path)
    d={}
    for name in fs:
        if name[-4:]!='.pkl':
            continue
        file_name=path+name[:-4]
        sd = Storage_dic.load(file_name)
        
        dd=sd.load_dic(*['Net_0', 'M1','mean_rate_slices'])
        d = misc.dict_update(d, {name[:-4]:dd['Net_0']})
    
#     ax=pylab.subplot(211)
    if d=={}:
        i=0
        for name in fs:
            if name[0:6]!='script':
                continue
#             i+=1
            ax=pylab.subplot(5,6,i)
            file_name=path+name+'/'+'Net_0'
            sd = Storage_dic.load(file_name)
            dd=sd.load_dic(*['Net_0', 'M1','mean_rate_slices', 'firing_rate'])
            print file_name
            
#             ax.plot(dd['Net_0']['M1']['firing_rate'].y)
            ax.text(0.1,0.1,name, transform=ax.transAxes, fontsize=7)
            d = misc.dict_update(d, {name:dd['Net_0']})
            
            
#     pylab.show()
    return d



def interpolate(d):
    dinter={}

    
    for keys, val in misc.dict_iter(d):
        key='_'.join(keys[0].split('_')[2:])
        dinter[key]=interp1d(val.y, val.x)
    return dinter

def process(path, freqs):
    d=gather(path)
    dinter=interpolate(d)
    damp=get_amlitudes(freqs, dinter)
    return damp
    
    


import unittest
class TestMethods(unittest.TestCase):     
    def setUp(self):
        self.path=('/home/mikael/results/papers/inhibition'+
                   '/network/simulate_inhibition_ZZZ/')
        self.freqs=[0,5,1.0]

    def test_gather(self):
        d=gather(self.path)
        self.assertTrue({}!=d)


    def test_interpolate(self):
        d=gather(self.path)
        dinter=interpolate(d)
        self.assertTrue({}!=dinter)
        
    def test_get_amplitudes(self):
        d=gather(self.path)
        dinter=interpolate(d)
        damp=get_amlitudes(self.freqs, dinter)
        self.assertTrue({}!=damp)

    def test_process(self):
        damp=process(self.path, self.freqs)
        self.assertTrue({}!=damp)
        
        
 

if __name__ == '__main__':
    test_classes_to_run=[
                         TestMethods
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)    



    
# pp(damp)