'''
Created on Nov 12, 2014

@author: mikael
'''
from toolbox.data_to_disk import pickle_load
import pprint
import pylab
pp=pprint.pprint

path=('/home/mikael/results/papers/inhibition/single'
      +'/single_FSN/IF/Net_2-FS-IF_curve-3935409.pkl')
d=pickle_load(path)
pp(d)
d.plot(pylab.subplot(111))
pylab.show()