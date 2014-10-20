'''
Created on Oct 18, 2014

@author: mikael
'''

import sys
from toolbox.data_to_disk import pickle_load, pickle_save
path_in,path_out, =sys.argv[1:]

net=pickle_load(path_in)
d=net.simulation_loop()
pickle_save(d, path_out)
