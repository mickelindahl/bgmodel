'''
Created on Sep 29, 2014

@author: mikael
'''

import sys
from toolbox import data_to_disk
from toolbox.misc import Stopwatch
fileName=sys.argv[1]
obj, script=data_to_disk.pickle_load(fileName)
# print 'Running '+str(obj)+' as ' + script
with Stopwatch('Running '+str(obj)+' as ' + script):
    obj.do()
