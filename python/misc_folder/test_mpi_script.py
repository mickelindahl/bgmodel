'''
Created on Sep 23, 2014

@author: mikael
'''
import sys
from core import data_to_disk
from core.misc import Stopwatch 
fileName=sys.argv[0]

obj, script=data_to_disk.pickle_load(fileName)

with Stopwatch('Running '+str(obj)+' as ' + script):
    obj.do()


