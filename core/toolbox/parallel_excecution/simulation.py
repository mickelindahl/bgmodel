'''
Created on Sep 29, 2014

@author: mikael
'''

import sys
from toolbox import data_to_disk
from toolbox.misc import Stopwatch
from toolbox.parallel_excecution import Mockup_class
from scripts_inhibition import Go_NoGo_compete

import pprint
pp=pprint.pprint
# pp(sys.modules)
fileName=sys.argv[1]
print fileName
obj, script=data_to_disk.pickle_load(fileName)

# print 'Running '+str(obj)+' as ' + script
with Stopwatch('Running '+str(obj)+' as ' + script):
    obj.do()
