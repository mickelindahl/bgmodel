'''
Created on Sep 29, 2014

@author: mikael
'''



import os
import sys
currdir=os.getcwd()
basedir='/'.join(currdir.split('/')[:-1])

from core import data_to_disk
from core.misc import Stopwatch
from core.parallel_excecution import Mockup_class
from scripts_inhibition import base_Go_NoGo_compete

import pprint
pp=pprint.pprint
# pp(sys.modules)
fileName=sys.argv[1]
print fileName
obj, script=data_to_disk.pickle_load(fileName)

# print 'Running '+str(obj)+' as ' + script
with Stopwatch('Running '+str(obj)+' as ' + script):
    obj.do()
