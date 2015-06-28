'''
Created on Jul 15, 2014

@author: mikael
'''
from core.data_to_disk import pickle_load
from core.parallelization import comm
import sys

path=sys.argv[1]+str(comm.rank())
# path='/home/mikael/git/bgmodel/core_old/misc_folder/test_subprocess/00'
print path


fun, args, kwargs=pickle_load(path)

fun(*args, **kwargs)
