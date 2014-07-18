'''
Created on Jul 15, 2014

@author: mikael
'''
from toolbox.data_to_disk import pickle_load
from toolbox.parallelization import comm
import sys

path=sys.argv[1]+str(comm.rank())
# path='/home/mikael/git/bgmodel/core/misc_folder/test_subprocess/00'
print path


fun, args, kwargs=pickle_load(path)

fun(*args, **kwargs)
