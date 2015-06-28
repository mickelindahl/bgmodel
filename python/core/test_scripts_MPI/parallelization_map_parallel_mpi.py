'''
Created on Sep 19, 2014

@author: mikael
'''

import numpy #just to not get segmentation fault
import sys

from toolbox.data_to_disk import pickle_save, pickle_load, mkdir
from toolbox.parallelization import (comm, Barrier, map_parallel,
                                     mockup_fun, mockup_fun_large_return_2,
                                     mockup_fun_large_return_1)
from toolbox import misc
np_local=2

fileName, fileOut, data_path =sys.argv[1:]

with Barrier():
    if comm.rank()==0:   
        out, mockup=pickle_load(fileName, all_mpi=True) 
    else:
        out=None
        mockup=None


out=comm.bcast(out, root=0)
mockup=comm.bcast(mockup, root=0)

print comm.rank()

with misc.Stopwatch('mpi'):
    a=map_parallel(mockup, out, out, **{'local_num_threads':np_local})  

pickle_save(a, fileOut)
