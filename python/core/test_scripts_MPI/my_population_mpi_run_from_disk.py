'''
Created on Sep 19, 2014

@author: mikael
'''

import numpy #just to not get segmentation fault
from toolbox.data_to_disk import pickle_save, pickle_load, mkdir
# import cPickle as pickle
import sys
from toolbox.my_population import sim_group
from toolbox import misc
from toolbox import my_nest
# Necessary for pickle se
# http://stefaanlippens.net/pickleproblem


from toolbox.parallelization import comm, Barrier
fileName, fileOut, data_path =sys.argv[1:]

with Barrier():
    if comm.rank()==0:   
        out=pickle_load(fileName, all_mpi=True) 
    else:
        out=None


out=comm.bcast(out, root=0)
sim_time, args, kwargs=out
d={'sd':{'active':True,
           'params':{'record_to':'file'}}}

kwargs=misc.dict_update(kwargs, d)

mkdir(data_path+'nest/')
my_nest.ResetKernel(display=False, data_path=data_path+'nest/')
my_nest.SetKernelStatus({'overwrite_files':True})
g=sim_group(sim_time, *args, **kwargs)
ss=g.get_spike_signal()

mr=ss.mean_rate()
print comm.rank,mr

pickle_save(ss, fileOut)
