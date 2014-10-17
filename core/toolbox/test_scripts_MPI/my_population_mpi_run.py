'''
Created on Sep 19, 2014

@author: mikael
'''

from toolbox.data_to_disk import pickle_save, pickle_load
# import cPickle as pickle
import sys
from toolbox.my_population import sim_group

# Necessary for pickle se
# http://stefaanlippens.net/pickleproblem
from toolbox.signal_processing import phases_diff

from toolbox.parallelization import comm, Barrier
fileName, fileOut =sys.argv[1:]

with Barrier():
    if comm.rank()==0:   
        out=pickle_load(fileName, all_mpi=True) 
    else:
        out=None


out=comm.bcast(out, root=0)
sim_time, args, kwargs=out

g=sim_group(sim_time, *args, **kwargs)

ss=g.get_spike_signal()
mr=ss.mean_rate()
fr=ss.firing_rate(1)

pickle_save([mr, fr], fileOut)
