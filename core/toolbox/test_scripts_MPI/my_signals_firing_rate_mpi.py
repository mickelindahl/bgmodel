'''
Created on Sep 22, 2014

@author: mikael
'''
# import pickle
import cPickle as pickle
import sys

# Necessary for pickle se
# http://stefaanlippens.net/pickleproblem
from toolbox.my_signals import MySpikeList

from toolbox.parallelization import comm, Barrier
fileName, fileOut =sys.argv[1:]

with Barrier():
    if comm.rank()==0:        
        f=open(fileName, 'rb') #open in binary mode
        signal_obj=pickle.load(f)
        f.close()
        
    else:
        signal_obj=None
        

signal_obj=comm.bcast(signal_obj, root=0)

fr=signal_obj.firing_rate(1)

if comm.rank()==0:
    f=open(fileOut, 'wb') #open in binary mode
    pickle.dump(fr, f, -1)
    f.close()