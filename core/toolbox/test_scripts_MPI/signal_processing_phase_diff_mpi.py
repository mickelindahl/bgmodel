'''
Created on Sep 22, 2014

@author: mikael
'''
import pickle
# import cPickle as pickle
import sys

# Necessary for pickle se
# http://stefaanlippens.net/pickleproblem
from toolbox.signal_processing import phases_diff

from toolbox.parallelization import comm, Barrier
fileName, fileOut =sys.argv[1:]

with Barrier():
    if comm.rank()==0:        
        f=open(fileName, 'rb') #open in binary mode
        x, y, kwargs=pickle.load(f)
        f.close()
        
    else:
        x,y, kwargs=None, None, None
        

x=comm.bcast(x, root=0)
y=comm.bcast(y, root=0)
kwargs=comm.bcast(kwargs, root=0)

fr=phases_diff(x,y,**kwargs)

if comm.rank()==0:
    f=open(fileOut, 'wb') #open in binary mode
    pickle.dump(fr, f, -1)
    f.close()
