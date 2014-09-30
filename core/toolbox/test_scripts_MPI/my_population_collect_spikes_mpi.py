'''
Created on Sep 22, 2014

@author: mikael
'''
import numpy
import pickle
import sys

from toolbox.data_to_disk import mkdir
from toolbox.my_population import collect_spikes_mpi
from toolbox.parallelization import comm
fileName, =sys.argv[1:]

fileName+='data'
s,e=numpy.ones(2)*comm.rank(),numpy.ones(2)*comm.rank()+1


s, e= collect_spikes_mpi(s, e)


if comm.rank()==0:
    mkdir('/'.join(fileName.split('/')[0:-1]))  
    
    print 'File name'
    print fileName
    
    if 4<len(fileName) and fileName[-4:]!='.pkl':
        fileName=fileName+'.pkl'
        f=open(fileName, 'wb') #open in binary mode
         
    pickle.dump([s,e], f, -1)
    f.close()
    



