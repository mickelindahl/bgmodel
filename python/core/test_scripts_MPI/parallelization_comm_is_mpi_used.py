'''
Created on Sep 19, 2014

@author: mikael
'''

from toolbox.data_to_disk import mkdir
from toolbox.parallelization import comm
from os.path import expanduser

import pickle
import mpi4py.MPI as MPI
import sys

fileName,=sys.argv[1:]

# Idividual name
fileName+='data'+str(MPI.COMM_WORLD.rank) 
 
fileName=fileName.split('/')
if  '~' in fileName:
    fileName[fileName.index('~')]=expanduser("~")
fileName='/'.join(fileName)    


if MPI.COMM_WORLD.rank==0:
    mkdir('/'.join(fileName.split('/')[0:-1]))    

MPI.COMM_WORLD.barrier()
print 'passed barrier'

if 4<len(fileName) and fileName[-4:]!='.pkl':
    fileName=fileName+'.pkl'
f=open(fileName, 'wb') #open in binary mode
 
# print f
 
# With -1 pickle the list using the highest protocol available (binary).
pickle.dump(comm.is_mpi_used(), f, -1)
#cPickle.dump(data, f, -1)
 
f.close()

