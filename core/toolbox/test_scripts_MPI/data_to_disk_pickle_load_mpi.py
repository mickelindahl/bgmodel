'''
Created on Sep 19, 2014

@author: mikael
'''
from toolbox import data_to_disk
from toolbox.parallelization import comm
import sys
# print sys.argv
_, path=sys.argv
# print path+'data.pkl'
data=data_to_disk.pickle_load(path+'data.pkl')
data_to_disk.pickle_save(data, path+'data'+str(comm.rank()), all_mpi=True)
