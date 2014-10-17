'''
Created on Oct 13, 2014

@author: mikael
'''
import sys
import numpy
import nest
import time

from toolbox import my_nest
from toolbox.parallelization import mpi_thread_tracker as mth
from toolbox.parallelization import comm

import pprint
pp=pprint.pprint

path_out,=sys.argv[1:]

n=nest.Create('iaf_neuron', 10)
model='static_synapse'

delays=numpy.random.random(10*10)+2
weights=numpy.random.random(10*10)+1
    
post=[]
for _id in n:
    post+=[_id]*10 

my_nest.Connect_DC(n*10, post,weights, delays,  model)
# nest.ConvergentConnect(n, n)
conn=nest.GetConnections(n)
stat=nest.GetStatus(conn)
time.sleep(0.2*comm.rank())
print 'hej', comm.rank(), len(conn), len(post)
# pp(stat)
# pp([[d['source'], d['target']] for d in stat ])
# print nest.GetKernelStatus().keys()
print nest.GetKernelStatus(['num_connections'])
comm.barrier()
# my_nest.Connect_DC(n,n,weights,delays,  model)


# nest_path, np,=sys.argv[1:]

# sim_group(nest_path, **{'total_num_virtual_procs':int(np)})
