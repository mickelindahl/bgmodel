'''
Created on Oct 16, 2014

@author: mikael
'''
import pprint
pp=pprint.pprint
from nest import *
# SetKernelStatus({"total_num_virtual_procs": 40})
SetKernelStatus({"local_num_threads": 10})


pp(GetKernelStatus())

pg = Create("poisson_generator", params={"rate": 50000.0})
n = Create('aeif_cond_exp', 20000)
RandomConvergentConnect(pg, n, 100)
Simulate(100.0)