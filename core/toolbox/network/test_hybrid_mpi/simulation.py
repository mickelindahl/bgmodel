'''
Created on Oct 7, 2014

@author: mikael
'''
from nest import *
SetKernelStatus({"total_num_virtual_procs": 4})
pg = Create("poisson_generator", params={"rate": 50000.0})
n = Create('aeif_cond_exp', 2000)
# sd = Create("spike_detector", params={"to_file": True})
RandomConvergentConnect(pg, n, 100)
# ConvergentConnect(n, sd)
Simulate(100.0)