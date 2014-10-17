'''
Created on Oct 7, 2014

@author: mikael
'''
import sys
import pprint
pp=pprint.pprint

from nest import *

local_th, path_nest, =sys.argv[1:]
 
SetKernelStatus({'data_path':path_nest,
                 "local_num_threads": int(local_th),
                 'overwrite_files':True,
                 })


# pp(GetKernelStatus())

pg = Create("poisson_generator", params={"rate": 50000.0})
n = Create('aeif_cond_exp', 20000)
sd = Create("spike_detector", params={"to_file": True, 
                                      "to_memory": False})
RandomConvergentConnect(pg, n, 100)
ConvergentConnect(n, sd)
Simulate(100.0)