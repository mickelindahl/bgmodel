'''
Created on Oct 20, 2014

@author: mikael
'''

import numpy
from toolbox import my_nest
my_nest.SetKernelStatus({"total_num_virtual_procs": 4})
pg = my_nest.Create("poisson_generator", 20, params={"rate": 50000.0})
n = my_nest.Create('aeif_cond_exp', 20)
# n = my_nest.Create('iaf_neuron', 20)
w=[10. for _ in n]
d=[1. for _ in n]
sd = my_nest.Create("spike_detector", params={"to_file": False,
                                              "to_memory": True})

post=n
post_ids=list(numpy.unique(post))
status=my_nest.GetStatus(list(numpy.unique(post)), 'local')
lockup=dict(zip(post_ids,status))
l=[1 if lockup[p] else 0 for p in post ]
print l

pre=[p for p, b in zip(pg,l) if b]
post=[p for p, b in zip(n,l) if b]
my_nest.Connect_speed(pre, post, w, d)
my_nest.ConvergentConnect(n, sd)
my_nest.Simulate(1000.0)

print len(my_nest.GetStatus(sd)[0]['events']['senders'])
