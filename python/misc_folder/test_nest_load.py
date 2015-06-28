'''
Created on Sep 13, 2013

@author: lindahlm
'''
from core import data_handling

file_names=['/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/inhibition/Inhibition_base/spike_detector-10033-0.gdf']
s, t=data_handling.nest_sd_load(file_names)

print s[0:10]
