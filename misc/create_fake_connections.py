'''
Created on Oct 14, 2013

@author: lindahlm
'''
import numpy
import random
from toolbox import data_handling
import os

n_states=10
n_actions=5

names=['CO_M1', 'CO_M2', 'FS_M1', 'FS_M2']


for name in names[0:2]:
    w=[]
    for i in range(n_states):
        for j in range(n_actions):
            w.append(random.random())
    
    w=numpy.array(w)
    fileName=os.getcwd()+'/conn-'+ name

    data_handling.pickle_save(w, fileName)
    
for name in names[2:]:
    w=[]
    for i in range(n_states*n_actions):
        w.append(random.random())
    
    w=numpy.array(w)
    fileName=os.getcwd()+'/conn-'+ name

    data_handling.pickle_save(w, fileName)