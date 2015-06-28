'''
Created on Sep 24, 2014

@author: mikael
'''

import os

from toolbox.network import default_params
from toolbox.network.structure import Conn, Surface_dic
from toolbox.misc import my_slice
from toolbox import data_to_disk
from toolbox.parallelization import comm, Barrier
HOME=default_params.HOME
import time

# with Barrier(True):
#     if comm.rank()==0:
#         time.sleep(10)


n_sets=3
n=1000
sets=[my_slice(s, n, n_sets) for s in range(n_sets)]
nd=Surface_dic()
nd.add('i1', **{'n':n, 'n_sets':n_sets, 'sets':sets })
nd.add('n2', **{'n':n, 'n_sets':n_sets, 'sets':sets})

surfs=nd
source=nd['i1']
target=nd['n2']


path=HOME+'/results/unittest/structure/set_save_load_mpi/'
path_conn=path+'conn/'
rules=[
       'divergent',
        '1-1', 
        'all-all', 
        'set-set', 
        'set-not_set', 
        'all_set-all_set',

       ]     
k={'fan_in':10.0}
l1=[]
l2=[]
for rule in rules:
    k.update({'display':False,
              'rule':rule,
              'source':source.get_name(),
              'target':target.get_name(),
              'save':{'active':True,
                      'overwrite':False,
                     'path':path_conn+rule}})

    c1=Conn('n1_n2', **k)
    c1.set(surfs, display_print=False)
    
    l1.append(c1.n)
    c2=Conn('n1_n2', **k)
    c2.set(surfs, display_print=False)
  
    l2.append(c2.n)
    
    
data_to_disk.pickle_save( [l1, l2], 
                              path+'data'+str(comm.rank()), 
                              all_mpi=True)
    
   

