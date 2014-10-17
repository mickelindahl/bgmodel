'''
Created on Oct 2, 2014

@author: mikael
'''

'''
Created on Sep 11, 2014

@author: mikael
'''

import numpy
import nest
import pprint
import time

pp=pprint.pprint

from toolbox import my_nest
from toolbox.misc import Stopwatch
from os.path import expanduser

import gc

from toolbox import nest_speed 


class Stopwatch():
    def __init__(self, *args):
        self.msg = args[0]
        self.args=args
        self.time=None
    def __enter__(self):
        print self.msg,
        self.time=time.time()
        
    def __exit__(self, type, value, traceback):
        t=round(time.time()-self.time,)
        print '... finnish {} {} sec'.format(self.msg, t)
        if len(self.args)>1:
            self.args[1][self.msg]=t    

def set_random_params(ids,vals, keys):
    for val, p in zip(vals, keys):
        local_nodes=[]
        for _id in ids:
            ni=nest.GetStatus([_id])[0]
            if ni['local']:
                local_nodes.append((ni['global_id'], ni['vp']))
                
        for gid, vp in local_nodes:
            val_rand=1+0.1*(numpy.random.random()-0.5)
            val_rand*=val
            nest.SetStatus([gid],{p:val_rand})     

def get_pre_post(n, sources, targets):

    pre=sources*n
    post=targets*n

    return pre, post


def connect(d, chunks, type_conn):
#     chunks = 1
    n=500
    if type_conn=='Connect':
        args = get_pre_post(n,d[0], d[0])
        print 'Connecting'
        print len(args[0])
        m = len(args[0])
        step = m / chunks
        for i in range(chunks):
            if i < chunks - 1:
                pre = args[0][i * step:(i + 1) * step]
                post = args[0][i * step:(i + 1) * step]
            else:
                pre = args[0][i * step:]
                post = args[0][i * step:]
            nest.Connect(pre, post, model='static')

        
        del pre
        del post
    if type_conn=='RandomConvergentConnect':
        nest.RandomConvergentConnect(d[0], d[0], n)
        n

def gen_network( chunks, type_conn):
    d={}
    d[0]=nest.Create('aeif_cond_exp', 20000)
    nest.CopyModel('static_synapse', 'static')
    connect(d, chunks, type_conn)


#     my_nest.Connect(args[0], args[1], model='static', chunks=chunks)
#     args=get_pre_post(500,d[0], d[1] )
#     nest.Connect(*args, model='tsodyks')
    
    
#     nest.Simulate(10000)
#     time.sleep(30)

if __name__=='__main__':
#     time.sleep(60)
    with Stopwatch('Connect 1'):
        gen_network(1, 'Connect')
#     nest.Simulate(1000)
    nest.ResetKernel()
    gc.collect()
    time.sleep(60)
    with Stopwatch('Connect 10'):
        gen_network(10, 'Connect')
        
#     nest.Simulate(1000)
    nest.ResetKernel()
    gc.collect()
    time.sleep(60)

    with Stopwatch('Connect 100'):
        gen_network(100, 'Connect')
        
#     nest.Simulate(1000)
    nest.ResetKernel()
    gc.collect()
    time.sleep(60)
    
    with Stopwatch('Connect RandomConvergentConnect'):
        gen_network(1, 'RandomConvergentConnect')
#     nest.Simulate(1000)
    nest.ResetKernel()
    gc.collect()
    time.sleep(60)

    
    
    