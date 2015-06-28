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
import time
import sys
from toolbox import my_nest


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

    n=500
    if type_conn=='Connect':
        args = get_pre_post(n,d[0], d[0])
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
        
    if type_conn=='Connect_DC':
        my_nest.RandomConvergentConnect(d[0], d[0], n)

def gen_network( chunks, type_conn):
    d={}
    d[0]=nest.Create('aeif_cond_exp', 20000)
    nest.CopyModel('static_synapse', 'static')
    connect(d, chunks, type_conn)


if __name__=='__main__':

    if sys.argv[1]=='0':
        time.sleep(20)
        with Stopwatch('Connect RandomConvergentConnect'):
            gen_network(1, 'RandomConvergentConnect')
            
        print 'Finnished',nest.GetKernelStatus(['num_connections'])[0], 'connections'
        time.sleep(20)
       
    if sys.argv[1]=='1':
        time.sleep(20)
        with Stopwatch('Connect 1'):
            gen_network(1, 'Connect')
        print 'Finnished',nest.GetKernelStatus(['num_connections'])[0], 'connections'
        time.sleep(20)
       
    if sys.argv[1]=='2':
        time.sleep(20)
        with Stopwatch('Connect 10'):
            gen_network(10, 'Connect')
        print 'Finnished',nest.GetKernelStatus(['num_connections'])[0], 'connections'        
        time.sleep(20)
    
    if sys.argv[1]=='3':
        time.sleep(20)
        with Stopwatch('Connect 10'):
            gen_network(1000, 'Connect')
        print 'Finnished',nest.GetKernelStatus(['num_connections'])[0], 'connections'        
        time.sleep(20)  

    if sys.argv[1]=='4':
        time.sleep(20)
        with Stopwatch('Connect DC'):
            gen_network(1000, 'Connect_DC')
        print 'Finnished',nest.GetKernelStatus(['num_connections'])[0], 'connections'        
        time.sleep(20)  
    
    
    