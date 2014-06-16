'''
Created on Apr 9, 2014

@author: lindahlm
'''
import misc
import nest #Need to be imported before MPI is impoorted!!!
import mpi4py.MPI as MPI


import math
import threading
import unittest

from multiprocessing import Pool

class comm(object):
    
    '''dependancy injection'''
    obj=MPI.COMM_WORLD
    
    @classmethod
    def is_mpi_used(cls):
        return cls.obj.size-1
     
    @classmethod
    def size(cls):
        return cls.obj.size

    @classmethod
    def barrier(cls):
        cls.obj.barrier()

    @classmethod
    def bcast(cls, *args, **kwargs):
        return comm.obj.bcast(*args, **kwargs)
        
    @classmethod
    def rank(cls):
        return comm.obj.rank
    
    @classmethod
    def recv(cls, *args, **kwargs):
        return comm.obj.recv(*args, **kwargs)
    
    @classmethod
    def send(cls, *args, **kwargs):
        return comm.obj.send(*args, **kwargs)

class Barrier():
    
    def __init__(self, *args):
        if len(args):
            self.display = args[0]
        else:
            self.display=False
    def __enter__(self, *args):
        if self.display:
            print 'Entering barrier'
        comm.barrier()
                
    def __exit__(self, type, value, traceback):
        comm.barrier()
        if self.display:
            print 'Exiting barrier'


class Wrap(object):
    
    def __init__(self, fun):
        self.fun=fun
    
    def use(self, args):
        return self.fun(*args)
        
        

def execute(fun, worker,  *args, **kwargs):
    if comm.is_mpi_used():
        return _mpi(fun, worker, *args, **kwargs)
    else:

        pool = Pool(processes=kwargs.get('threads',2))

        fun=Wrap(fun)
        args=zip(*args)
        
#         fun.use(args[0])

        r=pool.map(fun.use, args)
        
        # Necessary to shut threads down. Otherwise just more and more threads 
        # are created
        
        ''' 
        close() 
        Indicate that no more data will be put on this queue 
        by the current process. The background thread will quit once it 
        has flushed all buffered data to the pipe. This is called 
        automatically when the queue is garbage collected.
        '''
        pool.close() 
        
        '''
        join()
        Block until all items in the queue have been gotten and processed.
        '''
        pool.join()
        
        return r
        


def _fun_worker(fun, chunksize, i, outs, *args, **kwargs):
    """ The worker function, invoked in a thread. 'nums' is a
        list of numbers to factor. The results are placed in
        outdict.
    """
    a=[args[j][chunksize * i:chunksize * (i + 1)] 
           for j in range(len(args))]
    if outs=={}:
        outs[i]=fun(*a, **kwargs)
    elif outs==None:
        return map(fun, *a, **kwargs)
    
def fun_parallel(fun, *args, **kwargs):
    worker=_fun_worker
    return execute(fun, worker,  *args, **kwargs) 
            
def _map_worker(fun, chunksize, i, outs, *args):
    """ The worker function, invoked in a thread. 'nums' is a
        list of numbers to factor. The results are placed in
        outdict.
    """
    a=[args[j][chunksize * i:chunksize * (i + 1)] 
           for j in range(len(args))]
    if outs=={}:
        outs[i]=map(fun, *a)
    elif outs==None:
        return map(fun, *a)
          
def map_parallel(fun, *args, **kwargs):
    worker=_map_worker
    return execute(fun, worker,  *args, **kwargs)


def _mpi(fun, worker, *args, **kwargs):
    
    with Barrier():    
        chunksize = int(math.ceil(len(args[0]) / float(comm.size())))
        l=worker(fun, chunksize, comm.rank(), None, *args)
        #print l
        if comm.rank() == 0: 
            data=[l]+[comm.recv(source=i, tag=1) for i in range(1, comm.size())]
            data=reduce(lambda x,y:x+y, data )
              
        else:
            comm.send(l,dest=0, tag=1)
            data=None
            
        data=comm.bcast(data, root=0)
        
    return data


# def _threading(fun, worker, *args, **kwargs):
#     # Each thread will get 'chunksize' nums and its own output dict
#     nthreads=kwargs.get('threads',2)
#     if 'threads' in kwargs.keys():del kwargs['threads']
#     chunksize = int(math.ceil(len(args[0]) / float(nthreads)))
#     threads = []
#     outs = [{} for i in range(nthreads)]
#     
#     for i_thread in range(nthreads):
#         # Create each thread, passing it its chunk of numbers to factor
#         # and output dict.
#         defaults=[fun, chunksize, i_thread, outs[i_thread]]
#         t = threading.Thread( target=worker,
#                               args=tuple(defaults+list(args)) ,
#                               kwargs=kwargs)
#         threads.append(t)
#         t.start()
# 
#     # Wait for all threads to finish
#     for t in threads:
#         t.join()
# 
#     # Merge all partial output dicts into a single dict and return it
#     #for key in outs
#     
#     d={k: v for out_d in outs for k, v in out_d.iteritems()}
#     return reduce(lambda x,y:x+y, [d[i] for i in range(nthreads)])
        

class TestModule_functions(unittest.TestCase):

    def fun(self, *args):
        a,b=args 
        #time.sleep(0.001)
        if a % 2==0:
            return a*b
        else:
            return 0
        
    
    def setUp(self):
        pass
        

#     def test_map_parallel(self):
#         a=range(10**1)
#         l1= map(self.fun,a,a)
#         l2=map_parallel(self.fun, a, a, **{'threads':4})
#         self.assertListEqual(l1,l2)
                
#         a=range(10**7)
# 
#         with misc.Stopwatch('Seriell...'):
#             l1= map(self.fun,a,a)
#         
#         with misc.Stopwatch('Parallel...'):
#             l2=map_parallel(self.fun, a, a, **{'threads':8})    
# 
#     def test_nest_connect_parallel(self):
#         import numpy
#         n=nest.Create('iaf_neuron', 100)
#         pre=n*len(n)
#         post=reduce(lambda x,y:x+y, [[i]*len(n) for i in n])
#         weights=numpy.random.random(len(post))*10
#         delays=numpy.random.random(len(post))+1
#         kwargs={'model':'static_synapse','threads':4}
#         nest.Connect(pre, post)#, weights, delays, **{'model':'static_synapse'})
# #         fun_parallel(nest.Connect, pre, post, weights, delays, **kwargs)

                
if __name__ == '__main__':
    
    test_classes_to_run=[
                            TestModule_functions,
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)
    