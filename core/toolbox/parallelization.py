'''
Created on Apr 9, 2014

@author: lindahlm

MPI + threads is not possible in python due to GlobalInterpreterLock
https://wiki.python.org/moin/GlobalInterpreterLock
'''
import misc
import nest #Need to be imported before MPI is impoorted!!!
import mpi4py.MPI as MPI
import multiprocessing

import inspect
import math
import subprocess
import numpy
# import threading

import os
import time  

from itertools import izip
from multiprocessing import Pool, Process,  Queue, Array
from os.path import expanduser
from toolbox import misc
from toolbox import my_socket

class comm(object):
    
    '''dependancy injection'''
    obj=MPI.COMM_WORLD
    
    
    @classmethod
    def is_mpi_used(cls):  
        return cls.obj.size-1
        #return 1<nest.GetKernelStatus('total_num_virtual_procs')
     
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
    
    @classmethod
    def open(cls, name, mode):
        
        if mode in ['r', 'rb']:
            amode=MPI.MODE_RDONLY
        if mode in ['w','wb']:
            amode=(MPI.MODE_WRONLY + MPI.MODE_CREATE)
        
        return MPI.File.Open(MPI.COMM_WORLD, name, amode=amode)

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
    


def chunkit(chunksize, i, last, *args):
    if not last:
        a=[args[j][chunksize * i:chunksize * (i + 1)] 
            for j in range(len(args))]
    elif last:
        a=[args[j][chunksize * i:] 
            for j in range(len(args))]
    return a

def my_map(i, chunksize, last, fun, out, args):
    a=map(fun, *args)
    if not last:
        out[chunksize*i:chunksize*(i+1)]=numpy.array(a)
    else:
        out[chunksize*i:]=numpy.array(a)       
         
class MyMap(object):
    def __init__(self, fun, i):
        self.__fun = fun
        self.__id=i
    def __call__(self, *args):
        return  map(self.__fun, *args) 
     
def map_local_threads(fun, args, kwargs):


    local_th=int(kwargs.get('local_num_threads', 1))
    if (local_th==1) or my_socket.determine_computer()=='milner':
        return map(fun, *args)

    n=len(args[0])
    chunksize=int(math.floor(len(args[0]) / float(local_th)))

    jobs=[]
    
    tmp_args=[a[0] for a in args ]    
    tmp_r=fun(*tmp_args)
    
    
    if not type(tmp_r) in [list, tuple, numpy.ndarray]:
        print 'array', comm.rank()
        if type(tmp_r)==int:
            out = Array( 'l', numpy.zeros(n, dtype=numpy.int64) )
        else:
            out = Array( 'd', numpy.zeros(n) )
            
        for i in xrange(local_th):
            last=i==local_th-1
            a=chunkit(chunksize, i, last, *args)
            p=Process(target=my_map, args=(i, chunksize, last, fun, out, a))
            jobs.append(p)
            
        for job in jobs: job.start()
        for job in jobs: job.join()
    
        r=list(out[:])
         
    else:

        pool = Pool(processes=local_th)
     
        fun = Wrap(fun)
        args=izip(*args)    
        f=fun.use
        r = pool.map(f, args)#, chunksize=len(args)/local_th)

        # Necessary to shut threads down. Otherwise just more and more threads
        # are created
        comm.barrier()
        pool.close()
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
    
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno




#     if comm.is_mpi_used():
    
#     return map_mpi(fun, *args, **kwargs)

#     else:
#         return map_pool(fun, args, kwargs)
        
def map_parallel(fun, *args, **kwargs):
# def map_mpi(fun, *args, **kwargs):
    
    with Barrier():
        
        n=len(args[0]) 
        chunksize = int(math.floor(n / float(comm.size())))
        
        last=comm.rank()==comm.size()-1          
        a=chunkit(chunksize, comm.rank(), last, *args)

        out=map_local_threads(fun, a, kwargs) 
        if comm.rank() == 0: 
            data=[out]+[comm.recv(source=i, tag=1) for i in range(1, comm.size())]
            data=reduce(lambda x,y:x+y, data )  
        else:
            comm.send(out,dest=0, tag=1)
            data=None    
    
#     if data:
#         print 'len data', len(data)
#         for d in data:
#             print d.shape
    
    with Barrier():
        data=comm.bcast(data, root=0)

    return data


def mpi_thread_tracker(file_name, s='as line '+str(lineno())):

    with Barrier():
        if comm.size()>1:
            time.sleep(0.1)
#             for i in range
            print 'MPI rank:', comm.rank(), s, 'in', file_name         
            time.sleep(0.1)
            

def mockup_fun(*args):
    a,b=args 

    if a % 2==0:
        return a*b
    else:
        return a-b
    
import unittest
class TestModule_functions(unittest.TestCase):
    
    def setUp(self):
        self.home=expanduser("~")
        
    def test_map_parallel(self):
        a=range(10**3)
#         a=[float(aa) for aa in a]
        with misc.Stopwatch('Seriell'):
            l1= map(mockup_fun, a, a)
        with misc.Stopwatch('Seriell2s'):
            l2=map_parallel(mockup_fun, a, a, **{'local_num_threads':4})

        self.assertListEqual(l1,l2)
                 
    def test_map_parallel_mpi(self):
        from data_to_disk import pickle_save, pickle_load
        from toolbox.data_to_disk import read_f_name

#         a=range(10**7+5)
        a=range(72)
#         a=[float(aa) for aa in a]
        with misc.Stopwatch('Seriell'):
            l1= map(mockup_fun,a,a)
        
        data_path= self.home+'/results/unittest/parallelization/map_parallel_mpi/'
        script_name=os.getcwd()+'/test_scripts_MPI/parallelization_map_parallel_mpi.py'
        np=10
            
        for fname in read_f_name(data_path):
            os.remove(data_path+fname)
  
        fileName = data_path + 'data_in.pkl'
        fileOut = data_path + 'data_out.pkl'
        pickle_save(a, fileName)

        with misc.Stopwatch('Subp'):
            p = subprocess.Popen(['mpirun',  '-np', str(np), 'python', 
                                  script_name, fileName, fileOut, data_path], 
        #                            stdout=subprocess.PIPE,
        #                            stderr=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            
            out, err = p.communicate()
        
        print len(a)
        l2=pickle_load(fileOut) 
        print l1[0:20]
        print l2[0:20]
#         self.assertListEqual(l1,l2)
        self.assertListEqual(l1[0:20],l2[0:20])
        self.assertListEqual(l1[-20:],l2[-20:])
class TestComm(unittest.TestCase):

    def test_is_mpi_used(self):

        import pickle
        from toolbox.data_to_disk import read_f_name
        s = expanduser("~")
        data_path= s+'/results/unittest/parallelization/comm_is_mpi_used/'
        script_name=os.getcwd()+'/test_scripts_MPI/parallelization_comm_is_mpi_used.py'
        np=2
        
        
        for fname in read_f_name(data_path):
            os.remove(data_path+fname)
        
        files=[]
        for i in range(np):
            files.append(data_path+'data'+str(i)+'.pkl')
            
        p=subprocess.Popen(['mpirun', 
                         '-np',
                         str(np), 
                         'python', 
                         script_name,
                         data_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        
        out, err = p.communicate()
#         print out
#         print err

        data=[]
        for fn in files:
            f=open(fn, 'rb') #open in binary mode

            data.append(pickle.load(f))
            f.close()
            
        data2=[1 for _ in range(np)]
        

        self.assertListEqual(data, data2)          

                               
if __name__ == '__main__':
    d={
       TestModule_functions:[
                            'test_map_parallel',
                            'test_map_parallel_mpi',
                           ],
       TestComm:[
#                  'test_is_mpi_used'
                 ]
       } 
    
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:            
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    