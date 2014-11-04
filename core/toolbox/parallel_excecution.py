'''
Created on Mar 19, 2014

@author: lindahlm

'''

 
##################
# MONKEY PATHING #
##################


import my_socket
import monkey_patch as mp
mp.patch_for_milner()

import numpy
# print '1'
import os
import sys
import subprocess
import time 
import pprint
pp=pprint.pprint

from itertools import izip
from toolbox import misc
from toolbox import data_to_disk 
from toolbox.network import default_params
from toolbox.data_to_disk import make_bash_script
from toolbox.parallelization import comm


# if my_signal.determine_host() in ['milner', 'milner_login']:
#     HOME='/cfs/milner/scratch/l/lindahlm'
# else: 
#     HOME = expanduser("~")

HOME=default_params.HOME
print HOME
sys.path.append(HOME+'/tmp/') #for testing


def chunk(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def do(*args, **kwargs):
    
    path_out, path_err, args_call=args
        
    path='/'.join(path_out.split('/')[0:-1])
    if not os.path.isdir(path):
        data_to_disk.mkdir(path)
        
    f_out=open(path_out, "wb", 0)
    f_err=open(path_err, "wb", 0)
    
    if kwargs.get('debug', False):
        p=subprocess.Popen(args_call, stderr=subprocess.STDOUT)
    else:
        p=subprocess.Popen(args_call,
                            stdout=f_out,
                            stderr=f_err,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     stderr=subprocess.STDOUT,
               )
        f_out.close()
        f_err.close()
        
    # Needed such that process finish before evaluation in test
    # script
 
#     out, err = p.communicate()
#     print out
#     print err

    return p
  
def do_milner():
    pass
    
    
def get_loop_index(n, l=[1,1,1]):
    out=[]
    for m in l:
        a=[n for _ in range(m/n)]
        if m-sum(a):
            b=[m-sum(a)]
        else:
            b=[]
        out+=a+b
    return out    
  
def generate_args(*args):
    args=['python']+[__file__]+list(args)
    return ' '.join(args)


def loop(*args, **kwargs):
    chunks, args_list,  kwargs_list=args 
    
    if not isinstance(chunks, list):
        n=len(args_list)/5*2
        m=len(args_list)/5
        chunks=get_loop_index(chunks, [n,n,m])
        
        
    
#     args_list_chunked= chunk(args_list, chunks)  
#     kwargs_list_chunked= chunk(kwargs_list, chunks)
    host=my_socket.determine_computer()
    
    
    chunks_cumsum0=[0]+list(numpy.cumsum(chunks))
    for i_start, i_stop in zip(chunks_cumsum0[:-1], chunks_cumsum0[1:]):
        jobs=[]
#     for al, kl in izip(args_list_chunked, kwargs_list_chunked):
        al=args_list[i_start: i_stop]
        kl=kwargs_list[i_start: i_stop]
        for obj, kwargs in zip(al, kl):
              
            if not kwargs.get('active'):
                continue
                        
            path_code=kwargs.get('path_code')
            path_results=kwargs.get('path_results')
            
            if not os.path.isdir(path_results):
                data_to_disk.mkdir(path_results)
            
            num_mpi_task=kwargs.get('num-mpi-task', 2)     
            index=kwargs['index']
            
            path_out=path_results+"std/subp/out{0:0>4}".format(index)
            path_err=path_results+'std/subp/err{0:0>4}'.format(index)
            path_sbatch_out=path_results+"std/sbatch/out{0:0>4}".format(index)
            path_sbatch_err=path_results+'std/sbatch/err{0:0>4}'.format(index)
            path_tee_out=path_results+'std/tee/out{0:0>4}'.format(index)
            path_params=path_results+'params/run{0:0>4}.pkl'.format(index)
            path_script=path_code+'/core/toolbox/parallel_excecution/simulation.py'
            path_bash0=path_code+'/core/toolbox/parallel_excecution/jobb0.sh'
            path_bash=path_results+'/jobbs/jobb_{0:0>4}.sh'.format(index)
            
            data_to_disk.mkdir('/'.join(path_sbatch_out.split('/')[0:-1]))
            data_to_disk.mkdir('/'.join(path_tee_out.split('/')[0:-1]))

            save_params(path_params, path_script, obj)
            
            if host=='milner':

                o=generate_milner_bash_script(path_sbatch_err,
                                              path_sbatch_out,
                                              path_tee_out,
                                              path_params,
                                              path_script,
                                              path_bash0,
                                              path_bash,
                                               **kwargs )
                args_bash_call=o
                p=do(path_out, path_err, args_bash_call, **kwargs)

                
            if host=='supermicro':
                if num_mpi_task==1:
                    args_call=['python', path_script, path_params]
                else:
                    args_call=['mpirun', '-np', str(num_mpi_task), 'python', 
                               path_script, path_params,
                               '2>&1','|', 'tee', path_tee_out]
                    
                p=do(path_out, path_err, args_call,  **kwargs)
                
            
            jobs.append(p)
            
        if not jobs:
            continue
        
        print jobs
        s='Waiting for {} processes to complete ...'.format(chunks)
        with misc.Stopwatch(s):
            for p in jobs:    
                p.wait()
#                 p.terminate()

            if host=='milner':
                
                from toolbox import jobb_handler
                
                path=default_params.HOME_DATA+'/active_jobbs.pkl'
                
                obj=jobb_handler.Handler(path, 1)
                print obj
                time.sleep(len(al))
                obj.loop(loop_print=True)
                

def save_params(path_params, path_script, obj):
    data_to_disk.pickle_save([obj, 
                              path_script.split('/')[-1]], 
                              path_params)


def generate_milner_bash_script(*args, **kwargs):
    
    p_mil_err, p_mil_out, p_tee_out, p_par, p_script, p_bash0, p_bash=args
    local_threads=10
    _kwargs={'home':default_params.HOME,
             'hours':'00',
             'deptj':1,
             'job_name':'dummy_job',
             'cores_hosting_OpenMP_threads':40/local_threads,
             'local_num_threads':local_threads, 
             'memory_per_node':int(819*local_threads),
             'num-mpi-task':40/local_threads,
             'num-of-nodes':40/40,
             'num-mpi-tasks-per-node':40/local_threads,
             'num-threads-per-mpi-process':local_threads, 
             'minutes':'10',
             'path_sbatch_err':p_mil_err,
             'path_sbatch_out':p_mil_out,
             'path_tee_out':p_tee_out,
             'path_params':p_par,
             'path_script':p_script,
             'seconds':'00',
             
        }
    _kwargs.update(kwargs) 
     
    host=my_socket.determine_computer()
    if host=='milner':
#             call='aprun -n 2 -N 1 -d 20 python {SCRIPT} 2>&1 | tee delme_simulation'
        call=('aprun '
              +'-n {num-mpi-task} '
              +'-N {num-mpi-tasks-per-node} '
              +'-d {num-threads-per-mpi-process} '
              +'-m {memory_per_node} '
              +'python {path_script} {path_params} '
              +'2>&1 | tee {path_tee_out}')
 
#         call=('aprun -n {n_mpi_processes} -N {n_tasks_per_node} '
#               +'-d {depth} '
#               +'-m {memory_per_node} python {path_script}'
#               +' {path_params} 2>&1 | tee {path_tee_out}')
        _kwargs['on_milner']=1
        args_bash_call=['sbatch', p_bash]
        
    else:
        call=('mpirun -np 20 python {path_script}'
              +' {path_params} 2>&1 | tee {path_tee_out}')
        _kwargs['on_milner']=0
        args_bash_call=[p_bash]
            
    
    call=call.format(**_kwargs)
     
    _kwargs['call']=call
    
    make_bash_script(p_bash0, p_bash, **_kwargs)
    
    return args_bash_call
        

def fun1(d):
    print 'Works '+str(d)

def fun2(d):
    import time
    print "Works"+str(d)
    time.sleep(15)
    print "After sleep"

class Mockup_class():
    def __init__(self, path):
        self.path=path
        
    def __repr__(self):
        return __file__.split('/')[-1][0:-4]
    
    def __getstate__(self):
        #print '__getstate__ executed'
        return self.__dict__
    
    def __setstate__(self, d):
        #print '__setstate__ executed'
        self.__dict__ = d 
    
    def do(self):
        import pickle
        f=open(self.path,'wb')
        pickle.dump([1], f, -1)
        f.close()

import unittest
class TestModuleFuncions(unittest.TestCase):

    def setUp(self):
        self.m=misc.import_module('toolbox.network.default_params')

        host=my_socket.determine_host()

        self.dir=HOME+'/results/unittest/parallel_excecution/'+host

        sys.path.append(HOME+'/tmp/')
        
        self.threads=2
        self.path_code=default_params.HOME+'/git/bgmodel/core/toolbox'
        self.path_bash0=self.path_code+'/parallel_excecution/jobb0.sh'
        self.path_bash=(self.path_code+'/parallel_excecution/'
                        +'jobb_unittest_'+host+'.sh')
        self.path_script=self.path_code+'/parallel_excecution/simulation.py'
        self.path_params=self.dir+'/params_in.pkl'
        self.path_tee_out=self.dir+'/tee_mpi_out'
        self.path_sbatch_out=self.dir+'/sbatch_out'
        self.path_sbatch_err=self.dir+'/sbatch_err'
        self.path_subprocess_out=self.dir+'/subprocess_out'
        self.path_subprocess_err=self.dir+'/subprocess_err'
        
        self.path_out_data=self.dir+'/data_out.pkl'
  
        self.obj=Mockup_class(self.path_out_data)
        
        
        for path in [self.path_tee_out, self.path_subprocess_out,
                     self.path_subprocess_err, self.path_bash,
                     self.path_sbatch_out, self.path_sbatch_err]:
            if os.path.isdir(path):
                if comm.rank()==0:
                    os.remove(path)

                     
    def test_do_supermicro(self):
#         run_generate_script(self.path_script)
        save_params(self.path_params, 
                        self.path_script, 
                        self.obj)
        args_call=['mpirun', 
                   '-np', 
                   str(self.threads), 
                   'python', 
                   self.path_script, 
                   self.path_params]
        
        p=do(self.path_subprocess_out, 
             self.path_subprocess_err,  
             args_call,**{'debug':True})        
        p.wait()
          
        l=data_to_disk.pickle_load(self.path_out_data)
        self.assertListEqual(l, [1])
         
        self.assertTrue(os.path.isfile(self.path_subprocess_out))
        self.assertTrue(os.path.isfile(self.path_subprocess_err))

    def test_do_shared_memory(self):
#         run_generate_script(self.path_script)
        save_params(self.path_params, 
                        self.path_script, 
                        self.obj)
        args_call=['python', self.path_script, self.path_params]
        p=do(self.path_subprocess_out, self.path_subprocess_err, 
                            args_call, **{'debug':False})    
        p.wait()    
        l=data_to_disk.pickle_load(self.path_out_data)
        self.assertListEqual(l, [1])
          
        self.assertTrue(os.path.isfile(self.path_subprocess_out))
        self.assertTrue(os.path.isfile(self.path_subprocess_err))
        os.remove(self.path_subprocess_out)
        os.remove(self.path_subprocess_err)
        
    def test_do_milner(self):

        kwargs={'hours':'00',
                'job_name':'lindahl_test_job',
                'minutes':'10',
                'path_sbatch_err':self.path_sbatch_err,
                'path_sbatch_out':self.path_sbatch_out,
                'path_tee_out':self.path_tee_out,
                'path_params':self.path_params,
                'path_script':self.path_script,
                'seconds':'00',
                'threads':20
                }
        
        
#         print self.path_sbatch_err
#         print self.path_sbatch_out
#         print self.path_tee_out
#         print self.path_params
#         print self.path_script
#         print self.path_bash
#         print self.path_bash0
        
        save_params(self.path_params, 
                        self.path_script, 
                        self.obj)
        
        args_call=generate_milner_bash_script(self.path_sbatch_err,
                                    self.path_sbatch_out,
                                    self.path_tee_out,
                                    self.path_params,
                                    self.path_script,
                                    self.path_bash0,
                                    self.path_bash,
                                    **kwargs )

        p=do(self.path_subprocess_out, 
             self.path_subprocess_err, 
             args_call, 
            **kwargs)
        
        if my_socket.determine_host()=='milner_login':
            print 'waiting 20 s'
            time.sleep(20)

        l=data_to_disk.pickle_load(self.path_out_data)
        self.assertListEqual(l, [1])
          
        self.assertTrue(os.path.isfile(self.path_subprocess_out
                                       +'_mpi_milner'))
        self.assertTrue(os.path.isfile(self.path_subprocess_err
                                       +'_mpi_milner')) 
        self.assertTrue(os.path.isfile(self.path_tee_out))        
        
        if my_socket.determine_host() in ['milner_login', 'milner']:
            self.assertTrue(os.path.isfile(self.path_sbatch_out))
            self.assertTrue(os.path.isfile(self.path_sbatch_err)) 
            

    def test_mpi_script(self):
         
        data_to_disk.pickle_save([self.obj, 'Dummy_script_name'], 
                                 self.path_params)
         
        p=subprocess.Popen(['mpirun', '-np', str(self.threads), 'python', 
                            self.path_script, self.path_params],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)    
      
        out, err = p.communicate()
#         print out
#         print err
#          
        l=data_to_disk.pickle_load(self.path_out_data)
        self.assertListEqual(l, [1])

                               
if __name__ == '__main__':
    d={TestModuleFuncions:[
                            'test_do_supermicro',
                            'test_do_shared_memory',
                            'test_do_milner',
#                             'test_mpi_script',
                           ]} 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            host=my_socket.determine_host()
            bo=test in ['test_do_milner']
            if host=='milner_login' and not bo :
                continue
            
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # To run only a single specific test you can use:
    # python -m unittest parallel_excecution.TestModuleFuncions.test_do_milner()
    

