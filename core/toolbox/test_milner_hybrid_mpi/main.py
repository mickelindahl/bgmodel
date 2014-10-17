'''
Created on Sep 27, 2014

@author: mikael
'''

from toolbox import monkey_patch as mp
mp.patch_for_milner()

import unittest
import os
import subprocess
import time

from toolbox.network import default_params
from toolbox import data_to_disk
from toolbox import my_socket
from toolbox.data_to_disk import make_bash_script, mkdir




class TestMain(unittest.TestCase):
    def setUp(self):

        home=default_params.HOME
        code=home+'/git/bgmodel/core/toolbox/test_milner_hybrid_mpi'
        result=home+'/results/unittest/test_milner_hybrid_mpi/main'
        path_bash=code+'/jobb.sh'
        path_bash0=code+'/jobb0.sh'
        path_err=result+'/jobb_err'
        path_out=result+'/jobb_out'
        path_simu=code+'/simulation.py'
        path_data_nest=result+'/nest'
        path_data_sd=result+'/spike_dic'
        path_delme=result+'/delme_simulation'
        
        mkdir(result)
        
        cores=40*2 #multiple of 40
        local_threads=10

        d={'cores_hosting_OpenMP_threads':40/local_threads,
           'memory_per_node':int(819*local_threads), 
           'num-mpi-task':cores/local_threads,
           'num-of-nodes':cores/40,
           'num-mpi-tasks-per-node':40/local_threads,
           'num-threads-per-mpi-process':local_threads,
           'output':path_delme,
           'path_nest':path_data_nest,
           'script':path_simu,
            } 
        
#         if my_socket.determine_host()=='milner_login':
        fun='sbatch'
#             call='aprun -n 2 -N 1 -d 20 python {SCRIPT} 2>&1 | tee delme_simulation'
        call=('aprun -n {num-mpi-task} '
              +'-N {num-mpi-tasks-per-node} '
              +'-d {num-threads-per-mpi-process} '
              +'-m {memory_per_node} '
              +'python {script} {num-threads-per-mpi-process} {path_nest} 2>&1 | tee {output}')
        call=call.format(**d)
        print call
#         else:
#             fun='batch'
#             call='mpirun -np 2 python {script} 2>&1 | tee {output}'
        
        call=call.format(**d)
        
        on_milner=int(my_socket.determine_host() == 'milner_login')       
        
        k={
            'path_err':path_err,
            'path_out':path_out,
            'call':call,
            'home':home,
            'on_milner':on_milner}
        k.update(d)
        

        make_bash_script(path_bash0, 
                         path_bash,
                         **k)
        
#         if my_socket.determine_host()=='milner_login':
        args_call=[fun, 
                   path_bash
                   ]
#         else:
#             args_call=[
#                        path_bash
#                        ]
        self.path_err=path_err
        self.path_out=path_out
        self.path_simu=path_simu
        self.path_data_nest=path_data_nest
        self.path_data_sd=path_data_sd
        self.path_delme=path_delme
        
        self.args_call=args_call
    
    def test_call_subprocess(self):
        p=subprocess.Popen(self.args_call, stderr=subprocess.STDOUT)
        p.wait()
        
        
        if my_socket.determine_host()=='milner_login':
            print 'waiting 20 s'
            time.sleep(20)
        
        sd=data_to_disk.Storage_dic.load(self.path_data_sd)
        d=sd.load_dic()
#         self.assertAlmostEqual(d['mr'].y, 67,delta=2)
        
          
#         self.assertTrue(os.path.isfile(self.path_delme))
        
#         if my_socket.determine_host()=='milner_login':
#             self.assertTrue(os.path.isfile(self.path_err))  
#             self.assertTrue(os.path.isfile(self.path_err))  

if __name__ == '__main__':
    d={
        TestMain:[
                  'test_call_subprocess'
                  ]
       }
 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))
 
    unittest.TextTestRunner(verbosity=2).run(suite)