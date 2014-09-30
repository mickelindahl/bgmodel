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
from toolbox.data_to_disk import make_bash_script




class TestMain(unittest.TestCase):
    def setUp(self):

        home=default_params.HOME
        code=home+'/git/bgmodel/core/toolbox/test_milner'
        result=home+'/results/unittest/test_milner/main'
        path_bash=code+'/jobb.sh'
        path_bash0=code+'/jobb0.sh'
        path_err=result+'/jobb_err'
        path_out=result+'/jobb_out'
        path_simu=code+'/simulation.py'
        path_data_nest=result+'/nest'
        path_data_sd=result+'/spike_dic'
        path_delme=result+'/delme_simulation'
        

        d={'SCRIPT':path_simu,
           'ARGV':path_data_nest+' '+path_data_sd,
           'OUTPUT':path_delme}        
        
        if my_socket.determine_host()=='milner_login':
            fun='sbatch'
            call='aprun -n 20 python {SCRIPT} {ARGV} 2>&1 | tee {OUTPUT}'
        else:
            fun='batch'
            call='mpirun -np 10 python {SCRIPT} {ARGV} 2>&1 | tee {OUTPUT}'
        call=call.format(**d)
        
        on_milner=int(my_socket.determine_host()=='milner_login')
        make_bash_script(path_bash0, 
                         path_bash,
                         **{'path_err':path_err,
                            'path_out':path_out,
                            'call':call,
                            'home':home,
                            'on_milner':on_milner})
        
        if my_socket.determine_host()=='milner_login':
            args_call=[fun, 
                       path_bash
                       ]
        else:
            args_call=[
                       path_bash
                       ]
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
        self.assertAlmostEqual(d['mr'].y, 67,delta=2)
        
          
        self.assertTrue(os.path.isfile(self.path_delme))
        
        if my_socket.determine_host()=='milner_login':
            self.assertTrue(os.path.isfile(self.path_err))  
            self.assertTrue(os.path.isfile(self.path_err))  

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