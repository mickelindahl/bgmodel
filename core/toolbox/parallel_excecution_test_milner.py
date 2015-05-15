'''
Created on Mar 19, 2014

@author: lindahlm

'''

 
##################
# MONKEY PATHING #
##################


from toolbox import my_socket
from toolbox import monkey_patch as mp
import psycopg2
mp.patch_for_milner()

import os
import sys
import time
import unittest

from toolbox import data_to_disk 
from toolbox import directories as dr
from toolbox.parallelization import comm
from toolbox.parallel_excecution import (do, Job_admin_sbatch, Mockup_class, 
                                         Wrapper_process_sbatch)
class TestModuleFuncions(unittest.TestCase):

    def setUp(self):
        sys.path.append(dr.HOME+'/tmp/')
        
    def clear_paths(self, cb):
        for key in ['p_tee_out', 'p_subp_out','p_subp_err', 
                    'p_bash','p_sbatch_out', 'p_sbatch_err']: 
            if cb.get(key) and os.path.isdir(cb.get(key)):
                if comm.rank()==0:
                    os.remove(cb.get(key))

    def p_out_data(self, host):
        path=dr.HOME+'/results/unittest/parallel_excecution/'+host
        data_to_disk.mkdir(path)
        
        return path
    
    
    def create_obj(self, host):
        return Mockup_class(self.p_out_data(host)+'/data_out.pkl')

    def create_job_admin_milner(self, job_admin, host):
#         path=self.dir+host
        d={'job_admin':Job_admin_sbatch,
           'index':0,
           'num-mpi-task':2,
           'path_results':self.p_out_data(host),
           'hours':'00',
           'job_name':'lindahl_test_job',
           'minutes':'10',
           'seconds':'00',
           'wrapper_process':Wrapper_process_sbatch,
#            'threads':20
           }
                 
        cb=job_admin(**d)
        self.clear_paths(cb)
        return cb          
             
    def test_do_milner(self):

#         kw={'hours':'00',
#             'job_name':'lindahl_test_job',
#             'minutes':'10',
#             'path_sbatch_err':self.path_sbatch_err,
#             'path_sbatch_out':self.path_sbatch_out,
#             'path_tee_out':self.path_tee_out,
#             'path_params':self.path_params,
#             'path_script':self.path_script,
#             'seconds':'00',
# #                 'threads':20
#             }
        host='milner'
        cb=self.create_job_admin_milner(Job_admin_sbatch, host)
        obj=self.create_obj(host)
        cb.gen_job_script()
        
        args=cb.get_subp_args()
#         print self.path_sbatch_err
#         print self.path_sbatch_out
#         print self.path_tee_out
#         print self.path_params
#         print self.path_script
#         print self.path_bash
#         print self.path_bash0
        cb.save_obj(obj)
        
# #         save_params(self.path_params, 
# #                         self.path_script, 
# #                         self.obj)
#         
#         args_call=generate_milner_bash_script(self.path_sbatch_err,
#                                     self.path_sbatch_out,
#                                     self.path_tee_out,
#                                     self.path_params,
#                                     self.path_script,
#                                     self.path_bash0,
#                                     self.path_bash,
#                                     **kwargs )

#         p=do(self.path_subprocess_out, 
#              self.path_subprocess_err, 
#              args_call, 
#             **kwargs)
        
        p=do(*args,**{'debug':False})       
        p.wait()       
         
#         time.sleep(1) 
#         if my_socket.determine_host()=='milner_login':
        print 'waiting 20 s'
        time.sleep(20)

        l=data_to_disk.pickle_load(self.p_out_data(host)+'/data_out.pkl')
        print cb.p_subp_out
        print cb.p_tee_out
        print cb.p_sbatch_out
        self.assertListEqual(l, [1])
        self.assertTrue(os.path.isfile(cb.p_subp_out))
        self.assertTrue(os.path.isfile(cb.p_subp_err))
        self.assertTrue(os.path.isfile(cb.p_tee_out))
        self.assertTrue(os.path.isfile(cb.p_sbatch_out))
        self.assertTrue(os.path.isfile(cb.p_sbatch_err))
        
        
#         self.assertTrue(os.path.isfile(self.path_subprocess_out
#                                        +'_mpi_milner'))
#         self.assertTrue(os.path.isfile(self.path_subprocess_err
#                                        +'_mpi_milner')) 
#         self.assertTrue(os.path.isfile(self.path_tee_out))        
#         
#         if my_socket.determine_host() in ['milner_login', 'milner']:
#             self.assertTrue(os.path.isfile(self.path_sbatch_out))
#             self.assertTrue(os.path.isfile(self.path_sbatch_err)) 
                                     
if __name__ == '__main__':
    d={TestModuleFuncions:[
                            'test_do_milner',
                           ]} 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:            
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # To run only a single specific test you can use:
    # python -m unittest parallel_excecution.TestModuleFuncions.test_do_milner()
    

