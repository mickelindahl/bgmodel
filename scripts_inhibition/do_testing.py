'''
Created on Mar 19, 2014

@author: lindahlm

'''
import os
import sys

from toolbox import data_to_disk
from toolbox import misc

from subprocess import Popen
from os import mkdir
print os.curdir

import pprint
pp=pprint.pprint

from os.path import expanduser
HOME = expanduser("~")


def dummy_python_file_content():
    s='import sys\ndef main():\n   print "WWorks "+str(sys.argv[1:])'
    return s

def dummy_python_file_content2():
    s=('import sys, time\ndef main():\n'+
       '   print "WWorks"+str(sys.argv[1:])\n'+
       '   time.sleep(10)\n'+
       '   print "After sleep"+str(sys.argv[1:])\n')

    return s


import unittest
class TestModuleFuncions(unittest.TestCase):

    def setUp(self):
        self.m=misc.import_module('toolbox.network.default_params')
        self.file_name=HOME+'/tmp/do_unittest.py'
        self.file_name1=HOME+'/tmp/do_unittest1.py'
        self.dir=HOME+'/tmp/do_unittest/'
        sys.path.append(HOME+'/tmp/')
        
    def test_do(self):
         
        f=open(self.file_name, 'wb')
        f.write(dummy_python_file_content())
        f.close()
        from subprocess import call
        import subprocess
        self.assertTrue(os.path.isfile(self.file_name))
        cmd='python /home/mikael/git/bgmodel/scripts_inhibition/do.py do_unittest unittest'
        
        call(cmd, shell=True, stdout=subprocess.PIPE)     
        os.remove(self.file_name)
        os.remove(self.file_name+'c')
        self.assertFalse(os.path.isfile(self.file_name))        
        
    def test_do_with_subprocessors(self):

          
        f=open(self.file_name1, 'wb')
        f.write(dummy_python_file_content2())
        f.close()
        self.assertTrue(os.path.isfile(self.file_name1))
           
        mkdir(self.dir) 
        stdout=[]
        stderr=[]
        for i in range(2): 
            cmd='python /home/mikael/git/bgmodel/scripts_inhibition/do.py do_unittest1 unittest'
            stdout.append(self.dir+"stdout{}.txt".format(str(i)))
            stderr.append(self.dir+'stderr{}.txt'.format(str(i)))
            with open(stdout[i],"wb") as out, open(stderr[i],"wb") as err:
                Popen(cmd, shell=True, stdout=out, stderr=err)
           
        import time
        time.sleep(2) 
        os.remove(self.file_name1)
        os.remove(self.file_name1+'c')
        self.assertFalse(os.path.isfile(self.file_name1))
        
    def tearDown(self):
        self.clear()    
    
    def clear(self):
    
        path=self.dir
        if os.path.isdir(path):
            l=os.listdir(path)
            l=[path+ll for ll in l]
            for p in l:
                os.remove(p)
            os.rmdir(path)
                              
if __name__ == '__main__':
    
    test_classes_to_run=[
                         TestModuleFuncions
                       ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)