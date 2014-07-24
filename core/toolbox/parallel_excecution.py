'''
Created on Mar 19, 2014

@author: lindahlm

'''

import os
import sys


from os.path import expanduser
# from subprocess import Popen
from multiprocessing import Process
from toolbox import data_to_disk, misc

import pprint
pp=pprint.pprint

HOME = expanduser("~")
sys.path.append(HOME+'/tmp/') #for testing

def chunk(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def do( path_out, path_err, fun, script,  *args):
    stdout= sys.stdout
    stderr= sys.stderr
    
    path='/'.join(path_out.split('/')[0:-1])
    if not os.path.isdir(path):
        data_to_disk.mkdir(path)
    
    '''
    "The optional buffering argument specifies the 
    file's desired buffer size: 0 means unbuffered, 
    1 means line buffered, any other positive value 
    means use a buffer of (approximately) that size. 
    A negative buffering means to use the system 
    default, which is usually line buffered for tty 
    devices and fully buffered for other files. If 
    omitted, the system default is used."
    '''
    
    sys.stdout = open(path_out, "wb", 0)
    sys.stderr = open(path_err, "wb", 0)
#     
#     module=__import__(attr)
#     fun=module.main
#     try:
    print 'Running '+str(fun)+' as ' + script
    fun(*args)
#     print 'Success'
#     except Exception as e:       
#     print 'Fail '+e.message
    
    sys.stdout.close()
    sys.stderr.close()
     
    sys.stdout=stdout
    sys.stderr=stderr
#     
def generate_args(*args):
    args=['python']+[__file__]+list(args)
    return ' '.join(args)


def loop(args_list, path, chunks):

    if not os.path.isdir(path):
        data_to_disk.mkdir(path)
    
    args_list_chunked= chunk(args_list, chunks)  
    
    i=0
    p_list=[]
    for args_list in args_list_chunked:
        for args in args_list:
            path_out=path+"std/out{}.txt".format(str(i))
            path_err=path+'std/err{}.txt'.format(str(i))
            args=[path_out, path_err]+args
            #with open(stdout[i],"wb") as out, open(stderr[i],"wb") as err:
#                 p=Popen(cmd, shell=True, stdout=out, stderr=err)
#             do(*args)
            p=Process(target=do, args=args)  
            p.start()
            p_list.append(p)
            i+=1
            
        s='Waiting for {} processes to complete ...'.format(chunks)
        with misc.Stopwatch(s):
            for p in p_list:    
                p.join()
                p.terminate()
        import time
        time.sleep(1)
        


def fun1(d):
    print 'Works '+str(d)

def fun2(d):
    import time
    print "Works"+str(d)
    time.sleep(15)
    print "After sleep"

import unittest
class TestModuleFuncions(unittest.TestCase):

    def setUp(self):
        self.m=misc.import_module('toolbox.network.default_params')
        self.file_name=HOME+'/tmp/do_unittest.py'
        self.file_name1=HOME+'/tmp/do_unittest1.py'
        self.dir=HOME+'/tmp/do_unittest/'
        self.dir2=HOME+'/tmp/do_unittest2/'
        self.file_out=HOME+'/tmp/do_unittest/stdout.txt'
        self.file_err=HOME+'/tmp/do_unittest/stderr.txt'
        self.file_out0=HOME+'/tmp/do_unittest2/std/out0.txt'
        sys.path.append(HOME+'/tmp/')
        
    def test_do(self):
         
        args=[self.file_out, self.file_err, fun1, 'script_fun', {1:1}]
        
        do(*args)
        self.assertTrue(os.path.isfile(self.file_out))
        self.assertTrue(os.path.isfile(self.file_err))
        os.remove(self.file_out)
        os.remove(self.file_err)
        self.assertFalse(os.path.isfile(self.file_out))        
        self.assertFalse(os.path.isfile(self.file_err))    
                
    def test_do_with_subprocessors(self):
          
        args_list=[]
        for _ in range(6): 
            args_list.append([fun2, 'script_fun2', {1:1}])
         
        with misc.Stop_stdout(True):
            loop(args_list, self.dir2, 2)
                    
        self.assertTrue(os.path.isfile(self.file_out0))
        
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


