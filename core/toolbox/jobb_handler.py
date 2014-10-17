'''
Created on Sep 30, 2014

@author: mikael
'''
import os
import subprocess
import subprocess as sp
import time

from toolbox import data_to_disk
from toolbox.network import default_params

class Handler(object):
    
    def __init__(self, old_jobbs_path, loop_time):
        
        self.old_jobbs_path=old_jobbs_path
        self.loop_time=loop_time
        
        old_jobbs=read_old_jobbs(old_jobbs_path)
        current_jobbs=read_current_jobbs()
        print 'Old jobbs',old_jobbs
        print 'Current jobbs',current_jobbs
        print 'Diff', current_jobbs.difference(old_jobbs)
        self.handler_jobbs=current_jobbs.difference(old_jobbs)
        self.save_current_jobbs()
      
    def save_current_jobbs(self):  
        current_jobbs=read_current_jobbs()
#         print current_jobbs
        data_to_disk.pickle_save(current_jobbs, self.old_jobbs_path)
           
    def loop(self, loop_print=False):
        go=1
        i=0
        while go:
            if loop_print:
                print i*self.loop_time, 'seconds'
            current_jobbs=read_current_jobbs()
            print 'Current jobbs:',current_jobbs
            print 'Handler jobbs:', self.handler_jobbs
            go=self.handler_jobbs.intersection(current_jobbs)
            print 'Diff jobbs:', go
            time.sleep(self.loop_time)
            i+=1
        self.save_current_jobbs()

def read_old_jobbs(path):
    if os.path.isfile(path):
        active_jobbs=data_to_disk.pickle_load(path)
    else:
        active_jobbs=[]
    return set(active_jobbs)

def read_current_jobbs():
            
    p=subprocess.Popen(['squeue','-u','lindahlm'],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    out, _ = p.communicate()
    
    current_jobbs=[]
    for i, row in enumerate(out.split('\n')):
    
        if i==0:
            continue
        l=[v for v in row.split(' ') if v!=''][0:2]
        if len(l)==2:
            current_jobbs.append(l[0])
        
    return set(current_jobbs)   


class Mocke_up_subprocess():
    PIPE=0
    class Popen():
    
        def __init__(self,cls, *aegs,**kwargs):
            pass
        
        def communicate(self):
            path=default_params.HOME+'/results/unittest/jobb_handler'
            return data_to_disk.text_load(path+'/data2'), None

    


import unittest
class TestModuleFunctions(unittest.TestCase):

    def setUp(self):
        global subprocess
        subprocess=Mocke_up_subprocess
        self.old_jobbs=set(['28377', '28376', '28375',])
        self.path=default_params.HOME+'/results/unittest/jobb_handler'
        self.path_old_jobbs=self.path+'/old_jobbs.pkl'
        data_to_disk.pickle_save(self.old_jobbs, self.path_old_jobbs)        
        
    def test_read_current_jobbs(self):
        s1=set(['28373', '28372', '28375', '28374', '28377', '28376'])
        self.assertFalse(s1.difference(read_current_jobbs()))

    def test_read_old_jobbs(self):
        s1=read_old_jobbs(self.path_old_jobbs)
#         print s1
        self.assertFalse(s1.difference(read_current_jobbs()))
        
class TestHandler(unittest.TestCase):

    def setUp(self):
        global subprocess
        subprocess=Mocke_up_subprocess
        self.old_jobbs=set(['28377', '28376', '28375',])
        self.path=default_params.HOME+'/results/unittest/jobb_handler'
        self.path_old_jobbs=self.path+'/old_jobbs.pkl'
        data_to_disk.pickle_save(self.old_jobbs, self.path_old_jobbs)   
                  
    def test_loop(self):
        sp.Popen(['python', os.getcwd()+'/jobb_handler_test.py',
                          self.path],
                           stderr=sp.STDOUT)
        
        
        
        obj=Handler(self.path_old_jobbs, 1)
        obj.loop(loop_print=True)
        
        sp.Popen(['cp', self.path+'/data1', self.path+'/data2'])
        time.sleep(2)

if __name__ == '__main__':
    d={TestModuleFunctions:[
                            'test_read_current_jobbs',
                            'test_read_old_jobbs'
                           ],
      TestHandler:[
                   'test_loop'
                   ] } 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    