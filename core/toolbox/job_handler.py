'''
Created on Sep 30, 2014

@author: mikael
'''
import os
import subprocess
import subprocess as sp
import time

import pprint
pp=pprint.pprint

from toolbox import data_to_disk, my_socket
from toolbox.network import default_params



class Handler(object):
    
    def __init__(self, p_list=[], jobs=[], job_names=[],loop_time=1, display=False):
        
        self.loop_time=loop_time
        self.display=display
        self.jobs=set(jobs)         #current_jobbs.difference(old_jobbs)
        d={}
        for  p, j, jn in zip(p_list, jobs, job_names):
            d[jn]={'process':p,
                   'job':j}
        
        self.jobs_dic=d
        self.job_names=job_names
        self.processes=p_list
        if display:
            self.printme()
        
    def append_job(self, p, job_id, job_name=''): 
        self.jobs.add(job_id)
        self.jobs_dic.update({job_name:{'process':p,
                                         'job':job_id}})
        self.job_names.append(job_name)
        self.processes.append(p)
        
    def get_num_of_active_jobs(self):
        
        if my_socket.determine_computer()=='milner':
            current_jobbs=read_current_jobs()   
            go=current_jobbs.intersection(self.jobs)
            n=len(go)
        elif my_socket.determine_computer()=='supermicro':
            n=0
            for p in self.processes:
                if p.poll()==None:
                    n+=1
                       
        return n
    
    def clear(self):
        self.jobs=set([])
        self.jobs_dic={}
        self.job_names=[]
        self.processes=[]
    
    def clean_up(self):
        self.processes=[p for p in self.processes 
                                if p.poll()==None]
        
        for key in self.jobs_dic.keys():
            if my_socket.determine_computer()=='milner':
                current_jobs=read_current_jobs()
                self.jobs=current_jobs.intersection(self.jobs)
                if self.jobs_dic[key]['job'] not in self.jobs:
                    self.job_names.remove(key)
                    del self.jobs_dic[key]
                
            elif my_socket.determine_computer()=='supermicro':    
                if self.jobs_dic[key]['process'] not in self.processes:
                    self.job_names.remove(key)
                    del self.jobs_dic[key]       
                    
    def loop(self, loop_print=False):
        go=1
        i=0
        while go:
            if loop_print:
                print i*self.loop_time, 'seconds'
            current_jobs=read_current_jobs()
            go=current_jobs.intersection(self.jobs)
            print go
            self.printme()
            self.sleep()        
            i+=1
    
    def loop_with_queue(self, num_active_jobs, job_queue, 
                        epoch_fun, loop_print=False):
        q=job_queue
        m=num_active_jobs
        
        m_active=0
        i=0
        while not q.empty() or m_active!=0:
            if m_active<m and not q.empty() : 
                args=epoch_fun(*q.get())
                if args:
                    self.append_job(*args)
            m_active=self.get_num_of_active_jobs()
            self.sleep()
            self.clean_up()
          
            if loop_print:
                print i*self.loop_time, 'seconds'
                self.printme()
            i+=1
    
    def sleep(self):        
        time.sleep(self.loop_time)
    
    def printme(self):
        if my_socket.determine_computer()=='milner':
            current_jobs=read_current_jobs()
            print 'All jobs:    ',list(current_jobs)
            print 'Handler jobs:',list(self.jobs)
        elif my_socket.determine_computer()=='supermicro':    
            print 'Processes:', list(self.processes)
        print 'Job names:',self.job_names
    


def read_current_jobs():
    if my_socket.determine_computer()=='supermicro':
        return set([])
        
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
            current_jobbs.append(int(l[0]))
        
    return set(current_jobbs)   


class Mockup_subprocess():
    PIPE=0
    class Popen():
    
        def __init__(self,cls, *args,**kwargs):
            pass
        
        def communicate(self):
            path=default_params.HOME+'/results/unittest/job_handler'
            return data_to_disk.text_load(path+'/data2'), None
 
class Mockup_subprocess_milner_loop_w():
    PIPE=0
    class Popen():
    
        def __init__(self,cls, *args,**kwargs):
            pass
        
        def communicate(self):
#             time.sleep(0.5)
            path=default_params.HOME+'/results/unittest/job_handler'
            return data_to_disk.text_load(path+'/data3'), None
  
class Mockup_my_socket_milner(object):
    
    @classmethod
    def determine_computer(cls):
        return'milner'

class Mockup_my_socket_supermicro(object):
    
    @classmethod
    def determine_computer(cls):
        return'supermicro'
    
class Mockup_process():
    def __init__(self, name, alive):
        self.alive=alive
        self.name=name
        
#     def __hash__(self):
#         return self.name
        
    def __eq__(self, other):
        return self.name==other.name
    
    def __repr__(self):
        return self.name
        
    def poll(self):
        return self.alive
    
    def set_state(self, alive):
        self.alive=alive

def mockup_epoch_supermicro(*args):
    p=sp.Popen(['python', 'job_handler/epoch_supermicro.py']) 
    return p, args[0], args[1] 

def mockup_epoch_milner(*args):
    p=sp.Popen(['python', 'job_handler/epoch_milner.py', args[2], str(args[3])]) 
    return p, args[0], args[1] 

import unittest
class TestModuleFunctions(unittest.TestCase):

    def setUp(self):
        global subprocess
        subprocess=Mockup_subprocess
  
        self.path=default_params.HOME+'/results/unittest/job_handler'
      
    def test_read_current_jobbs(self):
        global my_socket
        my_socket=Mockup_my_socket_milner
        global subprocess
        subprocess=Mockup_subprocess
               
        p=sp.Popen(['cp', self.path+'/data1', self.path+'/data2']) 
        p.wait()
        s1=set([28373, 28372, 28375, 28374, 28377, 28376])
        self.assertFalse(s1.difference(read_current_jobs()))


        
class TestHandler(unittest.TestCase):

    def setUp(self):
        global subprocess
        subprocess=Mockup_subprocess

        self.path=default_params.HOME+'/results/unittest/job_handler'
        
        kwargs={'p_list':[Mockup_process('1',None),
                          Mockup_process('2',None),
                          Mockup_process('3',None)],
                'jobs': [28372, 28373, 28374],
                'job_names':['1', '2', '3'],
                'loop_time':1}
        self.obj=Handler(**kwargs)  
         
    def test_loop_milner(self):

        global my_socket
        my_socket=Mockup_my_socket_milner
        
        sp.Popen(['cp', self.path+'/data1', self.path+'/data2']) 
        sp.Popen(['python', os.getcwd()+'/job_handler/test.py',
                          self.path],
                           stderr=sp.STDOUT)
        
        self.obj.loop(loop_print=True)
        
        sp.Popen(['cp', self.path+'/data1', self.path+'/data2'])
        time.sleep(2)

    def test_loop_watch_supermicro(self):
        
        global subprocess
        subprocess=Mockup_subprocess
        global my_socket
        my_socket=Mockup_my_socket_supermicro
                
        num_active_jobs=5
        import Queue
        q=Queue.Queue()
        for a,b in [['',1], ['', 2]]:
            q.put([a,b])
            
        self.obj.clear()
        self.obj.loop_with_queue(num_active_jobs, q, mockup_epoch_supermicro,
                                  loop_print=True)

    def test_loop_watch_milner(self):
        
        global subprocess
        subprocess=Mockup_subprocess_milner_loop_w
        global my_socket
        my_socket=Mockup_my_socket_milner
        
        p=sp.Popen(['cp', self.path+'/data4', self.path+'/data3']) 
        
        num_active_jobs=5
        import Queue
        q=Queue.Queue()
        for a,b, c in [[28377,1, 0], [28378, 2, 1], [28379,3, 2]]:
            q.put([a,b, self.path, c])
            
        self.obj.clear()
        self.obj.loop_with_queue(num_active_jobs, q, mockup_epoch_milner,
                                  loop_print=True)


    def test_append_job_milner(self):
        
        global my_socket
        my_socket=Mockup_my_socket_milner
        
        self.obj.append_job(Mockup_process('4',None), 
                            28375, 
                            job_name='4')
        self.assertEqual(self.obj.jobs,set([28372, 28373, 28374]+[28375]))
        
        d1={
            '1': {'process': Mockup_process('1',None), 'job': 28372}, 
            '2': {'process': Mockup_process('2',None), 'job': 28373}, 
            '3': {'process': Mockup_process('3',None), 'job': 28374}, 
            '4': {'process': Mockup_process('4',None), 'job': 28375}
            }

        self.assertDictEqual(d1, self.obj.jobs_dic)

    def test_get_num_of_active_jobs_milner(self):
        
        global my_socket
        my_socket=Mockup_my_socket_milner
        
        n=self.obj.get_num_of_active_jobs()
        self.assertEqual(3,n)

    def test_get_num_of_active_jobs_supermicro(self):
        
        global my_socket
        my_socket=Mockup_my_socket_supermicro
        
        n=self.obj.get_num_of_active_jobs()
        self.assertEqual(3,n)            
 
    def test_clean_up_milner(self):
        
        global my_socket
        my_socket=Mockup_my_socket_milner
        
        sp.Popen(['cp', self.path+'/data0', self.path+'/data2'])
        
        self.obj.append_job(Mockup_process('4',None),  28375,  job_name='4')
        self.obj.append_job(Mockup_process('5',None),  28376,  job_name='5')
        self.obj.append_job(Mockup_process('6',None),  28377,  job_name='6')
        

        self.obj.clean_up()
        self.assertEqual(len(self.obj.jobs_dic.keys()), 6) 
        self.obj.jobs=[28375,28376,28377]
        self.obj.clean_up()
        self.assertEqual(len(self.obj.jobs_dic.keys()), 3) 
        
    def test_clean_up_supermicro(self):
        
        global my_socket
        my_socket=Mockup_my_socket_supermicro        
        
        self.obj.append_job(Mockup_process('4',None),  28375,  job_name='4')
        self.obj.append_job(Mockup_process('5',None),  28376,  job_name='5')
        self.obj.append_job(Mockup_process('6',None),  28377,  job_name='6')
        
        self.obj.processes[0].set_state(False)
        
        self.obj.clean_up()    
        
        self.assertEqual(len(self.obj.jobs_dic.keys()), 5)    
                        
if __name__ == '__main__':
    d={TestModuleFunctions:[
#                             'test_read_current_jobbs',
                           ],
      TestHandler:[
                   
#                     'test_loop_milner',  
                    'test_loop_watch_milner',
#                     'test_loop_watch_supermicro',    
#                     'test_append_job_milner',
#                     'test_get_num_of_active_jobs_milner',
#                     'test_get_num_of_active_jobs_supermicro', 
#                     'test_clean_up_milner',
#                     'test_clean_up_supermicro',
                   ] } 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    