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

from toolbox import directories as dr
from toolbox import data_to_disk

class Handler(object):
    
    def __init__(self, p_list=[], loop_time=1, **kwargs):
        
        self.display=kwargs.get('display')      
        self.log_to_file=kwargs.get('log_to_file')
        self.log_file_name=kwargs.get('log_file_name',os.getcwd()+'/job_handler/log')
        self.loop_time=loop_time
        self.processes=p_list
    
        if self.display:
            self.printme()
            
        if self.log_to_file:
            self.init_log()
         
    def append_job(self, p): 
        self.processes.append(p)

    def append_to_log(self, s):
        f=open(self.log_file_name,'a')
        f.write(s)
        f.close()
        
    def init_log(self):
        f=open(self.log_file_name,'w')
        s='Logfile init at '+str(time.strftime("%H:%M:%S"))+'\n'
        f.write(s)
        f.close()
        
      
    def get_num_of_active_jobs(self):
        '''
        A process need to implement the poll function which
        returns None if process still running and something else
        if it is not running
        '''
        
        n=0
        for p in self.processes:
            if p.poll()==None:
                n+=1
                       
        return n
    
    def clear(self):
        self.processes=[]
    
    def clean_up(self):
        self.processes=[p for p in self.processes if p.poll()==None]
                
    def loop(self, loop_print=False):
        go=1
        i=0
        while go:
            
            if self.processes==[]:
                go=0
            self.clean_up()

            if loop_print:
                self.printme(i*self.loop_time)
                
            self.sleep()        
            i+=1
    
    def loop_with_queue(self, max_num_active_jobs, job_queue, 
                            epoch_fun, loop_print=False):
        q=job_queue
        m=max_num_active_jobs
        
        m_active=0
        i=0
        while not q.empty() or m_active!=0:
            if m_active<m and not q.empty() : 
                out=epoch_fun(*q.get())
#                 if out:
#                 print out
                self.append_job(out)
            
            self.sleep()
            m_active=self.get_num_of_active_jobs()
            self.clean_up()
          
            if loop_print:
                self.printme(i*self.loop_time)
            i+=1
    
    def sleep(self):        
        time.sleep(self.loop_time)
    
    def printme(self, t=0):
        
        s=str(t)+' seconds\n'     
        s+='Processes:'+str(self.processes)+'\n'
        
        print s
        
        if self.log_to_file:
            self.append_to_log(s)
    

#TODO: move to setup file
def read_subp_jobs_milner():
    
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
            path=dr.HOME+'/results/unittest/job_handler'
            return data_to_disk.text_load(path+'/data2'), None
 
class Mockup_subprocess_milner_loop_w():
    PIPE=0
    class Popen():
    
        def __init__(self,cls, *args,**kwargs):
            pass
        
        def communicate(self):
#             time.sleep(0.5)
            path=dr.HOME+'/results/unittest/job_handler'
            return data_to_disk.text_load(path+'/data3'), None
 
 
class Mockup_subprocess_cleanup():
    PIPE=0
    class Popen():
    
        def __init__(self,cls, *args,**kwargs):
            pass
        
        def communicate(self):
            path=dr.HOME+'/results/unittest/job_handler'
            return data_to_disk.text_load(path+'/data5'), None 

    
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

class Mockup_wrap_process_milner():
    def __init__(self, p, *args):        
        self.p=p
        self.job_id=args[0]
        self.script_name=args[1]
        
    def __repr__(self):
        return self.script_name+'_id_'+str(self.job_id)
        
    def poll(self):
        '''
        should return None if process is not finnished
        '''
        jobs=read_subp_jobs_milner()
#         print jobs
        if  self.job_id in jobs:
            return None
        else:
            return 1
 
class Mockup_wrap_process_batch():
    def __init__(self, p, *args):        
        self.p=p
        self.job_id=args[0]
        self.script_name=args[1]
        
    def __repr__(self):
        return self.script_name+'_id_'+str(self.job_id)
        
    def poll(self):
        '''
        should return None if process is not finnished
        '''
        return self.p.poll()

    
    def set_state(self, alive):
        self.p.alive=alive

def mockup_epoch_supermicro(*args):
#     args, kw=args
    job_id, script_name,  kw=args
    
    wp=kw['wrapper_process']
    
    p=sp.Popen(['python', 'job_handler/epoch_supermicro.py']) 
    
    p=wp(p, job_id, script_name)

    return p
    
def mockup_epoch_milner(*args):
    
    job_id, script_name, path, _id, kw=args
    
    wp=kw['wrapper_process']
    
    p=sp.Popen(['python', 'job_handler/epoch_milner.py', path,str(_id),]) 
    
    p=wp(p, job_id, script_name)
#     print p
    return p

import unittest
class TestModuleFunctions(unittest.TestCase):

    def setUp(self):
        global subprocess
        subprocess=Mockup_subprocess
  
        self.path=dr.HOME+'/results/unittest/job_handler'
      
    def test_read_current_jobbs(self):
#         global my_socket
#         my_socket=Mockup_my_socket_milner
        global subprocess
        subprocess=Mockup_subprocess
               
        p=sp.Popen(['cp', self.path+'/data1', self.path+'/data2']) 
        p.wait()
        s1=set([28373, 28372, 28375, 28374, 28377, 28376])
        self.assertFalse(s1.difference(read_subp_jobs_milner()))

def data0():
    
    s=  '''JOBID  USER     ACCOUNT           NAME EXEC_HOST ST     REASON   START_TIME     END_TIME  TIME_LEFT NODES   PRIORITY
28377  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:03     3       1016
28376  tully    (null)       sequences    login1  R       None 2014-09-30T1 2014-09-30T1    3:21:02     3       1016
28375  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:01     3       1016
'''
    return s

def data1():
    s='''JOBID  USER     ACCOUNT           NAME EXEC_HOST ST     REASON   START_TIME     END_TIME  TIME_LEFT NODES   PRIORITY
28377  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:03     3       1016
28376  tully    (null)       sequences    login1  R       None 2014-09-30T1 2014-09-30T1    3:21:02     3       1016
28375  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:01     3       1016
28374  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:03     3       1016
28373  tully    (null)       sequences    login1  R       None 2014-09-30T1 2014-09-30T1    3:21:02     3       1016
28372  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:01     3       1016
'''
    return s
def data2():
    s='''JOBID  USER     ACCOUNT           NAME EXEC_HOST ST     REASON   START_TIME     END_TIME  TIME_LEFT NODES   PRIORITY
28377  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:03     3       1016
28376  tully    (null)       sequences    login1  R       None 2014-09-30T1 2014-09-30T1    3:21:02     3       1016
28375  tully    (null)       sequences    login2  R       None 2014-09-30T1 2014-09-30T1    3:21:01     3       101
'''
    return s
def data3():
    s='''JOBID  USER     ACCOUNT           NAME EXEC_HOST ST     REASON   START_TIME     END_TIME  TIME_LEFT NODES   PRIORITY
'''
    return s
class TestHandler(unittest.TestCase):

    def setUp(self):
        global subprocess
        subprocess=Mockup_subprocess
#         print data0()
        self.path=dr.HOME+'/results/unittest/job_handler'
        
        if not os.path.isdir(self.path):
            data_to_disk.mkdir(self.path)
        if not os.path.isfile(self.path+'/data0'):
            data_to_disk.txt_save(data0(), self.path+'/data0', file_extension='')
        if not os.path.isfile(self.path+'/data1'):
            data_to_disk.txt_save(data1(), self.path+'/data1', file_extension='')
        if not os.path.isfile(self.path+'/data2'):
            data_to_disk.txt_save(data2(), self.path+'/data2', file_extension='')
        if not os.path.isfile(self.path+'/data3'):
            data_to_disk.txt_save(data3(), self.path+'/data3', file_extension='')   
                      
        p_list=[Mockup_process('1',None), Mockup_process('2',None), Mockup_process('3',None)]
        jobs=[28372, 28373, 28374]
        names=['Net_0', 'Net_1', 'Net_2']
        
        wp_list=[Mockup_wrap_process_milner(p,j,n) for p,j,n in zip(p_list, jobs,names)]
        kw={'p_list':wp_list,
            'loop_time':1,
            'log_to_file':True,
            'log_file_name':os.getcwd()+'/job_handler/log'}
        
        self.obj_milner=Handler(**kw)  
        
        p_list=[Mockup_process('1',None), Mockup_process('2',None), Mockup_process('3',None)]
        wp_list=[Mockup_wrap_process_batch(p,j,n) for p,j,n in zip(p_list, jobs,names)]
        kw={'p_list':wp_list,
            'loop_time':1,
            'log_to_file':True,
            'log_file_name':os.getcwd()+'/job_handler/log'}
        
        self.obj_super=Handler(**kw)  
              
        self.obj_empty=Handler(kw={'p_list':[],
                                   'loop_time':1,
                                   'log_to_file':True,
                                   'log_file_name':os.getcwd()+'/job_handler/log'})
         
    def test_loop_milner(self):

        self.obj_milner.log_file_name+='_loop_milner'
        global subprocess
        subprocess=Mockup_subprocess

        
        sp.Popen(['cp', self.path+'/data1', self.path+'/data2']) 
        sp.Popen(['python', os.getcwd()+'/job_handler/sleep_five_then_switch_data.py',
                          self.path],
                           stderr=sp.STDOUT)
        
        self.obj_milner.loop(loop_print=True)
        
        sp.Popen(['cp', self.path+'/data1', self.path+'/data2'])
        time.sleep(2)

    def test_loop_with_queue_batch(self):
        
        self.obj_empty.log_file_name+='_loop_watch_supermicro'
                
        num_active_jobs=5
        import Queue
        q=Queue.Queue()
        d={'wrapper_process':Mockup_wrap_process_batch}
        
        for a,b,c in [[1000,'Net_0', d.copy()], [1001, 'Net_1', d.copy()]]:
            q.put([a,b,c])
            
        self.obj_empty.clear()
        self.obj_empty.loop_with_queue(num_active_jobs, q, mockup_epoch_supermicro,
                                  loop_print=True)

    def test_loop_with_queue_milner(self):
         
        self.obj_empty.log_file_name+='_loop_watch_milner'

        global subprocess
        subprocess=Mockup_subprocess_milner_loop_w

        p=sp.Popen(['cp', self.path+'/data4', self.path+'/data3']) 
        
        num_active_jobs=5
        import Queue
        q=Queue.Queue()
        
        d={'wrapper_process':Mockup_wrap_process_milner}
        for a, b, c in [[28377, 'Net_0', 0], [28378, 'Net_1', 1], [28379,'Net_2', 2]]:
            q.put([a,b, self.path, c, d.copy()])
            
        self.obj_empty.clear()
        self.obj_empty.loop_with_queue(num_active_jobs, q, mockup_epoch_milner,
                                  loop_print=True)


    def test_append_job_milner(self):
        p_list=[Mockup_process('1',None), Mockup_process('2',None), Mockup_process('3',None)]
        jobs=[28375, 28376, 28377]
        names=['Net_3', 'Net_4', 'Net_5']
        
        wp_list=[Mockup_wrap_process_milner(p,j,n) for p,j,n in zip(p_list, jobs,names)]
            
        for p in wp_list:
            self.obj_milner.append_job(p)
        s1=str(self.obj_milner.processes)
        s2='[Net_0_id_28372, Net_1_id_28373, Net_2_id_28374, Net_3_id_28375, Net_4_id_28376, Net_5_id_28377]'
        self.assertEqual(s1, s2)

    def test_get_num_of_active_jobs_milner(self):
        
        global subprocess
        subprocess=Mockup_subprocess_cleanup
        n=self.obj_milner.get_num_of_active_jobs()
        self.assertEqual(2,n)

    def test_get_num_of_active_jobs_supermicro(self):
        n=self.obj_super.get_num_of_active_jobs()
        self.assertEqual(3,n)            
 
    def test_clean_up_milner(self):
        
        global subprocess
        subprocess=Mockup_subprocess_cleanup
        self.obj_milner        
        self.assertEqual(len(self.obj_milner.processes), 3) 
        self.obj_milner.clean_up()
        self.assertEqual(len(self.obj_milner.processes), 2) 
        
    def test_clean_up_supermicro(self):

        self.obj_super 
        self.obj_super.processes[0].set_state(1)
        self.assertEqual(len(self.obj_super.processes), 3) 
        self.obj_super.clean_up()
        self.assertEqual(len(self.obj_super.processes), 2) 
 
                        
if __name__ == '__main__':
    d={TestModuleFunctions:[
                            'test_read_current_jobbs',
                           ],
      TestHandler:[             
                    'test_loop_milner',  
                    'test_loop_with_queue_batch',
                    'test_loop_with_queue_milner',    
                    'test_append_job_milner',
                    'test_get_num_of_active_jobs_milner',
                    'test_get_num_of_active_jobs_supermicro', 
                    'test_clean_up_milner',
                    'test_clean_up_supermicro',
                   ] } 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    