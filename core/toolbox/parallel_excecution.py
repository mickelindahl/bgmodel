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
import subprocess
import time 
import pprint
pp=pprint.pprint
import Queue

from toolbox import data_to_disk 
from toolbox import directories as dr
from toolbox import job_handler
from toolbox import postgresql as psql
 
from toolbox.data_to_disk import make_bash_script
from toolbox.parallelization import comm


def chunk(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def do(*args, **kwargs):
    
    path_out, path_err=args[0:2]
    args_call=args[2:]
        
    path='/'.join(path_out.split('/')[0:-1])
    if not os.path.isdir(path):
        data_to_disk.mkdir(path)
        
    f_out=open(path_out, "wb", 0)
    f_err=open(path_err, "wb", 0)
    
    if kwargs.get('debug', False):
        p=subprocess.Popen(args_call, stderr=subprocess.STDOUT)
    else:
        print args_call
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
#   
# def do_milner():
#     pass
#     
#     
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


def epoch(*args):
    obj, kw=args
    
#     if not kw.get('active'):
#         return False
         
    path_results=kw.get('path_results')
#     process_type=kw.get('process_type')
    
    if not os.path.isdir(path_results):
        data_to_disk.mkdir(path_results)   
        
         
    ja=kw.get('job_admin')(**kw)
    wp=kw.get('wrapper_process')
    
#     gen_subp_job_script=kwargs.get('subp_job_script', gen_subp_job_script_milner)
#     get_subp_job_id=kwargs.get('get_subp_job_id', get_subp_job_id)
#     save_obj=kwargs.get('save_obj', save_obj)
    
    
#     host=my_socket.determine_computer()
    
           
#     path_code=kwargs.get('path_code')

    
         
#     index=kwargs['index']
    
    
#     data_to_disk.mkdir('/'.join(path_sbatch_out.split('/')[0:-1]))
#     data_to_disk.mkdir('/'.join(path_tee_out.split('/')[0:-1]))

#     save_obj(path_params, path_script, obj)
    ja.save_obj(obj)
    ja.gen_job_script()
    args=ja.get_subp_args()
    
    p=do(*args, **kw)
    ja.process=p   
    
    job_id=ja.get_job_id()
    
#     if process_type=='mpi':
#         subp_job_script_milner(**kwargs)
#         o=generate_milner_bash_script(path_sbatch_err,
#                                       path_sbatch_out,
#                                       path_tee_out,
#                                       path_params,
#                                       path_script,
#                                       path_bash0,
#                                       path_bash,
#                                        **kwargs )
#     args=cb.gen_subp_job_script(**kwargs)
        
#     p=do(*args, **kwargs)
        
# 
#         
#     if host in ['supermicro', 'mikaellaptop', 'thalamus']:
#         num_mpi_task=kwargs.get('num-mpi-task')
#         if num_mpi_task==1:
#             args_call=['python', path_script, path_params]
#         else:
#             args_call=['mpirun', '-np', str(num_mpi_task), 'python', 
#                        path_script, path_params,
#                        '2>&1','|', 'tee', path_tee_out]
#             
#         p=do(path_out, path_err, args_call,  **kwargs)
        
    script_name=obj.get_name()
    
    p=wp(p, job_id, script_name)
    return p




def save_to_database(path_results):
    l=os.listdir(path_results)
    db=None
    for fn in l:
        path=path_results+fn
        if not os.path.isdir(path):
            continue
        if not fn.split('/')[-1][:6]=='script':
            continue
        ll=os.listdir(path)
        for fn2 in ll:
            if fn2.split('.')[-1]!='db_dump':
                continue
            data=data_to_disk.pickle_load(path+'/'+fn2, '.db_dump')
            db_name, db_table, keys_db, values_db, to_binary=data
            to_binary=data[-1]
            values_db=[psycopg2.Binary(a) if tb 
             else a 
             for a,tb in zip(values_db,to_binary)]
            s='Writing to database {} table {} for {}'
            print s.format(db_name, db_table, fn2)
            db=psql.insert(db_name, db_table, keys_db, values_db, db)
            print 'Removing '+path+'/'+fn2
#             subprocess.Popen(['rm', path+'/'+fn2])
    db.close()
    
def loop(*args, **kwargs):
    n, m_list, args_list,  kwargs_list=args 
    

    db_save=kwargs_list[0].get('database_save', False)
    path_results=kwargs_list[0].get('path_results')
    process_type=kwargs_list[0].get('process_type')
    read_subp_jobs=kwargs_list[0].get('read_subp_jobs')
    
    log_file_name=path_results+'/std/job_handler_log'
    data_to_disk.mkdir(path_results+'/std/')
    

    h=job_handler.Handler(loop_time=5,  
                          log_to_file=True,
                          log_file_name=log_file_name,
                          process_type=process_type,
                          read_subp_jobs=read_subp_jobs)
    
    for m in m_list:
        
        q=Queue.Queue()

        for _ in range(m): 
            if not kwargs_list:
                continue
            k=kwargs_list.pop(0)
            if k['active']:
                q.put([args_list.pop(0), k])
            else:
                args_list.pop(0)
    
        h.loop_with_queue(n, q,  epoch, 
                          loop_print=True)
     
             
    if db_save:
        save_to_database(path_results)
            
        
class Job_admin_abstract(object):
    '''
    Callback class needed to be provided when running epoch
    '''
    

    def __init__(self,**kw):
        
        self.index=kw['index']
        self.path_results=kw.get('path_results')
        self.p_par=self.path_results+'params/run{0:0>4}.pkl'.format(self.index)
        self.p_script=dr.HOME_CODE+'/core/toolbox/parallel_excecution/simulation.py'
        
        for key, value in kw.items():
            self.__dict__[key] = value


    
    def get(self,key):
        if key in self.__dict__.keys():
            return self.__dict__[key]
        else: return None
    
    def gen_job_script(self):   
        '''
        Creating a bash file if necessary. 
        ''' 
        pass
     
    def get_subp_args(self):
        '''
        Create arguments for subprocess call
        '''
        raise NotImplementedError 
        
          
    def get_job_id(self):
        '''
        Function that returns a identifier for the process that were started.
        Can be subprocess id or jobb id form supercomputer jobb list.
        Only neccesary for print outs by jobb handler. Can be empty string.
        '''
        return '' 

    def save_obj(self, obj):
        '''
        Function that save object. Can then be loaded by the simuation
        script. Way of passing parameters in to the simulation script
        '''

       
        data_to_disk.pickle_save([obj, 
                                  self.p_script.split('/')[-1]], 
                                  self.p_par)

class Wrapper_process_sbatch():
    
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
        jobs=job_handler.read_subp_jobs_milner()()
        if  self.job_id in jobs:
            return None
        else:
            return 1
    
class Wrapper_process_batch():
    
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
               
class Job_admin_sbatch(Job_admin_abstract):
    def __init__(self,**kw):
        
        index=kw.get('index') #simulation index
#         path_code=kw.get('path_code')
        pr=kw.get('path_results')
    
        self.local_threads=10
    
        self.p_subp_out=pr+"/std/subp/out{0:0>4}".format(index)
        self.p_subp_err=pr+'/std/subp/err{0:0>4}'.format(index)
        self.p_sbatch_out=pr+"/std/sbatch/out{0:0>4}".format(index)
        self.p_sbatch_err=pr+'/std/sbatch/err{0:0>4}'.format(index)
        self.p_tee_out=pr+'/std/tee/out{0:0>4}'.format(index)
        self.p_par=pr+'/params/run{0:0>4}.pkl'.format(index)
        self.p_script=dr.HOME_CODE+'/core/toolbox/parallel_excecution/simulation.py'
        self.p_bash0=dr.HOME_CODE+'/core/toolbox/parallel_excecution/jobb0_milner.sh'
        self.p_bash=pr+'/jobbs/jobb_{0:0>4}.sh'.format(index)
        
        data_to_disk.mkdir('/'.join(self.p_subp_out.split('/')[0:-1]))
        data_to_disk.mkdir('/'.join(self.p_sbatch_out.split('/')[0:-1]))
        data_to_disk.mkdir('/'.join(self.p_tee_out.split('/')[0:-1]))
                       
        for key, value in kw.items():
            self.__dict__[key] = value
            
    
    
    def gen_job_script(self, **kw):
        '''
        Creating a bash file, out and errr for subprocess call as well
        as the parameters for the subprocesses call. 
        
        Returns:
        path out
        path err
        *subp call, comma seperate inputs (se code) 
        '''
#         home=kw.get('home')
#         index=kw.get('index') #simulation index
# #         path_code=kw.get('path_code')
#         path_results=kw.get('path_results')
    
#         p_subp_out=path_results+"std/subp/out{0:0>4}".format(index)
#         p_subp_err=path_results+'std/subp/err{0:0>4}'.format(index)
#         p_sbatch_out=path_results+"std/sbatch/out{0:0>4}".format(index)
#         p_sbatch_err=path_results+'std/sbatch/err{0:0>4}'.format(index)
#         p_tee_out=path_results+'std/tee/out{0:0>4}'.format(index)
#         p_par=path_results+'params/run{0:0>4}.pkl'.format(index)
#         p_script=dr.HOME_CODE+'/core/toolbox/parallel_excecution/simulation.py'
#         p_bash0=dr.HOME_CODE+'/core/toolbox/parallel_excecution/jobb0_milner.sh'
#         p_bash=path_results+'/jobbs/jobb_{0:0>4}.sh'.format(index)
    

    
        
        kw_bash={'home':dr.HOME,
                 'hours':'00',
                 'deptj':1,
                 'job_name':'dummy_job',
                 'cores_hosting_OpenMP_threads':40/self.local_threads,
                 'local_num_threads':self.local_threads, 
                 'memory_per_node':int(819*self.local_threads),
                 'num-mpi-task':40/self.local_threads,
                 'num-of-nodes':40/40,
                 'num-mpi-tasks-per-node':40/self.local_threads,
                 'num-threads-per-mpi-process':self.local_threads, 
                 'minutes':'10',
                 'path_sbatch_err':self.p_sbatch_err,
                 'path_sbatch_out':self.p_sbatch_out,
                 'path_tee_out':self.p_tee_out,
                 'path_params':self.p_par,
                 'path_script':self.p_script,
                 'seconds':'00',
                 
            }
        kw_bash.update(kw) 
        make_bash_script(self.p_bash0, self.p_bash, **kw_bash) #Creates the bash file 
        
    def get_subp_args(self):
        args=[self.p_subp_out, self.p_subp_err,'sbatch', self.p_bash]
        
        return args
     
    def get_job_id(self, **kw):
        '''
        Function that returns a identifier for the process that were started.
        Can be subprocess id or jobb id form supercomputer jobb list.
        Only neccesary for print outs by jobb handler. Can be empty string.
        '''
        time.sleep(1)
        text=data_to_disk.text_load(self.p_subp_out)
        
        i=0
        while not text and i<10:
            time.sleep(1) # wait for file to be populated
            text=data_to_disk.text_load(self.p_subp_out)
            i+=1
            
        job_id=int(text.split(' ')[-1])
        
        return job_id    

 
class Job_admin_batch(Job_admin_abstract): 
    
    def __init__(self,**kw):
        
        index=kw.get('index') #simulation index
        pr=kw.get('path_results')
 
        self.p_subp_out=pr+"/std/subp/out{0:0>4}".format(index)
        self.p_subp_err=pr+'/std/subp/err{0:0>4}'.format(index)
        self.p_tee_out=pr+'/std/tee/out{0:0>4}'.format(index)
        self.p_par=pr+'/params/run{0:0>4}.pkl'.format(index)
        self.p_script=dr.HOME_CODE+'/core/toolbox/parallel_excecution/simulation.py'
        self.p_bash0=dr.HOME_CODE+'/core/toolbox/parallel_excecution/jobb0_supermicro.sh'
        self.p_bash=pr+'/jobbs/jobb_{0:0>4}.sh'.format(index)     
                
        data_to_disk.mkdir('/'.join(self.p_subp_out.split('/')[0:-1]))
        data_to_disk.mkdir('/'.join(self.p_tee_out.split('/')[0:-1]))
        data_to_disk.mkdir('/'.join(self.p_bash.split('/')[0:-1]))
            
        for key, value in kw.items():
            self.__dict__[key] = value
    
    def gen_job_script(self, **kw):

        kw_bash={
                 'path_tee_out':self.p_tee_out,
                 'path_params':self.p_par,
                 'path_script':self.p_script,  
                 }
        kw_bash.update(kw) 
        
        make_bash_script(self.p_bash0, self.p_bash, **kw_bash)
    
    def get_subp_args(self):
        args=[self.p_subp_out, self.p_subp_err, self.p_bash]
        
        return args
    
    def get_job_id(self):
        '''
        Function that returns a identifier for the process that were started.
        Can be subprocess id or jobb id form supercomputer jobb list.
        Only neccesary for print outs by jobb handler. Can be empty string.
        '''
        return self.process.pid   

class Job_admin_mpi_python(Job_admin_abstract):
    
    def __init__(self, **kw):
        
        index=kw.get('index') #simulation index
#         path_code=kw.get('path_code')
        pr=kw.get('path_results')
        self.num_mpi_task=kw.get('num-mpi-task')
#         self.local_threads=10
    
        self.p_subp_out=pr+"/std/subp/out{0:0>4}".format(index)
        self.p_subp_err=pr+'/std/subp/err{0:0>4}'.format(index)
        self.p_par=pr+'/params/run{0:0>4}.pkl'.format(index)
        self.p_script=dr.HOME_CODE+'/core/toolbox/parallel_excecution/simulation.py'     

        data_to_disk.mkdir('/'.join(self.p_subp_out.split('/')[0:-1]))
     
        for key, value in kw.items():
            self.__dict__[key] = value

        
    def get_subp_args(self):

        if self.num_mpi_task==1:
            args_call=['python', self.p_script, self.p_par]
        else:
            args_call=['mpirun', '-np', str(self.num_mpi_task), 'python', 
                       self.p_script, self.p_par]
            
        args=[self.p_subp_out, self.p_subp_err]+args_call
        
        return args
    
    def get_job_id(self):
        '''
        Function that returns a identifier for the process that were started.
        Can be subprocess id or jobb id form supercomputer jobb list.
        Only neccesary for print outs by jobb handler. Can be empty string.
        '''
        return self.process.pid         

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
    
    def get_name(self):
        return 'mockup_class'
    
    def do(self):
        import pickle
        f=open(self.path,'wb')
        print f
        pickle.dump([1], f, -1)
        f.close()

import unittest
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
        
    def create_job_admin(self, job_admin, host):
        d={'job_admin':Job_admin_batch,
           'index':0,
           'num-mpi-task':2,
           'path_results':self.p_out_data(host),
           'wrapper_process':Wrapper_process_batch,
           }
                 
        cb=job_admin(**d)
        self.clear_paths(cb)
        return cb     
       
    def test_do_mpi_python(self):
        host='mpi_python'
        cb=self.create_job_admin(Job_admin_mpi_python, host)
        obj=self.create_obj(host)
        cb.save_obj(obj)
        args=cb.get_subp_args()
        
        p=do(*args,**{'debug':False})        
        p.wait()
          
        l=data_to_disk.pickle_load(self.p_out_data(host)+'/data_out.pkl')
        self.assertListEqual(l, [1])
         
        self.assertTrue(os.path.isfile(cb.p_subp_out))
        self.assertTrue(os.path.isfile(cb.p_subp_err))

    def test_do_batch(self):
        host='batch'
        cb=self.create_job_admin(Job_admin_batch, host)
        obj=self.create_obj(host)

        cb.save_obj(obj)
        cb.gen_job_script()
        
        args=cb.get_subp_args()
        
        p=do(*args,**{'debug':False})       
        p.wait()       
         
        time.sleep(1) 
          
        l=data_to_disk.pickle_load(self.p_out_data(host)+'/data_out.pkl')
        self.assertListEqual(l, [1])
         
        self.assertTrue(os.path.isfile(cb.p_subp_out))
        self.assertTrue(os.path.isfile(cb.p_subp_err))
        self.assertTrue(os.path.isfile(cb.p_tee_out))
           
 
    def test_epoch_mpi_python(self):
        host='mpi_python2'
        obj=self.create_obj(host)
        kw={'job_admin':Job_admin_mpi_python,
            'index':0,
             'num-mpi-task':2,
             'path_results':self.p_out_data(host),
             'wrapper_process':Wrapper_process_batch,
             }
        args=[obj, kw]
        out=epoch(*args)
        print out
            
    def test_epoch_batch(self):
        host='batch2'
        obj=self.create_obj(host)
        kw={'job_admin':Job_admin_batch,
            'index':0,
             'num-mpi-task':2,
             'path_results':self.p_out_data(host),
             'wrapper_process':Wrapper_process_batch,
             }
        args=[obj, kw]
        out=epoch(*args)
        print out
                                     
if __name__ == '__main__':
    d={TestModuleFuncions:[
                            'test_do_mpi_python',
                            'test_do_batch',
                              'test_epoch_mpi_python',
                              'test_epoch_batch',

                           ]} 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # To run only a single specific test you can use:
    # python -m unittest parallel_excecution.TestModuleFuncions.test_do_milner()
    

