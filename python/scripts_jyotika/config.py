'''
Created on May 1, 2015

@author: mikael
'''

import subprocess
import time
from toolbox import data_to_disk
from toolbox import directories as dr
from toolbox import job_handler
from toolbox.parallel_excecution import Job_admin_abstract, make_bash_script

def read_subp_jobs_hambach():

    p=subprocess.Popen(['qstat','-u', 'bahuguna'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    out, _ = p.communicate()

    current_jobbs=[]
    for i, row in enumerate(out.split('\n')):
	if i <5:
		continue
	if len(row) == 0:
		continue
		
        if i==0:
            continue
#        l=[v for v in row.split('.') if v!=''][0:2]
#        if len(l)==2:
	l = row.split('.')
        current_jobbs.append(int(l[0]))

    return set(current_jobbs)

class Ja_hambach(Job_admin_abstract):
    def __init__(self, **kw):
        
        index=kw.get('index') #simulation index
#         path_code=kw.get('path_code')
        pr=kw.get('path_results')
    
#        self.local_threads=10
	    
        self.p_subp_out=pr+"/std/subp/out{0:0>4}".format(index)
        self.p_subp_err=pr+'/std/subp/err{0:0>4}'.format(index)
	'''
        self.p_sbatch_out=pr+"/std/sbatch/out{0:0>4}".format(index)
        self.p_sbatch_err=pr+'/std/sbatch/err{0:0>4}'.format(index)
	'''
        self.p_tee_out=pr+'/std/tee/out{0:0>4}'.format(index)
	
        self.p_par=pr+'/params/run{0:0>4}.pkl'.format(index)
        self.p_script=dr.HOME_CODE+'/core/toolbox/parallel_excecution/simulation.py'
        self.p_bash0=dr.HOME_CODE+'/2ndManuscript/20paramsfree/checkRelSWAACT/jc1NoEqjc2/jobb0_jyotika.sh'
        self.p_bash=pr+'/jobbs/jobb_{0:0>4}.sh'.format(index)
	self.p_qsubout = '/users/bahuguna/2ndManuscript/20paramsfree/checkRelSWAACT/jc1NoEqjc2/'
	
        
#        data_to_disk.mkdir('/'.join(self.p_subp_out.split('/')[0:-1]))
#        data_to_disk.mkdir('/'.join(self.p_sbatch_out.split('/')[0:-1]))
        data_to_disk.mkdir('/'.join(self.p_tee_out.split('/')[0:-1]))
                       
        for key, value in kw.items():
            self.__dict__[key] = value
        self.kw=kw
            
    def gen_job_script(self):
        '''
        Creating a bash file, out and errr for subprocess call as well
        as the parameters for the subprocesses call. 
        
        Returns:
        path out
        path err
        *subp call, comma seperate inputs (se code) 
        '''
        self.local_threads=self.local_threads_milner
        kw_bash={'home':dr.HOME,
                 'hours':'2',
#                 'deptj':1,
                 'job_name':self.job_name,
#                 'cores_hosting_OpenMP_threads':40/self.local_threads,
#                 'local_num_threads':self.local_threads, 
                 'memory_per_node':int(819*self.local_threads),
#                 'num-mpi-task':40/self.local_threads,
		'memsize':'3200mb',
  #               'num-of-nodes':40/40,
 #                'num-mpi-tasks-per-node':40/self.local_threads,
 #                'num-threads-per-mpi-process':self.local_threads, 
                 'minutes':'10',
		
#                 'path_sbatch_err':self.p_sbatch_err,
#                 'path_sbatch_out':self.p_sbatch_out,
                 'path_tee_out':self.p_tee_out,
                 'path_params':self.p_par,
                 'path_script':self.p_script,
		'path_qsubout':self.p_qsubout,
                 'seconds':'00',
                 
            }
        kw_bash.update(self.kw) 
        make_bash_script(self.p_bash0, self.p_bash, **kw_bash) #Creates the bash file 
        
    def get_subp_args(self):
        args=[self.p_subp_out, self.p_subp_err,'qsub', self.p_bash]
        
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
            
        job_id=int(text.split('.')[0])
        
        return job_id      

class Ja_else(Job_admin_abstract):
    
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

class Wp_hambach():
    
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
        jobs=read_subp_jobs_hambach()
        if  self.job_id in jobs:
            return None
        else:
            return 1
       
 
class Wp_else():
    
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
