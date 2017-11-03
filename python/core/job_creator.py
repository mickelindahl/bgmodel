'''
Created on Mar 19, 2014

@author: lindahlm

'''

import datetime
import os
import subprocess
import pprint

pp = pprint.pprint

HOME = os.getenv('BGMODEL_HOME')
HOME_DATA = os.getenv('BGMODEL_HOME_DATA')


def mkdir(path):
    # If a directory does not exist where a file is suppose to be stored  it is created
    path = path.split('/')
    i = len(path)
    while not os.path.isdir('/'.join(path[0:i])):
        i -= 1

    while i != len(path):
        if not os.path.isdir('/'.join(path[0:i + 1])):
            os.mkdir('/'.join(path[0:i + 1]))
        i += 1


def text_load(path):
    f = open(path, 'rb')
    s = f.read()
    f.close()

    return s


def text_save(data, filename):
    if not os.path.isdir(os.path.dirname(filename)):
        mkdir(os.path.dirname(filename))

    f = open(filename, 'wb')  # open in binary mode

    f.write(data)
    f.close()


def make_bash_script(path_bash0, path_bash,mode,  **kwargs):
    s = text_load(path_bash0)
    s = s.format(**kwargs)

    if mode=='string':
        return s

    text_save(s, path_bash)

    p = subprocess.Popen(['chmod', '777', path_bash])
    #     p=subprocess.Popen(['chmod', 'a+x', path_bash])


class JobAdminSbatch(object):
    """Timetable constructor

          Keyword arguments:
              job_name -- name of job
              nest_installation_path -- path to nest installation
              time_seconds -- job length in seconds
              time_minutes -- job length in minutes
              time_hours -- job length in hours
              local_threads -- local number of threads on each node
              max_local_threads -- max number of threads on each node
              cmd -- command to run script
              path_bash_template -- path to template for bash job
              path_bash -- path to store generated bash job
              path_out -- path to stdout
              path_err -- path to stderr
              path_tee -- path to piped output
         """

    def __init__(self, **kw):
        self.cmd = HOME + 'example/eneuro/python main.py 5000'

        fn = '-'.join(self.cmd.split(' ')[-2:])
        fn=''.join(fn.split('.py'))
        now = datetime.datetime.now()
        now = datetime.datetime.strftime(now, '%y%m%d-%H%M%S')

        batch_id = fn + '-' + now

        self.job_name = batch_id
        self.local_threads = 10
        self.max_local_threads = 40

        self.path_bash_template = os.path.join(HOME, 'python/core/job_template/job0-beskow-2.12.0.sh')
        self.path_bash = os.path.join(HOME_DATA, 'job', batch_id + '.out')

        self.path_out = os.path.join(HOME_DATA, 'std/sbatch', batch_id + '.out')
        self.path_err = os.path.join(HOME_DATA, 'std/sbatch', batch_id + '.err')

        self.path_tee = os.path.join(HOME_DATA, 'std/sbatch', batch_id + '.tee')

        mkdir(os.path.dirname(self.path_out))
        mkdir(os.path.dirname(self.path_err))
        mkdir(os.path.dirname(self.path_tee))

        for key, value in kw.items():
            self.__dict__[key] = value

    def gen_job_script(self, mode):
        '''
        Creating a bash file, out and errr for subprocess call as well
        as the parameters for the subprocesses call. 
        
        Returns:
        path out
        path err
        *subp call, comma seperate inputs (se code) 
        '''

        kw_bash = {
            # 'home': dr.HOME,
                   'hours': self.time_hours,
                   'nest_installation_path': self.nest_installation_path,
                   'root_model': HOME,
                   'job_name': self.job_name,

                   'cores_hosting_OpenMP_threads': 40 / self.local_threads,
                   'local_num_threads': self.local_threads,
                   'memory_per_node': int(819 * self.local_threads),
                   'num-mpi-task': 40 / self.local_threads,
                   'num-of-nodes': 40 / 40,
                   'num-mpi-tasks-per-node': 40 / self.local_threads,
                   'num-threads-per-mpi-process': self.local_threads,
                   'minutes': self.time_minutes,
                   'path_sbatch_err': self.path_err,
                   'path_sbatch_out': self.path_out,
                   'path_tee_out': self.path_out,
                   # 'path_params':self.p_par,
                   # 'path_script': self.p_script,
                   'cmd':self.cmd,
                   'seconds': self.time_seconds,

                   }
        # kw_bash.update(kw)
        return make_bash_script(self.path_bash_template, self.path_bash, mode, **kw_bash)  # Creates the bash file
