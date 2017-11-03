# Create by Mikael Lindahl on 4/27/17.
from core.job_creator import JobAdminSbatch
import os
import pprint

pp = pprint.pprint

size = 3000

root_model = '/cfs/klemming/nobackup/b/belic/mikael/bgmodel' #os.getenv('BGMODEL_HOME')

base_sim = os.path.join( root_model, 'python/examples/eneuro')
base_core = os.path.join(root_model, 'python/core')
base_result = os.path.join(root_model, 'results/examples/eneuro', str(size))
nest_installation_path = os.path.join(root_model, 'nest/dist/install/nest-simulator-2.12.0')

kw = {
    'nest_installation_path':nest_installation_path,
    'time_hours':'00',
    'time_minutes':'10',
    'time_seconds':'00',
    'path_results': '/home/mikael/git/bgmodel/results/jobs',
    'local_threads': 20,
}

job = JobAdminSbatch(**kw)
s = job.gen_job_script(mode='string')

print(s)
