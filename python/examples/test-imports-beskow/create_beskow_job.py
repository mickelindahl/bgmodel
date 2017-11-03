# Create by Mikael Lindahl on 4/27/17.
from core.parallel_excecution import Job_admin_sbatch
import os
import pprint

pp = pprint.pprint

size = 3000

root_model = os.getenv('BGMODEL_HOME') #'/cfs/klemming/nobackup/b/belic/mikael/bgmodel' #os.getenv('BGMODEL_HOME')

job_name='test-imports-beskow'

base_sim = os.path.join( root_model, 'python/examples/eneuro')
base_core = os.path.join(root_model, 'python/core')
base_result = os.path.join(root_model, 'results/examples',job_name, str(size))
nest_installation_path = os.path.join(root_model, 'nest/dist/install/nest-simulator-2.12.0')

kw = {
    'nest_installation_path':nest_installation_path,
    'root_model':root_model,
    'hours':'00',
    'minutes':'10',
    'seconds':'00',
    'path_results': '/home/mikael/git/bgmodel/results/jobs',
    'local_threads': 20,
    'job_name': job_name,
    'p_subp_out': os.path.join(base_result, "std/subp/out"),
    'p_subp_err': os.path.join(base_result, "std/subp/err"),
    'p_sbatch_out': os.path.join(base_result, "std/sbatch/out"),
    'p_sbatch_err': os.path.join(base_result, "std/sbatch/err"),
    'p_tee_out': os.path.join(base_result, "std/tee/err"),
    'p_par': os.path.join(base_result, "params/run"),
    'p_script': os.path.join(base_sim, "main.py " + str(size)),
    'p_bash0': os.path.join(os.getenv('BGMODEL_HOME'), 'python/core/parallel_excecution/job0-beskow-2.12.0.sh'),
    'p_bash': os.path.join(os.getenv('BGMODEL_HOME'), "python/examples/test-imports-beskow/jobs/job-"+job_name+'-' + str(size) + '.sh')
}
job = Job_admin_sbatch(**kw)
job.gen_job_script()
