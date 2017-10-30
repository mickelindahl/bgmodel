# Create by Mikael Lindahl on 4/27/17.
from core.parallel_excecution import Job_admin_sbatch


kw={
    'path_results':'/home/mikael/git/bgmodel/results/jobs',
    'local_threads':20,
    'job_name':'test'
}
job=Job_admin_sbatch(**kw)
job.gen_job_script()