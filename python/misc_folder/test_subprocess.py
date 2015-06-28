'''
Created on Jul 15, 2014

@author: mikael
'''
path='/home/mikael/git/bgmodel/core_old/misc_folder/test_subprocess/'
from core.data_to_disk import pickle_save

import subprocess


from core_old.misc_folder import test_subprocess_fun
    
    
la=range(4)
lb=range(4)
threads=2   

paths=[]

for i, v in enumerate(zip(la,lb)):
    for thread in range(threads):
        a,b=v
        c=test_subprocess_fun.fin().do
        data=[c, [a], {'b':b}]
        p=path+str(i)
        pickle_save(data, p+str(thread))
    paths.append(p)
        
processes=[]
f=[]
for i, p in enumerate(paths):
    
    #Popen do not block, call does

    f.append(open(path+'out'+str(i), "wb", 0))
    f.append(open(path+'err'+str(i), "wb", 0))
#     
    
    pr=subprocess.Popen(['mpirun', 
                        '-np',
                        str(threads),
                        'python', 
                        'test_subprocess_wrap_fun.py',
                        p],
                        stdout=f[-2],
                        stderr=f[-1])
#     pr.append(p)

for _f in f:
    _f.close()

    