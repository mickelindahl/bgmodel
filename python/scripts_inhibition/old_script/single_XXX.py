'''
Created on Mar 19, 2014

@author: lindahlm

'''
import os

from core import data_to_disk

import pprint
pp=pprint.pprint

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


i=0
cmd_lines=[]
# for from_disk in [0,1,2]:
#     for rand_nodes in [1,0]:
for from_disk, name, rand_nodes in [
#                      ['MSN_D1',1], 
#                      ['MSN_D2',1], 
#                      ['FSN',1], 
             [0, 'GPe_TA',1], 
             [0, 'GPe_TI',1],
             [0, 'SNR',1], 
             [0, 'STN',1],
             [0, 'GPe_TA_diff_rates',1], 
             [0, 'GPe_TI_diff_rates',1],
    
             ]:
    attr='single_'+name
    
    if rand_nodes:
        s='_rand'
    else:
        s=''

    script_name=__file__.split('/')[-1][0:-3]+'/'+name+s
    args=[attr, script_name, 
          str(rand_nodes), script_name, str(from_disk)]
    
    cmd_lines.append(' '.join(args))
#     do(attr, l, script_name, *args )
    i+=1

pp(cmd_lines)
chunked= chunks(cmd_lines,4)
      

from subprocess import Popen, call
from os import mkdir
   
for cmd_lines in chunked:

    for number, line in enumerate(cmd_lines):   
        
        script_name=line.split(' ')[1]
         
        newpath = script_name
        
        if not os.path.isdir('./'+newpath):
            data_to_disk.mkdir('./'+newpath)

            
        cmd = ('python /home/mikael/git/bgmodel/'+
               'scripts_inhibition/do.py ' 
               + line.strip())
        print 'Running %r in %r' % (cmd, newpath)
        
        stdout= newpath+"/stdout.txt"
        stderr=newpath+'/stderr.txt'
        with open(stdout,"wb") as out, open(stderr,"wb") as err:
            Popen(cmd, shell=True, stdout=out, stderr=err)
    
    
    
    