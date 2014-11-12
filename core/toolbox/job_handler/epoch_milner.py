'''
Created on Sep 30, 2014

@author: mikael
'''
import sys
import time
import subprocess

print sys.argv[1:]
path,id,=sys.argv[1:]
id=int(id)
# time.sleep(id*2)
header=('JOBID  USER     ACCOUNT           NAME EXEC_HOST ST     REASON   '
        +'START_TIME     END_TIME  TIME_LEFT NODES   PRIORITY\n')

if id==0:
    row=('28377  tully    (null)       sequences    login2  R       '
         +'None 2014-09-30T1 2014-09-30T1    3:21:03     3       1016\n')
    
if id==1:
    row=('28378  tully    (null)       sequences    login2  R       '
     +'None 2014-09-30T1 2014-09-30T1    3:21:03     3       1016\n')
if id==2:
    row=('28379  tully    (null)       sequences    login2  R       '
     +'None 2014-09-30T1 2014-09-30T1    3:21:03     3       1016\n')

f=open(path+'/data3','r')
rows=f.readlines()
rows.append(row)
f.close()
f=open(path+'/data3','w')
for _row in rows:
    f.write(_row)
f.close()

time.sleep(3)

f=open(path+'/data3','r+')
rows=f.readlines()
rows.remove(row)


f.close()
 
f=open(path+'/data3','w')
for _row in rows:
    f.write(_row)
f.close()

