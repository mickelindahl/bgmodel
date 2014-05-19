'''
Created on Aug 12, 2013

@author: lindahlm
'''
import os
import pprint
pp=pprint.pprint

def do(fun, l, s, *args):
#     try:
        print 'running '+s
        fun(*args)
        l[s]='success'
#     except Exception as e:
        
        l[s]='fail '+e.message
        print l[s] 

l={}
pre='python '+os.getcwd()


i=0

for size in [3000.0]:
        from_disk=0
        sim_time=5000.0
        module=__import__('simulate_slow_wave')
        print module
        fun=module.main
        script_name=(__file__.split('/')[-1][0:-3]
                     +'/size_'+str(size)
                     +'_time_'+str(sim_time))
        
        args=[from_disk, script_name, sim_time, size]
        do(fun, l, script_name, *args )
        i+=1

pp(l)
        