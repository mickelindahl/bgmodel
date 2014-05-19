'''
Created on May 13, 2014

@author: mikael
'''
import os
import pprint
pp=pprint.pprint

def do(fun, l, s, *args):
    try:
        print 'running '+s
        fun(*args)
        l[s]='success'
    except Exception as e:
        
        l[s]='fail '+e.message
        print l[s] 


# def perturbations():
#     
#     d={'nest':{'GA_M1_gaba':{'weight':v1},
#                'GA_M2_gaba':{'weight':v2}
#                }}
    

l={}
pre='python '+os.getcwd()


i=0

for size in [10000.0, 20000.0, 40000.0]:
        from_disk=0
        sim_time=60000.0
        module=__import__('simulate_inhibition_striatum')
        print module
        fun=module.main
        script_name=(__file__.split('/')[-1][0:-3]
                     +'/size_'+str(size)
                     +'_time_'+str(sim_time))
        
        args=[from_disk, script_name, sim_time, size]
        do(fun, l, script_name, *args )
        i+=1

pp(l)