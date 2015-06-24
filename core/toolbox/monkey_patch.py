'''
Created on Sep 27, 2014

@author: mikael
'''

import sys
import types

from toolbox import my_socket

PATCH_MODULE='toolbox.monkey_patch_empty'

def patch_nest():
    _=__import__(PATCH_MODULE)
    
    _patch_nest(sys.modules[PATCH_MODULE] )
    
    modules=['nest', 'my_nest', 'nest.topology',
             'nest.pynestkernel']
    for module in modules:
        sys.modules[module] = sys.modules[PATCH_MODULE]
        
def _patch_nest(target):
    def sr(target, *args, **kwargs):
        pass
        
    def version(target, *args, **kwargs):
        return 'NEST 2.2.2'

    def Models(target, *args, **kwargs):
        return []


    def Install(target, *args, **kwargs):
        pass
    
    def GetDefaults(target, *args, **kwargs):
        return {'receptor_types':{'AMPA_1':0,
                                  'AMPA_2':0,
                                  'NMDA_1':0,
                                  'NMDA_2':0,
                                  'GABAA_1':0,
                                  'GABAA_2':0,
                                  'GABAA_3':0}}
        
    def GetKernelStatus(target, *args, **kwargs):
        return 1
    
    target.GetKernelStatus = types.MethodType(GetKernelStatus, target)
    target.sr = types.MethodType(sr, target)
    target.version = types.MethodType(version, target)
    target.Models = types.MethodType(Models, target)
    target.Install = types.MethodType(Install, target)
    target.GetDefaults = types.MethodType(GetDefaults, target)
    
    d={'pushsli':None,
       'runsli':None}
    target.pynestkernel = types.ClassType('pynestkernel',(),d)
    
def patch_NeuroTools():
    _=__import__(PATCH_MODULE)
    
    _patch_NeuroTools(sys.modules[PATCH_MODULE] )
        
    modules=['NeuroTools', 'NeuroTools.stgen', 'NeuroTools.io',
             'NeuroTools.signals', 'NeuroTools.plotting']
    for module in modules:
        sys.modules[module] = sys.modules[PATCH_MODULE]
    
def _patch_NeuroTools(target):     
    
#     return
    
    def StGen():
        pass 

    def StandardPickleFile():
        pass 

    def signals():
        pass      
 
    def get_display():
        pass    

    def set_labels():
        pass   

    def set_axis_limits():
        pass   
    target.StGen = types.MethodType(StGen, target)
    target.StandardPickleFile = types.MethodType(StandardPickleFile, target)
    target.signals = types.MethodType(signals, target)
    target.ConductanceList = types.ClassType('ConductanceList',(),{})
    target.CurrentList = types.ClassType('CurrentList',(),{})
    target.VmList = types.ClassType('VmList',(),{})
    target.SpikeList = types.ClassType('SpikeList',(),{})
    target.get_display = types.MethodType(get_display, target) 
    target.set_labels = types.MethodType(set_labels, target) 
    target.set_axis_limits = types.MethodType(set_axis_limits, target) 


def patch_mpi4py():
    _=__import__(PATCH_MODULE)
    
    _patch_mpi4py(sys.modules[PATCH_MODULE] )
        
    modules=['mpi4py', 'mpi4py.MPI']
    for module in modules:
        sys.modules[module] = sys.modules[PATCH_MODULE]

def _patch_mpi4py(target): 
    class COMM_WORLD():
        size=1
    
    def barrier(target, *args, **kwargs):
        pass
    
    def bcast(target, *args, **kwargs):
        pass
    
    d={'size':1,
       'rank':0,
       'bcast':types.MethodType(bcast, target),
       'barrier': types.MethodType(barrier, target)}  

    dd={'COMM_WORLD': types.ClassType('COMM_WORLD',(),d)}
    target.MPI = types.ClassType('MPI',(),dd)

def patch_for_milner():
    
    if my_socket.determine_host()=='milner_login':
        print 'Monkey patch:', my_socket.determine_host()
        patch_nest()
        patch_NeuroTools()
        patch_mpi4py()

import unittest
class TestModuleFuncions(unittest.TestCase):

    def setUp(self):
        pass
                  
    def test_patch_nest(self):
        a=True
        try:
            patch_nest()
            
            import numpy #need to imprt before nest
            import nest
            import my_nest
            import parallelization
            import misc
            from toolbox.network import default_params
        except Exception, err:
            a=False
            print err
            
        self.assertTrue(a)
            
    def test_patch_NeuroTools(self):
        a=True
        try:
            patch_NeuroTools()
            
            import my_population #need to imprt before nest
            import my_signals
            import misc

        except Exception, err:
            a=False
            print err            
            
        self.assertTrue(a)

    def test_patch_mpi4py(self):
        a=True
        try:
            patch_mpi4py()
            
#             import numpy
#             import nest
            import parallelization
            import my_nest
            from toolbox import misc
            from toolbox import data_to_disk 

        except Exception, err:
            a=False
            print err            
            
        self.assertTrue(a)
        
if __name__ == '__main__':
    d={TestModuleFuncions:[
                           'test_patch_nest',
                           'test_patch_NeuroTools',
                           'test_patch_mpi4py',
                           ]} 
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    




