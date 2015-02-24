'''
Created on Sep 28, 2014

@author: mikael
'''

from socket import *
import socket

EXCLUDE=['supermicro','mikaellaptop']
    

def determine_host():

    HOST=socket.gethostname().split('.')
    if (len(HOST)==1 and HOST[0] not in EXCLUDE):
        return 'milner'
    if HOST[0][0:6]=='milner':
        return 'milner_login'
    else:
        return HOST[0]

def determine_computer():

    HOST=socket.gethostname().split('.')
    if (len(HOST)==1 and HOST[0] not in EXCLUDE) or (HOST[0][0:6]=='milner'):
        return 'milner'
    else:
        return HOST[0]
    
    
import unittest
class ModuleFunctions(unittest.TestCase):     
    def setUp(self):
        
        pass


        
        
if __name__ == '__main__':
    d={
        ModuleFunctions:[
                         
                         
                    ],

       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)

# conn=psycopg2.connect("dname=test")