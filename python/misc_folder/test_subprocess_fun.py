'''
Created on Jul 15, 2014

@author: mikael
'''
import time  
class fin():
    
    def __getstate__(self):
        #print '__getstate__ executed'
        return self.__dict__
    
    def __setstate__(self, d):
        #print '__setstate__ executed'
        self.__dict__ = d   
    
    def do(self, a, b=1):
        print 'a+b=',a+b
        time.sleep(3)
        print 'a*b=', a*b