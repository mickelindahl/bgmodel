'''
Created on Aug 27, 2013

@author: lindahlm
'''



import inspect
import types

class Test(object):
    def methodOne(self):
        print('one')
    def methodTwo(self):
        print('two')

a = Test()    
methodList = [n for n, v in inspect.getmembers(a, inspect.ismethod)
              if isinstance(v,types.MethodType)]

for methodname in methodList:
    func=getattr(a,methodname)
    func()
    
    