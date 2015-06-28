'''
Created on Sep 26, 2013

@author: lindahlm
'''
class RevealAccess(disc):
    """A data descriptor that sets and returns values
       normally and prints a message logging their access.
    """

    def __init__(self, initval=None, name='var'):
        self.val = initval
        self.name = name

    def __get__(self, obj, objtype):
        print 'Retrieving', self.name
        return self.val

    def __set__(self, obj, val):
        print 'Updating' , self.name
        self.val = val

class MyClass(object):
    def __init__(self):
        self.x = RevealAccess(10, 'var "x"')
        self.y = 5
    
    
    
x=RevealAccess(10, 'var "x"')    
m = MyClass()
print m.x
print x
#Retrieving var "x"
#10
m.x = 20
x=20
#Updating var "x"
print m.x
#Retrieving var "x"
#20
m.y
