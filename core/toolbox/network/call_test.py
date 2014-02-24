'''
Created on Feb 13, 2014

@author: lindahlm
'''
class Call():

    def __init__(self, method, *args, **kwargs):

        self.method=method
        self.args=args
        self.kwargs=kwargs  
        self.mul=None 
        self.sub=None     
       
    def do(self, obj):
        
        call=getattr(obj, self.method)
        
    
        if not self.mul==None:
            return call(*self.args, **self.kwargs)*self.mul
        elif not self.sub==None:
            return call(*self.args, **self.kwargs)-self.sub
        else:
            return call(*self.args, **self.kwargs)
        
    def __repr__(self):
        return self.__class__.__name__ +':'+self.method       

    def __eq__(self, other):
        if hasattr(other, 'method'):
            return self.method==other.method
        else:
            False
            
    def __mul__(self, other):
        self.mul=other
        return self

#    def __rsub_(self, other):
#        self.sub=other
                    
    def __sub__(self, other):
        self.sub=other
        return self
        
    def __ne__(self, other):
        if hasattr(other, 'method'):
            return self.method!=other.method
        else:
            True
            

def test():
    return 3

class Double(object) :
    def __init__(self, nr) :
        self.value = nr

    def __add__(self, value) :
        return self.value + 2 * value

    def __sub__(self, value) :
        return self.value - 2 * value

    def __mul__(self, value) :
        return self.value * (2*value)

    def __div__(self, value) :
        return self.value / (2*value)

    def __cmp__(self, value) :
        if self.value < (value*2) :
            return -1
        elif self.value == value*2 :
            return 0
        else :
            return 1
print dir()
print __package__
print __name__
print __file__
d = Double(11)
d.value = 10
print d - 2
print d + 2

print d * 2
print d / 1

if d < 5 : print "Less than 5"
else : print "Not less than 5"

d.value = 10
if d == 5 :
       print "Yes!"
else :
       print "no..."
obj=__import__('call_test')

c=Call(*['test'])
print c.do(obj)
print c*3
print c.do(obj)

