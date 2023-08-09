"""
NeuroTools.random
=====================

A set of classes representing statistical distributions, with an interface that
is compatible with the ParameterSpace class in the parameters module.

Classes
-------

GammaDist   - gamma.pdf(x,a,b) = x**(a-1)*exp(-x/b)/gamma(a)/b**a
NormalDist  - normal distribution
UniformDist - uniform distribution

"""

from NeuroTools import check_dependency

import numpy, numpy.random

  
class ParameterDist(object):

    def __init__(self,**params):
        self.params = params
        self.dist_name = 'ParameterDist'
    
    def __repr__(self):
        if len(self.params)==0:
            return '%s()'% (self.dist_name,)
        s = '%s('% (self.dist_name,)
        for key in self.params:
            s+='%s=%s,' % (key,str(self.params[key]))
        return s[:-1]+')'

    def next(self,n=1):
        raise NotImplementedError('This is an abstract base class and cannot be used directly')

    def from_stats(self,vals,bias=0.0,expand=1.0):
        self.__init__(mean=numpy.mean(vals)+bias, std=numpy.std(vals)*expand)

    def __eq__(self, o):
        # should we track the state of the rng and return False if it is different between self and o?
        if (type(self) == type(o) and
            self.dist_name == o.dist_name and
            self.params == o.params):
            return True
        else:
            return False

class GammaDist(ParameterDist):
    """
    gamma.pdf(x,a,b) = x**(a-1)*exp(-x/b)/gamma(a)/b**a

    Yields strictly positive numbers.
    Generally the distribution is implemented by scipy.stats.gamma.pdf(x/b,a)/b
    For more info, in ipython type:
    >>> ? scipy.stats.gamma 

    """
    
    def __init__(self,mean=None,std=None,repr_mode='ms',**params):
        """
        repr_mode specifies how the dist is displayed,
        either mean,var ('ms', the default) or a,b ('ab')
        """

        if check_dependency('scipy'):
            self.next = self._next_scipy

        self.repr_mode = repr_mode
        if 'm' in params and mean==None:
            mean = params['m']
        if 's' in params and std==None:
            std = params['s']

        # both mean and std not specified
        if (mean,std)==(None,None):
            if 'a' in params:
                a = params['a']
            else:
                a = 1.0
            if 'b' in params:
                b = params['b']
            else:
                b = 1.0
        else:
            if mean==None:
                mean = 0.0
            if std==None:
                std=1.0
            a = mean**2/std**2
            b = mean/a    
        ParameterDist.__init__(self,a=a,b=b)
        self.dist_name = 'GammaDist'

    def _next_scipy(self,n=1):
        import scipy.stats
        return scipy.stats.gamma.rvs(self.params['a'],size=n)*self.params['b']
    def _next_no_scipy(self,n=1):
        raise Exception('Error scipy was not found at import time.  GammaDist realization disabled.')

    next = _next_no_scipy
    
    def mean(self):
        return self.params['a']*self.params['b']

    def std(self):
        return self.params['b']*numpy.sqrt(self.params['a'])

    def __repr__(self):
        if self.repr_mode == 'ms':
            return '%s(m=%f,s=%f)' % (self.dist_name,self.mean(),self.std())
        else:
            return '%s(a=%f,b=%f)' % (self.dist_name,self.params['a'],self.params['b'])
        

class NormalDist(ParameterDist):
    """
    normal distribution with parameters
    mean + std
    
    """
    
    def __init__(self,mean=0.0,std=1.0):
        ParameterDist.__init__(self,mean=mean,std=std)
        self.dist_name = 'NormalDist'
        
    def next(self,n=1):
        return numpy.random.normal(loc=self.params['mean'],scale=self.params['std'],size=n)
        

class UniformDist(ParameterDist):
    """
    uniform distribution with min,max
    """

    def __init__(self,min=0.0,max=1.0, return_type=float):
        ParameterDist.__init__(self,min=min,max=max)
        self.dist_name = 'UniformDist'
        self.return_type = return_type
        
    def next(self,n=1):
        vals = numpy.random.uniform(low=self.params['min'],high=self.params['max'],size=n)
        if self.return_type != float:
            vals = vals.astype(self.return_type)
        return vals

    def from_stats(self,vals,bias=0.0,expand=1.0):
        mn = numpy.min(vals)
        mx = numpy.max(vals)
        center = 0.5*(mx+mn)+bias
        hw = 0.5*(mx-mn)*expand
        self.__init__(min=center-hw,max=center+hw)
