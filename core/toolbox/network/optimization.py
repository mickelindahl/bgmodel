'''
Created on Jan 23, 2014

@author: lindahlm
'''


import numpy
import scipy.optimize as opt
from toolbox import data_to_disk, misc
from toolbox.misc import Stop_stdout, Stopwatch
import unittest
from toolbox.network.engine import Network_list
import sys

class Fmin(object):
    
    def __init__(self, name, **kwargs):
   
        self.call_get_x0=kwargs.get('call_get_x0', None)
        self.call_get_error=kwargs.get('call_get_error', None)            
        self.data={'fopt':[],
                   'ftol':[],
                   'xopt':[],
                   'xtol':[],
                   'funcalls':None,
                   'warnflag':None,
                   'allvecs':[]}
        
        self.kwargs_fmin={'maxiter':40, 
                       'maxfun':40,
                       'full_output':1,
                       'retall':1,
                       'xtol':10.0,
                       'ftol':.5,
                       'disp':0}
        
        self.kwargs_fmin.update(kwargs.get('kwargs_fmin',{}))
        self.name=name
        self.model=kwargs.get('model', None)
        self.verbose=kwargs.get('verbose', True)
        
        assert self.model!=None, 'need to provide model to optimize'
        assert self.call_get_x0!=None, 'need to provide call for get x0'
        assert self.call_get_error!=None, 'need to provide call for get error'

    @property 
    def fsim(self):
        ind = numpy.argsort(self.fopt)
        return numpy.array(self.fopt)[ind[0:3]]
    
    @property
    def sim(self):
        ind = numpy.argsort(self.fopt)
        return numpy.array(self.xopt)[ind[0:3],:]
        
    def __getattr__(self, name):
        if name in self.data.keys():
            return self.data[name]
        else:
            raise AttributeError
               
    def fmin(self):
                        
        self._fmin()    
        return self.data
                       
    def _fmin(self):
        '''
        xtol - difference between x for last and second last iteration
        ftol - same but for error function
        
        The nedler-mead simplex algorithm keeps track of the three 
        best solutions. xtol and ftol is calculated base on the three 
        last best solutions. Best minus the value of the second and 
        third. Max of this.
        '''    
        with Stopwatch('\nOptimizing...'):    
            
            out=opt.fmin(self.fmin_error_fun, self.get_x0(), args=(),
                         **self.kwargs_fmin )
            
            [xopt, fopt, iterations, funcalls , warnflag, allvecs] = out
            
            #print xopt, fopt, allvecs
            self.data['xopt'].append(xopt) # append last
            self.data['fopt'].append(fopt)
            self.data['iterations']=iterations
            self.data['funcalls']=funcalls
            self.data['warnflag']=warnflag
            self.data['allvecs']=allvecs
            
            self.print_last()
#             self.print_summary()
               
    def fmin_error_fun(self, x, *arg):
        with Stop_stdout(not self.verbose):
            e=self.get_error(x)
            e=numpy.array(e)**2
            fopt=numpy.sum(e)
    
            self.data['xopt'].append(numpy.array(x))
            self.data['fopt'].append(fopt)
            
            self.get_tol()     
            self.print_last()
        
        return fopt
    
    def get_path_data(self):
        return self.model.get_path_data()
    
    def get_name(self):
        return self.name
    
    def get_x0(self):
        call=getattr(self.model, self.call_get_x0)
        return call()
    
    def get_error(self, x):
        call=getattr(self.model, self.call_get_error)
        return call(x)
    
    def get_tol(self):
        if len(self.xopt)>2:
            
            xtol=max(numpy.ravel(abs(self.sim[1:]-self.sim[0]))) 
            ftol=max(abs(self.fsim[0]-self.fsim[1:]))
            
            self.xtol.append(xtol)
            self.ftol.append(ftol)
        else:
            self.xtol.append(None)
            self.ftol.append(None)

    def header(self):
        s=('time opt={0:6} xopt={1:10} fopt={2:6} iter={3:6}'
                   +' funcalls={4:6} warnflag={5:2}')
        s=s.format(str(self.data['xopt'][-1]), 
                   str(self.data['fopt'][-1]), 
                   str(self.data['iterations']), 
                   str(self.data['funcalls']), 
                   str(self.data['warnflag']))
        return s
          
    def print_last(self):
        s='{0:<40}{1:<10}{2:<10}{3:<10}'
        if len(self.xopt)==1:
            print ''
            print s.format('xopt', 'xtol', 'fopt', 'ftol')
            
        ind=numpy.argsort(self.fopt)
        xopt=self.xopt[ind[0]]
        xopt=[round(x,1) for x in xopt]     
        print s.format(str(xopt), str(self.xtol[-1])[0:6], 
                       str(self.fopt[ind[0]])[0:6], str(self.ftol[-1])[0:6])
        
    def print_summary(self):
        print ''
        if self.warnflag == 1:
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
        elif self.warnflag == 2:
                print "Warning: Maximum number of iterations has been exceeded"
        else:
                print "Optimization terminated successfully."
        
        #print "         time: %s second" % str(self.time)          
        print "         xopt: %s" % str(self.xopt[-1])     
        print "         xtol: %s" % str(self.xtol[-1])             
        print "         fopt: %f" % self.fopt[-1]
        print "         ftol: %f" % self.ftol[-1]
        print "         Iterations: %d" % self.iterations
        print "         Function evaluations: %d" % self.funcalls
        


def fmin_error_fun_wrap(self, x, *arg):
    with Stop_stdout(not self.verbose):
        e=self.model_wrap.sim_optimization(x)
        e=numpy.array(e)**2
        fopt=numpy.sum(e)

        self.data['xopt'].append(numpy.array(x))
        self.data['fopt'].append(fopt)
        
        self.get_tol()     
        self.print_last()
    
    return fopt


from toolbox.network.engine import Unittest_net
class TestFmin(unittest.TestCase):

    
    
    def setUp(self):
        from default_params import Unittest, Unittest_extend        
        opt={'f':['n1', 'n2'],
             'x':['node.i1.rate', 'node.i2.rate'],
             'x0':[2600., 2400.]}
        
        dic_rep={}
        dic_rep.update({'simu':{'sim_time':1000.0,
                                'sim_stop':1000.0,
                             'mm_params':{'to_file':False, 'to_memory':True},
                             'sd_params':{'to_file':False, 'to_memory':True}},
                             'netw':{'size':20.,
                                     'optimization':opt}})
#        p=Perturbation_list('n1_n4', [['node.n1.rate', 3000.0,'='],
#                                      ['node.n4.rate', 2000.0,'=']])
        net_kwargs={'save_conn':False, 
                'sub_folder':'unit_testing', 
                'verbose':False,
                 'par':Unittest_extend(**{'dic_rep':dic_rep,
                                          'other':Unittest()})}
        name='net1'
        self.Network=Unittest_net
        net=[self.Network(name, **net_kwargs)]
        self.nd=Network_list(net)
        #nd.add(name, **kwargs)
     
        self.net_kwargs=net_kwargs
        self.kwargs={'model':self.nd,
                     'call_get_x0':'get_x0',
                     'call_get_error':'sim_optimization', 
                     'verbose':True,}
        
    def test_1_fmin_error_fun(self):
        
        
        f=Fmin('n1_n2_opt', **self.kwargs)
        e1=f.fmin_error_fun([3100., 3100.])
        e2=f.fmin_error_fun([3100., 3100.])
        e3=f.fmin_error_fun([3000., 3000.])
        
        self.assertAlmostEqual(e1,e2,delta=2)
        self.assertFalse(e2==e3)

    def test_2__fmin(self):
        
        f=Fmin('n1_n2_opt', **self.kwargs)
        h=f._fmin()
        
    def test_3__fmin(self):
        self.nd.append(self.Network('net2', **self.net_kwargs))
        #self.nd.add('net2', **self.net_kwargs)
        f=Fmin('n1_n2_opt', **self.kwargs)
        f._fmin()
        xopt=f.xopt[0]
        self.assertAlmostEqual(xopt[0], xopt[2], delta=50)
        self.assertAlmostEqual(xopt[1], xopt[3], delta=50)

    def test_4__fmin_single_network(self):
        self.kwargs['model']=self.nd.l[0]
        f=Fmin('n1_n2_opt', **self.kwargs)
        h=f._fmin()        
        
if __name__ == '__main__':
    
    test_classes_to_run=[TestFmin,
                       ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)  
    
    #unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()      
    
        