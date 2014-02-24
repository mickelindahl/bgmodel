'''
Created on Aug 7, 2013

@author: lindahlm
'''
import copy
import numpy
from toolbox import my_nest
import pylab
import os
import random
from toolbox import signal_processing as sp
from toolbox import misc, data_to_disk
from toolbox.my_signals import SpikeListMatrix, VmListMatrix 
import unittest


#class Data_entry():
#    pass
    
    


class Data_unit(object):
    '''
    classdocs
    Class that neural data produced
    '''
    
    def __init__(self, name, *args, **kwargs):
        '''
        Constructor
        Wraps SpikeListMatrix and VmListMatrix. Use dependency injection
        as design pattern.
        '''
        
        self._recorded={}
        
        self.ids=None
        d={'coherences':{'ids_pairs':[], 'x':[], 'y':[], 'other':''},
           'firing_rate':{'ids':[], 'x':[], 'y':[]},
           'isi':{'ids':[],  'x':[], 'y':[]},
           'isi_IF':{'ids':[],  'x':[], 'y':[]},
           'mean_rates':{'ids':[],   'x':[], 'y':[]},
           'mean_rate':{ 'ids':[],  'x':[],  'y':[]},
           'psd':{'ids':[],'x':[],'y':[]},
           'phase':{'ids':[],'x':[],'y':[]},
           'phases':{'ids':[],'x':[],'y':[]},
           'raster':{'ids':[],  'x':[], 'y':[],},
           'spike_stats':{'rates':{}, 'isi':{}},
           'voltage_trace':{'ids':[],  'x': [], 'y':[] },
           
            }
        self.data=d
        
        self.name=name

        self.merge_runs=False #sets if spike wrap should be merge over runs   
        self.target_rate=0.0
        
        spike_class=kwargs.get('spike_class', SpikeListMatrix)
        voltage_class=kwargs.get('voltage_class', VmListMatrix)      
        self.wrap={'s':{'obj':None, 'class':spike_class},
                   'v':{'obj':None, 'class':voltage_class}}
    
    
    @property
    def recorded(self):
        if self._recorded=={}:
            for key in self.data.keys():
                self._recorded[key]=False
        
        return self._recorded

    
                                         
    def __getattr__(self, name):
        if name in self.data.keys():
            return self.data[name]
        else:
            raise AttributeError

    
        
    def __getstate__(self):
        #print 'im being pickled'
        return self.__dict__
    def __setstate__(self, d):
        #print 'im being unpickled with these values'
        self.__dict__ = d

    def __repr__(self):
        return self.__class__.__name__+':'+self.name    
        
    def add(self, name, l):
        if name=='spike_signal':
            name='s'
            
        d=self.wrap[name]
        
        if d['obj']==None:
            d['obj']=d['class'](l)
        else:    
            d['obj'].concatenate(l)
     
    def add_data_attr(self, name, attr, val):
        self.data[name][attr]=val
                 

    def compute(self, attr, *args, **kwargs):
           
        d=self._compute(attr, *args, **kwargs)
        d=to_single_dic(d)
        return d
 
    def _compute(self, attr, *args, **kwargs):
        
        ''' Attributes are based on my_signal get functions''' 
        if attr in ['firing_rate',
                    'isi',
                    'psd',
                    'phase', 
                    'phases',
                    'raster',
                    'spike_stats',
                    'isi_IF',
                    'mean_coherence',
                    'mean_rate',
                    'mean_rates', 
                    'phase_diff',
                   ]:
            w=self.wrap['s']['obj']
    
            if attr=='isi_IF':
                attr='isi'
                               
            
            w=self.wrap['s']['obj']  
            
            if self.merge_runs:
                w=w.merge(axis=0)
        if attr in ['voltage_trace',
                    ]:
            w=self.wrap['v']['obj']  



        call=getattr(w, 'get_'+attr)
        return call(*args, **kwargs)    

    def compute_set(self, attr, *args, **kwargs):
        val=self.compute(attr, *args, **kwargs)
        self.set(attr, val)

    
    def get(self, attr, *args, **kwargs):
        keys=kwargs.get('attr_list', self.data[attr].keys())
        args=[]
        for key in keys: 
            if attr in ['mean_rate', 'isi_IF']:
                args.append(self.data[attr][key])
            else:
                args.append(self.data[attr][key].ravel())
                            
        return args   

       
    def get_IF_curve(self, set=0):
        
        if self.isi_IF['y']==[]:
            raise RuntimeError('isis IF are not recorded')
        
        if 'x' not in self.isi_IF.keys():
            raise RuntimeError('need to add x attribute')
                
        isi={'curr':[], 'first':[], 'mean':[], 'last':[]}
        x, y=self.get('isi_IF', attr_list=['x', 'y'])
        
        # To get all aligned
        x,y=to_3d_array(*[x,y])
        y=y.reshape((x.shape[0], x.shape[1]*x.shape[2])) 
        x=x.reshape((x.shape[0], x.shape[1]*x.shape[2]))   
                 
        for xx, yy in zip(x,y):
            #if type(yy)!=list:
            #    yy=[yy]
                
            for xxx,yyy in zip(xx, yy):
                if not yyy.any():
                    isis=[1000000.]
                isi['first'].append( yyy[ 0 ] )            # retrieve first isi
                isi['mean'].append( numpy.mean( yyy ) )   # retrieve mean isi
                isi['last'].append( yyy[ -1 ] )           # retrieve last isi
                isi['curr'].append(xxx)
                
                if isi['last'][-1]==0:
                    print 'je'
            n=len(yy)
           
        for key in isi.keys():
            a=numpy.array(isi[key])
            if a.shape[0]/n>=2:
                a=a.reshape((a.shape[0]/n, n))
            
            
            if key!='curr':
                a=1000./a #Convert to firing rate
            isi[key]=a
        return isi['curr'], isi['first'], isi['mean'], isi['last']
   
 
    def get_mean_rate_error(self, *args, **kwargs):
        e=self.data['mean_rate']['y']-self.target_rate
        return numpy.mean(numpy.mean(e))
            
    def get_spike_stats_text(self, **kwargs):
        
        mr,_,_,CV=self.get_spike_stats(**kwargs)
        s='Rate: {0} (Hz) ISI CV: {1}'
        s=s.format(round(mr,2),round(CV,1))
 
    def get_wrap(self,key):
        w=self.wrap[key]['obj']
        if self.merge_runs:
            w=w.merge(axis=0)
        return w
        
 
    def isrecorded(self, name):
        return self.recorded[name]
        
    def plot_firing_rate(self, ax=None, sets=[], win=100, **k):
        if not ax:
            ax=pylab.subplot(111)

        x, y=self.get('firing_rate', attr_list=['x','y'], )
        x,y=to_2d_array(*[x,y])
        x,y=x.transpose(), y.transpose()
        x, y=get_sets_or_single('firing_rate', sets, *[ x, y])
        y=misc.convolve(y, **{'bin_extent':win, 'kernel_type':'triangle',
                              'axis':1})     
        add_labels(sets, k)
        
        
        plot(ax, x, y, **k)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (spike/s)') 
        ax.legend()

    
    def plot_hist_isis(self, ax=None, sets=[], **k):
        if not ax:
            ax=pylab.subplot(111)

        y,=self.get('isi',  attr_list=['y'])
        #y,=merge_runs(*[y])
        
        y=[reduce(lambda a, b:list(a)+list(b), yy ) for yy in y]
        if not sets:
            y=[reduce(lambda a, b:list(a)+list(b), y)]
        add_labels(sets, k)
        for yy in y:
            ax.hist(numpy.array(yy), **k)
        ax.set_xlabel('Time (ms)')     
        ax.set_ylabel('Count (#)')
               
    def plot_mean_rate(self, ax=None, sets=[],  **k):
        if not ax:
            ax=pylab.subplot(111) 
        
        x,y=self.get('mean_rate', attr_list=['x','y'])
        x,y=get_sets_or_single('mean_rate', sets, *[ x, y])
                
        add_labels(sets,  k)
        
        plot(ax, x, y, **k)
        
        ax.set_ylabel('Frequency (spike/s)') 
        ax.set_xlabel('Stimuli')
        
        
    def plot_IF_curve(self, ax=None, set=0, **k):
        if not ax:
            ax=pylab.subplot(111) 
        c, _, _, lisi=self.get_IF_curve()
        ax.plot(c, lisi, **k)
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Rate (spike/s)') 

    def plot_FF_curve(self, ax=None, sets=[], **k):
        if not self.isrecorded('mean_rate'):
            return       
        if not ax:
            ax=pylab.subplot(111) 
        
        x, y=self.get('mean_rate', attr_list=['x','y'])
        x,y=get_sets_or_single('mean_rate', sets, *[ x, y])
        add_labels(sets,  k)
        plot(ax, x, y, **k)
        ax.set_xlabel('Response (spike/s)') 
        ax.set_ylabel('Stimuli (spike/s)') 

    def plot_voltage_trace(self, ax=None, index=[[0, 0]], **k):
        
        if not ax:
            ax=pylab.subplot(111) 
        x, y=self.get('voltage_trace', attr_list=['x','y'])
        #x, y=merge_runs(*[x, y]) 
        
        for i,j in index:
            ax.plot(x[i][j], y[i][j], **k)
        ax.set_xlabel('Time (ms)') 
        ax.set_ylabel('Membrane potential (mV)') 

    def reshape(self, *args, **kwargs):
        return misc.vector2matrix(*args, **kwargs)      
        
    
    def sample(self, n_sample, seed=1):
        random.seed(seed)
        ids=numpy.unique(self.rasters[1])
        if len(ids)<n_sample:
            sample=ids
        else:
            sample=random.sample(ids, n_sample)
        #sample=random.sample(self.raster_ids, n_sample)
    
        truth_val=numpy.zeros(len(self.rasters[1] ))==1
        for s in sample:
            truth_val+=self.rasters[1] == s           
        
        self.truth_val_sample=truth_val
        self.idx_sample=self.rasters[1,truth_val]
        
        if len(self.truth_val_sample):
            return self.rasters[:, self.truth_val_sample ]
        else: 
            return []

    def set(self, attr, val):   
        
        if hasattr(self, attr):
            if attr in self.data.keys():
                cond=(sorted(getattr(self, attr).keys())
                      ==sorted(val.keys()))
                assert cond, 'keys missing in data for attr {}'.format(attr)
                self.data[attr]=val
                self.recorded[attr]=True
            else:
                setattr(self, attr, val)
        else:
            raise RuntimeError('attr {} do not exist'.format(attr))
 


    def set_stimulus(self, name, attr, vals):
        x=self.data[name][attr]
        out=[]
        assert len(x)==len(vals), 'stim need to be {} long'.format(len(x))
        for v, e in zip(vals, x):
            
            
            
                    
            for i, a in enumerate(e):
                if a.shape:
                    a[:]=v
                else:
                    e[i]=v               
    def set_times(self, times):
        self.times=times

  
class Data_units_dic(object):
    
     
    def __init__(self, **kwargs):
        #OBS, having dic={} do not ensure clearance of the dictionary.
        self.dic=kwargs.get('dic',{})
        self.allowed=allowed_Data_units_dic()
        self.attr=''
        
    @property
    def models(self):
        return sorted(self.dic.keys())

    def __repr__(self):
        return self.__class__.__name__+':'+str([str(d) for d in self.dic])     
    
    def __getstate__(self):
        #print '__getstate__ executed'
        return self.__dict__
    
    def __setstate__(self, d):
        #print '__setstate__ executed'
        self.__dict__ = d               
   
    def __getattr__(self, attr):    
        if attr in self.allowed:
            self.attr=attr
            return self._caller
        else:
            raise AttributeError
            
    def __getitem__(self, model):
        if model in self.models:
            return self.dic[model]
        else:
            raise Exception("Model %d is not present in the Data_units_dic. See models()" %model)

    def __iter__(self):
        for key in self.models:
            yield key, self.dic[key]
            
    def __setitem__(self, model, val):
        assert isinstance(val, Data_unit), "An Data_units_dic object can only contain Data_unit objects"
        self.dic[model] = val        
    
    def _caller(self, *args, **kwargs):
        out=[]
        models=kwargs.get('models', self.models)
        for name, obj in self:
            if not name in models:
                continue
            call=getattr(obj, self.attr)
            out.append(call(*args, **kwargs))
        return out
    
    def add(self, attr, d):
        for key, val in d.items():
            if not key in self.dic.keys():
                self.dic[key]=Data_unit(key)
            self.dic[key].add(attr, val)    
            
    def get_model(self, name):
        return self.dic[name]

            
            
class Dud_list(object):
    def __init__(self, dud_list, **kwargs):
        self.allowed=allowed_Dud_list()
        self.l=dud_list

    def __getattr__(self, name):
        if name in self.allowed:
            self.attr=name
            return self._caller
        else:
            raise AttributeError(name)  

    def __iter__(self):
        
        for val in self.l:
            yield val
    
    def __repr__(self):
        return self.__class__.__name__+':'+str([str(l) for l in self]) 
    
    def _caller(self, *args, **kwargs):
        a=[]
        
        for obj in self:
            call=getattr(obj, self.attr)
            d=call(*args, **kwargs)
            if d:
                a.append(d)               
        return a
                  
                
class Data_units_relation(object):
    '''
    classdocs
    Class that represent coherence between two data units (can be same data units)
    '''
    
        
    def __init__(self, name, du1, du2, *args, **kwargs):
        '''
        Constructor
        '''
        
        d={'mean_coherence':{'ids1':[], 'ids2':[], 'x':[], 'y':[]},
           'phase_diff':{'ids1':[], 'ids2':[], 'x':[], 'y':[]},
            }
           
        self.data=d  
        
        # wraps data units      
        self.du1=du1 
        self.du2=du2
        self.name=name
        
    def compute(self, attr, *args, **kwargs):
        kwargs['other']=self.du2.get_wrap(kwargs.get('wrap_type','s'))
        d=self.du1.compute(attr, *args, **kwargs)
        return d
    
    def compute_set(self, attr, *args, **kwarg):
        d=self.compute(attr, *args, **kwarg)
        self.set(attr, d)

    def get(self, attr, *args, **kwargs):
        keys=kwargs.get('attr_list', self.data[attr].keys())
        args=[]
        for key in keys: 
            args.append(self.data[attr][key].ravel())                            
        return args  


    def plot_mean_coherence(self, ax=None, sets=[], rem_first=True,  **k):
        if not ax:
            ax=pylab.subplot(111)

        x, y=self.get('mean_coherence', attr_list=['x','y'])
        x,y=to_2d_array(*[x,y])
        x,y=x.transpose(), y.transpose()
        x, y=get_sets_or_single('mean_coherence', sets, *[ x, y])
   
        add_labels(sets, k)
        
        if rem_first:
            x,y=x[1:],y[1:]
                
        plot(ax, x, y, **k)
        ax.set_xlabel('Frequency (Hz)') 
        ax.set_ylabel('Coherence') 
        ax.legend()

    def plot_phase_diff(self, ax=None, num=100, sets=[], rem_first=True,  **k):
        if not ax:
            ax=pylab.subplot(111)

        y,=self.get('phase_diff', attr_list=['y'])
        y,=to_2d_array(*[y])
        
        y,=get_sets_or_single('phase_diff', sets, *[ y])
   
        bins=numpy.linspace(-numpy.pi, numpy.pi, num)
        add_labels(sets, k)
        
        if rem_first:
            y=y[1:]
                
        ax.hist(y, bins, **k)
        ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_xlabel('Angle (Rad)') 
        ax.set_ylabel('Count') 
        ax.legend()
    
    
    def set(self, attr, val):   
        
        if hasattr(self, attr):
            setattr(self, attr, val)
        elif attr in self.data.keys():
            for key in self.data[attr].keys():
                self.data[attr][key]=val[key]
        elif attr=='merge_runs':
            self.du1.set(attr,val)
            self.du2.set(attr,val)
        else:
            raise RuntimeError('attr {} do not exist'.format(attr))
            

class Data_units_relation_dic(object):    
    def __init__(self, models_list, **kwargs):
        

        self.dic={}
        for model in models_list:
            self.dic[model]=Data_units_relation(model, **kwargs)
    
        self.relation_list=copy.deepcopy(models_list)
    
        
    def __getitem__(self, relation):
        if relation in self.relation_list:
            return self.dic[relation]
        else:
            raise Exception("Model %d is not present in the Data_units_dic. See models()" %relation)

    def __setitem__(self, i, val):
        assert isinstance(val, Data_unit), "An Data_units_dic object can only contain Data_unit objects"
        self.dic[i] = val    

def add_labels(sets, k):
    if sets:
        l= ['Set {}'.format(s) for s in sets]   
    else:
        l=['Mean']
        
    if 'label' not in k.keys():
        k['label']=l          
 
def allowed_Data_units_dic():
    l=['compute_set',
      'get',
      'get_mean_rate_error',
      'get_spike_stat', 
      'plot_firing_rate',
      'plot_hist_isis',
      'plot_mean_rate',    
      'plot_IF_curve',
      'plot_FF_curve',
      'plot_voltage_trace',
      'set',
      'set_stimulus',
                  ]
    return l

def allowed_Dud_list():
    return allowed_Data_units_dic()
        
def get_sets_or_single(name, sets, *args):
    
    if sets: 
        args=[numpy.array([arg[:,s] for s in sets]).transpose() for arg in args]
    else:
        args=reduce_sets(name, *args)
    return args

def iter2d(m):
    for i in xrange(m.shape[0]):
        for j in xrange(m.shape[1]):
            yield i, j, m[i,j]



def plot(ax, x, y, **k):
    '''
    lines arguments can be provided as lists
    '''   
    lines=ax.plot(x,y)
    
    n=len(lines)
    
    for key, val in k.items():
        if type(val) == list:
            if len(val)==1:
                val=val[0]
            else:
                assert len(val)==n, 'List need to be fo length {}'.format(n)    
                continue
        k[key]=[val]*n
    
    for i, l in enumerate(lines):
        d=[(key, val[i]) for key, val in k.items()]
        kk=dict(d)
        pylab.setp(l,**kk)
    ax.legend()

def ravel(*args):
    out=[]
    for a in args:
        d=numpy.ravel(a)
        if type(d[0])==numpy.ndarray:
            d=numpy.concatenate(list(d))
        out.append(d)
    return out

def reduce_sets(flag, *args):
    if flag in [
                'voltage_trace']:    
        out=[numpy.mean(a, axis=0) for a in args]
    if flag in ['firing_rate',  'mean_rate', 'mean_coherence',
               ]:
        out=[numpy.mean(a, axis=1) for a in args]
    if flag in ['phase_diff']:
        out=[a.ravel() for a in args]
    
    return out

def to_2d_array(*args):
    out=[]
    for a in args:
        a=numpy.array([list(aa) for aa in a])
        out.append(a)
    return out 

def to_3d_array(*args):
    out=[]
    for a in args:
        a=numpy.array([[list(aaa)for aaa in aa] for aa in a])
        out.append(a)
    return out     

def to_single_dic( a):
    keys=a.ravel()[0].keys()
    l=[]
    for key in keys:
        #dtp=type(a.ravel()[0][key])
        l.append(numpy.empty(a.shape, dtype=object))
    
    d=dict(zip(keys, l))
    
    for i, j, _ in iter2d(a):
        for key in keys:
            d[key][i,j]=a[i,j][key]
    return d
from toolbox.my_signals import dummy_data as dd

def dummy_data(**kwargs):
        
        n_runs=kwargs.get('n_runs', 3)
        n_sets=kwargs.get('n_sets', 2)
        n_pop_spk=kwargs.get('n_pop_spk',6)
        n_pop_vm=kwargs.get('n_pop_vm',4)
        shift=kwargs.get('shift',0.)
        
        
        l1,l2=[],[]
        for i in xrange(n_runs):
            s,v=[],[]
            for j in xrange(n_sets):
                kwargs={'run':i, 'set':j, 'n_sets':n_sets, 
                        'stim_time':200.0, 'n_pop':n_pop_spk, 'scale':0.5,
                        'shift':shift,
                        }
                
                s.append(dd('spike',**kwargs))    
                
                kwargs['n_pop']= n_pop_vm
                kwargs['sim_time']=200.0
                v.append(dd('voltage', **kwargs))
            
            l1.append(s)
            l2.append(v)
        return l1,l2
    
def dummy_data_du(**kwargs):
    name=kwargs.get('name','unittest')
    du=Data_unit(name)
    l1,l2=dummy_data(**kwargs)    
    for s, v in zip(l1,l2):
        du.add('s',s) 
        du.add('v',v)   
    return du

def dummy_data_dud(names, **kwargs):
        
        d1,d2={},{}
        dud=Data_units_dic()        
        for name in names:
            kwargs['name']=name
            d1[name], d2[name]=dummy_data(**kwargs)  
           
                
        dud.add('s',d1)
        dud.add('v',d2)
        return dud
        
class TestData_unit(unittest.TestCase):
    def setUp(self):
        self.longMessage=True
        self.n_runs=3

        self.obj=dummy_data_du(**{'n_runs':self.n_runs})
        
#    def test_1_compute_set_get(self):
#        for attr, a, k in [['firing_rate',[100],{'average':True}], 
#                           ['isi', [],{}],
#                           ['mean_rate', [],{}],
#                           ['mean_rates',[],{}],
#                           ['psd',[256,1000.0],{}],
#                           ['phase',[10,20,3,1000.0],{}],
#                           ['phases',[10,20,3,1000.0],{}],
#                           ['raster',[], {}],
#                           ['voltage_trace', [], {}]]:
#
#            self.obj.compute_set(attr,*a,**k)            
#            self.obj.get(attr)
#
#        
#    def test_2_plot_firing_rate(self):
#        self.obj.set('merge_runs', True) 
#        self.obj.compute_set('firing_rate',*[1],**{'average':True})
#
#        pylab.figure()
#        self.obj.plot_firing_rate()    
#        self.obj.plot_firing_rate(sets=range(2))  
#        #pylab.show()
#        
#    def test_3_plot_hist_isis(self):
#        self.obj.set('merge_runs', True) 
#        self.obj.compute_set('isi',*[],**{})
#
#        pylab.figure()
#        self.obj.plot_hist_isis()    
#        #pylab.show()
#
#    def test_4_plot_mean_rate(self):
#        self.obj.compute_set('mean_rate',*[],**{})
#        
#        pylab.figure() 
#        self.obj.plot_mean_rate()
#        self.obj.plot_mean_rate(sets=range(2)) 
#          
#        #pylab.show()
#        
#    def test_5_get_IF_curve(self):
#        self.obj.compute_set('isi_IF',*[],**{})
#        self.obj.set_stimulus('isi_IF', 'x',  range(100,100*(1+self.n_runs),100))
#        self.obj.get_IF_curve()
#        #pylab.show()


    def test_6_plot_IF_curve(self):
        self.obj.compute_set('isi_IF',*[],**{})
        self.obj.set_stimulus('isi_IF', 'x',  range(100,100*(1+self.n_runs),100))
        pylab.figure()      
        self.obj.plot_IF_curve()#**{'color':'b'})
        pylab.show()
 

#    def test_7_plot_FF_curve(self):
#        self.obj.compute_set('mean_rate',*[],**{})
#        self.obj.set_stimulus('mean_rate', 'x',  range(100,100*(1+self.n_runs),100))
#        pylab.figure()      
#        self.obj.plot_FF_curve()#**{'color':'b'})
##        #pylab.show(        d1, d2={}, {}
#
#
#    def test_8_plot_voltage_trace(self):  
#        self.obj.compute_set('voltage_trace',*[],**{})
#        pylab.figure() 
#        self.obj.plot_voltage_trace(**{'index':[[0,0], [1,0], [2,0]]})
#        pylab.show()   
#       
#    def test_9_get_spike_stats(self):
#        self.obj.compute_set('spike_stats',*[],**{})


class TestData_units_dic(TestData_unit):
    def setUp(self):

        self.longMessage=True
        names=['u1','u2']
        self.n_runs=3
        self.obj=dummy_data_dud(names, **{'n_runs':3})


    #Blocked    
    def test_5_get_IF_curve(self):
        pass  

class TestDud_list(TestData_units_dic):
    def setUp(self):
        self.longMessage=True
        names=[['u1','u2'],['u3']]
        self.n_runs=3
        l=[]
        for n in names:
            l.append(dummy_data_dud(n, **{'n_runs':3}))
        self.obj=Dud_list(l)       
         
class TestData_unit_relation(unittest.TestCase):
    def setUp(self):
        self.longMessage=True
        self.n_pop_spk=30
        self.n_pop_vm=12 #speed sensitive

        self.n_sets=3
        self.n_runs=4
        
        kwargs={'n_pop_spk':self.n_pop_spk,
                'n_pop_vm':self.n_pop_vm,
                'n_sets':self.n_sets,
                'n_runs':self.n_runs}

        kwargs['name']='u1'
        du1=dummy_data_du(**kwargs)
        kwargs['name']='u2'
        kwargs['shift']=1.
        du2=dummy_data_du(**kwargs)
        self.dur=Data_units_relation('u1_u2', du1, du2)
    def test_1_compute_set_get(self):
        for attr, a, k in [['mean_coherence',[],{'fs':1000.0,
                                                 'NFFT':256,
                                                 'noverlap':int(256/2),
                                                 'sample':2.}], 
                           ['phase_diff', 
                            [10,20,3,1000.0],
                            {'bin_extent':10.,
                             'kernel_type':'gaussian',
                             'params':{'std_ms':5.,
                             'fs': 1000.0}}],
                           ]:

            self.dur.compute_set(attr,*a,**k)
            self.dur.get(attr)        
            
    def test_2_plot_mean_coherence(self):
        self.dur.set('merge_runs', True)
        self.dur.compute_set('mean_coherence',*[],**{'fs':1000.0,
                                                 'NFFT':256,
                                                 'noverlap':int(256/2),
                                                 'sample':2.} )
        pylab.figure()
        self.dur.plot_mean_coherence()    
        self.dur.plot_mean_coherence(sets=range(3))  
        #pylab.show()

    def test_3_plot_phase_difft(self):
        self.dur.set('merge_runs', True)
        self.dur.compute_set('phase_diff',*[10,20,3,1000.0],
                                                **{'bin_extent':10.,
                                             'kernel_type':'gaussian',
                                             'params':{'std_ms':5.,
                                             'fs': 1000.0}} )
        pylab.figure()
        self.dur.plot_phase_diff()    
        #self.dur.plot_mean_coherence(sets=range(3))  
        #pylab.show()
            
if __name__ == '__main__':
    test_classes_to_run=[TestData_unit,
                         #TestData_units_dic,
                         #TestDud_list,
                         #TestData_unit_relation,
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)
    
    #unittest.main()           
   
           