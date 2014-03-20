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
    
    
class Data_unit_base(object):
    def __init__(self, name, wrap, *args, **kwargs):
        '''
        Constructor
        Wraps SpikeListMatrix or VmListMatrix. Use dependency injection
        as design pattern.
        '''
        
        self._recorded={}
        self.data={}
        self.name=name
        self.merge_runs=False #sets if spike wrap should be merge over runs   
        self.merge_sets=False
        self.target_rate=0.0
        self.wrap=wrap
        
        self._init_extra_attributes( *args, **kwargs)
        
    @property
    def recorded(self):
        if self._recorded=={}:
            for key in self.data.keys():
                self._recorded[key]=False      
        return self._recorded
    
    def __getitem__(self, key):        
        wrap=self.wrap.__getitem__(key)       
        return self.__class__(self.name, wrap)
    
    def __getattr__(self, name):
        if name in self.data.keys():
            return self.data[name]
        elif name in self.wrap.allowed:
            return getattr(self.wrap, name)
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
        
    def add(self, l):
        self.wrap.concatenate(l)
     
    def add_data_attr(self, name, attr, val):
        self.data[name][attr]=val
                 

    def compute(self, attr, *args, **kwargs):      
        d=self._compute(attr, *args, **kwargs)
        return d

    def _compute(self, attr, *args, **kwargs):
        call=getattr(self.wrap, 'get_'+attr)
        return call(*args, **kwargs) 


    def compute_set(self, attr, *args, **kwargs):
        val=self.compute(attr, *args, **kwargs)
        self.set(attr, val)

    
    def get(self, attr, *args, **kwargs):
        keys=kwargs.get('attr_list', self.data[attr].keys())
        args=[]
        for key in keys: 
#             if attr in ['mean_rate', 'isi_IF', 'firing_rate', 'mean_rate_parts']:
                args.append(self.data[attr][key])
#             else:
#                 args.append(self.data[attr][key].ravel())
                            
        return args   

    def isrecorded(self, name):
        return self.recorded[name]
        
    def reshape(self, *args, **kwargs):
        return misc.vector2matrix(*args, **kwargs)      
        
    
    def reset(self, attr):
        self.recorded[attr]=False

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
        
    def set_times(self, times):
        self.times=times   
                
class Mixin_spk(object):
    '''
    classdocs
    Class that neural data produced
    '''
    
    def _init_extra_attributes(self, *args, **kwargs):
        '''
        Constructor
        Wraps SpikeListMatrix and VmListMatrix. Use dependency injection
        as design pattern.
        '''
    
        d={'firing_rate':{'ids':[], 'x':[], 'y':[]},
           'isi':{'ids':[],  'x':[], 'y':[]},
           'isi_parts':{'ids':[],  'x':[], 'y':[]},
           'mean_rate':{ 'ids':[],  'x':[],  'y':[]},
           'mean_rate_parts':{'ids':[],   'x':[], 'y':[]},
           'mean_rates':{ 'ids':[],  'x':[],  'y':[]},
           'psd':{'ids':[],'x':[],'y':[]},
           'phase':{'ids':[],'x':[],'y':[]},
           'phases':{'ids':[],'x':[],'y':[]},
           'raster':{'ids':[],  'x':[], 'y':[],},
           'spike_stats':{'rates':{}, 'isi':{}},
            }
        self.data=d
        

               
    def get_IF_curve(self):
        
        if self.isi_parts['y']==[]:
            self.compute_set('isi_parts',*[],**{})
        
        isi={'curr':[], 'first':[], 'mean':[], 'last':[]}
        x, y=self.get('isi_parts', attr_list=['x', 'y'])
        
        y=y.ravel()
        for xx, yy in zip(x,y):
            #if type(yy)!=list:
            #    yy=[yy]
                
            for yyy in yy:
                if not yyy.any():
                    yyy=[1000000.]
                isi['first'].append( yyy[ 0 ] )            # retrieve first isi
                isi['mean'].append( numpy.mean( yyy ) )   # retrieve mean isi
                isi['last'].append( yyy[ -1 ] )           # retrieve last isi
                isi['curr'].append(xx[0])
                
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
 
#     def get_wrap(self,key):
#         w=self.wrap[key]['obj']
#         if self.merge_runs:
#             w=w.merge(axis=0)
#         if self.merge_sets:
#             w=w.merge(axis=1)
#         return w
        
 
    def plot_firing_rate(self, ax=None, win=100, **k):
        if not ax:
            ax=pylab.subplot(111)
        
        if not self.isrecorded('firing_rate'):
            self.compute_set('firing_rate',*[1],**{'average':True})

        x, y=self.get('firing_rate', attr_list=['x','y'], )
    
        y=misc.convolve(y, **{'bin_extent':win, 'kernel_type':'triangle',
                          'axis':1})     
        ax.plot(x, y, **k)
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (spike/s)') 
        ax.legend()

    
    def plot_hist_isis(self, ax=None, sets=[], **k):
        if not ax:
            ax=pylab.subplot(111)

        if not self.isrecorded('isi'):
            self.compute_set('isi',*[],**{})     

        y,=self.get('isi',  attr_list=['y'])
        y=reduce(lambda a, b:list(a)+list(b), y )
        #y,=merge_runs(*[y])
        
#         
#         if not sets:
#             y=[reduce(lambda a, b:list(a)+list(b), y)]
#         add_labels(sets, k)
        ax.hist(numpy.array(y), **k)
        ax.set_xlabel('Time (ms)')     
        ax.set_ylabel('Count (#)')
        ax.legend()
    def plot_mean_rate_parts(self, ax=None, x=[],  **k):
        if not ax:
            ax=pylab.subplot(111) 

        if not self.isrecorded('mean_rate_parts'):
            self.compute_set('mean_rate_parts',*[],**{})     
        
        _x,y=self.get('mean_rate_parts', attr_list=['x','y'])
#         x,y=get_sets_or_single('mean_rate', sets, *[ x, y])
        
        if not x:
            x=_x
              
#         add_labels(sets,  k)
        
        ax.plot(x, y, **k)
        
        ax.set_ylabel('Frequency (spike/s)') 
        ax.set_xlabel('Stimuli')
        
        
    def plot_IF_curve(self, ax=None, x=[], part='last', **k):
        if not ax:
            ax=pylab.subplot(111) 
        c, fisi, misi, lisi=self.get_IF_curve()
        
        if not x:
            x=numpy.mean(c,axis=1)
       
        if part=='first':isi=fisi
        if part=='mean':isi=misi
        if part=='last':isi=lisi    
        
        
        std=numpy.std(isi,axis=1)
        
        m=numpy.mean(isi,axis=1)
        color=pylab.getp(ax.plot(x, m, marker='o', **k)[0], 'color')    
        
        p= ax.fill_between(x, m-std,m+std, facecolor=color,  
                        alpha=0.5)  
        ax.plot(x, lisi , **{'color':color})    
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Rate (spike/s)') 
        ax.legend()
        
    def plot_FF_curve(self, ax=None, sets=[], x=[], **k):
        if not self.isrecorded('mean_rate_parts'):
            self.compute_set('mean_rate_parts',*[],**{})          
        if not ax:
            ax=pylab.subplot(111) 
        
        _x, y=self.get('mean_rate_parts', attr_list=['x','y'])
        if not x:
            x=_x
        #x,y=get_sets_or_single('mean_rate', sets, *[ x, y])
        #add_labels(sets,  k)
        ax.plot(x, y, **k)
        ax.set_xlabel('Stimuli (spike/s)') 
        ax.set_ylabel('Response (spike/s)') 
        ax.legend()
    
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

class Data_unit_spk(Data_unit_base, Mixin_spk):
    pass

class Mixin_vm(object):
    
    def _init_extra_attributes(self, *args, **kwargs):
        '''
        Constructor
        Wraps SpikeListMatrix and VmListMatrix. Use dependency injection
        as design pattern.
        '''
    
        d={'voltage_trace':{'ids':[], 'x':[], 'y':[]},
           'mean_voltage_parts':{'ids':[], 'x':[], 'y':[]},
            }
        self.data=d    
   
    def plot_IV_curve(self, ax=None, sets=[], x=[], **k):
        if not self.isrecorded('mean_voltage_parts'):
            self.compute_set('mean_voltage_parts',*[],**{})          
        if not ax:
            ax=pylab.subplot(111) 
        
        _x, y=self.get('mean_voltage_parts', attr_list=['x','y'])
        if not len(x):
            x=_x

        ax.plot(x, y, **k)
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Membrane potential (mV)') 
        ax.legend()
                
    def plot_voltage_trace(self, ax=None, index=[[0, 0]], **k):
        
        if not ax:
            ax=pylab.subplot(111) 
        x, y=self.get('voltage_trace', attr_list=['x','y'])
        #x, y=merge_runs(*[x, y]) 
        
        for i,j in index:
            ax.plot(x[i][j], y[i][j], **k)
        ax.set_xlabel('Time (ms)') 
        ax.set_ylabel('Membrane potential (mV)') 

class Data_unit_vm(Data_unit_base, Mixin_vm):
    pass
  
class Data_units_dic(object):
    
     
    def __init__(self, factory_class, **kwargs):
        #OBS, having dic={} do not ensure clearance of the dictionary.
        self.dic=kwargs.get('dic',{})
        self.allowed=allowed_Data_units_dic()
        self.attr=''
        self.factory_class=factory_class
        
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
            s="Model '{}' is not present in {}"
            raise Exception(s.format(model, self))

    def __iter__(self):
        for key in self.models:
            yield key, self.dic[key]
            
    def __setitem__(self, model, val):
        assert isinstance(val, Data_unit_base), "An Data_units_dic object can only contain Data_unit objects"
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
                self.dic[key]=self.factory_class(key, val)
            else:
                self.dic[key].add(val)    
            
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
        elif attr=='merge_sets':
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
        assert isinstance(val, Data_unit_base), "An Data_units_dic object can only contain Data_unit objects"
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


def get_labels(shape):
    a=numpy.empty(shape, dtype=object)
    if shape[1]==1:
        s1='Mean'
    else:
        s1='Set'
    if shape[0]==1:
        s2=''
    else:
        s2=' run'
        if s1=='Mean':
            s1=''
            s2='Run'
        
    
    n1=''
    n2=''            
    for i in range(shape[0]):
        for j in range(shape[1]):
            if shape[1]>1:
                n1=' '+str(j)
            if shape[0]>1:
                n2=' '+str(i)
            a[i,j]=s1+n1+s2+n2    
              
    return a
    

def iter2d(m):
    for i in xrange(m.shape[0]):
        for j in xrange(m.shape[1]):
            yield i, j, m[i,j]

def iter3d(m):
    for i in xrange(m.shape[0]):
        for j in xrange(m.shape[1]):
            for k in xrange(m.shape[2]):
                yield i, j, k, m[i,j,k]
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
        if len(a.shape)==4:
            aa=numpy.empty(a.shape[0:3], dtype=object)
            for i, j, k, _ in iter3d(a):
                aa[i,j,k]=a[i,j,k]
            a=aa    
        
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
        reset=kwargs.get('reset',False)
        shift=kwargs.get('shift',0.)
        
        l1,l2=[],[]
        for i in xrange(n_runs):
            s,v=[],[]
            for j in xrange(n_sets):

                kwargs={'run':i, 'set':j, 'n_sets':n_sets, 
                        'stim_time':200.0, 'n_pop':n_pop_spk, 'scale':0.5,
                        'shift':shift, 'reset':reset,
                        }
                
                s.append(dd('spike',**kwargs))    
                
                kwargs['n_pop']= n_pop_vm
                kwargs['sim_time']=200.0
                v.append(dd('voltage', **kwargs))
            
            l1.append(s)
            l2.append(v)
            
            
        return SpikeListMatrix(l1),VmListMatrix(l2)
    
def dummy_data_du(**kwargs):
    name=kwargs.get('name','unittest')

    l1,l2=dummy_data(**kwargs)    
    du_spk=Data_unit_spk(name, l1)
    du_vm=Data_unit_vm(name, l2)

    return du_spk, du_vm

def dummy_data_dud(names, **kwargs):
        
        d1,d2={},{}
        dud=Data_units_dic()        
        for name in names:
            kwargs['name']=name
            d1[name], d2[name]=dummy_data(**kwargs)  
           
                
        dud.add('s',d1)
        dud.add('v',d2)
        return dud
        
class TestData_unit_spk(unittest.TestCase):
    def setUp(self):
        self.longMessage=True
        self.n_runs=3

        self.obj, _=dummy_data_du(**{'n_runs':self.n_runs})
         
    def test_1_compute_set_get(self):
        for attr, a, k in [['firing_rate',[100],{'average':True}], 
                           ['isi', [],{}],
                           ['mean_rate', [],{}],
                           ['mean_rates',[],{}],
                           ['psd',[256,1000.0],{}],
                           ['phase',[10,20,3,1000.0],{}],
                           ['phases',[10,20,3,1000.0],{}],
                           ['raster',[], {}],
#                            ['voltage_trace', [], {}],
                           ]:
  
            self.obj.compute_set(attr,*a,**k)            
            self.obj.get(attr)


    def test_2_plot_firing_rate(self):
           
        self.obj, _=dummy_data_du(**{'n_runs':self.n_runs, 'reset':False})
   
        pylab.figure()
        ax=pylab.subplot(211)
        self.obj.plot_firing_rate(ax, **{'label':'Mean'})
        self.obj[:,0].plot_firing_rate(ax, **{'label':'Set 1'})    
        self.obj[:,1].plot_firing_rate(ax, **{'label':'Set 2'})   
# 
 
        self.obj, _=dummy_data_du(**{'n_runs':self.n_runs, 'reset':True})
        self.obj.compute_set('firing_rate',*[1],**{'average':True})
        ax=pylab.subplot(212)
        self.obj[0,:].plot_firing_rate(ax, **{'label':'Run 1'})
        self.obj[1,:].plot_firing_rate(ax, **{'label':'Run 2'})
        self.obj[2,:].plot_firing_rate(ax, **{'label':'Run 3'})  
#         pylab.show()                
#   
    def test_3_plot_hist_isis(self):
 
        pylab.figure()
        self.obj.plot_hist_isis( **{'label':'Mean'})    
        self.obj[:,0].plot_hist_isis( **{'label':'Set 1'}) 
#         pylab.show()
 
    def test_4_plot_mean_rate_parts(self):
        x= [100,200, 300]
        pylab.figure() 
        self.obj.plot_mean_rate_parts(x=x, **{'label':'Mean'})
        self.obj[:,0].plot_mean_rate_parts(x=x, **{'label':'Set 1'})
# 
#            
#         pylab.show()
#         
    def test_5_get_IF_curve(self):
        self.obj.compute_set('isi',*[],**{})
#         self.obj.set_stimulus('isi', 'x',  range(100,100*(1+self.n_runs),100))
        self.obj.get_IF_curve()
        #pylab.show()
# 
# 
    def test_6_plot_IF_curve(self):
        pylab.figure()  
        x=[100,200,300]   
        ax=pylab.subplot(211) 
        self.obj.plot_IF_curve(ax, x=x)#**{'color':'b'})
        ax.set_title('Mean')
        ax=pylab.subplot(212)    
        self.obj[:,0].plot_IF_curve(ax,x=x)#**{'color':'b'})
        ax.set_title('Set 1')
#         pylab.show()
#  
    def test_7_plot_FF_curve(self):
        pylab.figure()      
        self.obj.plot_FF_curve(**{'label':'Mean'})#**{'color':'b'})
        self.obj[:,0].plot_FF_curve(**{'label':'Set 1'})
#         pylab.show()#        d1, d2={}, {}
# 
#     def test_9_get_spike_stats(self):
#         self.obj.compute_set('spike_stats',*[],**{})

class TestData_unit_vm(unittest.TestCase):
    def setUp(self):
        self.longMessage=True
        self.n_runs=3
        _, self.obj=dummy_data_du(**{'n_runs':self.n_runs})
        
    def test_1_compute_set_get(self):
        for attr, a, k in [
                            ['voltage_trace', [], {}],
                            ['mean_voltage_parts', [], {}],
                           ]:
   
            self.obj.compute_set(attr,*a,**k)            
            self.obj.get(attr)
  
    def test_1_plot_voltage_trace(self):  
        pylab.figure() 
        ax=pylab.subplot(211) 
        self.obj.plot(**{'ax':ax})
        ax.set_title('All traces')
         
        ax=pylab.subplot(212)   
        self.obj[:,0].plot(**{'ax':ax})
        ax.set_title('Set 1 traces')
#         pylab.show()   
        
    def test_2_plot_IV(self):  

        pylab.figure() 
        
        ax=pylab.subplot(111) 
        self.obj.plot_IV_curve(**{'ax':ax, 'label':'Mean'})
        self.obj[:,0].plot_IV_curve(**{'ax':ax, 'label':'Set 1'})                          
        ax.set_title('All traces')
#         pylab.show()  

class TestData_units_dic(TestData_unit_spk):
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
    test_classes_to_run=[
                        TestData_unit_spk,
                        TestData_unit_vm, 
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
   
           