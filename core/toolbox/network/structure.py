'''
Created on Jun 24, 2013

@author: lindahlm

Tools for handling/setting up set/action based connections in models

'''
from copy import deepcopy
import numpy
import os
import random
import time

from toolbox import data_to_disk, my_nest, my_population, misc
from toolbox.misc import my_slice

import unittest
import pylab


class Dic(object):
    
    def __repr__(self):
        return self.dic
    
    def __getitem__(self,a):
        return self.dic[a]
    
    def __len__(self):
        return len(self.dic.values())  

    def __iter__(self):
        for d in self.dic:
            yield d
    
class Base_dic(Dic):
    def __init__(self):
        
        self.dic={}   
        self.attr=None
          
    @property
    def size(self):
        size=0
        for val in self.dic.values():
            size+=val.size
        return size

    @property
    def name(self):
        return str(self.dic.keys())
        
    
    def __iter__(self):
        l=sorted(self.dic.values(),key=lambda x:x.name)
        for v in l:
            yield v


    def __repr__(self):
        return self.__class__.__name__+':'+self.name
        
    def generic_iterator(self, *a, **k):
        for val in self.dic:
            func=getattr(val, self.attr)
            func(*a, **k)
            
    def add(self, *a, **k):
        pass


class Surface(Dic):
    '''
    As represents either group of input nodes, neuron model nodes, 
    or background nodes. It has different properties that define
    it, such as extent, edge_wrap, neuron model, number of neurons. 
    '''
    def __init__(self, name, **kwargs):
        
        self.edge_wrap=kwargs.get('edge_wrap', True)
        self.extent=kwargs.get('extend', [-0.5, 0.5])
        self.input=None
        self.model=kwargs.get('model', 'CO') # nest 
        self.name=name
        self.n=kwargs.get('n', 10)
        self.n_sets=kwargs.get('n_sets',1)
        self.sets=kwargs.get('sets',[my_slice(s, 10, 1) for s in range(1)] )
        assert self.n>=1.0, "Unit %s needs to have at least one node"%(name)
    
#    @property
#    def sets(self):
#        sets=[my_slice(s, self.n, self.n_sets) for s in range(self.n_sets)]
#        return sets
    

    
    @property
    def idx(self): 
        return range(self.n)
    
    @property
    def idx_edge_wrap(self):
        return numpy.array(self.idx[self.n/2:]+self.idx+self.idx[:self.n/2])
    
    @property
    def pos(self):
        step=1/float(self.n)
        return numpy.linspace(step*0.5+self.extent[0], 
                              self.extent[1]-step*0.5, self.n)
    
    @property
    def pos_edge_wrap(self):
        step=1/float(self.n)
        
        first=self.pos[0]
        last=self.pos[-1]
        upper=self.pos[:self.n/2]
        lower=self.pos[self.n/2:]
        pos_edge_wrap=numpy.array(list(-step/2+first+upper)+
                           list(self.pos)+
                           list(+step/2+last+lower))
        
        return pos_edge_wrap    
    
    @property
    def size(self):
        return self.n
    
    def __str__(self):
        return self.name+':'+str(self.n)

    def __repr__(self):
        return self.__class__.__name__+':'+self.name

    def _apply_mask(self, p0, mask_dist=None, mask_ids=None):
        '''
        Picks out the indecies from the unit that are constrained by
        distance from point p and mask_ids idx  
        
        Arguments:
        p0 - is the position from where distance is measured. (The driver
              node) 
        mask_distance - max distance from p to pick idx for the poos
        mask_ids - mask_ids index of the pool '''
           
        if self.edge_wrap:
            idx=self.idx_edge_wrap
            pos=self.pos_edge_wrap

        else:
            idx=numpy.array(self.idx)
            pos=numpy.array(self.pos)
        
        
        if not mask_dist: r=(self.extent[1]-self.extent[0])/2.
        else: r=(mask_dist[1]-mask_dist[0])/2.
              
        dist=numpy.abs(pos-p0)
        idx=idx[dist<=r] 
         
        if not mask_ids:pass
        else: idx=set(idx).intersection(mask_ids)         
        
        return list(idx)
     
    def _apply_kernel(self, idx, fan):
        '''
        Pickes out n nodes from the pool such that it approximately equals the
        fan. 
        idx - the set of pool the idx to considered
        fan - governs the probability of connections. Approximately fan 
              connections will be made'''
        
        n=len(idx)
        
        #p=float(fan)/n
        if n>=1: p=float(fan)/n
        else: return []
        rb=numpy.random.binomial
        rs=random.sample
      
        # For binomal distribution we have
        # Mean = n*p
        # Variance= n*p(1-p)
        if p<1:
            n_sample=rb(n,p)
            return rs(idx, n_sample)
        else:
            sple=[]
            while p>0:
                n_sample=rb(n,min(p,1))
                sple.extend(rs(idx, n_sample))
                p-=1
            return sple


    def get_connectables(self, p0, fan, mask_dist=None, mask_ids=None):
        
        idx=self._apply_mask(p0, mask_dist, mask_ids)
        idx=self._apply_kernel(idx, fan)
        return idx
    
    
    def get_idx(self, slice_objs=None):
        
        if not slice_objs:
            return self.idx
        
        if not type(slice_objs)==list:
            slice_objs=[slice_objs]
            
        idx=[]
        for so in slice_objs:
            idx+=self.idx[so.slice]
        return idx
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def get_n(self):
        return self.n
    
    def get_n_sets(self):
        return self.n_sets
    
    def get_pos(self, slice_objs=None):
        if not slice_objs:
            return self.pos
        
        if not type(slice_objs)==list:
            slice_objs=[slice_objs]
            
        pos=[]
        for so in slice_objs:
            pos.extend(self.pos[so.slice])
        return pos
    
    
    def get_sets(self, rule):
        
        
        if rule =='all':
            return [misc.my_slice(0,self.n)]
        
        # Randomly connects presynaptic neurons of set i with postsynaptic 
        # neurons of set i. Constrained by k_source, mask_dist and the sets.
        if rule=='set':
            return self.sets
        
        # Randomly connects presynaptic neurons from source set i to all 
        # postsynaptic sets except to target set i.         
        if rule=='not_set':
            sets=[]
            
            if self.n_sets==1:
                return self.sets
            
            for s in self.sets:
                sets.append(list(set(self.sets).difference(set([s]))))
                
            return sets
        
        # Connect all sets to all other
        if rule=='all_set':
            return self.sets*self.n_sets

class Population_dic(Base_dic):
    
    def __init__(self, *args, **kwargs):
        super( Population_dic, self ).__init__(*args, **kwargs)
        self.recorded_models=kwargs.get('recorded_models',[])
        
    def __str__(self):
        s='\n****************\nSurface info\n****************'
        s+='\n{0:14}{1}'.format('Total number:', self.size)
        s+='\n{0:14}'.format('By node:')
        
        text_lists=[]
        for struc in sorted(self.dic.values(), key=lambda x:x.name):
            text_lists.append('{0:>5}:{1:<8}'.format(struc.name, len(struc.conn_pre)))
        n=len(text_lists)
        text_lists=[text_lists[i:n:6] for i in range(6)] 

        for i, text_list in enumerate(text_lists):
            if i==0: pass
            else:s+='\n{0:14}'.format('')
            s+=''.join(text_list)
        
        return s
    
    def add(self, *a, **k):
        class_name=k.get('class', 'MyGroup') #default is MyGroup
        the_class=misc.import_class('toolbox.my_population.'+class_name)

        # del k['class']
        self.dic[a[0]]=the_class(*a, **k)

    def get(self, attr, **k):
        d={}
        
        for model in self.dic.keys():
            
            v=self.dic[model].get(attr, **k)    

            if v!=None:
                d[model]=v
        return d
    
    def get_mean_rate_error(self, models):
        e=[]
        for name in models:
            e.append(self.dic[name].get_mean_rate_error())
                      
class Surface_dic(Base_dic):
    
    def __str__(self):
        s='\n****************\nSurface info\n****************'
        s+='\n{0:14}{1}'.format('Total number:', self.size)
        s+='\n{0:14}'.format('By node:')
        
        text_lists=[]
        for struc in sorted(self.dic.values(), key=lambda x:x.name):
            text_lists.append('{0:>5}:{1:<8}'.format(struc.name, 
                                                     struc.n))
        n=len(text_lists)
        text_lists=[text_lists[i:n:6] for i in range(6)] 

        for i, text_list in enumerate(text_lists):
            if i==0: pass
            else:s+='\n{0:14}'.format('')
            s+=''.join(text_list)
        
        return s
    
    def add(self, *a, **k):

        class_name=k.get('structure_class', 'Surface')
        the_class=misc.import_class('toolbox.network.structure.'+class_name)
        self.dic[a[0]]=the_class(*a, **k)
    

            
            
                          
class Conn(Dic):
    '''
    A structure object defines the connectivity between a group of source units
    and a group of target units. 
    
    Convergent connection:
    When creating a divergent connection, each node in the source units 
    is visited in turn and selects target nodes from the target units. Masks, kernels,
    and boundary conditions are applied in the target units
         
    Divergent connection:               
    When creating a convergent connection between units, each node in the 
    target units are visited in turn and sources are selected for it in the
    source units. Masks and kernels are applied to the source units, and periodic
    boundary conditions are applied in the source units, provided that the source
    units has periodic boundary conditions.  
    '''
    
    def __init__(self, name, **kwargs):
        
        #self.dic=par['conn'][name]
        self.delay_setup=kwargs.get('delay_setup', {'type':'constant', 
                                                    'params':1.0})
        self.fan_in=kwargs.get('fan_in', 10)
        self.name=name            
        self.netw_size=kwargs.get('netw_size', 'unknown')
        self.mask=kwargs.get('mask', [-0.25, 0.25])
        self.pre=[]
        self.post=[]
        self.rule=kwargs.get('rule','all-all')
        self.sets=[]    
        self.save_path=(kwargs.get('save_path', '~/temp/unittest/conn'))
        self.syn=kwargs.get('syn', 'CO_M1_ampa') # nest 
        self.tata_dop=kwargs.get('tata_dop', 0.8)
        self.weight_setup=kwargs.get('weight_setup', {'type':'constant', 
                                                      'params':1.0})
    @property
    def n(self):
        return len(self.pre)
         
    @property
    def size(self):
        return self.n
    
    
    def __repr__(self):
        return self.__class__.__name__+':'+self.name    
    
    def __str__(self):
        return self.name+':'+str(len(self.pre))
    
    def _add_connections(self, d_slice, p_slice, driver, pool, fan_in, 
                         flag='Convergent'):
        '''
        For each node driver not nodes from the pool is considered 
        '''
        
        # For each driver as set of pool nodes considered depending on
        # driver position and allowed idx. Then the fan governs the probability
        # of makeing a connection to the set of pool nodes.
        d_idx=driver.get_idx(d_slice)
        p_idx=pool.get_idx(p_slice)
        d_pos=driver.get_pos(d_slice)
        
        
        # Short cuts for speed up
        get_connectables=pool.get_connectables
        fun_sum=lambda x,y: list(x)+list(y)
        fun_mul=lambda x,y: (x,)*y 

        #nodes=[gc(pool.get_pos(v), *a, **k) for v in p_idx]
        
        
        n=len(d_idx)#pool.get_n()
        
        #if self.rule=='all_set-all_set':
            
        fan_driver_to_pool=[fan_in]*n
        mask_dist_pool=[self.mask]*n 
        mask_ids_pool=[p_idx]*n 
        arg=[d_pos, fan_driver_to_pool, mask_dist_pool, mask_ids_pool]
        pool=map(get_connectables, *arg)
        
        
        n_of_conn_per_driver=map(len, pool)
        driver=map(fun_mul, d_idx, n_of_conn_per_driver)
        
        n1=len(self.pre)
        if flag=='Convergent':
            pre, post=pool, driver
        if flag=='Divergent':
            pre, post=pool, driver            
            
        self.pre+=reduce(fun_sum, pre)
        self.post+=reduce(fun_sum, post) 
        n2=len(self.pre)
        self.sets.append(slice(n1, n2, 1))
    
                       
    def get_fan_in(self, driver):
        fan_in=self.fan_in 
        if self.rule=='all_set-all_set':
            fan_in=self.fan_in/driver.get_n_sets()
            if fan_in<1:
                raise Exception(('For rule "all_set-all_set" the fan in has'
                                 +'to be bigger than number driver sets'))
        return fan_in

                 
    def get_delays(self):
        x=self.delay_setup
        if 'constant' == x['type']:
            return numpy.ones(self.n)*x['params']
        elif 'uniform' == x['type']:
            return list(numpy.random.uniform(low=x['params']['min'], 
                                             high=x['params']['max'], 
                                             size=self.n))      

    def get_post(self):
        return self.post

    def get_pre(self):
        return self.pre

    def get_syn(self):
        return self.syn
    
    def get_source(self):
        return self.name.split('_')[0]
    
    def get_target(self):
        return self.name.split('_')[1]
    
    def get_weights(self):
        x=self.weight_setup
        if 'constant' == x['type']:
            return numpy.ones(self.n)*x['params']
        if 'uniform' == x['type']:
            return list(numpy.random.uniform(low=x['params']['min'], 
                                             high=x['params']['max'], 
                                             size=self.n))    
        if 'learned' == x['type']:
            weights=data_to_disk.pickle_load(x['path'])
            
            conns=numpy.zeros(self.n)
            for i, sl in enumerate(self.sets):
                conns[sl]=weights[i]
            return list(conns*x['params'])    
    

    
    def set(self, driver, pool, save_mode=True, display_print=True):
        t=time.time()
        
        if save_mode and os.path.isfile(self.save_path+'.pkl'): 
            d=data_to_disk.pickle_load(self.save_path)
            self.pre, self.post, self.sets=d
        
        else:
            self._set(driver, pool)
            if save_mode:
                data_to_disk.pickle_save([self.pre, self.post, 
                                          self.sets], self.save_path)         
                   
        t=time.time()-t
        if display_print:
            s='Conn: {0:18} Connections: {1:8} Fan pool:{2:6} Time:{3:5} sec'
            a=[self.name, 
               len(self.pre), 
               round(float(len(self.pre))/pool.get_n(),0),
               round(t,2)]
            print s.format(*a)
            
    
    def _set(self, driver, pool):   
        '''
        For each node driver not nodes from the pool is considered
        '''   
        fan_in=self.get_fan_in(driver)
        args=[driver, pool, fan_in]
        if self.rule=='1-1':
            # Each presynaptic node connects to only one postsynaptic node.
            if driver.get_n()!=pool.get_n(): 
                raise Exception(('number of pre nodes needs to'
                                 + 'equal number of post'))
            self.pre, self.post= driver.get_idx(), pool.get_idx()

        elif self.rule in ['all-all', 'set-set', 
                           'set-not_set', 'all_set-all_set']:
            
            driver_sets=driver.get_sets(self.rule.split('-')[0]) 
            pool_sets=pool.get_sets(self.rule.split('-')[0]) 
            for slice_dr, slice_po in zip(driver_sets, pool_sets):     
                self._add_connections(slice_dr, slice_po, *args)
         
        assert isinstance(self.pre, list)            
        assert isinstance(self.post, list) 
         
            
    def plot_hist(self, ax):
        if self.n_conn==0:
            self.set_conn()
        pre, post=self.pre, self.post
        #ax.hist(post,self.target.get_n() )
        # numpy.histogram()
        n_pre, bins =   numpy.histogram(pre, bins=self.source.get_n() )
        n_post, bins = numpy.histogram(post, bins=self.target.get_n() )
        
        x_pre=numpy.linspace(0, 1, self.source.get_n())
        x_post=numpy.linspace(0, 1, self.target.get_n())
        
        ax.plot(x_pre,n_pre)
        ax.twinx().plot(x_post, n_post, 'r')
        
        fan_in=numpy.mean(n_post) # Number of incomming MSN connections per SNr
        n_out=numpy.mean(n_pre) 
        fan_in_std=numpy.std(n_post)
        n_out_std=numpy.std(n_pre)
        a=[self.name, round(fan_in,1),round(fan_in_std,1), 
                round(n_out,1), round(n_out_std,1)]
        ax.set_title('{0}\nIn/out:{1}({2})/{3}({4})'.format(*a))

                          
class Conn_dic(Base_dic):
    
    def __str__(self):
        s='\n****************\nConnection info\n****************'
        s+='\n{0:14}{1}'.format('Total number:', self.size)
        s+='\n{0:14}'.format('By connection:')
        
        text_lists=[]
        for struc in sorted(self.dic.keys(), key=lambda x:x.name):
            text_lists.append('{0:>5}:{1:<8}'.format(struc.name, len(struc.n)))
        n=len(text_lists)
        text_lists=[text_lists[i:n:6] for i in range(6)] 

        for i, text_list in enumerate(text_lists):
            if i==0: pass
            else:s+='\n{0:14}'.format('')
            s+=''.join(text_list)
        
        return s
    
    def add(self, *a, **k):
          
        class_name=k.get('structure_class', 'Conn')
        the_class=misc.import_class('toolbox.network.structure.'+class_name)
        self.dic[a[0]]=the_class(*a, **k)


def build(params_nest, params_node, params_pop):   
    nodes=create_struc(params_node)
    pops=create_pops(params_nest, params_pop)      
    return nodes, pops

def connect(pops, nodes, params_nest, params_conn, save_path, display_print=False):
    conns=create_conns(nodes, params_conn, save_path, display_print=False)
    connect_conns(params_nest, conns, pops, display_print)



def connect_conns(params_nest, conns, pops, display_print=False):
    for c in conns:
        if display_print:
            print 'Connecting '+str(c)
        my_nest.MyCopyModel( params_nest[c.get_syn()], c.get_syn())
        #c.copy_model( params_nest )
                    
        sr_ids=numpy.array(pops[c.get_source()].ids)
        tr_ids=numpy.array(pops[c.get_target()].ids)

        weights=list(c.get_weights())
        delays=list(c.get_delays())
        pre=list(sr_ids[c.get_pre()])
        post=list(tr_ids[c.get_post()])

        my_nest.Connect(pre, post , weights, delays, model=c.get_syn())    

def create_pops(params_nest, params_pop):
    #! Create input nodes
    pops=Population_dic()
    for name in params_pop.keys():
 
        model=params_pop[name]['model']
        my_nest.MyCopyModel( params_nest[model], model)
      
        args=[name]
        kwargs=deepcopy(params_pop[name])                            
        pops.add(*args, **kwargs)
            
    return pops

def create_conns(nodes, params_conn, save_path, display_print=False):
    conns=Conn_dic()
    for k,v in params_conn.iteritems(): 
        
        source=v['source']
        target=v['target']
        #kwargs={'source':source, 'target':target}
        v['save_path']=save_path
        conns.add(k, **v)
        
        # Convergent connections when driver = target and  pool = source.
        # The for each node driver not nodes from the pool is considered
        
        # Divergent connections when driver = source and  pool = target.
        # The for each node driver not nodes from the pool is considered
        conns[k].set(driver=nodes[target], 
                     pool=nodes[source],
                     save_mode=False, 
                     display_print=False)
    return conns      
      
def create_struc(params_node):
    nodes=Surface_dic()       
    for k,v in params_node.iteritems(): 
        nodes.add(k, **v)
    return nodes

           
class TestSurface(unittest.TestCase):
        
    def test_create(self):     
        n=Surface('unittest', **{})

    def test_get_set(self):
        m=5
        n=Surface('unittest', **{'n_sets':m,
                              'sets':[my_slice(s, 10, m) for s in range(m)] })
        s=[]
        s.append(n.get_sets('set'))
        s.append(n.get_sets('not_set'))
        s.append(n.get_sets('all_set'))
        
        self.assertEqual(len(s[0]),m)
        self.assertEqual(len(s[1]),m)
        self.assertEqual(len(s[1][0]),4)
        self.assertEqual(len(s[2]),m*m)
    
    def test_apply_kernel(self):
        n=Surface('unittest', **{})
        fan=40
        m=100
        p=fan/float(m)
        l=[]
        for _ in range(200):
            l.append(len(n._apply_kernel(range(m), fan)))
            
        self.assertAlmostEqual(m*p, numpy.mean(l), delta=1.0)
        self.assertAlmostEqual(numpy.sqrt(m*p*(1-p)), numpy.std(l), delta=1.0)

    def test_apply_mask(self):
        m=100
        n=Surface('unittest', **{'n':100, 'extent':[-0.5,0.5]})
        idx=n._apply_mask(n.get_pos()[1] , mask_dist=[-0.25,0.25], 
                          mask_ids=None)
        self.assertEqual(len(idx),  m/2+1)
        
    def test_get_connectables(self):
        m=100
        fan=40
        p=fan/float(m)
        mask_dist=[-0.25,0.25]
           
        n=Surface('unittest', **{'n':100, 'extent':[-0.5,0.5]})
        p0=n.get_pos()[1] 
        l=[]
        for _ in range(200):
            l.append(len(n.get_connectables( p0, fan, mask_dist, 
                                             mask_ids=None)))
        
        self.assertAlmostEqual(m*p, numpy.mean(l), delta=1.0)
        var=(m/2+1)*p*(1-p) #m/2+1 equals the number of ids returned by mask
        self.assertAlmostEqual(numpy.sqrt(var), numpy.std(l), delta=1.0)         
   
   
    def test_pos_edge_wrap(self):
        n=Surface('unittest', **{'n':100, 'extent':[-0.5,0.5]})
       
        p=n.pos_edge_wrap     
        
        m1=numpy.mean(numpy.diff(p))
        m2=numpy.diff(p)[0]
        self.assertAlmostEqual(m1,m2,10)

class TestSurface_dic(unittest.TestCase):
    def test_add(self):
        nd=Surface_dic()
        names=['unittest1, unittest2']
        for name in names:
            nd.add(name, **{})
        
class TestConn(unittest.TestCase):
        
    def setUp(self):
        sets=[my_slice(s, 100, 2) for s in range(2)]
        self.source=Surface('n1', **{'n':100, 
                                  'n_sets':2,
                                  'sets':sets })
        self.target=Surface('n2', **{'n':100, 
                                  'n_sets':2, 
                                  'sets':sets})
        self.home_path=('/afs/nada.kth.se/home/w/u1yxbcfw/'
                        +'results/papers/inhibition')
        self.save_path_conn=self.home_path+'/unittest/conn/'
        
        self.path_ext_learned='/unittest/learned'
    def test_create(self):
        c=Conn('n1_n2', **{'fan_in':10.0})

    
    def test_set_con(self):
        rules=['1-1', 'all-all', 'set-set', 'set-not_set', 'all_set-all_set']     
        l=[]

        for rule in rules:
            c=Conn('n1_n2', **{'fan_in':10.0, 'rule':rule})
            c._set(self.source, self.target)
            l.append(c)
            self.assertEqual(len(c.pre), len(c.post))
           
            if rule=='1-1':
                self.assertEqual(len(c.pre), self.target.get_n())
            else:
                self.assertAlmostEquals(len(c.pre), self.target.get_n()*10, delta=70)
#            colors=['b','g','r','m']*10
#            for i, s in enumerate(c.sets):
#                pylab.plot(c.pre[s], c.post[s], colors[i], marker='*',ls='')
#            pylab.show()
#            
    def test_set_save_load(self):
        rules=['1-1', 'all-all', 'set-set', 'set-not_set', 'all_set-all_set']     
        k={'fan_in':10.0, 
           'save_path':self.save_path_conn}
        l1=[]
        l2=[]
        for rule in rules:
            k.update({'rule':rule,
                      'save_path':self.save_path_conn+rule})
            c1=Conn('n1_n2', **k)
            c1.set(self.source, self.target, display_print=False)
            l1.append(c1.n)
            c2=Conn('n1_n2', **k)
            c2.set(self.source, self.target,  display_print=False)
            l2.append(c2.n)


        path=self.save_path_conn+'*'
        os.system('rm ' + path  + ' 2>/dev/null' )  
            
        self.assertListEqual(l1, l2)     
            
    def test_get_weight(self):
        m=1.
        c1=Conn('n1_n2', **{'weight_setup':{'params':m, 
                                            'type':'constant'}})
        c2=Conn('n1_n2', **{'weight_setup':{'params':{'min':m-0.5,
                                                         'max':m+0.5}, 
                                              'type':'uniform'}})      
        c3=Conn('n1_n2', **{'weight_setup':{'params':1, 
                                              'type':'learned', 
                                              'path':(self.home_path
                                                      +self.path_ext_learned)}})   
        c1._set(self.source, self.target) 
        c2._set(self.source, self.target) 
        d1=c1.get_weights()
        d2=c2.get_weights()
        
        self.assertAlmostEqual(numpy.mean(d1), m, 10)
        self.assertAlmostEqual(numpy.mean(d2), m, 1)  
        self.assertAlmostEqual(numpy.std(d2), numpy.sqrt(1/12.), delta=0.02)

    def test_get_delays(self):
        m=1.
        c1=Conn('n1_n2', **{'delay_setup':{'params':m, 
                                            'type':'constant'}})
        c2=Conn('n1_n2', **{'delay_setup':{'params':{'min':m-0.5,
                                                         'max':m+0.5}, 
                                              'type':'uniform'}})      
        c1._set(self.source, self.target) 
        c2._set(self.source, self.target) 
        d1=c1.get_delays()
        d2=c2.get_delays()
        
        self.assertAlmostEqual(numpy.mean(d1), m, 10)
        self.assertAlmostEqual(numpy.mean(d2), m, 1)  
        self.assertAlmostEqual(numpy.std(d2), numpy.sqrt(1/12.), delta=0.02)  
      
class TestConn_dic(unittest.TestCase):
    def test_add(self):
        cd=Conn_dic()
        names=['unittest1, unittest2']
        for name in names:
            cd.add(name, **{})
                           
class TestModuleFunctions(unittest.TestCase):
    def setUp(self):
        
        from toolbox.network.default_params import Unittest
        self.par=Unittest()
        self.home_path=('/afs/nada.kth.se/home/w/u1yxbcfw/'
                        +'results/papers/inhibition')
        self.save_path_extension='/unittest/conn/'
        self.save_path=self.home_path+self.save_path_extension
    @property           
    def params_conn(self):
        return self.par.get_conn()
    
    @property    
    def params_nest(self):
        return self.par.get_nest()
    
    @property
    def params_stru(self):
        return self.par.get_stru()
    
    @property
    def params_popu(self):
        return self.par.get_popu()

           
    def test_1_create_nodes(self):
        nodes=create_struc(self.params_stru)
        #print nodes
        
    def test_2_create_pops(self):
        nodes=create_struc(self.params_stru)
        pops=create_pops(self.params_nest, self.params_popu)


    def test_3_create_conns(self):
        nodes=create_struc(self.params_stru)
        conns=create_conns(nodes, self.params_conn, self.save_path)
    

    def test_4_connect_conns(self):
        nodes=create_struc(self.params_stru)
        pops=create_pops(self.params_nest, self.params_popu)
        conns=create_conns(nodes, self.params_conn, self.save_path)
        connect_conns(self.params_nest, conns, pops)
        
        
    def test_5_build(self):
        args=[self.params_nest, self.params_stru, self.params_pop]
        nodes, pops=build(*args)
        
    def test_6_connect(self):    
        args=[self.params_nest, self.params_stru, self.params_popu]
        nodes, pops=build(*args)
        conns=connect(pops, nodes, self.params_nest, self.params_conn,
                      self.save_path)
        
        
if __name__ == '__main__':
    test_classes_to_run=[TestSurface,
                         TestSurface_dic,
                         TestConn,
                         TestConn_dic,
                         TestModuleFunctions,
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    
    #suite =unittest.TestLoader().loadTestsFromTestCase(TestSurface)
    #suite =unittest.TestLoader().loadTestsFromTestCase(TestSurface_dic)
    #suite =unittest.TestLoader().loadTestsFromTestCase(TestStructure)
    unittest.TextTestRunner(verbosity=2).run(big_suite)
    
    #unittest.main() 
