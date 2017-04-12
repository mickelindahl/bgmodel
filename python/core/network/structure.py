'''
Created on Jun 24, 2013

@author: lindahlm

Tools for handling/setting up set/action based connections in models

'''
import numpy
import os
import random
import time
import unittest
import pylab
import gc

from copy import deepcopy
from itertools import izip
from core import data_to_disk, my_nest, my_population, misc, directories

from core.misc import my_slice, Base_dic
from core.parallelization import map_parallel, comm, Barrier
from core.network import default_params
from scipy import sparse

import pprint
pp=pprint.pprint



HOME = directories.HOME

    
class Surface(object):
    '''
    As represents either group of input surfs, neuron model surfs, 
    or background surfs. It has different properties that define
    it, such as extent, edge_wrap, neuron model, number of neurons. 
    '''
    def __init__(self, name, **kwargs):
        
        self.edge_wrap=kwargs.get('edge_wrap', True)
        self.extent=kwargs.get('extend', [-0.5, 0.5])
        self.fan_in_distribution=kwargs.get('fan_in_distribution', 'binomial')
        self.input=None
        self.model=kwargs.get('model', 'CO') # nest 
        self.name=name
        self.n=kwargs.get('n', 1000)
        self.n_sets=kwargs.get('n_sets',1)
        self.sets=kwargs.get('sets',[my_slice(s, 1000, 1) for s in range(1)] )
        
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


    def _apply_boundary_conditions(self):
        ''' self is the pool, the surface nodes are chosen from for each 
        node in the driver'''
        if self.edge_wrap:
            idx=self.idx_edge_wrap
            pos=self.pos_edge_wrap

        else:
            idx=numpy.array(self.idx)
            pos=numpy.array(self.pos)
        return idx, pos

    def _apply_mask(self, idx, pos, p0, mask_dist=None, mask_ids=None):
        ''' 
        self is the pool, the surface nodes are chosen from for each 
        node in the driver
        
        Picks out the indecies from the unit that are constrained by
        distance from point p and mask_ids idx  
        
        Arguments:
        p0 - is the position from where distance is measured. (The driver
              node) 
        mask_distance - max distance from p to pick idx for the poos
        mask_ids - mask_ids index of the pool '''
           

        if not mask_dist: r=(self.extent[1]-self.extent[0])/2.
        else: r=(mask_dist[1]-mask_dist[0])/2.
              
        dist=numpy.abs(pos-p0)
        idx=idx[dist<=r] 
         
        if not mask_ids:pass
        else: 
            if type(mask_ids)!=list:
                mask_ids=mask_ids.get_as_list()
            idx=set(idx).intersection(mask_ids)         
#             idx[mask_ids.get_slice()]
        
        return list(idx)
     
    def _apply_kernel(self, idx, fan):
        '''
        self is the pool, the surface nodes are chosen from for each 
        node in the driver
        
        Pickes out n surfs from the pool such that it approximately equals the
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
            if self.fan_in_distribution == 'binomial':
                n_sample=rb(n,p)
            elif self.fan_in_distribution == 'constant':
                n_sample=int(fan)
            return rs(idx, n_sample)
        else:
            sple=[]
            while p>0:
                n_sample=rb(n,min(p,1))
                sple.extend(rs(idx, n_sample))
                p-=1
            return sple


    def get_connectables(self, p0, fan, mask_dist=None, mask_ids=None):
        
        idx, pos=self._apply_boundary_conditions()
        idx=self._apply_mask(idx, pos, p0, mask_dist, mask_ids)
        idx=self._apply_kernel(idx, fan)
        return idx
    
    
    def get_idx(self, slice_objs=None, index=None, as_slice=False):
        
        if index!=None:
            return [self.idx[index]] 
           
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
    
    def get_pos(self, slice_objs=None,  index=None):
        
        if index!=None:
            return [self.pos[index]] 
        
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
        
        import pprint
        s=pprint.pformat(self.dic)
   
        return s
    
    def add(self, *a, **k):
        class_name=k.get('class', 'MyGroup') #default is MyGroup
        the_class=misc.import_class('core.my_population.'+class_name)

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
        the_class=misc.import_class('core.network.structure.'+class_name)
        self.dic[a[0]]=the_class(*a, **k)
    
                          
class Conn(object):
    '''
    A structure object defines the connectivity between a group of source units
    and a group of target units. 
    
    Convergent connection:
    When creating a divergent connection, each node in the source units 
    is visited in turn and selects target surfs from the target units. Masks, kernels,
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
        self.delay=kwargs.get('delay', {'type':'constant', 
                                              'params':1.0})
        self.display=kwargs.get('display', True)
        self.fan_in=kwargs.get('fan_in', 10)
        self.local_lockup=None
        self.name=name            
        self.netw_size=kwargs.get('netw_size', 'unknown')
        self.mask=kwargs.get('mask', [-0.25, 0.25])
        self.pre=None #sparse.coo_matrix()
        self.post=None #sparse.coo_matrix()
        self.rule=kwargs.get('rule','all-all')

        self.save=kwargs.get('save', {'active':False,
                                      'overwrite':False,
                                      'path':''})
        self.sets=[]    
        self.source=kwargs.get('source', 'C0')
        self.syn=kwargs.get('syn', 'CO_M1_ampa') # nest 
        self.target=kwargs.get('target', 'CO')
        self.tata_dop=kwargs.get('tata_dop', 0.8)
        
        
        self.local_num_threads=kwargs.get('local_num_threads', 2) 
        
        self.weight=kwargs.get('weight', {'type':'constant', 
                                                'params':1.0})


    def set_local_lockup(self, tr_id):
        post=self.get_post()
        post_ids=list(numpy.unique(post))
        status=my_nest.GetStatus(list(numpy.unique(tr_id[post])), 'local')
        
        lockup=dict(zip(post_ids,status))

        l=[1 if lockup[p] else 0 for p in post ]

        self.local_lockup=sparse.coo_matrix(l)

        
    @property
    def n(self):
        if self.pre==None:
            return 0
        else:
            return self.pre.shape[1]
    @property
    def n_pre(self):
        if self.pre==None:
            return 0
        else:
            return self.pre.shape[1]
    @property
    def n_post(self):
        if self.pre==None:
            return 0
        else:
            return self.post.shape[1]
                 
    @property
    def size(self):
        return self.n
    
    
    def __repr__(self):
        return self.__class__.__name__+':'+self.name    
    
    def __str__(self):
        return self.name+':'+str(self.n)
    
    
#     def _add_connections_parallell
    
    def _add_connections(self, d_slice, p_slice, driver, pool, fan_in, 
                         flag='Convergent'):
        '''
        For each node driver not surfs from the pool is considered 
        
        d_slice - slice for drivers (should be target=convergent connect). 
                  Drivers if the surface in which each
                  node is considered. Thus for each node in driver a set of 
                  pool nodes are chosen. get_connectables is applied on
                  each driver node.
        '''
        
        # For each driver as set of pool surfs considered depending on
        # driver position and allowed idx. Then the fan governs the probability
        # of making a connection to the set of pool surfs.
        d_idx=driver.get_idx(d_slice)
        d_pos=driver.get_pos(d_slice)
        d_n=len(d_idx)#pool.get_n()        
        
        # Short cuts for speed up
        pool_get_connectables=pool.get_connectables
        fun_sum=lambda x,y: list(x)+list(y)
        fun_mul=lambda x,y: (x,)*y 

        # All of length d_n    
        pool_fan_driver_to=[fan_in]*d_n
        pool_mask_dist=[self.mask]*d_n 
        pool_mask_ids=[pool.get_idx(p_slice)]*d_n 
        
#         pool_mask_ids=[p_slice]*d_n 
        # arg consists of list of length of driver nodes. Thus 
        # get_connectables is applied to each driver with arguments from the
        # lists
        
        arg=[d_pos, pool_fan_driver_to, pool_mask_dist, pool_mask_ids]
       # arg=zip(*arg)    

#         pool_conn=[]
#         for a in arg:
#             pool_conn.append(pool_get_connectables(*a))
        # For each driver nod get_connectales is applied
#         pool_conn=map(pool_get_connectables, *arg) #get connectables belong to pool

        # Necessary to ensure all mpi proc gets the same data
        pool_conn=map_parallel(pool_get_connectables, 
                          *arg, **{'local_num_threads':self.local_num_threads,
                                   'display':self.display})
        
        # For each driver get number of pool neurons that have been chosen
        # and then expand d_idx accordingly to get 
        n_of_conn_per_driver=map(len, pool_conn)
        driver_conn=map(fun_mul, d_idx, n_of_conn_per_driver)
        
        n1=self.n
        if flag=='Convergent':
            pre, post=pool_conn, driver_conn
        if flag=='Divergent':
            pre, post=driver_conn, pool_conn            
        
        pre_add=sparse.coo_matrix(reduce(fun_sum, pre))
        post_add=sparse.coo_matrix(list(reduce(fun_sum, post))) 
        if self.pre==None:
            self.pre= pre_add 
        else:
            self.pre=sparse.hstack([self.pre, pre_add])
            
        if self.post==None:    
            self.post=post_add
        else: 
            self.post=sparse.hstack([self.post, post_add])
        n2=self.n
        
        self.sets.append(slice(n1, n2, 1))
    
    def clear(self):
        del self.pre
        del self.post
                       
    def get_fan_in(self, driver):
        fan_in=self.fan_in 
        if self.rule=='all_set-all_set':
            fan_in=self.fan_in/driver.get_n_sets()
            if fan_in<1:
                raise Exception(('For rule "all_set-all_set" the fan in has'
                                 +'to be bigger than number driver sets'))
        return int(fan_in)


                 
    def _get_delays(self):
        x=self.delay
        if 'constant' == x['type']:
            return numpy.ones(self.n)*x['params']
        elif 'uniform' == x['type']:
            
            return numpy.random.uniform(low=x['params']['min'], 
                                             high=x['params']['max'], 
                                             size=self.n)  
    def get_delays(self):
        return list(self._get_delays())
    
    def get_delays_local(self):
        a=self._get_delays()
        b=self._get_local_lockup()
        return list(a[b==1])
       
    def _get_local_lockup(self):
        return numpy.asarray((self.local_lockup.todense())).ravel()
   
    def get_post(self):
        return list(self._get_post())
    
    def _get_post(self):
        return numpy.asarray((self.post.todense())).ravel()
    
    def get_post_local(self):
        a=self._get_post()
        b=self._get_local_lockup()
        return list(a[b==1])
    
    def get_pre(self):
        return list(self._get_pre())
    
    def _get_pre(self):
        return numpy.asarray((self.pre.todense())).ravel()
 
    def get_pre_local(self):
        a=self._get_pre()
        b=self._get_local_lockup()
        return list(a[b==1])
    
    def get_syn(self):
        return self.syn
    
    def get_source(self):
        return self.name.split('_')[0]
    
    def get_target(self):
        return self.name.split('_')[1]
    
    def _get_weights(self):
        x=self.weight
        if 'constant' == x['type']:
            return numpy.ones(self.n)*x['params']
        if 'uniform' == x['type']:
#             low=min(x['params']['min'],1.0) # nest scheduler
            low=x['params']['min']
            return numpy.random.uniform(low=low, 
                                             high=x['params']['max'], 
                                             size=self.n)
        if 'learned' == x['type']:
            weights=data_to_disk.pickle_load(x['path'])
            
            conns=numpy.zeros(self.n)
            for i, sl in enumerate(self.sets):
                conns[sl]=weights[i]
            return conns*x['params']    

    def get_weights(self):
        return list(self._get_weights())
    
    def get_weights_local(self):
        a=self._get_weights()
        b=self._get_local_lockup()
        return list(a[b==1])

    def _save_mpi(self):
        with Barrier():
            if comm.rank()==0:
                self._save()
    
    def _save(self):
        data_to_disk.pickle_save([self.pre, self.post, 
                                  self.sets], self.save['path'] ) 
        comm.barrier()

    
    def set(self, surfs, display_print=True):
        t=time.time()
        
        # Convergent connections when driver = target and  pool = source.
        # The for each node driver not surfs from the pool is considered
        
        # Divergent connections when driver = source and  pool = target.
        # Then for each node driver not surfs from the pool is considered
        driver=surfs[self.target]
        pool=surfs[self.source]
        
        if not (self.save['active'] 
                and not self.save['overwrite']
                and os.path.isfile(self.save['path'] +'.pkl')): 
            
            self._set(driver, pool)
           
            self._save()

        else:
            d=data_to_disk.pickle_load(self.save['path'] )
            
            
            self.pre=sparse.coo_matrix(d[0])
            self.post=sparse.coo_matrix(d[1])
            self.sets=d[2]
#             self.pre=numpy.array([[0]])#sparse.coo_matrix(d[0])
#             self.post=numpy.array([[0]])#sparse.coo_matrix(d[1])
#             self.sets=[0]#d[2]
#                 
                   
        t=time.time()-t
        if display_print and comm.rank()==0:
            s='Conn: {0:18} Connections: {1:8} Fan pool:{2:6} ({3:6}) Time:{4:5} sec Rule:{5}'
            a=[self.name, 
               self.n, 
               round(float(self.n)/driver.get_n(),0),
               self.fan_in,
               round(t,2),
               self.rule]
            print s.format(*a)
            
    def _set_mpi(self, *args):
#         with Barrier(True):
        if comm.rank()==0:
            self._set(*args)
                
        comm.barrier()
    
    def _set(self, driver, pool):   
        '''
        For each node driver nodes from the pool is considered
        '''   
        fan_in=self.get_fan_in(driver)
        args=[driver, pool, fan_in]
        if self.rule=='1-1':
            # Each presynaptic node connects to only one postsynaptic node.
            if driver.get_n()!=pool.get_n(): 
                raise Exception(('number of pre surfs needs to'
                                 + 'equal number of post'))
            self.pre= sparse.coo_matrix(pool.get_idx())
            self.post= sparse.coo_matrix(driver.get_idx())
            
        elif self.rule=='1-all':
            # Each presynaptic node connects to only one postsynaptic node.
            if pool.get_n()!=1 : 
                raise Exception(('Pool population has to of size 1'))
            
            post=[driver.get_idx()[0] for _ in range(len(pool.get_idx()))]
            self.pre= sparse.coo_matrix(pool.get_idx())
            self.post= sparse.coo_matrix(post)

        elif self.rule=='divergent':
            # Each presynaptic node connects to only one postsynaptic node.
            
            post=list(driver.get_idx())*len(pool.get_idx())
            pre=[len(driver.get_idx())*[idx] for idx in pool.get_idx()]
            pre=reduce(lambda x,y:x+y,pre)
            
            
#             post=[driver.get_idx()[0] for _ in range(len(pool.get_idx()))]
            self.pre = sparse.coo_matrix(pre)
            self.post = sparse.coo_matrix(post)
        elif self.rule =='fan-1':
            
            if driver.get_n()*fan_in!=pool.get_n(): 
                raise Exception(('number of pre surfs needs to'
                                 + 'equal number of post'))
            self.pre=sparse.coo_matrix(pool.get_idx())
            self.post=sparse.coo_matrix(driver.get_idx()*int(fan_in))
         
        elif self.rule in ['all-all', 'set-set', 
                           'set-not_set', 'all_set-all_set']:
     
            driver_sets=driver.get_sets(self.rule.split('-')[0]) 
            pool_sets=pool.get_sets(self.rule.split('-')[1]) 
            pre,post=[],[]
            for slice_dr, slice_po in zip(driver_sets, pool_sets): 
                
                self._add_connections(slice_dr, slice_po, *args)    
                          
#                 list_dr=slice_dr.get_as_list()
#                 n=len(list_dr)
#                 map_args=([list_dr, 
#                               [slice_po]*n]
#                               +[[a]*n for a in args]
#                               +[['Convergent']*n, 
#                                 [self]*n])
#  
#                 fun=_add_connections
#                 k={'local_num_threads':self.local_num_threads}
#                 l=map_parallel(fun, *map_args, **k)
#                 for v in l:
#                     pre+=reduce(lambda x,y:x+y,v[0])
#                     post+=reduce(lambda x,y:x+y,v[1])
#                       
#             v=add_pre_post(self.pre, self.post, self.sets, 
#                                    pre, post)
#              
#             self.pre, self.post, self.sets=v
  
         
        assert isinstance(self.pre, sparse.coo_matrix)            
        assert isinstance(self.post, sparse.coo_matrix) 
         
            
    def plot_hist(self, ax):
        if self.n_conn==0:
            self.set_conn()
        pre, post=self.get_pre(), self.get_post()
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

        
        return str(self.dic)
    
    def add(self, *a, **k):
          
        class_name=k.get('structure_class', 'Conn')
        the_class=misc.import_class('core.network.structure.'+class_name)
        self.dic[a[0]]=the_class(*a, **k)


def _add_connections(d_id, p_slice, driver, pool, fan_in, flag, obj):
    '''
    For each node driver not surfs from the pool is considered 
    
    d_slice - slice for drivers (should be target=convergent connect). 
              Drivers if the surface in which each
              node is considered. Thus for each node in driver a set of 
              pool nodes are chosen. get_connectables is applied on
              each driver node.
    '''
    
    # For each driver as set of pool surfs considered depending on
    # driver position and allowed idx. Then the fan governs the probability
    # of making a connection to the set of pool surfs.
    d_idx=driver.get_idx(index=d_id)
    d_pos=driver.get_pos(index=d_id)
    d_n=len(d_idx)#pool.get_n()        
    
    # Short cuts for speed up
    pool_get_connectables=pool.get_connectables
    fun_mul=lambda x,y: (x,)*y 

    # All of length d_n    
    pool_fan_driver_to=[fan_in]*d_n
    pool_mask_dist=[obj.mask]*d_n 
    pool_mask_ids=[pool.get_idx(p_slice)]*d_n 
    
    # arg consists of list of length of driver nodes. Thus 
    # get_connectables is applied to each driver with arguments from the
    # lists
    arg=[d_pos, pool_fan_driver_to, pool_mask_dist, pool_mask_ids]

    pool_conn=map(pool_get_connectables, *arg) #get connectables belong to pool


    # For each driver get number of pool neurons that have been chosen
    # and then expand d_idx accordingly to get 
    n_of_conn_per_driver=map(len, pool_conn)
    driver_conn=map(fun_mul, d_idx, n_of_conn_per_driver)
    
   
    if flag=='Convergent':
        pre, post=pool_conn, driver_conn
    if flag=='Divergent':
        pre, post=driver_conn, pool_conn            

    
    return pre, post
    
    
def add_pre_post(pre, post, sets, pre_add, post_add):
    
    n1=pre.shape[1]if pre else 0
    if type(pre_add)==list:
        pre_add=sparse.coo_matrix(pre_add)
    if type(post_add)==list:
        post_add=sparse.coo_matrix(post_add) 
    if not pre:
        pre= pre_add 
    else:
        pre=sparse.hstack([pre, pre_add])
         
    if not post:    
        post=post_add
    else: 
        post=sparse.hstack([post, post_add])
    
    n2=pre.shape[1]
#     print n2
    sets.append(slice(n1, n2, 1))
    return pre, post, sets
    
    
def build(params_nest, params_surf, params_popu): 
 
    surfs=create_surfaces(params_surf)
    popus=create_populations(params_nest, params_popu)      

    return surfs, popus

def connect(popus, surfs, params_nest, params_conn, display_print=False):
    for source in popus:
        if not isinstance(source, my_population.VolumeTransmitter):
            continue
        name=source.get_syn_target()
        my_nest.MyCopyModel( params_nest[name], name)
        my_nest.SetDefaults(name, {'vt':source.ids[0]}) 
    
    conns=create_connections(surfs, params_conn, display_print)
    connect_conns(params_nest, conns, popus, display_print)
    
    for s in surfs:
        del s
        gc.collect()
#     return conns

def connect_conns(params_nest, conns, popus, display_print=False):
    for c in conns:

        # if c.name[0] not in ['E','C']:
        # if c.name[0:5] not in ['C1_M1', 'C2_M2']:
        #if c.name[0:5] not in ['E']:
        #
        # continue

        if display_print and comm.rank()==0:
            print 'Connecting '+str(c), params_nest[c.get_syn()]
        my_nest.MyCopyModel( params_nest[c.get_syn()], c.get_syn())
        #c.copy_model( params_nest )


        sr_ids=numpy.array(popus[c.get_source()].ids)
        tr_ids=numpy.array(popus[c.get_target()].ids)


        c.set_local_lockup(tr_ids)

        weights=list(c.get_weights_local())
        delays=list(c.get_delays_local())
        pre=list(sr_ids[c.get_pre_local()])
        post=list(tr_ids[c.get_post_local()])
         
        import time
#         time.sleep(0.1*my_nest.Rank())
#         print c,'lu',c._get_local_lockup()
#         print c,'pre', c.get_pre_local()[0:10], len(pre) 
#         print c,'post', c.get_post_local()[0:10], len(pre)  
#         print sr_ids[0:10]
#         print tr_ids[0:10]
#         print c,'pre2', pre[0:10], len(pre) 
#         print c,'pos2t', post[0:10], len(pre)  
#         comm.barrier()
#         weights_input_rates=list(c.get_weights())
#         delays=list(c.get_delays())
#         pre=list(sr_ids[c.get_pre()])
#         post=list(tr_ids[c.get_post()])
        model=c.get_syn()

        c.clear()
        del c
        gc.collect()
         
#         delete_and_gc_collect(c)
        
#         my_nest.Connect_DC(pre, post , weights_input_rates, delays, model, only_local=True)
        my_nest.Connect_speed(pre, post , weights, delays, model=model)
#
        # syn_dict={
        #     "model":model,
        #     "weight":weights,
        #     "delay":delays,
        # }
        # conn_dict={
        #     "rule": "one_to_one",
        # }
        # import nest
        # nest.Connect(pre, post, conn_dict, syn_dict)
        # pp(my_nest.GetConnections(list(set(pre)))[0])
        # pp(my_nest.GetConnections(post)[0])
        # print pre[0], post[0]
        # print model, len(pre), len(post), len(weights), len(delays)
        # print 'Connecting ' + ' my_nest.GetConnections ', len(my_nest.GetConnections(list(set(pre)))), len(pre)
        # print 'Connecting ' + ' my_nest.GetConnections ', len(my_nest.GetConnections(post)), len(post)

        #         delete_and_gc_collect(weights_input_rates, delays, pre, post)
#         
        del weights
        del delays
        del pre
        del post
        gc.collect()
        

def delete_and_gc_collect(*args):
    for arg in args:
        del arg
    gc.collect()    


def create_populations(params_nest, params_popu):
    #! Create input surfs
    popus=Population_dic()
    for name in sorted(params_popu.keys()):
        
        model=params_popu[name]['model']
        my_nest.MyCopyModel( params_nest[model], model)

        args=[name]
        kwargs=deepcopy(params_popu[name])                          
        
        

        popus.add(*args, **kwargs)
        print name, popus[name].n
    return popus

def create_connections(surfs, params_conn, display_print=False):
    conns=Conn_dic()
    for k in sorted(params_conn.keys()): 
        v=params_conn[k]
        conns.add(k, **v)
        
        conns[k].set(surfs, display_print)
    return conns      
      
def create_surfaces(params_surf):
    surfs=Surface_dic()       
    for k,v in params_surf.iteritems():
        #
        # if k not in ['SN', 'ES']:
        #     continue

        surfs.add(k, **v)
    return surfs

def create_dummy_learned_weights(path_file, n_sets):

    w=[]
    for _ in range(n_sets):
        w.append(random.random())
    
    w=numpy.array(w)+0.5

    data_to_disk.pickle_save(w, path_file)
        
           
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
        for i, l in enumerate(s[1]):
            self.assertTrue(not my_slice(i, 10, m) in l)
        self.assertEqual(len(s[2]),m*m)
    
    def test_apply_boundary_conditions(self):
        m=100
        n=Surface('unittest', **{'edge_wrap':True, 'n':m})
        idx, pos=n._apply_boundary_conditions()
        self.assertEqual(len(idx),m*2)
        self.assertEqual(len(pos),m*2)
        
        n=Surface('unittest', **{'edge_wrap':False, 'n':m})
        idx, pos=n._apply_boundary_conditions()
        self.assertEqual(len(idx),m)
        self.assertEqual(len(pos),m)
        
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
        idx, pos=n._apply_boundary_conditions()
        idx=n._apply_mask(idx, pos, n.get_pos()[1] , mask_dist=[-0.25,0.25], 
                          mask_ids=None)
        self.assertEqual(len(idx),  m/2+1)
        
    def test_get_connectables(self):
        m=300
        fan=40
        p=fan/float(m)
        mask_dist=[-0.25,0.25]
            
        n=Surface('unittest', **{'n':m, 'extent':[-0.5,0.5]})
        p0=n.get_pos()[1] 
        l=[]
        for _ in range(1000):
            l.append(len(n.get_connectables( p0, fan, mask_dist, 
                                             mask_ids=None)))
         
        self.assertAlmostEqual(m*p, numpy.mean(l), delta=1.0)
#         var=(m/2+1)*p*(1-p) #m/2+1 equals the number of ids returned by mask
#         self.assertAlmostEqual(numpy.sqrt(var), numpy.std(l), delta=1.0)         
     
    
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
        n_sets=3
        sets=[my_slice(s, 100, n_sets) for s in range(n_sets)]
        self.n_sets=len(sets)
        nd=Surface_dic()
        nd.add('i1', **{'n':100, 'n_sets':n_sets, 'sets':sets })
        nd.add('i2', **{'n':200, 'n_sets':n_sets, 'sets':sets })
        nd.add('i3', **{'n':50, 'n_sets':n_sets, 'sets':sets })
        nd.add('n2', **{'n':100, 'n_sets':n_sets, 'sets':sets})
        self.surfs=nd
        self.source=nd['i1']
        self.source2=nd['i2']
        self.source3=nd['i3']
        self.target=nd['n2']
        self.path_base=HOME+'/results/unittest/structure/'
        self.path_conn=HOME+'/results/unittest/structure/conn/'
        self.path_learned=HOME+'/results/unittest/structure/learned.pkl'

    def test_create(self):
        _=Conn('n1_n2', **{'fan_in':10.0})

    
    def test_set_con(self):
        rules=[ 
                'all-all', 
                '1-1', 
                'set-set', 
                'set-not_set', 
                'all_set-all_set',
                'divergent'
                ]     
        l=[]
        fan_in=15
        for rule in rules:
            c=Conn('n1_n2', **{'display':False,
                               'fan_in':fan_in,
                               'local_num_threads':2,
                                'rule':rule})
            
            l.append(c)
   
           
            if rule=='1-1':
                c._set(self.source, self.target)
                self.assertEqual(c.n_pre, self.target.get_n())
            elif rule=='divergent':
                c._set(self.source2, self.target)
                self.assertEqual(c.n_pre, 
                                 self.target.get_n()*self.source2.get_n())
            else:
#                 print rule
                c._set(self.target, self.source2)
#                 print c.n_pre
                self.assertAlmostEquals(fan_in, 
                                        float(c.n_pre)/self.target.get_n(), 
                                        delta=3)
                c=Conn('n1_n2', **{'display':False,
                                   'fan_in':fan_in, 'rule':rule})
                c._set(self.target, self.source3)
                self.assertAlmostEquals(fan_in, 
                                        float(c.n_pre)/self.target.get_n(), 
                                        delta=3)          
                
                if rule=='set-not_set':
                    l=[]
                    for pre, post in zip(c.get_pre(), c.get_post()):
                        n_sets=self.target.get_n_sets()
                        self.assertTrue(pre % n_sets != post % n_sets)
                if rule=='set-set':
                    l=[]
                    for pre, post in zip(c.get_pre(), c.get_post()):
                        n_sets=self.target.get_n_sets()
#                         print pre % n_sets, pre
#                         print post % n_sets, post
                        self.assertTrue(pre % n_sets == post % n_sets)
         
        self.assertEqual(c.n_pre, c.n_post)
                  
#            for i, s in enumerate(c.sets):
#                pylab.plot(c.pre[s], c.post[s], colors[i], marker='*',ls='')
#            pylab.show()
#            
    def test_set_save_load(self):
        rules=['1-1', 'all-all', 'set-set', 'set-not_set', 'all_set-all_set',
               'divergent']     
        k={'fan_in':10.0}
        l1=[]
        l2=[]
        for rule in rules:
            k.update({'display':False,
                      'rule':rule,
                      'source':self.source.get_name(),
                      'target':self.target.get_name(),
                      'save':{'active':True,
                              'overwrite':False,
                             'path':self.path_conn+rule}})
            c1=Conn('n1_n2', **k)
            c1.set(self.surfs, display_print=False)
            l1.append(c1.n)
            c2=Conn('n1_n2', **k)
            c2.set(self.surfs, display_print=False)
            l2.append(c2.n)


        path=self.path_conn+'*'
        os.system('rm ' + path  + ' 2>/dev/null' )  
            
        self.assertListEqual(l1, l2)     


    def test_set_save_load_mpi(self):
        import subprocess
        rules=['1-1', 'all-all', 'set-set', 'set-not_set', 'all_set-all_set',
               'divergent'] 
        
        data_path= self.path_base+'set_save_load_mpi/'
        script_name=os.getcwd()+('/test_scripts_MPI/'
                                 +'structure_set_save_load_mpi.py')
        
        np=1

        
        p=subprocess.Popen(['mpirun', '-np', str(np), 'python', 
                            script_name, data_path],
#                             stdout=subprocess.PIPE,
#                             stderr=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                           )
        
        out, err = p.communicate()
#         print out
#         print err
        
        threads=np
        
        for i in range(threads):
            fileName= data_path+'data'+str(i)
            l1,l2=data_to_disk.pickle_load(fileName)
            print l1
            print l2
            self.assertListEqual(l1, l2) 
                
        path_clear=data_path+'*'
        os.system('rm ' + path_clear  + ' 2>/dev/null' )  


        path_clear=data_path+'conn/*'
        os.system('rm ' + path_clear  + ' 2>/dev/null' ) 
          
    def test_get_weight(self):
        m=1.
        c1=Conn('n1_n2', **{'display':False,
                            'weight':{'params':m, 
                                      'type':'constant'}})
        c2=Conn('n1_n2', **{'display':False,
                            'weight':{'params':{'min':m-0.5,
                                                'max':m+0.5}, 
                                      'type':'uniform'}})      
         
        create_dummy_learned_weights(self.path_learned, self.n_sets )
         
        c3=Conn('n1_n2', **{'display':False,
                            'weight':{'params':1, 
                                      'type':'learned', 
                                      'path':(self.path_learned)}})   
        c1._set(self.source, self.target) 
        c2._set(self.source, self.target) 
        c3._set(self.source, self.target) 
        d1=c1.get_weights()
        d2=c2.get_weights()
        d3=c3.get_weights()
         
        self.assertAlmostEqual(numpy.mean(d1), m, 10)
        self.assertAlmostEqual(numpy.mean(d2), m, 1)  
        self.assertAlmostEqual(numpy.std(d2), numpy.sqrt(1/12.), delta=0.02)
 
    def test_get_delays(self):
        m=1.
        c1=Conn('n1_n2', **{'display':False,
                            'delay':{'params':m, 
                                            'type':'constant'}})
        c2=Conn('n1_n2', **{'display':False,
                            'delay':{'params':{'min':m-0.5,
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
        from core.network.default_params import Unittest
        self.par=Unittest() 
        self.path=HOME+'/results/unittest/conn/'
                
    def params_conn(self, save):
        p=self.par.get_conn()
        for c in p.keys():
            if save:
                continue
            
            p[c]['save_path']={'active':False,
                               'path':''}
        
        return p
    
    @property    
    def params_nest(self):
        return self.par.get_nest()
    
    @property
    def params_surfs(self):
        return self.par.get_surf()
    
    @property
    def params_popu(self):
        return self.par.get_popu()

           
    def test_1_create_surfaces(self):
        surfs=create_surfaces(self.params_surfs)
        #print surfs
        
    def test_2_create_populations(self):
        surfs=create_surfaces(self.params_surfs)
        popus=create_populations(self.params_nest, self.params_popu)


    def test_3_create_connections(self):
        surfs=create_surfaces(self.params_surfs)
        conns=create_connections(surfs, self.params_conn(False))
    

    def test_4_connect_conns(self):
        surfs=create_surfaces(self.params_surfs)
        popus=create_populations(self.params_nest, self.params_popu)
        conns=create_connections(surfs, self.params_conn(False))
        connect_conns(self.params_nest, conns, popus)
        
        
    def test_5_build(self):
        args=[self.params_nest, self.params_surfs, self.params_popu]
        surfs, popus=build(*args)
        
    def test_6_connect(self):    
        args=[self.params_nest, self.params_surfs, self.params_popu]
        surfs, popus=build(*args)
        conns=connect(popus, surfs, self.params_nest, self.params_conn(False))
        
    def test_7_connect_with_save(self):
        
        args=[self.params_nest, self.params_surfs, self.params_popu]
        surfs, popus=build(*args)
        t=time.time()
        conns=connect(popus, surfs, self.params_nest, self.params_conn(True))
        for v in self.params_conn(True).values():
            filename=v['save']['path']+'.pkl' 
#             print 'Deleting: ' +filename
            os.remove(filename)
            
        conns=connect(popus, surfs, self.params_nest, self.params_conn(True))
        t1=time.time()-t
        conns=connect(popus, surfs, self.params_nest, self.params_conn(True))
        t2=time.time()-t-t1
        self.assertTrue(t1*10>t2)
        
if __name__ == '__main__':
    d={
#         TestSurface:[
#                      'test_create',
#                      'test_get_set',
#                      'test_apply_boundary_conditions',
#                      'test_apply_kernel',
#                      'test_apply_mask',
#                      'test_get_connectables',
#                      'test_pos_edge_wrap',
#                      ],
#         TestSurface_dic:[
#                          'test_add',
#                          ],
       TestConn:[
#                  'test_create',
#                 'test_set_con',
#                 'test_set_save_load',
                'test_set_save_load_mpi',
#                 'test_get_weight',
#                 'test_get_delays',
                 ],
        TestConn_dic:[
                      'test_add'
                      ],
        TestModuleFunctions:[
                            'test_1_create_surfaces',
                            'test_2_create_populations',
                            'test_3_create_connections',
                            'test_4_connect_conns',
                            'test_5_build',
                             'test_6_connect',
                             'test_7_connect_with_save',
                             ],
       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
