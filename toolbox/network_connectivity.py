'''
Created on Jun 24, 2013

@author: lindahlm

Tools for handling/setting up set/action based connections in models

'''
import copy
import numpy
import os
import random
import time
from toolbox import data_to_disk
from toolbox.default_params import Par, Par_bcpnn
import unittest

class Units(object):
    '''
    A s represents either group of input nodes, neuron model nodes, 
    or background nodes. It has different properties that define
    it, such as extent, edge_wrap, neuron model, number of neurons. 
    '''
    def __init__(self, name, dic):
        
        par=dic['par']
        
        self.collected_spikes=False
        self.collected_votage_traces=False
        self.name=name
        self.n=par['node'][name]['n']
        self.model=par['node'][name]['model']
        self.lesion=par['node'][name]['lesion']
        self.extent=par['node'][name]['extent']
        self.edge_wrap=par['node'][name]['edge_wrap']
        self.population=None # Empty container to put a group or input from population module
        self.type=par['node'][name]['type']

        self.proportion_of_network=par['node'][name]['prop']

    
    @property
    def idx(self): 
        return range(self.n)
    
    @property
    def idx_edge_wrap(self):
        return numpy.array(self.idx[self.n/2:]+self.idx+self.idx[:self.n/2])
    
    @property
    def pos(self):
        step=1/float(self.n)
        return numpy.linspace(step*0.5+self.extent[0],+self.extent[1]-step*0.5, self.n)
    
    @property
    def pos_edge_wrap(self):
        step=1/float(self.n)
        pos_edge_wrap=numpy.array(list(self.pos[0]+self.pos[:self.n/2])+
                           list(self.pos)+
                           list(self.pos[-1]+step+self.pos[self.n/2:]))
        return pos_edge_wrap    
    def __str__(self):
        return self.name

    def apply_mask(self, p0, mask_dist=None, allowed=None):
        '''
        Picks out the indecies from the unit that are constrained by
        distance from point p and allowed idx  
        
        Arguments:
        p0 - is the position from where distance is measured. 
        mask_distance - max distance from p to pick idx
        mask_idx - allowed index '''
           
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
         
        if not allowed:pass
        else: idx=set(idx).intersection(allowed)         
        
        return list(idx)
     
    def apply_kernel(self, idx, fan):
        n=len(idx)
        p=float(fan)/n
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
        
    
    def copy(self):

        return copy.deepcopy(self) # deep (recursive) copy
    
    def pprint(self):
        s='name:{0:10} n:{1:6} model:{2:16} lesion:{3:6}'
        return s.format(self.name, str(self.n), self.model, str(self.lesion), 
                        str(round(self.I_e,0)), str(round(self.proportion_of_network,3)))
    
    def set_population(self, population):
        self.population=population


class Units_input(Units):
    def __init__(self, name, dic):
        super( Units_input, self ).__init__(name, dic)
        self._init_extra_attributes(dic)
           
    def _init_extra_attributes(self, dic):
        # Add new attribute
        self.rate=dic['rate']

class Units_background(Units):
    
    def __init__(self, name, dic):
        super( Units_background, self ).__init__(name, dic)
        self._init_extra_attributes(dic)
           
    def _init_extra_attributes(self, dic):
        # Add new attribute
        self.proportion_of_network=dic['prop']
        self.rate=dic['rate']
        
class Units_neuron(Units):
    
    def __init__(self, name, dic):
        super( Units_neuron, self ).__init__(name, dic)
        self._init_extra_attributes(dic)
                 
    def _init_extra_attributes(self, dic):
        # Add new attribute
        self.proportion_of_network=dic['prop']
        self.I_vitro=dic['I_vitro']
        self.I_vivo=dic['I_vivo']
        self.randomization=dic['randomization']
        self.target_rate=dic['target_rate']
       
    def randomize_params(self, params):
        setup={}
        for key in params:
            setup[key]=self.randomization[key]
        
        pop=self.population
        pop.model_par_randomize(setup)
        
  
class Structure(object):
    '''
    A structure object defines the connectivity between a group of source units
    and a group of target units. 
    '''
    
    def __init__(self, name, dic):
        
        # Need to set them first
        self.source=dic['source'] # As units
        self.target=dic['target']
        
        par=dic['par']
        
        self.connection_type=par['conn'][name]['connection_type']
        self.conn_pre=[]
        self.conn_post=[]
        self.delay_val=par['conn'][name]['delay_val']
        self.delay_setup=par['conn'][name]['delay_setup']
        self.lesion=par['conn'][name]['lesion']
        
        
        self.fan_in=int(par['conn'][name]['fan_in']*(1-(par['netw']['tata_dop']-par['netw']['tata_dop0'])*par['conn'][name]['beta_fan_in']))
        self.beta_fan_in=par['conn'][name]['beta_fan_in']
                                 
        # So that the fan always is less or equal to number of targets
        if self.fan_in >self.target.n:
            self.source.n=self.fan_in
            
        self.name=name
        self.n_conn=0
        self.network_size=par['netw']['size']
        
        
        #Defines which pool nodes are considered as potential targets based on distance
        self.mask=par['conn'][name]['mask'] 
        
        # Defines which pool nodes are considered based on hierarchy 
        self.rule=par['conn'][name]['rule']
        if self.rule=='set-not_set' and par['conn'][name]['sets'][0]==1: # If we only have one set
            self.rule='set-set' 
        
        self.data_path_learned_conn=par['conn'][name]['data_path_learned_conn']
        
        self.save_at=dic['save_at']
        
        # Source and target are unit objects
        self.sets=par['conn'][name]['sets']
        
        self.sets_mapping_pre={}
        self.sets_mapping_post={}
        
        
        self.sets_mapping_driver={}
        self.sets_mapping_pool={}
        self.sets_driver=[]
        self.sets_pool=[]
        
        
        sets_source=[slice(s, self.source.n, self.sets[0]) for s in range(self.sets[0])]
        sets_target=[slice(s, self.target.n, self.sets[1]) for s in range(self.sets[1])]
        
        
        # When creating a divergent connection, each node in the source units 
        # is visited in turn and selects target nodes from the target units. Masks, kernels,
        # and boundary conditions are applied in the target units
        if self.connection_type=='divergent':
            self.driver=self.source
            self.pool=self.target
            self.fan_pool=self.fan_in
            self.sets_driver=sets_source
            self.sets_pool=sets_target
                
        # When creating a convergent connection between units, each node in the 
        # target units are visited in turn and sources are selected for it in the
        # source units. Masks and kernels are applied to the source units, and periodic
        # boundary conditions are applied in the source units, provided that the source
        # units has periodic boundary conditions.    
        elif self.connection_type=='convergent':
            self.driver=self.target
            self.pool=self.source 
            self.fan_pool=int(round(self.fan_in*self.target.n/float(self.source.n)))
            self.sets_driver=sets_target
            self.sets_pool=sets_source
        
        # The sets are mixed up. With three sets then every third
        # node starting from 0 will be from the first set, every
        # third node starting from 1 will be from the second set 
        # and the rest will be from the third set.
        self.sets_source=sets_source
        self.sets_target=sets_target
     
        self.syn=par['conn'][name]['syn']     
        self.tata_dop=par['netw']['tata_dop']
        self.weight_val=par['conn'][name]['weight_val']
        self.weight_setup=par['conn'][name]['weight_setup']
        
    def __str__(self):
        return self.name
    
    def _add_connections(self, d_out, p_out, dr, po):
        '''
        Retrieve pre and post neurons that connect given a pool of source and target nodes. 
        Take source and target nodes as parameter. 
        '''
        # Short cuts for speed up
        am=self.driver.apply_mask
        ak=self.driver.apply_kernel
        fan=self.fan_pool

        # Get connections
        nodes=[ak(am(self.pool.pos[v], self.mask, dr), fan) for v in po]
        n_bp=map(len, nodes)
   
        d_out.extend(reduce(lambda x,y: list(x)+list(y), nodes))
        p_out.extend(reduce(lambda x,y: list(x)+list(y), [(v,)*n_bp[i] for i, v in enumerate(po) ]))

    def set_connections(self, save_mode=True):
        
        s=''
        if self.beta_fan_in:
            s='-dop-'+str(self.tata_dop)
           
        save_path=self.save_at+'conn-'+str(int(self.network_size))+'/'+self.name+s
        t=time.time()
        if os.path.isfile(save_path+'.pkl') and save_mode: 
            self.n_conn, self.conn_pre, self.conn_post=data_to_disk.pickle_load(save_path)
        
        else:
        
            
            driver_out=[]
            pool_out=[]
            # Each presynaptic node connects to only one postsynaptic
            # node. Requires that the number of presynaptic nodes 
            # equals the number of postsynaptic node.
            if self.rule=='1-1':
                if self.source.n!=self.target.n: 
                    raise Exception('number of pre nodes needs to equal number of post')
                driver_out, pool_out= self.driver.idx, self.pool.idx
                self.sets_mapping_driver=None
                self.sets_mapping_pool=None
            # Connects all presynaptic neurons randomally with connection
            # probability k_source to the pool of postsynaptic neurons 
            # defined by edge_wrap and mask_dist 
            elif self.rule=='all':
                #if self.driver.name=='STN':
                #    pass
                dr=self.driver.idx
                po=self.pool.idx
                self._add_connections(driver_out, pool_out, dr, po)
                self.sets_mapping_driver=None
                self.sets_mapping_pool=None
                
            # Randomly connects presynaptic neurons of set i with postsynaptic 
            # neurons of set i. Constrained by k_source, mask_dist and the sets. 
            elif self.rule in 'set-set':
                i=0
                for se_dr, se_po in zip(self.sets_driver, self.sets_pool):        
                  
                    dr=self.driver.idx[se_dr]
                    po=self.pool.idx[se_po]
                    
                    n_pre=len(driver_out)
                    self._add_connections(driver_out, pool_out, dr, po)
                    n_post=len(driver_out)
                
                    self.sets_mapping_driver={i: slice(n_post,n_post, 1)}
                    self.sets_mapping_pool={i: slice(n_post,n_post, 1)}
                    i+=1
            # Randomly connects presynaptic neurons from set i to all postsynaptic sets except
            # to set i.     
            elif self.rule=='set-not_set':
                                     
                for se_dr, se_po in zip(self.sets_driver, self.sets_pool):       
                  
                    dr=self.driver.idx[se_dr]
                    po=set(self.pool.idx).difference(self.pool.idx[se_po])    
                    
                    self._add_connections(driver_out, pool_out, dr, po)
      
            # Load learned connections     
            elif self.rule=='set-all_to_all':
                if self.connection_type=='divergent':
                    pass
       
                elif self.connection_type=='convergent':
                    raise Exception('set-all_to_all connections need to be divergent')  
                
                
                for i, se_pre in enumerate(self.sets_driver):
                    for j, se_post in enumerate(self.sets_pool):
                        pre =self.pre.idx[se_pre]
                        post=self.pool.idx[se_post]
                        
                        self._add_connections(driver_out, pool_out, pre, post)
                        self.sets_mapping_driver={i+j: slice(n_post,n_post, 1)}
                        self.sets_mapping_pool={i+j: slice(n_post,n_post, 1)}
  
            if self.connection_type=='divergent':
                self.conn_pre, self.conn_post= driver_out, pool_out
                
            elif self.connection_type=='convergent':
                self.conn_pre, self.conn_post= pool_out, driver_out 
            
            self.n_conn=len(pool_out)
            
            if save_mode:
                data_to_disk.pickle_save([self.n_conn, self.conn_pre, self.conn_post], save_path) 
        
        t=time.time()-t
        print 'Structure: {0:18} Connections: {1:8} Fan pool:{2:6} Time:{3:5} sec'.format(self.name, len(self.conn_pre), 
                                                                                            round(float(len(self.conn_pre))/self.pool.n,0), 
                                                                                            round(t,2))
           
    def get_delays(self):
        x=self.delay_setup
        if 'constant' in x.keys():
            return numpy.ones(self.n_conn)*x['constant']
        elif 'uniform' in x.keys():
            return list(numpy.random.uniform(low=x['uniform']['min'], 
                                             high=x['uniform']['max'], 
                                             size=self.n_conn))      
    
    def get_weights(self):
        x=self.weight_setup
        if 'constant' in x.keys():
            return numpy.ones(self.n_conn)*x['constant']
        if 'uniform' in x.keys():
            return list(numpy.random.uniform(low=x['uniform']['min'], 
                                             high=x['uniform']['max'], 
                                             size=self.n_conn))    
        if 'learned' in x.keys():
            weights=data_to_disk.pickle_load(self.data_path_learned_conn)
                 
            n_set_pre, n_set_post=weights.shape # Driver is pre
            n_pre=self.soruce.n 
            n_post=self.target.n    
                
            # Get range of sets of presynaptic subpoulations
            step=n_pre/n_set_pre
            sub_pops_pre=numpy.array([[step*i,step*(i+1)]  for i in range(n_set_pre)])
                
            # Get range of sets of postsynaptic subpoulations
            step=n_post/n_set_post
            sub_pop_pool=[[step*i, step*(i+1)]  for i in range(n_set_post)]
            
            conns=numpy.zeros(self.n_conn)
            for i in range(n_set_pre):
                se_pre=sub_pops_pre[i]
                for j in range(n_set_post):
                    se_post=sub_pop_pool[j]
                    
                    # For each weight pick out the specific pre and post connections
                    idx_conn_bool=(self.conn_pre>se_pre[0])*(self.conn_pre<se_pre[1])
                    idx_conn_bool*=(self.conn_post>se_post[0])*(self.conn_post<se_post[1])
                    conns[idx_conn_bool]= weights[i]
                    
            
    def plot_hist(self, ax):
        if self.n_conn==0:
            self.set_connections()
        pre, post=self.conn_pre, self.conn_post
        #ax.hist(post,self.target.n )
        # numpy.histogram()
        n_pre, bins =   numpy.histogram(pre, bins=self.source.n )
        n_post, bins = numpy.histogram(post, bins=self.target.n )
        
        x_pre=numpy.linspace(0, 1, self.source.n)
        x_post=numpy.linspace(0, 1, self.target.n)
        
        ax.plot(x_pre,n_pre)
        ax.twinx().plot(x_post, n_post, 'r')
        
        fan_in=numpy.mean(n_post)#float(len(post))/self.target.n # Number of incomming MSN connections per SNr
        n_out=numpy.mean(n_pre)#float(len(pre))/self.source.n # 
        fan_in_std=numpy.std(n_post)
        n_out_std=numpy.std(n_pre)
        ax.set_title('{0}\nIn/out:{1}({2})/{3}({4})'.format(self.name, round(fan_in,1),round(fan_in_std,1), 
                                                            round(n_out,1), round(n_out_std,1)))

    def pprint(self):
        s='name:{0:20} source:{1:7} target:{2:7} fan_in:{3:4} syn:{4:16} lesion:{5:6} sets:{6:3} rule:{7:10} mask:{8:6}'
        return s.format(self.name, self.source.name, self.target.name, str(self.fan_in), self.syn, 
                        str(self.lesion), str(self.sets), self.rule,[round(self.mask[0],1),round(self.mask[1],1)]) 
    
       
class Structure_list(object):
    
    def __init__(self, setup=None):
        
        self.list=[]        
        if setup:
            for name, params in setup: 
                self.append(name, params) 
    
    def __getitem__(self,a):
        return self.list[a]
    
    def __len__(self):
        return len(self.list)    
    
    def __str__(self):
        return str(['conn_obj:'+str(i) for i in self.list])
        
    def append(self,k,v):
        self.list.append(Structure( k,v ))
        
    def extend(self,k,v):
        self.list.extend(Structure( k,v))
    
    def pprint(self):
        pos='{0:20}{1:10}{2:9}{3:6}{4:21}{5:8}{6:5}{7:14}{8:6}'
        print pos.format('name','source','target','fan_in','syn','lesion','sets','rule', 'mask')    
        for u in self.list:
            print pos.format(u.name, u.source.name, u.target.name, str(u.fan_in), u.syn, 
                        str(u.lesion), str(u.sets), u.rule,[round(u.mask[0],1),round(u.mask[1],1)]) 
    
    def recover_units(self):
        
        units_dic={}
        for s in self.list:
            units_dic[s.source.name]=s.source
            units_dic[s.target.name]=s.target
            
        units_list=units_dic.values()
    
        
        # So that there is only one units object per units. When loading
        # a structure multiple copies of units object will exist 
        for s in self.list:
            s.source=units_dic[s.source.name]
            s.target=units_dic[s.target.name]
            
        return units_dic, units_list
        
        
    
    def sort(self):
        self.list=sorted(self.list,key=lambda x:x.name)   
        
        


class TestStructure(unittest.TestCase):
    
    def setUp(self):
        
        self.par_bcpnn=Par_bcpnn(par_rep={'netw':2000.0})
        
        self.u1=Units('CO', {'par':self.par_bcpnn })   
        self.u1=Units('M1', {'par':self.par_bcpnn }) 
        
        self.s1=Structure('CO_M1_ampa', {'source':self.u1, 'target':self.u2, 'par':self.par_bcpnn})
        
        
                
    def test_set_all_to_all(self):
        self.s1.set_connections(False)
        
        
            
    

        
if __name__ == '__main__':
    unittest.main() 