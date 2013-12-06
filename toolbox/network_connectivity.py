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

from toolbox import data_to_disk, my_nest
import unittest

class Units(object):
    '''
    A s represents either group of input nodes, neuron model nodes, 
    or background nodes. It has different properties that define
    it, such as extent, edge_wrap, neuron model, number of neurons. 
    '''
    def __init__(self, name, dic, par):
        
        par=par['node'][name]
        
        self.collected_spikes=False
        self.collected_votage_traces=False
        self.name=name
        self.n=par['n']
        self.n_sets=par['n_sets']
        self.model=par['model']
        self.lesion=par['lesion']
        self.extent=par['extent']
        self.edge_wrap=par['edge_wrap']
        self.population=None # Empty container to put a group or input from population module
        self.sets=[slice(s, self.n, self.n_sets) for s in range(self.n_sets)]
        self.type=par['type']

        if 'prop' in par.keys():
            self.proportion_of_network=par['prop']
        else: 
            self.proportion_of_network=0.0

    
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

    def __repr__(self):
        return self.__class__.__name__+':'+self.name

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
        
    
    def copy(self):

        return copy.deepcopy(self) # deep (recursive) copy
    
    def pprint(self):
        s='name:{0:10} n:{1:6} model:{2:16} lesion:{3:6}'
        return s.format(self.name, str(self.n), self.model, str(self.lesion), 
                        str(round(self.I_e,0)), str(round(self.proportion_of_network,3)))
    
    def set_population(self, population):
        self.population=population


class Units_input(Units):
    def __init__(self, name, dic, par):
        super( Units_input, self ).__init__(name, dic, par)
        self._init_extra_attributes(par)
           
    def _init_extra_attributes(self, par):
        # Add new attribute
        self.rate=par['node'][self.name]['rate']

        
class Units_neuron(Units):
    
    def __init__(self, name, dic, par):
        super( Units_neuron, self ).__init__(name, dic, par)
        self._init_extra_attributes(par)
                 
    def _init_extra_attributes(self, par):
        # Add new attribute
        par=par['node'][self.name]
        self.proportion_of_network=par['prop']
        self.I_vitro=par['I_vitro']
        self.I_vivo=par['I_vivo']
        self.randomization=par['randomization']
        self.target_rate=par['target_rate']
       
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
    
    def __init__(self, name, dic, par):
        
        # Need to set them first
        self.source=dic['source'] # As units
        self.target=dic['target']
        
        self.connection_type=par['conn'][name]['connection_type']
        self.conn_pre=[]
        self.conn_post=[]
        self.delay_val=par['conn'][name]['delay_val']
        self.delay_setup=par['conn'][name]['delay_setup']
        self.lesion=par['conn'][name]['lesion']
        
        
        self.fan_in=int(par['conn'][name]['fan_in'])
        self.beta_fan_in=par['conn'][name]['beta_fan_in']
                                 
        # So that the fan always is less or equal to number of targets
        #if self.fan_in >self.target.n:
        #    self.source.n=self.fan_in
            
        self.name=name
        self.n_conn=0
        self.network_size=par['netw']['size']
        
        
        #Defines which pool nodes are considered as potential targets based on distance
        self.mask=par['conn'][name]['mask'] 
        
        # Defines which pool nodes are considered based on hierarchy 
        self.rule=par['conn'][name]['rule']
        
        if self.rule=='set-not_set' and self.source.n_sets==1: # If we only have one set
            self.rule='set-set' 
        
        self.data_path_learned_conn=par['conn'][name]['data_path_learned_conn']
        
        self.save_at=dic['save_at']
        
        self.sets_conn=[]
        
           
        # When creating a divergent connection, each node in the source units 
        # is visited in turn and selects target nodes from the target units. Masks, kernels,
        # and boundary conditions are applied in the target units
        if self.connection_type=='divergent':
            self.driver=self.source
            self.pool=self.target
            self.fan_pool=self.fan_in

                
        # When creating a convergent connection between units, each node in the 
        # target units are visited in turn and sources are selected for it in the
        # source units. Masks and kernels are applied to the source units, and periodic
        # boundary conditions are applied in the source units, provided that the source
        # units has periodic boundary conditions.    
        elif self.connection_type=='convergent':
            self.driver=self.target
            self.pool=self.source 
            self.fan_pool=int(round(self.fan_in*self.target.n/float(self.source.n)))

        
        # The sets are mixed up. With three sets then every third
        # node starting from 0 will be from the first set, every
        # third node starting from 1 will be from the second set 
        # and the rest will be from the third set.
     
        self.syn=par['conn'][name]['syn']     
        self.tata_dop=par['netw']['tata_dop']
        self.weight_val=par['conn'][name]['weight_val']
        self.weight_setup=par['conn'][name]['weight_setup']
    
    def __repr__(self):
        return self.__class__.__name__+':'+self.name    
    
    def __str__(self):
        return self.name+':'+str(len(self.conn_pre))
    
    def _add_connections(self, d_out, p_out, dr, po):
        '''
        Retrieve pre and post neurons that connect given a pool of source and target nodes. 
        Take source and target nodes as parameter. 
        '''
        # Short cuts for speed up
        am=self.driver.apply_mask
        ak=self.driver.apply_kernel
        fan=self.fan_pool
        
        if self.rule=='set-all_to_all':
            fan=fan/self.driver.n_sets
            if fan<1:
                raise Exception('For rule "set-all_to_all" the fan in has to be bigger than number driver sets')

        if self.rule=='set-not_set':
            fan=fan/(self.driver.n_sets-1)
        # Get connections
        nodes=[ak(am(self.pool.pos[v], self.mask, dr), fan) for v in po]
        n_bp=map(len, nodes)
   
        d_out.extend(reduce(lambda x,y: list(x)+list(y), nodes))
        p_out.extend(reduce(lambda x,y: list(x)+list(y), [(v,)*n_bp[i] for i, v in enumerate(po) ]))

    def set_connections(self, save_mode=True, display_print=True):
        
        s=''
        if self.beta_fan_in:
            s='-dop-'+str(self.tata_dop)
           
        save_path=self.save_at+'conn-'+str(int(self.network_size))+'/'+self.name+s+str(self.source.n+self.target.n)
        t=time.time()
        if os.path.isfile(save_path+'.pkl') and save_mode: 
            self.n_conn, self.conn_pre, self.conn_post, self.sets_conn=data_to_disk.pickle_load(save_path)
        
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

                for se_dr, se_po in zip(self.driver.sets, self.pool.sets):        
                  
                    dr=self.driver.idx[se_dr]
                    po=self.pool.idx[se_po]
                    
                    n_pre=len(driver_out)
                    self._add_connections(driver_out, pool_out, dr, po)
                    n_post=len(driver_out)
                
                    self.sets_conn.append(slice(n_pre, n_post, 1))


            # Randomly connects presynaptic neurons from set i to all postsynaptic sets except
            # to set i.     
            elif self.rule=='set-not_set':
                                     
                for se_dr, se_po in zip(self.driver.sets, self.pool.sets):       
                  
                    dr=self.driver.idx[se_dr]
                    po=list(set(self.pool.idx).difference(self.pool.idx[se_po]) )   
                    n_pre=len(driver_out)
                    self._add_connections(driver_out, pool_out, dr, po)
                    n_post=len(driver_out)
                    
                    self.sets_conn.append(slice(n_pre, n_post, 1))
            # Load learned connections     
            elif self.rule=='set-all_to_all':
                
                for se_driver in self.driver.sets:
                    for se_pool in self.pool.sets:
                        driver =self.driver.idx[se_driver]
                        pool=self.pool.idx[se_pool]
                        
                        n_before=len(driver_out)
                        self._add_connections(driver_out, pool_out, driver, pool)
                        n_after=len(driver_out)
                        
                        self.sets_conn.append(slice(n_before, n_after, 1))
  
            if self.connection_type=='divergent':
                self.conn_pre, self.conn_post= driver_out, pool_out
                
            elif self.connection_type=='convergent':
                self.conn_pre, self.conn_post= pool_out, driver_out 
            
            self.n_conn=len(pool_out)
            
            assert isinstance(self.conn_pre, list)
            
            if save_mode:
                data_to_disk.pickle_save([self.n_conn, self.conn_pre, self.conn_post, self.sets_conn], save_path) 
            
            
        t=time.time()-t
        if display_print:
            print 'Structure: {0:18} Connections: {1:8} Fan pool:{2:6} Time:{3:5} sec'.format(self.name, len(self.conn_pre), 
                                                                                              round(float(len(self.conn_pre))/self.pool.n,0), 
                                                                                              round(t,2))
           
    def get_delays(self):
        x=self.delay_setup
        if 'constant' == x['type']:
            return numpy.ones(self.n_conn)*x['params']
        elif 'uniform' == x['type']:
            return list(numpy.random.uniform(low=x['params']['min'], 
                                             high=x['params']['max'], 
                                             size=self.n_conn))      
    
    def get_weights(self):
        x=self.weight_setup
        if 'constant' == x['type']:
            return numpy.ones(self.n_conn)*x['params']
        if 'uniform' == x['type']:
            return list(numpy.random.uniform(low=x['params']['min'], 
                                             high=x['params']['max'], 
                                             size=self.n_conn))    
        if 'learned' == x['type']:
            weights=data_to_disk.pickle_load(self.data_path_learned_conn)
            
            conns=numpy.zeros(self.n_conn)
            for i, sl in enumerate(self.sets_conn):
                conns[sl]=weights[i]
            return list(conns*x['params'])    
            
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
            for k, dic, par in setup: 
                self.append(k, dic, par) 
    
    @property
    def size(self):
        size=0
        for s in self.list:
            size+=len(s.conn_pre)
        return size
    
    def __getitem__(self,a):
        return self.list[a]
    
    def __len__(self):
        return len(self.list)    

    def __repr__(self):
        return self.__class__.__name__+':'+self.name
    
    def __str__(self):
        s='\n****************\nConnection info\n****************'
        s+='\n{0:14}{1}'.format('Total number:', self.size)
        s+='\n{0:14}'.format('By connection:')
        
        text_lists=[]
        for struc in sorted(self.list, key=lambda x:x.name):
            text_lists.append('{0:>5}:{1:<8}'.format(struc.name, len(struc.conn_pre)))
        
        text_lists=[text_lists[i:-1:6] for i in range(6)] 

        for i, text_list in enumerate(text_lists):
            if i==0: pass
            else:s+='\n{0:14}'.format('')
            s+=''.join(text_list)
        
        return s
        
    def append(self,k,v, p):
        self.list.append(Structure( k, v, p ))
        
    def extend(self,k,v,p):
        self.list.extend(Structure( k,v,p))
    
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
        
        
class TestUnitsPar(unittest.TestCase):
    
    def setUp(self):
        from toolbox.default_params import Par
        self.par=Par()

    def test_units_create(self):
        units=[]
        for k, val in self.par['node'].items():
            dic={}
            units.append(val['unit_class'](k, dic, self.par))

class TestUnitsPar_bcpnn_h0(TestUnitsPar):
    
    def setUp(self):
        from toolbox.default_params import Par_bcpnn_h0
        self.par=Par_bcpnn_h0()
        
class TestStructurePar_bcpnn(unittest.TestCase):
    
    def setUp(self):
        
        #To avoid circular dependency
        from toolbox.default_params import Par_bcpnn_h0
        
        self.units_dic={}
        self.par=Par_bcpnn_h0(dic_rep={'netw':{'size':2000.0}})
        for k, v in self.par['node'].items():
            self.par['node'][k]['n']=10
            self.units_dic[k]=(v['unit_class'](k, {}, self.par))
        
        
        dplc='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-fake/CO_M1.pkl'
        
        self.par_bcpnn=Par_bcpnn_h0(dic_rep={'netw':{'size':2000.0}, 
                                          'node':{'CO':{'n':150},'M1':{'n':150}, 'SN':{'n':100}},
                                          'conn':{'CO_M1_ampa':{'sets':[10,5],
                                                                'fan_in':15.,
                                                                'rule':'set-all_to_all',
                                                                'data_path_learned_conn':dplc},
                                                  'M1_SN_gaba':{'sets':[2,2],
                                                                'fan_in':10.,
                                                                'rule':'set-set'}}})
        
        self.u1=Units('CO', {}, self.par_bcpnn )   
        self.u2=Units('M1', {}, self.par_bcpnn ) 
        self.u3=Units('SN', {}, self.par_bcpnn ) 
        
        self.s1=Structure('CO_M1_ampa', {'source':self.u1, 'target':self.u2, 'save_at':''}, 
                                         self.par_bcpnn)
        self.s2=Structure('M1_SN_gaba', {'source':self.u2, 'target':self.u3, 'save_at':''}, 
                                         self.par_bcpnn)        
        
               
    def test_set_all_to_all(self):
        self.s1.set_connections(False, display_print=False)
        set_source=self.par_bcpnn['conn']['CO_M1_ampa']['sets'][0]
        n_target=self.par_bcpnn['node']['M1']['n']
        self.assertAlmostEqual(set_source*n_target/float(self.s1.n_conn),1.0,1)
        
    def test_set_set(self):
        self.s2.set_connections(False, display_print=False)
        fan_in=self.par_bcpnn['conn']['M1_SN_gaba']['fan_in']
        n_target=self.par_bcpnn['node']['SN']['n']
        self.assertAlmostEqual(fan_in*n_target/self.s2.n_conn,1.0,1)        
    
    def test_get_weights_learned(self):
        self.s1.set_connections(False, display_print=False)
        w=self.s1.get_weights()
        weights=data_to_disk.pickle_load(self.s1.data_path_learned_conn)*self.s1.weight_setup['params']
        self.assertSetEqual(set(w), set(weights))
    
    def test_get_weights_learned2(self):

        for k, v in self.par['conn'].items():
            if v['weight_setup']['type']=='learned':
                dic={}
                soruce, target=k.split('_')[0:2]
                dic['source']=self.units_dic[soruce]
                dic['target']=self.units_dic[target]
                dic['save_at']=''
                s=Structure(k, dic, self.par)
                s.set_connections(False, display_print=False)
                w1=s.get_weights()
                w2=data_to_disk.pickle_load(s.data_path_learned_conn)*s.weight_setup['params']
                
                self.assertSetEqual(set(w1), set(w2))
      
          
    def test_get_weights_uniform(self):
        self.s2.set_connections(False, display_print=False)
        w=self.s2.get_weights()
        mw=numpy.mean(w)
        self.assertAlmostEqual(.1, self.s2.weight_val/mw/10.0,1)    
        
    
        
        
    def test_save_conn_change_sub_sampling(self):    
        self.par_bcpnn
        
        pass


class TestStructuresPar(unittest.TestCase):
    
    def build(self):
        # Create input units
        self.units_dic={}
        for k,v in self.par['node'].iteritems():   
            dic={}
            
            # Change neuron numbers for speed 
            #self.par['node'][k]['n']=10
            self.units_dic[k]=v['unit_class'](k, dic, self.par)

        
        setup_structure_list=[]
        for k, v in self.par['conn'].iteritems(): 
            s=k.split('_')
            keys=self.units_dic.keys()
            if (s[0] in keys) and (s[1] in keys):
                # Add units to dictionary
                dic={}
                dic['source']=self.units_dic[s[0]]
                dic['target']=self.units_dic[s[1]]
                dic['save_at']=''
                setup_structure_list.append([k, dic, self.par])
    
        self.structures=Structure_list(setup_structure_list)
        for s in sorted(self.structures,key=lambda x:x.name):
            s.set_connections(save_mode=False, display_print=False)

    
    def setUp(self):
        #To avoid circular dependency
        from toolbox.default_params import Par   
        self.par=Par(dic_rep={'netw':{'size':500.0}})
        self.build()   
                         
    def test_str_method(self):
        s=str(self.structures)
        self.assertTrue(isinstance(s, str))     
        
    def test_sets(self):
    
        for key, u in self.units_dic.items():
            n=0
            max_idx=0
            for se in u.sets:
                n+=len(u.idx[se])
                max_idx=max([max_idx, max(u.idx[se])])
            self.assertEqual(n, self.par['node'][key]['n'])
            self.assertEqual(max_idx, self.par['node'][key]['n']-1)     

class TestStructuresPar_slow_wave(TestStructuresPar): 
    def setUp(self):
        #To avoid circular dependency
        from toolbox.default_params import Par_slow_wave   
        self.par=Par_slow_wave(dic_rep={'netw':{'size':500.0}})
        self.build()   
                      
class TestStructuresPar_bcpnn_h0(TestStructuresPar): 
    def setUp(self):
        #To avoid circular dependency
        from toolbox.default_params import Par_bcpnn_h0   
        self.par=Par_bcpnn_h0(dic_rep={'netw':{'size':1500.0},
                                       'sub_sampling':{'M1':500.0}}) 
        #When sub sampling apply_mask can return zero nodes. This is now handled.
        self.build()          
        
if __name__ == '__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestStructuresPar_bcpnn_h0)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    
    unittest.main() 
