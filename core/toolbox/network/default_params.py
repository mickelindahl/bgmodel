'''
Module:
lines

Non dependable parameters are defined in konstructor of
the Par class. Dependable parameters are the initilized to
None. The class method update_dependable then updates all the 
dependable parameters.

Parameters are devided hierarchally into chategories:
simu - for simulation
netw - for network
conn - for connections
node - for a node in the networks
nest - parameters for available models and synapses 

simu:
Includes properties related to the  simulation, like start and stop of simulation,
nest kernel values, number of threads.

netw:
This cathegory includes properties related to the network as a whole. 
For example size, dopamine level,    

conn:
Inclues parameters related to connectivity. Each connection parameter have a
name.

node:
Parameters related to neurons and inputs.

nest:
Parameters defining nest models, synapses, input and neurons

All parameters are stored in a single dictionary 'dic'. Example:
dic['netw']['size']=1000.0
dic['conn']['C1_M2_ampa']['syn']='C1_M2_ampa' - nest synapses
dic['node']['M1']['model']='M1' - copied nest model

with
dic['nest']['C1_M2_ampa']['weight']=1.0
then variations can be created
dic['nest']['C1_M2_ampa_s']['weight']=1.0*2 #a for ampa, s for static


Abrevation for readability:
C1 - Cortex input MSN D1 node
C2 - Cortex input MSN D2 node
CF - Cortex input FSN node
CS - Cortex input STN node
EA - External input GPe A node
EI - External input GPe I node
ES - External input SNr node
M1 - MSN D1 node
M2 - MSN D2 node
B1 - MSN D1 background node
B2 - MSN D2 background node
FS - FSN node
ST - STN node
GA - GPe type A node
GI - GPe type I node
SN - SNr node

Functions:
models   - define neuron and synapse models
network  - define layers and populations
'''

from copy import deepcopy
from toolbox import directories as dr
from toolbox import misc
from toolbox.misc import my_slice
import sys
import os

from toolbox import my_nest as nest # Has to after misc. 

import numpy
import operator
import warnings
import pprint
pp=pprint.pprint


#@todo: change name on Node to Surface
class Call(object):

    obj=None
    def __init__(self, *args, **kwargs):

        if type(args[0])==list:
            args=args[0]

        self.dtype=kwargs.get('dtype', None)
        
        if 'dtype' in kwargs.keys():
            del kwargs['dtype']

        self.method=args[0]
        self.args=args[1:]
        self.kwargs=kwargs  
        self.op=None 
        self.parent=None
        self.val=None 
        self.order=''
        
    
    @classmethod
    def set_obj(cls, obj):
        cls.obj=obj #caller
       
    def do(self, obj=None):
        try:
            return self._do(obj)
        except Exception as e:
            s='\nTrying to do Call {} (id {}) with parent ->'
            s=s.format(self, id(self), self.parent)
            s+=self.parent_chain('\n\t')
            s+='\nWith object:'+str(obj.__repr__())
            raise type(e)(e.message + s), None, sys.exc_info()[2]
           
    def parent_chain(self, t):
        if self.parent:
            s=t+'with parent {} (id {}) ->'.format( self.parent, id(self.parent))
            s+=self.parent.parent_chain(t=t+'\t')
        else:
            s='parent {}'.format(self.parent)
        return s
            
    def _do(self, obj=None):

        if obj==None:
            try:
                obj=self.obj
            except:
                raise RuntimeError(('Need to set class variable self.obj in'+
                                    ' Call class or provide obj at do call'))
            
        call=getattr(obj, self.method)

        if id(self)==id(self.val):
            s='val can not be a reference to self: {}'
            raise RuntimeError(s.format(self))

        if id(self)==id(self.parent):
            s='parent can not be a reference to self: {}'
            raise RuntimeError(s.format(self))

        if not self.val:
            a=call(*self.args, **self.kwargs)
        else:
            #print self.parent!=None
            if self.parent!=None:
                val_parent=self.parent.do(obj)  #ensures obj integrity  
                   
            if isinstance(self.val, Call):
                val=self.val.do(obj)
            else:
                val=self.val
            
            try:
                if self.order=='right':    
                    a=self.op(val, val_parent)
                    
                elif self.order=='left':
                    a=self.op(val_parent, val)
                    
            except Exception as e:
                s='\nTrying to to operation {} on {} and {} \nself for {} (id {})'
                s+=' and parent {} (id {})'
                s=s.format(self.op.__doc__, val, val_parent, self, id(self),
                           self.parent, id(self.parent))
                raise type(e)(e.message + s), None, sys.exc_info()[2]
        if self.dtype:
            a=self.dtype(a)        
        return a

             
# import sys
#             print self
#             print "Unexpected error:", sys.exc_info()[0]
#             raise
                         
    def __repr__(self):
        s=[str(a) for a in self.args[1:]]
        return self.__class__.__name__ +':'+self.method+'_'+'_'.join(s)       

    def __eq__(self, other):
        if hasattr(other, 'method'):
            return self.method==other.method
        else:
            False
       
    def add_left(self, new, other):
        new.val=other
        new.order='left'
        new.parent=self
        return new

    def add_right(self, new, other):
        new.val=other
        new.order='right'
        new.parent=self
        return new
            
    def __add__(self, other):
        new=deepcopy(self)
        new.op=operator.__add__
        return self.add_left(new, other)

    def __radd__(self, other):
        new=deepcopy(self)
        new.op=operator.__add__
        return self.add_right(new, other)
    
    def __div__(self, other):
        new=deepcopy(self)
        new.op=operator.__div__
        return self.add_left(new, other)
    
    def __rdiv__(self, other):
        new=deepcopy(self)
        new.op=operator.__div__
        return self.add_right(new, other)


    def __mul__(self, other):
        new=deepcopy(self)
        new.op=operator.__mul__
        return self.add_left(new, other)
    
    def __rmul__(self, other):
        new=deepcopy(self)
        new.op=operator.__mul__
        return self.add_right(new, other)  
    
    def __sub__(self, other):
        new=deepcopy(self)
        new.op=operator.__sub__
        return self.add_left(new, other)   
    
    def __rsub__(self, other):
        new=deepcopy(self)
        new.op=operator.__sub__
        return self.add_right(new, other)
    
    def __ne__(self, other):
        if hasattr(other, 'method'):
            return self.method!=other.method
        else:
            return True

class DepConn(Call):
    def __init__(self, *a, **k):
        super(DepConn, self).__init__( *(['_dep', 'conn']+list(a)), **k)

class DepNest(Call):
    def __init__(self, *a, **k):
        super(DepNest, self).__init__( *(['_dep', 'nest']+list(a)), **k)

class DepNetw(Call):
    def __init__(self, *a, **k):
        super(DepNetw, self).__init__( *(['_dep', 'netw']+list(a)), **k)
        
class DepNode(Call):
    def __init__(self, *a, **k):
        super(DepNode, self).__init__( *(['_dep', 'node']+list(a)), **k)

class GetConn(Call):
    def __init__(self, *a, **k):
        super(GetConn, self).__init__( *(['_get', 'conn']+list(a)), **k)

class GetNest(Call):
    def __init__(self, *a, **k):
        super(GetNest, self).__init__( *(['_get', 'nest']+list(a)), **k)

class GetNetw(Call):
    def __init__(self, *a, **k):
        super(GetNetw, self).__init__( *(['_get', 'netw']+list(a)), **k)
        
class GetNode(Call):
    def __init__(self, *a, **k):
        super(GetNode, self).__init__( *(['_get', 'node']+list(a)), **k)

class GetSimu(Call):
    def __init__(self, *a, **k):
        super(GetSimu, self).__init__( *(['_get', 'simu']+list(a)), **k)

class Perturbation(object):   
     
    def __init__(self, keys, val, op):
        if isinstance(keys, str):
            keys=keys.split('.')
        self.op=op 
        self.keys=keys
        self.val=[val]
 
    def __eq__(self, other):
        return self.__repr__()==other.__repr__()

    def __lt__(self, other):
        return self.__repr__()<other.__repr__()
              
    def __hash__(self):
        return hash((tuple(self.keys), tuple(self.val), self.op))
     
    def __repr__(self):
        s=''
        for v in self.val:
            s+=self.op+str(v)
        
        # Such that equal comes before multiplying in sorting
        #perturbations
        if self.op=='=':
            priority='0'
        else:
            priority='1'
         
        return  priority+'.'+'.'.join(self.keys)+s
        return  '.'.join(self.keys)+s
    
    def __str__(self):
        return self.__repr__()
    
    def _get_val(self):
        return self.val 
    
    def set_val(self, val):
        
        if type(val)!=list:
            val=[val]
        self.val=val

    def apply(self, dic):
        for v in self.val:
            dic=misc.dict_apply_operation(dic, 
                                          self.keys, 
                                          v, 
                                          self.op)
      
            
class Perturbation_list(object):
                                                                                             
    def __init__(self, *args, **kwargs):
        ''' 
        args as (dictionary, operation, dictionary, 
        operation, ...  
        '''
        
        self.name=kwargs.get('name', '')
#         self.list=[]
        self.dic={}
        iterator=[args[i:i+2] for i in range(0,len(args),2)]
 
        for keys, val, op in self.iterate(iterator):
            p=Perturbation(keys, val, op)     
            key='.'.join(keys+[op])
            self._add_to_dic(key, p)
    
    @property
    def list(self):
        return self.dic.values() 

    def __eq__(self, other):
        return not set(self.list).isdisjoint(set(other.list))
        
    def __str__(self):
        return  self.name+':'+str(sorted(self.list))

    def __repr__(self):
        return  self.name+':'+str(sorted(self.list))
    
    def __getitem__(self, val):
        return self.list[val]
        
    def __add__(self, other):       
        new=deepcopy(self)
        new.append(other)
        return new
    
    def __radd__(self, other): 
        new=deepcopy(self)
        if not isinstance(other, Perturbation_list):
            return new
        
        new.append(other)
        return new
    
    def _add_to_dic(self, key, p):
        if key in self.dic:
            if p.op=='*':
                warnings.warn('Existing perturbation {} adding {}'.format(self.dic[key],p))
                self.dic[key].val+=p.val
            elif p.op=='=':
                warnings.warn('Exsisting perturbation {} overwriting to {}'.format(self.dic[key],p))
                self.dic[key]=p
            else:
                raise RuntimeError('Perturbation {} already exist'.format(self))
        else:
            self.dic[key]=p
      
    def append(self, other):
        assert type(other) is Perturbation_list,  "Not Perturbation_list object"
        if other.name:
            if self.name:
                self.name+='-'+other.name
            else:self.name=other.name
        
        for p in other.list:
            key='.'.join(p.keys+[p.op])
            self._add_to_dic(key, p)

    
    def apply_pertubations(self, dic, display=False):

        for p in sorted(self.list):
#             print p.keys
            if not misc.dict_haskey(dic, p.keys):
                warnings.warn('\nKey not in par: '+'.'.join(p.keys))
                continue
            
            try:
                p.apply(dic)
#                 dic=misc.dict_apply_operation(dic, p.keys, p.val, p.op)
            except Exception as e:
                s=('\nTrying to apply perturbation {} \n'+
                   'with keys {} with current value\n{}')
                s=s.format(p, p.keys, misc.dict_recursive_get(dic, p.keys))
                raise type(e)(e.message + s), None, sys.exc_info()[2]
                
        if display:
            print 'perturbations applied ' 
                   
        return dic        
    
    def has(self, key):
        return key in self.dic.keys()

    def get(self, key):
        return self.dic[key].val
    
    def info(self):
        d={}
        for p in self.list:
            d=misc.dict_recursive_add(d, p.keys, p.op+str(p.val))
            
        s= self.name+'\n'
        s+=pprint.pformat(d)
        return s
        
        
    def iterate(self, iterator):
        for d, op in iterator:
            for key, val in misc.dict_iter(d):
                yield key, val, op
   
    def update_single(self, val):
        key='.'.join(val.keys+[val.op])
        self.dic[key]=val
        
        
    def update_list(self, val):
        for p in val:
            key='.'.join(p.keys+[p.op])
            self.dic[key]=p
                
class Par_base(object):
    
    def __init__(self, **kwargs):
        
        self.update_triggered=False
        
        self._dic={}
        self._dic_con = {}
        self._dic_dep = {}
        self._dic_rep = kwargs.get('dic_rep', {})
        self._per = kwargs.get('perturbations', Perturbation_list())
        
        self.changed={'dic': True,
                      'dic_con': True,
                      'dic_dep':True,             
                      'dic_rep':True}

        self.dep={} #dic storing dependable calculations, dynamically growing

        self.other=kwargs.get('other', None)    
        
        path, sli_path=nest.get_default_module_paths(dr.HOME_MODULE)
        nest.install_module(path, sli_path, model_to_exist='my_aeif_cond_exp' )
        
        self.rec={}
        df=nest.GetDefaults('my_aeif_cond_exp')['receptor_types']
        self.rec['aeif']=df    # get receptor types
        df=nest.GetDefaults('izhik_cond_exp')['receptor_types']
        self.rec['izh']=df     # get receptor types  
        self.unittest=kwargs.get('unittest',False)

        Call.set_obj(self)
        self.trigger_dic_update()
    
    def __getitem__(self, key):
        return self.dic[key]

    def __setitem__(self, key, val):
        self.dic[key] = val  
            
    def __str__(self):
        
        nodes=sorted(self.dic['node'].keys())
        s='\n****************\nSimulation info\n****************'
        s+='\n{0:13}{1}'.format('Network size:', str(self.dic['netw']['size']))
        s+='\n{0:13}'.format('Nuclei sizes:')
       
        text_lists=[]
        for name in nodes:
            s='{0:>5}:{1:<6}'
            text_lists.append(s.format(name, self.dic['node'][name]['n']))
        text_lists.append('')
        text_lists=[text_lists[i:-1:5] for i in range(5)] 

        for i, text_list in enumerate(text_lists):
            if i==0: pass
            else:s+='\n{0:13}'.format('')
            s+=''.join(text_list)
        
        return self.__class__.__name__+'\n'+s     
    
    def __repr__(self):
        return  self.__class__.__name__
            
    @property
    def dic(self):
        return self._dic   

    @dic.setter
    def dic(self, val):
        self._dic=val
      
    @property 
    def dic_con(self):       
        return self._dic_con
    
    @dic_con.setter
    def dic_con(self, val):
        self._dic_con=val
        
    @property 
    def dic_dep(self):        
        return self._dic_dep   

    @dic_dep.setter
    def dic_dep(self, val):
        self._dic_dep=val
    
    @property
    def dic_rep(self):
        return self._dic_rep
    
    @dic_rep.setter
    def dic_rep(self, value):
        self._dic_rep=value            

    @property
    def keys_conn(self):
        l=['beta_fan_in', 
           'class_surface', 
           'delay', 
           'fan_in', 
           'fan_in0', 
           'netw_size', 
           'lesion', 
           'mask', 
           'rule', 
           'save', 
           'source',
           'syn', 
           'target',
           'tata_dop',
           'local_num_threads', 
           'weight']
        return l
    
    @property
    def keys_node_shared(self):
        l=['class_population',
           'class_surface', 
           'edge_wrap', 
           'extent', 
           'fan_in_distribution',
           'lesion', 
           'model', 
           'n', 
           'n_sets', 
           'nest_params', 
           'sets', 
           'rate', 
           'type']
        return l      

    @property
    def keys_node_network(self):

        l=['I_vitro', 
           'I_vivo', 
           'mm', 
           'rand', 
           'rate_in_vitro', 
           'sd', 
           ]
        l.extend(self.keys_node_shared)
        return sorted(l)
    
    @property
    def keys_node_input(self):
        l=['spike_setup',
           'target']
        l.extend(self.keys_node_shared)
        return sorted(l)


    @property
    def per(self):
        return self._per

    @per.setter
    def per(self, val):
        self._per=val

    def add(self, *args):
        if args[1] in ['netw',',simu']:
            a=[args[1:]]
        else:
            a=[]
            for name in self._get(args[1]).keys():
                a.append([args[1], name, args[2]])
        
        
        d=args[0]
        return self._add(d, *a)
        
    def _add(self, d, *args):

        
        for k1 in args:     
            
            if not misc.dict_haskey(self.dic_con, k1):
                continue
             
            val=misc.dict_recursive_get(self.dic_con, k1)
            dd=misc.dict_recursive_add({}, k1, val)

            for k2, val in misc.dict_iter(dd):
                #print k2, val
                if not isinstance(val, Call):
                    continue
                #print k2
                try:
                    d=misc.dict_recursive_add(d, k2, val.do(self))
                except Exception as e:
                    s='\nTrying to add:{}'
                    raise type(e)(e.message + s.format(str(k2))), None, sys.exc_info()[2]
                    
        return d

    def apply_pertubations(self, dic):
        return self.per.apply_pertubations(dic)

    def call_n_sum(self, *args):
        n=0.0
        for name in args[0]:
            n+=self.compute_node_sizes()[name]['n']
        return n
    
    def calc_fan_in(self, name):
        fan_in0=self._get('conn', name, 'fan_in0')
        beta=self._get('conn', name, 'beta_fan_in')
        dop=self.calc_tata_dop()
        
        return calc_fan_in(beta, dop, fan_in0)

    def calc_sets(self, name):
        n_sets=self._get('node', name, 'n_sets')
        n=self._get('node', name, 'n')
        return calc_sets(n_sets, n)
      
    def calc_n(self, name):
        d1=self._get('netw', 'n_nuclei')      
        d2=self._get('netw', 'sub_sampling')
        size=self._get('netw', 'size')
        
        return calc_n(d1,d2, name, size, obj=self)
        
    def calc_spike_setup(self, name):
        n=self._get('node', name, 'n')
        p=self._get('netw', 'input', name, 'params')
        r=self._get('node', name, 'rate')
        start=self.get_start_rec()
        stop=self.get_sim_stop()
        t=self._get('netw', 'input', name, 'type')
        
        return calc_spike_setup(n, p, r, start, stop, t)
        
    def calc_tata_dop(self, *args):
        return self._get('netw','tata_dop')-self._get('netw','tata_dop0')   
         
         
    def _dep_update(self, keys, val, _id):
        misc.dict_recursive_add(self.dep, keys+['val'], val)
        misc.dict_recursive_add(self.dep, keys+['id'], _id)
         
    def _dep(self, *args):
        ''' 
        Exampel:
        
        _dep('node','n1','calc_n')
        _dep('node','n1','calc_spike_setup')
        _dep('node','i1_n1','calc_fan_in')
        
        '''
        if type(args)!=list:
            args=list(args)
            
            
        args1=args[:-1]
        if type(args1)!=list:
            args1=list(args)
                    
        keys=args1+ [args[-1][5:]]

        obj=misc.dict_recursive_get(self.dic_con, keys) 
          
        call=Call(args[-1], args[-2])
        if not misc.dict_haskey(self.dep, args):
            val=call.do(self)
            self._dep_update(args, val, id(obj))
#             misc.dict_recursive_add(self.dep, args, val)
        
        _id = misc.dict_recursive_get(self.dep, args+['id'])
        if _id!=id(obj):
            val=call.do(self)
            self._dep_update(args, val, id(obj))             

        return misc.dict_recursive_get(self.dep, args+['val'])

    def dic_print_change(self, s, d1, d2):
        
        d1_reduced=misc.dict_reduce(d1, {}, deliminator='.')
        d2_reduced=misc.dict_reduce(d2, {}, deliminator='.')
                
        for key, val in d1_reduced.iteritems():
            if key in d2_reduced.keys():
                val_old=d2_reduced[key]
                if val_old!=val and val_old!=None:
                    print 'Change '+s +' '+ key +' '+ str(val_old)+'->'+str(val)
                    
    def _get(self, *args):
        
        if not args: 
            return self.dic_con
        if type(args[0])==list:
            args=args[0]
        try:
            args=[a.do(self) if isinstance(a, Call) else a for a in args]
        except Exception as e:
            s='\nTrying to do _get for self: {}'
            s=s.format(self)
            raise type(e)(e.message + s), None, sys.exc_info()[2]
        
        v=misc.dict_recursive_get(self.dic_con, args) 
        while isinstance(v,Call):
            v=v.do(self) 

        return v  

    def get(self, *args):
        return self._get(*args) 
     
    def _get_conn(self):
        return self.dic_con['conn']
        
    def get_conn(self):
        d=self.dic['conn']
        nodes=self.get_node().keys()
        
        dic={}
        for k, v in d.items():
            source, target=k.split('_')[0:2]
#             print v
            if v['lesion']:
                continue
                #dic[k]=d[k]
            if not ((source in nodes) and (target in nodes)):
                continue
            dic[k]=d[k] 
            
        return dic

    
    def get_dic(self):
        return self.dic
    
    def get_dic_con(self):
        return self.dic_con  
    
    def get_par_constant(self):
        return self._get_par_constant() 
  

    def get_do_reset(self):
        return self.dic_con['simu']['do_reset']
   
    def get_FF_input(self):
        return self.dic_con['netw']['FF_curve']['input']
   
    def get_FF_output(self):
        return self.dic_con['netw']['FF_curve']['output']

    def get_fopt(self):
        return self.dic_con['netw']['optimization']['f']

    def get_local_num_threads(self):
        return self.dic_con['simu']['local_num_threads']  

    def _get_nest(self):
        return self.dic_con['nest']
        
    def get_nest(self):
        return self.dic['nest']

    def _get_nest_setup(self, model):
        p=deepcopy(self.dic['nest'][model])
        return p

    def _get_netw_dependable_rates(self):
        return self.dic_con['netw']['dependable_rates']
                   
    def _get_netw_plast_syn_map(self):
        return self._dic_con['netw']['plast_syn_map'] 
                
    def _get_node(self):
        return self.dic_con['node']
    
    def get_node(self):
        dic=self.dic['node']

        l=[]
        for k, v in dic.items():
            if v['lesion']:
                l.append(k)
        for k in l: del dic[k]
        
        return dic

    def get_path_data(self):
        return self.dic['simu']['path_data']     
      
    def get_path_figure(self):
        return self.dic['simu']['path_figure']

    def get_path_nest(self):
        return self.dic['simu']['path_nest']

    def get_popu(self):
        dic={}
        for key in self.get_node().keys():
            d=self.dic['node'][key]
            keys=set(self.dic['netw']['attr_popu']).intersection(d.keys())
            dic[key]=misc.dict_slice(d, keys)
            
            dic[key]['class']=dic[key]['class_population']
            del dic[key]['class_population']

            dic[key]['params']=dic[key]['nest_params']
            del dic[key]['nest_params']
        return dic
    
    def get_popu_nest_params(self):
        d=self.get_popu()
        dic={}
        for key in d.keys():
            dic[key]=d[key]['params']
        return dic
    
    def get_spike_setup(self):
        d=self.get_node()
        dic={}
        for key, val in d.items():
            if 'spike_setup' not in val.keys():
                continue
            
            dic[key]=d[key]['spike_setup']
        return dic
        
    def get_sim_stop(self):
        return self.dic_con['simu']['sim_stop']
     
    def get_sim_time(self):
        return self.dic_con['simu']['sim_time'] 

    def get_start_rec(self):
        return self.dic_con['simu']['start_rec'] 
    
    def get_stop_rec(self):
        return self.dic_con['simu']['stop_rec']  

    
    def get_surf(self):
        dic={}
        for key in self.get_node().keys():
            d=self.dic['node'][key]
            keys=set(self.dic['netw']['attr_surf']).intersection(d.keys())
            dic[key]=misc.dict_slice(d, keys)
            dic[key]['class']=dic[key]['class_surface']
            del dic[key]['class_surface']
        
        return dic
 
    def get_to_file(self):
        return self.dic_con['simu']['sd_params']['to_file']
 
    def get_to_memory(self):
        return self.dic_con['simu']['sd_params']['to_memory']
        
    def get_unittest(self):
        return self.unittest

    def get_xopt(self):
        return self.dic_con['netw']['optimization']['x']

    def get_x0opt(self):
        return self.dic_con['netw']['optimization']['x0']
        
    def iter_plast_syn_map(self):
        for conn, val in self._get_netw_plast_syn_map.keys():
            models=val['models']
            syn=val['syn']
            for model in models:
                yield conn, syn, model    
 
    def set_print_time(self, val):
        d= {'simu':{'print_time':val}}
        self.update_perturbations(Perturbation_list(d,  '='))
#         self.update_dic_rep( {'simu':{'print_time':val}})
 
    def set_sim_time(self, val):
        d={'simu':{'sim_time':val}}
        self.update_perturbations(Perturbation_list(d,  '='))
        
#         self.update_dic_rep( {'simu':{'sim_time':val}})
 
    def set_sim_stop(self, val): 
        
        d={'simu':{'sim_stop':val,
                   'stop_rec':val}}
        self.update_perturbations(Perturbation_list(d,  '='))
        
#         self.update_dic_rep({'simu':{'sim_stop':val}})
#         self.update_dic_rep( {'simu':{'stop_rec':val}})

    def set_path_nest(self, val):
        d={'simu':{'path_nest':val}}
        self.update_perturbations(Perturbation_list(d,  '='))
        
    def clear_dep(self):
        self.dep={}
    
    def set_perturbation_list(self, val):
        self.per.update_list(val)    
#         self.per=val        
        self.trigger_dic_update()    

    def set_dic_rep(self, d):      
        self.dic_rep=d
        self.trigger_dic_update()

    def trigger_dic_update(self):
        self.clear_dep() 
        self.update_dic_con()
        self.update_dic_dep()
        self.update_dic()
#         self.update_triggered=True
    
    
    def update_dic(self):
        d={}
        d=misc.dict_update(d, self.dic_con)
        self.dic=misc.dict_update(d, self.dic_dep)
           
    def update_dic_rep(self,d):     
        self.dic_rep=misc.dict_update(self.dic_rep, d) 
        self.trigger_dic_update()
 
    def update_dic_con(self):
        dic=self._get_par_constant()     
        for keys, val in misc.dict_iter(self.dic_rep):
            try:
                dic=misc.dict_recursive_set(dic, keys, val)
            except Exception as e:
                s=('\nTrying to set dic_con at key {} with current'+
                   ' value\n{} to {}')
                s=s.format(keys, misc.dict_recursive_get(dic, keys), val )
                raise type(e)(e.message + s), None, sys.exc_info()[2]
                         
        self.apply_pertubations(dic)     
        self.dic_con=dic
    
    def update_dic_dep(self):                
        self.dic_dep=self._get_par_dependable()               

    def update_perturbation(self, perturbation):
        self.per.update_single(perturbation)
        self.trigger_dic_update() 
        
    def update_perturbations(self, perturbation_list):
        self.per.update_list(perturbation_list)
        self.trigger_dic_update()        
    
class Par_base_mixin_dummy(object):
    def _get_par_constant(self):
        return {}
    def _get_par_dependable(self):
        return {}
    
class Par_base_dummy(Par_base, Par_base_mixin_dummy):    
    pass

class Par_base_mixin(object):
    
    def build_setup_mm_params(self):
        mm_p = {'interval':GetSimu('mm_params', 'interval'), 
                'start':Call('get_start_rec'), 
                'stop':Call('get_stop_rec'), 
                'to_file':GetSimu('mm_params', 'to_file'), 
                'to_memory':GetSimu('mm_params', 'to_memory'), 
                'record_from':GetSimu('mm_params', 'record_from')}
        return mm_p
    
    def build_setup_node_rand(self, model):
        ra = {}
        ra['C_m'] = {'active':GetNetw('rand_nodes', 'C_m'), 
                     'gaussian':{'my':GetNest(model, 'C_m'), 
                                 'sigma':GetNest(model, 'C_m') * 0.2,
                                 'cut':True, 
                                 'cut_at':3},}
        
        ra['I_e'] = {'active':GetNetw('rand_nodes', 'I_e'), 
                     'gaussian':{'my':GetNest(model, 'I_e'), 
                                 'sigma':GetNest(model, 'I_e') * 0.5,
                                 'cut':True, 
                                 'cut_at':3},}
        
        ra['V_th'] = {'active':GetNetw('rand_nodes', 'V_th'), 
                      'gaussian':{'my':GetNest(model, 'V_th'), 
                      'sigma':GetNetw('V_th_sigma'), 
                      'cut':True, 
                      'cut_at':3.}}
        
        ra['V_m'] = {'active':GetNetw('rand_nodes', 'V_m'), 
            'uniform':{'min':GetNest(model, 'V_th') - 20, 
                'max':GetNest(model, 'V_th')}}
        return ra


    def build_setup_sd_params(self):
        sd_p = {'start':Call('get_start_rec'), 
                'stop':Call('get_stop_rec'), 
                'to_file':GetSimu('sd_params', 'to_file'), 
                'to_memory':GetSimu('sd_params', 'to_memory')}
        return sd_p

    def _get_attr_popu(self):
        p = ['class_population', 'model', 
            'n', 
            'mm',  
            'nest_params', 
            'rand', 
            'sd', 
            'sd_params', 
            'sets', 
            'spike_setup', 
            'syn_target',
            'rate',
            'type']
        return p
        
    def _get_attr_surf(self):
        p = ['class_surface', 
             'extent', 
            'edge_wrap', 
            'fan_in_distribution',
            'lesion', 
            'model', 
            'n', 
            'n_sets', 
            'sets',
            'type']
        return p
    
    
    
    def _get_defaults_conn(self, name, syn, **kwargs ):
        source, target=name.split('_')[0:2]
        
        syn_call=GetConn(name, 'syn')
        pr = 0.5
        if kwargs.get('conn_type','uniform')=='uniform':
            delay = {'type':'uniform', 
                     'params':{'min':GetNest(syn_call, 'delay') * pr, 
                               'max':GetNest(syn_call, 'delay') * (1 + pr)}}
            weight = {'type':'uniform', 
                      'params':{'min':GetNest(syn_call, 'weight') * pr, 
                                'max':GetNest(syn_call, 'weight') * (1 + pr)}}
        else:
            delay = {'type':'constant', 
                     'params':GetNest(syn_call, 'delay')}
            weight = {'type':'constant', 
                      'params':GetNest(syn_call, 'weight')}
        
        
        rule=kwargs.get('rule','1-1')
        
        k={'size':GetNetw('size', dtype=str),
            'source':source,
#             'source_edge_wrap':GetNode(source,'edge_wrap',dtype=str),
            'source_n':GetNode(source,'n', dtype=str),
            'source_n_sets':GetNode(source,'n_sets', dtype=str),
            'target':target,
#             'target_edge_wrap':GetNode(target,'edge_wrap',dtype=str),
            'target_n':GetNode(target,'n', dtype=str),
            'target_n_sets':GetNode(target,'n_sets', dtype=str),
      
            'beta_fan_in':GetConn(name,'beta_fan_in', dtype=str),
            'rule':GetConn(name,'rule', dtype=str),
            'fan_in':GetConn(name,'fan_in', dtype=str)}
        
        file_name=format_connectionn_save_path(**k)
        
        sp=GetSimu('path_conn')+file_name
        
        d = {'beta_fan_in':0.0, 
            'class_surface':'Conn', 
            'delay':delay, 
            'fan_in':DepConn(name, 'calc_fan_in'), 
            'fan_in0':kwargs.get('fan_in0',1), 
            'netw_size':GetNetw('size'), 
            'lesion':False, 
            'mask':[-0.5, 0.5], 
            'rule':rule, 
            'save':{'active':GetSimu('save_conn', 'active'),
                    'overwrite':GetSimu('save_conn', 'overwrite'),
                     'path':sp}, 
            'source':source,
            'syn':syn, 
            'target':target,
            'local_num_threads':GetSimu('local_num_threads'),
            'tata_dop':DepNetw('calc_tata_dop'), 
            'weight':weight}
        
        return d
            
    def _get_defaults_node_input(self, key, target):
        d = {'class_population':'MyPoissonInput', 
             'class_surface':'Surface', 
            'extent':[-0.5, 0.5], 
            'edge_wrap':True, 
            'fan_in_distribution':GetNetw('fan_in_distribution'),
            'lesion':False, 
            'model':key, 
            'n':GetNode(target, 'n'), 
            'n_sets':1, 
            'nest_params':{}, 
            'rate':0.0, 
            'sets':DepNode(key, 'calc_sets'), 
            'spike_setup':DepNode(key, 'calc_spike_setup'), 
            'type':'input',
            'target':target}
        return d    


    def _get_defaults_node_network(self, key):
        d = {'class_population':'MyNetworkNode',
             'class_surface':'Surface', 
            'edge_wrap':True, 
            'extent':[-0.5, 0.5], 
            'fan_in_distribution':GetNetw('fan_in_distribution'),
            'lesion':False, 
            'I_vitro':0.0, 
            'I_vivo':0.0, 
            'model':key, 
            'mm':{'active':False, 
                  'params':self.build_setup_mm_params()}, 
            'n':DepNode(key, 'calc_n'), 
            'n_sets':1, 
            'nest_params':{'I_e':GetNode(key, 'I_vivo')}, 
            'rate_in_vitro':0.0, 
            'rand':self.build_setup_node_rand(GetNode(key, 'model')), 
            'sd':{'active':True,
                  'params':self.build_setup_sd_params()}, 
            'sets':DepNode(key, 'calc_sets'), 
            'rate':0.5, 
            'type':'network'}
        return d

    def _get_defaults_node_volume_transmitter(self,  key, nest_model, syn_target):
        d = {'class_population':'VolumeTransmitter',
             'class_surface':'Surface',
             'extent':[-0.5, 0.5], 
             'edge_wrap':True,  
             'lesion':False, 
             'model':nest_model, 
             'n':1, 
             'n_sets':1, 
             'nest_params':{}, 
             'sets':[],
             'syn_target':syn_target,
             'type':'volume_transmitter'}
        return d
    
    def _get_par_dependable(self):
        d={}        
        d=self.add(d, 'netw', 'n_nuclei')    
        
        d=self.add(d, 'nest', 'GABAA_1_Tau_decay')    
        d=self.add(d, 'nest', 'GABAA_2_Tau_decay')           
        d=self.add(d, 'nest', 'tata_dop')
        
        d=self.add(d, 'node', 'fan_in_distribution')
        d=self.add(d, 'node', 'mm')
        d=self.add(d, 'node', 'n')
        d=self.add(d, 'node', 'n_sets')
        d=self.add(d, 'node', 'nest_params')
        d=self.add(d, 'node', 'rand')
        d=self.add(d, 'node', 'sd')
        d=self.add(d, 'node', 'sets')
        d=self.add(d, 'node', 'spike_setup')
        d=self.add(d, 'node', 'rate')
                        
        d=self.add(d, 'conn', 'delay')          
        d=self.add(d, 'conn', 'fan_in')  
        d=self.add(d, 'conn', 'fan_in0')
        d=self.add(d, 'conn', 'mask') 
        d=self.add(d, 'conn', 'netw_size')
        d=self.add(d, 'conn', 'save')
        d=self.add(d, 'conn', 'tata_dop')
        d=self.add(d, 'conn', 'local_num_threads')
        d=self.add(d, 'conn', 'weight') 

        
        return d

    def unittest_add_on(self, d):

        do=self.other.get_par_constant()

        if not 'simu' in d.keys():
            d['simu']={}
        for key in ['mm_params',
                    'path_conn',
                    'path_data',
                    'path_figure',
                    'path_nest',
                    'save_conn',
                    'sd_params',
                    'sim_stop',
                    'start_rec',
                    'stop_rec',
                    'sim_time',
                    ]:
            d['simu'][key]=do['simu'][key]
        
        for key in ['attr_popu',
                    'attr_surf',
                    'fan_in_distribution',
                    'n_nuclei',
                    'rand_nodes',
                    'size',
                    'sub_sampling',
                    'tata_dop',
                    'tata_dop0',
                    'V_th_sigma',
                    ]:
            if not key in d['netw'].keys():
                d['netw'][key]=do['netw'][key]
            else:
                d['netw'][key]=misc.dict_update(d['netw'][key], do['netw'][key])
        
        d['nest']=misc.dict_update(d['nest'], do['nest'])    

    def unittest_return(self, dic_other, dic):
        if self.unittest:
            self.unittest_add_on(dic)
        else:
            dic = misc.dict_update(dic_other, dic)
        return dic


class Par_base_mixin_bcpnn(Par_base_mixin):
    
    def _get_par_dependable(self):
        d=super(Par_base_mixin_bcpnn, self)._get_par_dependable()

        d=self.add(d, 'nest', 'delay')           
        d=self.add(d, 'nest', 'weight')    
        d=self.add(d, 'node', 'n_sets')            
        
        return d     
            
class Par_base_single_unit(Par_base):
    def __init__(self, **kwargs ):
        
        super( Par_base_single_unit, self ).__init__( **kwargs  )     
        self.network_node=kwargs.get('network_node', 'M1')
        
class Unittest_base(object):
    
    def _get_par_constant(self):
        dic={}
        # ========================
        # Default simu parameters 
        # ========================          
        stop=1000.0
        inp='i1'
        net='n1'
        dic['simu']={}
        dic['simu']['do_reset']=False
        dic['simu']['mm_params']={'interval':0.5, 
                                   'to_file':True, 
                                   'to_memory':False,
                                   'record_from':['V_m'] }
        
        dic['simu']['print_time']=False
        dic['simu']['local_num_threads']=4
#         dic['simu']['threads_local']=1
        dic['simu']['save_conn']={'active':False, 'overwrite':False}
        dic['simu']['start_rec']=0.0
        dic['simu']['stop_rec']=numpy.Inf
        dic['simu']['sim_time']=stop
        dic['simu']['sim_stop']=3*stop
        dic['simu']['sd_params']={'to_file':True, 'to_memory':False}
        

        dc=dr.HOME+'/results/unittest/conn/'
        dp=dr.HOME+'/results/unittest/' +self.__class__.__name__+'/'
        df=dp
        dn=dp+'/nest/'
        
        dic['simu']['path_conn']=dc   
        dic['simu']['path_data']=dp
        dic['simu']['path_figure']=df   
        dic['simu']['path_nest']=dn   
        
    
        dic['netw']={}        
        dic['netw']['attr_popu']=self._get_attr_popu()
        dic['netw']['attr_surf']=self._get_attr_surf()
        dic['netw']['fan_in_distribution']='constant'
        dic['netw']['FF_curve']={'input':inp,
                                 'output':net}
        
         
        dic['netw']['input']={inp:{'type':'constant','params':{}}}
        dic['netw']['n_nuclei']={net:1000.,}
        dic['netw']['optimization']={'f':[net],
                                     'x':['node.'+inp+'.rate'],
                                     'x0':[3000.0]}
        
 
        dic['netw']['rand_nodes']={'C_m':True, 'V_th':True, 'V_m':True, 'I_e':True}
        dic['netw']['size']=10
        dic['netw']['sub_sampling']={net:1.0} 
        dic['netw']['tata_dop']=0.8
        dic['netw']['tata_dop0']=0.8
        dic['netw']['V_th_sigma']=1.0
        
        # ========================
        # Default nest parameters 
        # ========================
        d={inp+'_nest':{'type_id':'poisson_generator'}, 
           net+'_nest':{'type_id':'iaf_cond_exp',
                 'C_m':200.0, 'V_th':-50.,'I_e':0.,
                     'tau_minus':20.0, },
           inp+'_'+net+'_nest':{'type_id':'static_synapse',
                                    'delay':1.0,
                                    'weight':10.0}}
        dic['nest']=d


        # ========================
        # Default node parameters 
        # ========================  
        
        d1=self._get_defaults_node_input( inp, net)
        d1['rate']=10.
        d1['model']=inp+'_nest'
     
        
#         model=GetNode(net, 'model')
        d2= self._get_defaults_node_network(net)
        d2['n_sets']=3
        d2['model']=net+'_nest'
        d2['rate']=10.0
        
        dic['node']= {inp:d1, net:d2}
        
        # ========================
        # Default conn parameters 
        # ========================
        c=inp+'_'+net
        syn=GetConn(c, 'syn')
        d1=self._get_defaults_conn( c, syn)
        d1['delay']['params']=GetNest(syn,'delay')
        d1['delay']['type']='constant'  
        d1['fan_in0']=1.
        d1['mask']=[-0.25,0.25]
        d1['rule']='1-1'
        d1['syn']=inp+'_'+net+'_nest'
        d1['weight']['params']=GetNest(syn,'weight')   
        d1['weight']['type']='constant'        
        
        dic['conn']={inp+'_'+net:d1}   

        return dic    
        
class Unittest(Par_base, Unittest_base, Par_base_mixin):
    pass              
 
class Unittest_extend_base(object):
                            
    def _get_par_constant(self):     

        dic_other=self.other.get_par_constant()
        
        
        dic={}
        
        # ========================
        # Default simu parameters 
        # ========================        
        dic['simu']={}
        dic['simu']['local_num_threads']=1
#         dic['simu']['threads_local']=1
        
        
        # ========================
        # Default netw parameters 
        # ========================        
        dic['netw']={}  
        
#         dic['netw']['fan_in_distribution']='constant'        
        dic['netw']['input']={'i2':{'type':'constant','params':{}}}
        dic['netw']['n_nuclei']={'n2':100.}
        dic['netw']['optimization']={'f':['n1', 'n2'],
                                     'x':['node.i1.rate',
                                             'node.i2.rate'],
                                     'x0':[3000.0, 2000.0]}
        dic['netw']['FF_curve']={'input':'i1',
                                 'output':'n1'} 
        dic['netw']['n1_rate']=10.0
        # ========================
        # Default nest parameters 
        # ========================
        
        d={'i2_nest':{'type_id':'poisson_generator'}, 
           'n2_nest':{'type_id':'iaf_cond_exp',
                 'C_m':100.0, 'V_th':-50.},
           'n1_n2_nest':{'type_id':'static_synapse',
                    'delay':1.0,
                    'weight':10.0},
           'i2_n2_nest':{'type_id':'static_synapse',
                    'delay':1.0,
                    'weight':10.0}}
        dic['nest']=d


        # ========================
        # Default node parameters 
        # ========================        
        # ========================
        # Default node parameters 
        # ========================  
        dic['node']={}     

#         #n1
#         net='n1'
#         d= self._get_defaults_node_network(net)
#         d['n_sets']=1
#         d['model']=net+'_nest'
#         d['rate']=30.0
#         dic['node']['n1']=d
        
        #n2
        net='n2'
        d= self._get_defaults_node_network(net)
        d['n_sets']=3
        d['model']=net+'_nest'
        d['rate']=30.0-GetNetw('n1_rate')
        dic['node']['n2']=d      
        
        #i2     
        d=self._get_defaults_node_input( 'i2', 'n2')
        d['rate']=10.
        d['model']='i2_nest'       
        dic['node']['i2']=d

    
        # ========================
        # Default conn parameters 
        # ========================

        e=min(0.5,50./(GetNode('n1','n')+GetNode('n2','n')))
        c='n1_n2'
        syn=GetConn(c, 'syn')
        d2=self._get_defaults_conn( c, syn)
        d2['delay']['params']=GetNest(syn,'delay')
        d2['delay']['type']='constant'  
        d2['fan_in0']=10.
        d2['mask']=[-e,e]
        d2['rule']='set-set'
        d2['syn']='n1_n2_nest'
        d2['weight']['params']=GetNest(syn,'weight')   
        d2['weight']['type']='constant'     
                
        c='i2_n2'
        syn=GetConn(c, 'syn')
        d3=self._get_defaults_conn( c, syn)
        d3['delay']['params']=GetNest(syn,'delay')
        d3['delay']['type']='constant'  
        d3['fan_in0']=1.
        d3['mask']=[-0.25,0.25]
        d3['rule']='1-1'
        d3['syn']='i2_n2_nest'
        d3['weight']['params']=GetNest(syn,'weight')   
        d3['weight']['type']='constant'     
                
              
        dic['conn']={'n1_n2':d2, 'i2_n2':d3}
               
        dout=misc.dict_update(dic_other, dic)
               
#         dout=self.unittest_return(dic_other, dic)
        return dout
        
class Unittest_extend(Par_base, Unittest_extend_base, Par_base_mixin):
    pass 

class Unittest_bcpnn_dopa_base(object):
    def _get_par_constant(self):     

        dic_other=self.other.get_par_constant()
        dic={}
        dic['netw']={}
        dic['netw']['size']=52
        dic['netw']['n_nuclei']={'n1':1, 'n2':1, 'm1':50}
        
        params1={
                'duration':[120.,100.]+[190,100]*9,
                'rates':[2000.0, 3500.0]*10,
                'repetitions':70}
         
        params2={
                'duration':[120.,100.]+[190,100]*9,
                'rates':[2000.0, 3500.0]*10,
                'repetitions':70}
         
        params3={
                 'duration':[220.,70.]+[220.,70.]*9,
                 'rates':([2500.0, 2800.0]*2+[2500.0, 0.0]*1
                         +[2500.0, 2800.0]*1+[2500.0, 0.0]*2
                        +[2500.0, 2800.0]*3+[2500.0, 0.0]*1), 
                 'repetitions':70}

        dic['netw']['input']={'i1':{'type':'burst','params':params1},
                              'i2':{'type':'burst','params':params2},
                              'i3':{'type':'burst','params':params3}}
        # ========================
        # Default nest parameters 
        # ========================
        f=.5
        d={'bias':0.0,   #ANN interpretation. Only calculated here to demonstrate match to rule. 
                           # Will be eliminated in future versions, where bias will be calculated postsynaptically       
           'b':-0.4,
           'delay':1.,
           'dopamine_modulated':True,
           'e_i':0.01,
           'e_j':0.01,
           'e_j_c':1.0,
           'e_ij':0.0001,
           'epsilon':0.001, #lowest possible probability of spiking, e.g. lowest assumed firing rate
           'fmax':50.0,    #Frequency assumed as maximum firing, for match with abstract rule
           'gain':0.0,    #Coefficient to scale weight as conductance, can be zero-ed out
           'gain_dopa':2.0, 
           'k_pow':1.0,
           'K':1.0,         #Print-now signal // Neuromodulation. Turn off learning, K = 0
           'n':0.4,
           'p_i':0.01,
           'p_j':0.01,
           'p_ij':0.0001,
           'reverse':-1., 
           'tau_i':5.*f,     #Primary trace presynaptic time constant
           'tau_j':5.*f,      #Primary trace postsynaptic time constant
           'tau_e':50.*f,      #Secondary trace time constant
           'tau_p':500.*f,     #Tertiarty trace time constant
           'tau_n':100.,
           'type_id':'bcpnn_dopamine_synapse',
           'weight':10.0,
           'z_i':0.01,
           'z_j':0.01,
           'sigmoid':True,
           'sigmoid_slope':20.,
           'sigmoid_mean':0.5}
          
        dic['nest']={}
        dic['nest']['n1_n2_nest_bcpnn_dopa']=d  
        
        d={'bias':0.0,   #ANN interpretation. Only calculated here to demonstrate match to rule. 
                           # Will be eliminated in future versions, where bias will be calculated postsynaptically       
           'delay':1.,
           'epsilon':0.001, #lowest possible probability of spiking, e.g. lowest assumed firing rate
           'fmax':50.0,    #Frequency assumed as maximum firing, for match with abstract rule
           'gain':0.,    #Coefficient to scale weight as conductance, can be zero-ed out
           'K':1.,         #Print-now signal // Neuromodulation. Turn off learning, K = 0
           'tau_i':10.,     #Primary trace presynaptic time constant
           'tau_j':10.,      #Primary trace postsynaptic time constant
           'tau_e':100.,      #Secondary trace time constant
           'tau_p':1000.,     #Tertiarty trace time constant
           'type_id':'bcpnn_synapse',
           'weight':0.0}
                
        dic['nest']['n1_n2_nest_bcpnn']=d  
        

        #i1_n1_nest
        d={ 'type_id':'static_synapse',
           'delay':1.0,
            'weight':10.0}
        dic['nest']['i1_n1_nest']=d
        
        #i2_n2_nest
        d={'type_id':'static_synapse',
           'delay':1.0,
            'weight':10.0}
        dic['nest']['i2_n2_nest']=d

        #i3_m1_nest
        d={'type_id':'static_synapse',
            'delay':1.0,
            'weight':10.0}
        dic['nest']['i3_m1_nest']=d

        #m1_v1_nest
        d={'type_id':'static_synapse',
            'delay':1.0,
            'weight':1.0}
        dic['nest']['m1_v1_nest']=d
        

        #v1       
        d={'type_id':'volume_transmitter'}
        dic['nest']['v1_nest']=d

        #m1_nest
        d={'type_id':'iaf_cond_exp',
            'C_m':200.0, 'V_th':-50.}
        dic['nest']['m1_nest']=d

        #n2_nest
        d={'type_id':'iaf_cond_exp',
            'C_m':200.0, 'V_th':-50.}
        dic['nest']['n2_nest']=d
        
        #i2_nest
        d={'type_id':'poisson_generator'}
        dic['nest']['i2_nest']=d

        #i3_nest
        d={'type_id':'poisson_generator'}
        dic['nest']['i3_nest']=d
        # ========================
        # Default node parameters 
        # ========================  
        dic['node']={}
        
        #n1
        net='n1'
        d= self._get_defaults_node_network(net)
        d['n_sets']=1
        d['model']=net+'_nest'
        d['rate']=10.0
        dic['node']['n1']=d
        
        #n2
        net='n2'
        d= self._get_defaults_node_network(net)
        d['n_sets']=1
        d['model']=net+'_nest'
        d['rate']=10.0

        dic['node']['n2']=d
           
        #i2     
        d=self._get_defaults_node_input( 'i2', 'n2')
        d['rate']=10.
        d['model']='i2_nest'       
        dic['node']['i2']=d

        #i3
        d=self._get_defaults_node_input( 'i3', 'm1')
        d['rate']=10.
        d['model']='i3_nest'       
        dic['node']['i3']=d


        #Volume transmitter
        d=self._get_defaults_node_volume_transmitter( 'v1', 'v1_nest',
                                                      'n1_n2_nest_bcpnn_dopa')
        dic['node']['v1']=d
                
        # Modulator neurons
        d=self._get_defaults_node_network( 'm1')    
        d['model']='m1_nest' 
        dic['node']['m1']=d

        # ========================
        # Default conn parameters 
        # ========================
        
        dic['conn']={}      

        #n1_n2
        d=self._get_defaults_conn( 'n1_n2', 'n1_n2_nest_bcpnn_dopa',
                                   **{'conn_type':'constant',
                                      'rule':'1-1'})   
        dic['conn']['n1_n2_dop']=d   

        #n1_n2
        d=self._get_defaults_conn( 'n1_n2', 'n1_n2_nest_bcpnn',
                                   **{'conn_type':'constant',
                                      'rule':'1-1'})   
        dic['conn']['n1_n2']=d 
   
        #i1_n1
        d=self._get_defaults_conn( 'i1_n1', 'i1_n1_nest',
                                   **{'conn_type':'constant',
                                      'rule':'1-1'}) 
        dic['conn']['i1_n1']=d        
        
        #i2_n2
        d=self._get_defaults_conn( 'i2_n2', 'i2_n2_nest',
                               **{'conn_type':'constant',
                                  'rule':'1-1'}) 
        dic['conn']['i2_n2']=d
        
        #i3_m1
        d=self._get_defaults_conn( 'i3_m1', 'i3_m1_nest',
                               **{'conn_type':'constant',
                                  'rule':'1-1'})         
        dic['conn']['i3_m1']=d

        #m1_v1
        d=self._get_defaults_conn( 'm1_v1', 'm1_v1_nest',
                                **{'conn_type':'constant',
                                  'rule':'divergent',
                                  'fan_in0':1})         
        dic['conn']['m1_v1']=d


    
        del dic_other['node']['n1']
        del dic_other['conn']['i1_n1']
        
        dic=misc.dict_update(dic_other, dic)
        
        return misc.dict_update(dic_other, dic)


class Unittest_bcpnn_dopa(Par_base, Unittest_bcpnn_dopa_base, Par_base_mixin):
    pass


class Unittest_stdp_base(object):
    def _get_par_constant(self):     
        # Need the Unittest_bcpnn
        dic_other=self.other.get_par_constant()

        dic={}
        d={'A_plus':0.01,
           'A_minus':0.1,
           'c':1.,
           'n':0.1,
           'tau_plus':20.0,   
           'tau_c':1000.0,    
           'tau_n':200.,   
           'type_id':'stdp_dopamine_synapse',      
           'Wmin':0.,     
           'Wmax':200.,   
           'weight':10.0,
           'delay':1.0,
           'b':0.03,
           }
                
        dic['nest']={}
        dic['nest']['n1_n2_nest_stdp']=d        
        
        # ========================
        # Default node parameters 
        # ========================          
        dic['node']={}
        dic['node']['v1']={'syn_target':'n1_n2_nest_stdp'}
        
        # ========================
        # Default conn parameters 
        # ========================        
        dic['conn']={}
        dic['conn']['n1_n2']={'syn':'n1_n2_nest_stdp'}
        
        
            
 
        
        
        return misc.dict_update(dic_other, dic)
    
class Unittest_stdp(Par_base, Unittest_stdp_base, Par_base_mixin):
    pass


class Unittest_bcpnn_base(object):
    def _get_par_constant(self):     
        # Need the Unittest_bcpnn
        dic_other=self.other._get_par_constant()

        dic={}
        dic['netw']={'size':2}

        params1={'start':1.0,
                'time_first':1000.0,
                'time_second':1000.0 ,
                'rate_first':3500.0,
                'rate_second':2600.0,
                'repetitions':20}
        
        params2={'start':1.0,
                'time_first':1000.0,
                'time_second':1000.0 ,
                'rate_first':3500.0,
                'rate_second':2600.0,
                'repetitions':20}
        
        dic['netw']['input']={'i1':{'type':'burst','params':params1},
                              'i2':{'type':'burst','params':params2}}
                
        d={'bias':0.0,   #ANN interpretation. Only calculated here to demonstrate match to rule. 
                           # Will be eliminated in future versions, where bias will be calculated postsynaptically       
           'delay':1.,
           'epsilon':0.001, #lowest possible probability of spiking, e.g. lowest assumed firing rate
           'fmax':50.0,    #Frequency assumed as maximum firing, for match with abstract rule
           'gain':20.,    #Coefficient to scale weight as conductance, can be zero-ed out
           'K':1.,         #Print-now signal // Neuromodulation. Turn off learning, K = 0
           'tau_i':10.,     #Primary trace presynaptic time constant
           'tau_j':10.,      #Primary trace postsynaptic time constant
           'tau_e':100.,      #Secondary trace time constant
           'tau_p':1000.,     #Tertiarty trace time constant
           'type_id':'bcpnn_synapse',
           'weight':10.0}
                
        dic['nest']={}
        dic['nest']['n1_n2_nest_bcpnn']=d        
        
        # ========================
        # Default node parameters 
        # ========================          
#         dic['node']={}
#         dic['node']['v1']={'syn_target':'n1_n2_nest_bcpnn'}
        
        # ========================
        # Default conn parameters 
        # ========================        
        dic['conn']={}
        dic['conn']['n1_n2']={'syn':'n1_n2_nest_bcpnn'}
        
        del dic_other['netw']['input']['i3']
        del dic_other['netw']['n_nuclei']['m1']
        del dic_other['node']['i3']
        del dic_other['node']['m1']
        del dic_other['node']['v1']        
        del dic_other['conn']['i3_m1']
        del dic_other['conn']['m1_v1']       
        
        return misc.dict_update(dic_other, dic)
    
class Unittest_bcpnn(Par_base, Unittest_bcpnn_base, Par_base_mixin):
    pass


class Single_unit_base(object):
  
    def get_single_unit(self):
        return self.dic['netw']['single_unit']

    def get_single_unit_input(self):
        return self.dic['netw']['single_unit_input']
    
    def _get_par_constant(self):
    
#         dic_other=self.other._get_par_constant()
        dic_other=self.other.dic
    
        if misc.dict_haskey(self.dic_rep, ['netw', 'single_unit']):
            single_unit=misc.dict_recursive_get(self.dic_rep, 
                                            ['netw', 'single_unit'])
        if self.per.has('.'.join(['netw', 'single_unit', '='])):
            single_unit=self.per.get('.'.join(['netw', 'single_unit', '=']))
            
        else:
            for key in dic_other['node'].keys():
                if dic_other['node'][key]['type']=='network':
                    single_unit=key
                    break
               
        dic={}

        # ========================
        # Default simu parameters 
        # ========================        
        dic['simu']=dic_other['simu']
        

        dc=(dr.HOME+'/results/papers/inhibition/single/conn/')        
        dp=(dr.HOME+'/results/papers/inhibition/single/'
            +self.other.__class__.__name__)+'/'
        df=(dr.HOME+'/projects/papers/inhibition/figures/'
            +self.other.__class__.__name__)+'/'
        dn=(dr.HOME+'/results/papers/inhibition/single/'
            +self.other.__class__.__name__+'/nest/')         
        
        dic['simu']['path_conn']=dc
        dic['simu']['path_data']=dp
        dic['simu']['path_figure']=df
        dic['simu']['path_nest']=dn
        # ========================
        # Default netw parameters 
        # ========================
        
        dic['netw']=dic_other['netw']
        dic['netw']['input']={}
        dic['netw']['n_nuclei']={single_unit:1.}
        dic['netw']['single_unit']=single_unit
        dic['netw']['single_unit_input']=None
        dic['netw']['sub_sampling']={}
#       
        dic['node']={}
        dic['conn']={}
        dic['nest']={}
        dic['nest']={'my_poisson_generator': {'rate':0.0, 
                                              'type_id':'poisson_generator'} }
        dic['nest'].update(dic_other['nest'])
        
        # Since dic_other is only constant values 
        for key in dic['nest'].keys():
            if not 'tata_dop' in dic['nest'][key].keys():
                continue
            dic['nest'][key]['tata_dop'] = DepNetw('calc_tata_dop')
        
        dic['node'][single_unit]=dic_other['node'][single_unit]
        dic['node'][single_unit]['n']=GetNetw('size', dtype=int)
        dic['node'][single_unit]['mm']['params']=self.build_setup_mm_params() 
        dic['node'][single_unit]['sd']['params']=self.build_setup_sd_params()
        rand=self.build_setup_node_rand(GetNode(single_unit, 'model'))
        dic['node'][single_unit]['rand']=rand 
        dic['node'][single_unit]['sets']=DepNode( single_unit,'calc_sets')
        dic['node'][single_unit]['nest_params']={'I_e':GetNode(single_unit, 
                                                               'I_vivo')} 
        d=dic_other['conn']
        for key, _, source,tp in iter_node_connected_to(d,single_unit):
            new_name=source+'p'      
            new_conn=new_name+'_'+single_unit+'_'+tp
            vn=deepcopy(dic_other['node'][source])
            
            dic['node'][new_name]=vn
            
            dic['netw']['input'][new_name]={'type':'constant', 
                                            'params':{}}
            
            vn['model']  = 'my_poisson_generator'
            vn['sets']=DepNode( new_name,'calc_sets')
            vn['spike_setup']= DepNode( new_name, 'calc_spike_setup')
            
#             if vn['type']=='input':
#                 vn['n']=1
#                 continue
            
            vn['class_population']='MyPoissonInput'    
            vn['n']=(GetConn(new_conn, 'fan_in', dtype=int)
                     *GetNode(single_unit, 'n'))  
            vn['nest_params']={}
            vn['type']='input'
            vn['target']=single_unit
            for par in ['I_vivo',  
                        'I_vitro', 
                        'mm',
                        'rand', 
                        'sd',
                        'rate_in_vitro',
                         ]:
                if par in vn.keys():
                    del vn[par]          

        # can be either AMPA and NMDA or GABAA
        for key, _, source, tp in iter_conn_connected_to(d, single_unit):
            new_name=source+'p' 
            new_conn=new_name+'_'+single_unit+'_'+ tp
            #Create connections  
            vc=deepcopy(dic_other['conn'][key])
                                          
            # Add units to dictionary
            syn=GetConn(new_conn, 'syn')
            vc['delay']  = {'type':'constant',
                            'params':GetNest(syn,'delay') }    
#             vc['delay'] = {'type':'uniform',
#                             'params':{'min':0.5*GetNest(syn,'delay'),
#                                       'max':1.5*GetNest(syn,'delay')} }                             
            vc['mask']= [-0.5, 0.5]
            

            vc['rule']    ='fan-1' 
            vc['netw_size']=GetNetw('size')
            vc['source']=new_name

            # OBS weights_input_rates need to be varied such that simulation of
            # a single group get variations in firing rate since they recieve
            # same poisson input. 
            vc['weight'] = {'type':'constant',
                            'params':GetNest(syn,'weight') }
#             vc['weight'] = {'type':'uniform',
#                             'params':{'min':0.5*GetNest(syn,'weight'),
#                                       'max':1.5*GetNest(syn,'weight')} }
#             
            
            file_name=(new_conn+'_'
                   +GetNode(new_name,'n', dtype=str)+'_'
                   +GetNode(single_unit,'n', dtype=str)+'_beta_fan_in_'
                   +GetConn(new_conn,'beta_fan_in', dtype=str))
         
            sp=GetSimu('path_conn')+GetNetw('size', dtype=str)+'_'+file_name
            
            vc['save']={'active':GetSimu('save_conn', 'active'),
                         'overwrite':GetSimu('save_conn', 'overwrite'),
                         'path':sp}
                    
            dic['conn'][new_conn]=vc
        
        return dic
    
class Single_unit(Par_base, Single_unit_base, Par_base_mixin):
    pass

class InhibitionPar_base(object):
    
    def _get_par_constant(self):
        dic={}
        
        # ========================
        # Default simu parameters 
        # ========================
        
        dic['simu']={}
        dic['simu']['do_reset']=False
        dic['simu']['mm_params']={'interval':0.5, 
                                   'to_file':True, 
                                   'to_memory':False,
                                   'record_from':['V_m'] }

        dp=dr.HOME_DATA+'/'
        dco= dp + 'conn/'       
        dcl= dp + self.__class__.__name__+'/'
        df=  dp + 'fig/'
        dn=  dp +  self.__class__.__name__+'/nest/'        
        
        dic['simu']['path_data']=dp
        dic['simu']['path_conn']=dco
        dic['simu']['path_class']=dcl
        dic['simu']['path_figure']=df
        dic['simu']['path_nest']=dn
        dic['simu']['print_time']=True
        dic['simu']['save_conn']={'active':True, 'overwrite':False}
        dic['simu']['sd_params']={'to_file':False, 'to_memory':True}
        dic['simu']['sim_time']=2000.0
        dic['simu']['sim_stop']=2000.0
        dic['simu']['stop_rec']=2000.0
        dic['simu']['start_rec']=1000.0
        dic['simu']['local_num_threads']=1
#         dic['simu']['threads_local']=1
        
        # ========================
        # Default netw parameters 
        # ========================
        
        dic['netw']={} 

        dic['netw']['attr_popu']=self._get_attr_popu()
        dic['netw']['attr_surf']=self._get_attr_surf()


        dic['netw']['fan_in_distribution']='constant'
        dic['netw']['FF_curve']={'input':'C1',
                                 'output':'M1'}
        dic['netw']['GP_fan_in']=30 
        dic['netw']['GP_rate']=30.
        dic['netw']['GP_fan_in_prop_GA']=1/17.
        dic['netw']['GA_prop']=0.2
        dic['netw']['GI_prop']=0.8
        d={'type':'constant', 'params':{}}
        dic['netw']['input']={}
        for key in ['C1', 'C2', 'CF', 'CS','EA', 'EI', 'ES']: 
            dic['netw']['input'][key]=deepcopy(d)        
        
        dic['netw']['n_actions']=1
           
        dic['netw']['n_nuclei']={'M1':2791000*0.425,
                                 'M2':2791000*0.425,
                                 'FS':2791000* 0.02, # 2 % if MSN population
                                 'ST': 13560.,
                                 'GI': 45960.*(1.-GetNetw('GA_prop')),
                                 'GA': 45960.*GetNetw('GA_prop'),
                                 'SN': 26320.}
        
        '''
        n_nuclei={'M1':15000,
               'M2':15000,
               'FS': 0.02*30000, # 2 % if MSN population
               'ST': 100,
               'GP': 300/.8,
               'SN': 300}   
        '''
        dic['netw']['optimization']={'f':['M1'],
                                     'x':['node.C1.rate'],
                                     'x0':[700.0]}
                 
        dic['netw']['rand_nodes']={'C_m':True, 'V_th':True, 'V_m':True, 
                                   'I_e':False #need to be false so that I_vivo can be modified
                                   }
        
        
        dic['netw']['size']=10000.0 
        dic['netw']['sub_sampling']={'M1':1.0,'M2':1.0} 
        dic['netw']['tata_dop']  = 0.8
        dic['netw']['tata_dop0'] = 0.8  
        dic['netw']['V_th_sigma']=1.0
                
         
        # ========================
        # Default nest parameters 
        # ========================
        # Defining default parameters
        dic['nest']={}
        
        # CTX-FSN 
        dic['nest']['CF_FS_ampa']={}
        dic['nest']['CF_FS_ampa']['weight']   = 0.25    # n.d. set as for CTX to MSN   
        dic['nest']['CF_FS_ampa']['delay']    = 12.0    # n.d. set as for CTX to MSN   
        dic['nest']['CF_FS_ampa']['type_id'] = 'static_synapse'   
        dic['nest']['CF_FS_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']     # n.d. set as for CTX to MSN   
    
    
        # FSN-FSN
        dic['nest']['FS_FS_gaba']={}
        dic['nest']['FS_FS_gaba']['weight']  = 1./0.29     # five times weaker than FSN-MSN, Gittis 2010
        dic['nest']['FS_FS_gaba']['delay']   = 1.7   # n.d.same asfor FSN to MSN    
        dic['nest']['FS_FS_gaba']['U']       = 0.29
        dic['nest']['FS_FS_gaba']['tau_fac'] = 53.   
        dic['nest']['FS_FS_gaba']['tau_rec'] = 902.   
        dic['nest']['FS_FS_gaba']['tau_psc'] = 6.     #   Gittis 2010 have    
        dic['nest']['FS_FS_gaba']['type_id'] = 'tsodyks_synapse'                
        dic['nest']['FS_FS_gaba']['receptor_type'] = self.rec['izh']['GABAA_1']
        
        
        # GPE-FSN
        dic['nest']['GA_FS_gaba']={}
        dic['nest']['GA_FS_gaba']['weight']   = 1./5     # n.d. inbetween MSN and FSN GABAergic synapses
        dic['nest']['GA_FS_gaba']['delay']    = 7.  # n.d. same as MSN to GPE Park 1982
        dic['nest']['GA_FS_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GA_FS_gaba']['receptor_type'] = self.rec['izh']['GABAA_2']

        dic['nest']['GI_FS_gaba']={}
        dic['nest']['GI_FS_gaba']['weight']   = 1./5     # n.d. inbetween MSN and FSN GABAergic synapses
        dic['nest']['GI_FS_gaba']['delay']    = 7.  # n.d. same as MSN to GPE Park 1982
        dic['nest']['GI_FS_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GI_FS_gaba']['receptor_type'] = self.rec['izh']['GABAA_2']
          
        
        # CTX-MSN D1
        dic['nest']['C1_M1_ampa']={}
        dic['nest']['C1_M1_ampa']['weight']   = .5     # constrained by Ellender 2011
        dic['nest']['C1_M1_ampa']['delay']    = 12.    # Mallet 2005
        dic['nest']['C1_M1_ampa']['type_id'] = 'static_synapse'
        dic['nest']['C1_M1_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']
        
        dic['nest']['C1_M1_nmda'] = deepcopy(dic['nest']['C1_M1_ampa'])
        dic['nest']['C1_M1_nmda']['weight'] =  0.11   # (Humphries, Wood, and Gurney 2009)
        dic['nest']['C1_M1_nmda']['receptor_type'] = self.rec['izh'][ 'NMDA_1' ]
        
        
        # CTX-MSN D2
        dic['nest']['C2_M2_ampa'] = deepcopy(dic['nest']['C1_M1_ampa'])
        dic['nest']['C2_M2_ampa']['weight'] =  .41     # constrained by Ellender 2011
        
        dic['nest']['C2_M2_nmda'] = deepcopy(dic['nest']['C1_M1_nmda'])
        dic['nest']['C2_M2_nmda']['weight'] =  0.019   # (Humphries, Wood, and Gurney 2009)
        
        
        # MSN-MSN    
        dic['nest']['M1_M1_gaba'] = {}
        dic['nest']['M1_M1_gaba']['weight']   =  0.6    # Taverna 2008
        dic['nest']['M1_M1_gaba']['delay']    =  1.7    # Taverna 2004          
        dic['nest']['M1_M1_gaba']['type_id'] = 'static_synapse'
        dic['nest']['M1_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_2']
        
        dic['nest']['M1_M2_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M1_M2_gaba']['weight']   =  1.5    # Taverna 2008
        
        dic['nest']['M2_M1_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M2_M1_gaba']['weight']   =  1.8    # Taverna 2008

        dic['nest']['M2_M2_gaba'] = deepcopy(dic['nest']['M1_M1_gaba'])
        dic['nest']['M2_M2_gaba']['weight']   =  1.4    # Taverna 2008
        
        # FSN-MSN
        dic['nest']['FS_M1_gaba']={}
        dic['nest']['FS_M1_gaba']['weight']  = round(6./0.29,1)     # Gittie #3.8    # (Koos, Tepper, and Charles J Wilson 2004)
        dic['nest']['FS_M1_gaba']['delay']   = 1.7    # Taverna 2004       
        dic['nest']['FS_M1_gaba']['U']       = 0.29     # GABAA plastic
        dic['nest']['FS_M1_gaba']['tau_fac'] = 53.0
        dic['nest']['FS_M1_gaba']['tau_rec'] = 902.0
        dic['nest']['FS_M1_gaba']['tau_psc'] = 8.0    # ?  Gittis 2010
        dic['nest']['FS_M1_gaba']['type_id'] = 'tsodyks_synapse' 
        dic['nest']['FS_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_1']   
        
        dic['nest']['FS_M2_gaba'] = deepcopy(dic['nest']['FS_M1_gaba'])

        # FSN-MSN static
        dic['nest']['FS_M1_gaba_s']={}
        dic['nest']['FS_M1_gaba_s']['weight']  =6.*0.3
        dic['nest']['FS_M1_gaba_s']['delay']   = 1.7    # Taverna 2004   
        dic['nest']['FS_M1_gaba_s']['type_id'] = 'static_synapse'  
        dic['nest']['FS_M1_gaba_s']['receptor_type'] = self.rec['izh']['GABAA_1']   
        
        dic['nest']['FS_M2_gaba_s'] = deepcopy(dic['nest']['FS_M1_gaba_s'])
        
        # GPE-MSN    
        dic['nest']['GA_M1_gaba']={}
        dic['nest']['GA_M1_gaba']['weight']   = 1./5  # Glajch 2013
        dic['nest']['GA_M1_gaba']['delay']    = 1.7 
        dic['nest']['GA_M1_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GA_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_3']   
        
        dic['nest']['GA_M2_gaba'] = deepcopy(dic['nest']['GA_M1_gaba'])
        dic['nest']['GA_M2_gaba']['weight']   = 1.*2/5  ## Glajch 2013
 
 
        dic['nest']['GI_M1_gaba']={}
        dic['nest']['GI_M1_gaba']['weight']   = 1./5  # Glajch 2013
        dic['nest']['GI_M1_gaba']['delay']    = 1.7 
        dic['nest']['GI_M1_gaba']['type_id'] = 'static_synapse'
        dic['nest']['GI_M1_gaba']['receptor_type'] = self.rec['izh']['GABAA_3']   
        
        dic['nest']['GI_M2_gaba'] = deepcopy(dic['nest']['GA_M1_gaba'])
        dic['nest']['GI_M2_gaba']['weight']   = 1.*2/5  ## Glajch 2013
      
            
        # CTX-STN
        dic['nest']['CS_ST_ampa']={}
        dic['nest']['CS_ST_ampa']['weight']   = 0.25
        dic['nest']['CS_ST_ampa']['delay']       = 2.5  # Fujimoto and Kita 1993
        dic['nest']['CS_ST_ampa']['type_id'] = 'static_synapse'  
        dic['nest']['CS_ST_ampa']['receptor_type'] = self.rec['aeif'] ['AMPA_1']  
        
        dic['nest']['CS_ST_nmda'] = deepcopy(dic['nest']['CS_ST_ampa'])
        dic['nest']['CS_ST_nmda']['weight'] = 0.00625   # n.d.; same ratio ampa/nmda as MSN
        dic['nest']['CS_ST_nmda']['receptor_type'] = self.rec['aeif'] ['NMDA_1']  
        
        
        # GPe I-STN  
        dic['nest']['GI_ST_gaba']={}
        dic['nest']['GI_ST_gaba']['weight'] = .08    # n.d.
        dic['nest']['GI_ST_gaba']['delay'] =  5.
        dic['nest']['GI_ST_gaba']['type_id'] = 'static_synapse' 
        dic['nest']['GI_ST_gaba']['receptor_type'] = self.rec['aeif'] [ 'GABAA_1' ]  
        
        
        # EXT-GPe
        dic['nest']['EA_GA_ampa']={}
        dic['nest']['EA_GA_ampa']['weight']   = 0.167
        dic['nest']['EA_GA_ampa']['delay']    = 5.
        dic['nest']['EA_GA_ampa']['type_id'] = 'static_synapse'  
        dic['nest']['EA_GA_ampa']['receptor_type'] = self.rec['aeif']['AMPA_2']  
        
        dic['nest']['EI_GI_ampa'] = deepcopy(dic['nest']['EA_GA_ampa'])
    
        
        # GPe-GPe
        dic['nest']['GA_GA_gaba']={}
        dic['nest']['GA_GA_gaba']['weight']   = 1.3    # constrained by (Sims et al. 2008)
        dic['nest']['GA_GA_gaba']['delay']    = 1.     #n.d. assumed due to approximity   
        dic['nest']['GA_GA_gaba']['type_id'] = 'static_synapse'  
        dic['nest']['GA_GA_gaba']['receptor_type'] = self.rec['aeif'][ 'GABAA_2' ]
        
        dic['nest']['GA_GI_gaba'] = deepcopy(dic['nest']['GA_GA_gaba'])
        dic['nest']['GI_GA_gaba'] = deepcopy(dic['nest']['GA_GA_gaba'])
        dic['nest']['GI_GI_gaba'] = deepcopy(dic['nest']['GA_GA_gaba']) 
    
         
        # MSN D2-GPe I 
        dic['nest']['M2_GI_gaba']={}
        dic['nest']['M2_GI_gaba']['weight']  = 2./0.24   # constrained by (Sims et al. 2008)
        dic['nest']['M2_GI_gaba']['delay']   = 7.       # Park 1982
        dic['nest']['M2_GI_gaba']['U']       = 0.24                                                   # GABAA plastic                   
        dic['nest']['M2_GI_gaba']['tau_fac'] = 13.0
        dic['nest']['M2_GI_gaba']['tau_rec'] = 77.0
        dic['nest']['M2_GI_gaba']['tau_psc'] = 6.    # (Shen et al. 2008)
        dic['nest']['M2_GI_gaba']['type_id'] = 'tsodyks_synapse' 
        dic['nest']['M2_GI_gaba']['receptor_type'] = self.rec['aeif']['GABAA_1']         
     

        # MSN D2-GPe A 
        dic['nest']['M2_GA_gaba']={}
        dic['nest']['M2_GA_gaba']['weight']  = 2./0.24   # constrained by (Sims et al. 2008)
        dic['nest']['M2_GA_gaba']['delay']   = 7.       # Park 1982
        dic['nest']['M2_GA_gaba']['U']       = 0.24                                                   # GABAA plastic                   
        dic['nest']['M2_GA_gaba']['tau_fac'] = 13.0
        dic['nest']['M2_GA_gaba']['tau_rec'] = 77.0
        dic['nest']['M2_GA_gaba']['tau_psc'] = 6.    # (Shen et al. 2008)
        dic['nest']['M2_GA_gaba']['type_id'] = 'tsodyks_synapse' 
        dic['nest']['M2_GA_gaba']['receptor_type'] = self.rec['aeif']['GABAA_1']         

        
        # STN-GPe
        dic['nest']['ST_GA_ampa']={}
        dic['nest']['ST_GA_ampa']['weight']   = 0.35     # constrained by (Hanson & Dieter Jaeger 2002)
        dic['nest']['ST_GA_ampa']['delay']    = 5.       # Ammari 2010
        dic['nest']['ST_GA_ampa']['type_id'] = 'static_synapse' 
        dic['nest']['ST_GA_ampa']['receptor_type'] = self.rec['aeif']['AMPA_1']         
        
        dic['nest']['ST_GI_ampa'] = deepcopy(dic['nest']['ST_GA_ampa']) 
        
        
        # EXR-SNr
        dic['nest']['ES_SN_ampa']={}
        dic['nest']['ES_SN_ampa']['weight']   = 0.5
        dic['nest']['ES_SN_ampa']['delay']    = 5.0
        dic['nest']['ES_SN_ampa']['type_id'] = 'static_synapse'  
        dic['nest']['ES_SN_ampa']['receptor_type'] = self.rec['aeif']['AMPA_2']  
     
        
        # MSN D1-SNr 
        dic['nest']['M1_SN_gaba']={}
        dic['nest']['M1_SN_gaba']['weight']   = 2./0.0192 # Lower based on (Connelly et al. 2010) = [4.7, 24.], 50 Hz model = [5.8, 23.5]
        dic['nest']['M1_SN_gaba']['delay']    = 7.3 
        dic['nest']['M1_SN_gaba']['U']        = 0.0192
        dic['nest']['M1_SN_gaba']['tau_fac']  = 623. 
        dic['nest']['M1_SN_gaba']['tau_rec']  = 559. 
        dic['nest']['M1_SN_gaba']['tau_psc']  = 5.2   
        dic['nest']['M1_SN_gaba']['type_id'] = 'tsodyks_synapse' 
        dic['nest']['M1_SN_gaba']['receptor_type'] = self.rec['aeif']['GABAA_1'] 
    
        
        # STN-SNr
        dic['nest']['ST_SN_ampa']={}
        dic['nest']['ST_SN_ampa']['weight']   = 0.91*3.8/0.35      # (Shen and Johnson 2006)
        dic['nest']['ST_SN_ampa']['delay']    = 4.6 
        dic['nest']['ST_SN_ampa']['U']        = 0.35 # AMPA plastic 2   
        dic['nest']['ST_SN_ampa']['tau_fac']  = 0.0
        dic['nest']['ST_SN_ampa']['tau_rec']  = 800.0
        dic['nest']['ST_SN_ampa']['tau_psc']  = 12.   # n.d.; set as for STN to GPE,
        dic['nest']['ST_SN_ampa']['type_id'] = 'tsodyks_synapse'
        dic['nest']['ST_SN_ampa']['receptor_type']= self.rec['aeif']['AMPA_1'] 
     
        
        # GPe-SNr
        dic['nest']['GI_SN_gaba']={}
        dic['nest']['GI_SN_gaba']['weight']   = 76./0.196  #0.152*76., (Connelly et al. 2010)
        dic['nest']['GI_SN_gaba']['delay']    = 3.  
        dic['nest']['GI_SN_gaba']['U']        = 0.196 # GABAA plastic,                   
        dic['nest']['GI_SN_gaba']['tau_fac']  = 0.0
        dic['nest']['GI_SN_gaba']['tau_rec']  = 969.0
        dic['nest']['GI_SN_gaba']['tau_psc']  = 2.1    # (Connelly et al. 2010),
        dic['nest']['GI_SN_gaba']['type_id'] = 'tsodyks_synapse'   
        dic['nest']['GI_SN_gaba']['receptor_type']= self.rec['aeif']['GABAA_2']          
    
        # ============        
        # Input Models
        # ============ 
        
        dic['nest']['poisson_generator']={}
        dic['nest']['poisson_generator']['type_id']='poisson_generator'
        dic['nest']['poisson_generator']['rate']=0.0
        
        #CTX-MSN D1
        dic['nest']['C1']={}
        dic['nest']['C1']['type_id']='poisson_generator'
        dic['nest']['C1']['rate']=0.0
        
        #CTX-MSN D2
        dic['nest']['C2']={}
        dic['nest']['C2']['type_id']='poisson_generator'
        dic['nest']['C2']['rate']=0.0
    
        #CTX-FSN
        dic['nest']['CF']={}
        dic['nest']['CF']['type_id']='poisson_generator'
        dic['nest']['CF']['rate']=0.0
        
        #CTX-STN
        dic['nest']['CS']={}
        dic['nest']['CS']['type_id']='poisson_generator' 
        dic['nest']['CS']['rate']=0.0
      
        #EXT-GPe type A
        dic['nest']['EA']={}
        dic['nest']['EA']['type_id']='poisson_generator' 
        dic['nest']['EA']['rate']=0.0
        
        #EXT-GPe type I
        dic['nest']['EI']={}
        dic['nest']['EI']['type_id']='poisson_generator' 
        dic['nest']['EI']['rate']=0.0
    
        #EXT-SNr
        dic['nest']['ES']={}
        dic['nest']['ES']['type_id']='poisson_generator' 
        dic['nest']['ES']['rate']=0.0
    
    
        # =============        
        # Neuron Models
        # =============
    
        # MSN
        # ===
        dic['nest']['MS']={}    
        dic['nest']['MS']['type_id'] = 'izhik_cond_exp'
    
        dic['nest']['MS']['a']      =  0.01      # (E.M. Izhikevich 2007)
        dic['nest']['MS']['b_1']    = -20.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['b_2']    = -20.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['c']      = -55.       # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['C_m']    =  15.2      # (Humphries, Lepora, et al. 2009) # C izh
        dic['nest']['MS']['d']      =  66.9      # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['E_L']    = -81.85     # (Humphries, Lepora, et al. 2009) # v_r in izh
        dic['nest']['MS']['I_e']    =  0.
        dic['nest']['MS']['k']      =   1.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_peak'] =  40.       # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_b']    = dic['nest']['MS']['E_L']    # (E.M. Izhikevich 2007)
        dic['nest']['MS']['V_th']   = -29.7      # (Humphries, Lepora, et al. 2009)
        dic['nest']['MS']['V_m']    =  80.
        
        # CTX_MSN
        dic['nest']['MS']['AMPA_1_Tau_decay'] = 12.  # (Ellender 2011)
        dic['nest']['MS']['AMPA_1_E_rev']     =  0.  # (Humphries, Wood, et al. 2009)
        
        dic['nest']['MS']['NMDA_1_Tau_decay'] = 160. # (Humphries, Wood, et al. 2009)
        dic['nest']['MS']['NMDA_1_E_rev']     =  dic['nest']['MS']['AMPA_1_E_rev']    
        dic['nest']['MS']['NMDA_1_Vact']      = -20.0
        dic['nest']['MS']['NMDA_1_Sact']      =  16.0
        
        # From MSN
        dic['nest']['MS']['GABAA_2_Tau_decay'] = 12.  
        dic['nest']['MS']['GABAA_2_E_rev']     = -74. # Koos 2004
        dic['nest']['MS']['beta_I_GABAA_2'] = 0.56#-0.625 #Dopamine leads to weakening of MSN synspase
    
        # From FSN
        dic['nest']['MS']['GABAA_1_E_rev']     = -74. # Koos 2004
        dic['nest']['MS']['GABAA_1_Tau_decay'] =GetNest('FS_M1_gaba', 'tau_psc')
        
        # From GPE
        dic['nest']['MS']['GABAA_3_Tau_decay'] = 12*5.          
        dic['nest']['MS']['GABAA_3_E_rev']     = -74. # n.d. set as for MSN and FSN
        
        dic['nest']['MS']['tata_dop'] = DepNetw('calc_tata_dop')
        
        
        dic['nest']['M1']=deepcopy(dic['nest']['MS'])
        dic['nest']['M1']['d']      =  66.9      # (E.M. Izhikevich 2007)
        dic['nest']['M1']['E_L']    = -81.85     # (E.M. Izhikevich 2007)
        
        dic['nest']['M1']['beta_d']        = 0.45
        dic['nest']['M1']['beta_E_L']      = -0.0282 #Minus size it is plus in Humphrie 2009
        dic['nest']['M1']['beta_V_b']      = dic['nest']['M1']['beta_E_L'] 
        dic['nest']['M1']['beta_I_NMDA_1'] = 1.04 #-1.04 #Minus size it is plus in Humphrie 2009
        
        dic['nest']['M1_low']  = deepcopy(dic['nest']['M1'])
        dic['nest']['M1_high'] = deepcopy(dic['nest']['M1'])
        dic['nest']['M1_high']['GABAA_1_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M1_high']['GABAA_2_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M1_high']['GABAA_3_E_rev']  = -64.     # n.d. set asfor MSN and FSN

        
        dic['nest']['M2']   = deepcopy(dic['nest']['MS'])
        dic['nest']['M2']['d']    =  91.       # (E.M. Izhikevich 2007)
        dic['nest']['M2']['E_L']  = -80.       # (E.M. Izhikevich 2007)
        dic['nest']['M2']['V_b']  =  dic['nest']['M2']['E_L']
        dic['nest']['M2']['beta_I_AMPA_1'] =-0.26 # 0.26    
    
    
        dic['nest']['M2_low']  = deepcopy(dic['nest']['M2'])
        dic['nest']['M2_high'] = deepcopy(dic['nest']['M2'])
        dic['nest']['M2_high']['GABAA_1_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M2_high']['GABAA_2_E_rev']  = -64.     # (Bracci & Panzeri 2006)
        dic['nest']['M2_high']['GABAA_3_E_rev']  = -64.     # n.d. set asfor MSN and FSN
    
        
      
        # FSN
        # ===
        dic['nest']['FS']={}
        dic['nest']['FS']['type_id'] = 'izhik_cond_exp'
    
        dic['nest']['FS']['a']      = 0.2    # (E.M. Izhikevich 2007)
        dic['nest']['FS']['b_1']    = 0.0  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['b_2']    = 0.025  # (E.M. Izhikevich 2007)
        dic['nest']['FS']['c']      = -60.   # (Tateno et al. 2004)
        dic['nest']['FS']['C_m']    = 80.    # (Tateno et al. 2004)
        dic['nest']['FS']['d']      = 0.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['E_L']    = -70.   #*(1-0.8*0.1)   # (Tateno et al. 2004)
        dic['nest']['FS']['I_e']    =  0. 
        dic['nest']['FS']['k']      = 1.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['p_1']    = 1.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['p_2']    = 3.     # (E.M. Izhikevich 2007)
        dic['nest']['FS']['V_b']    = -55.   # Humphries 2009
        dic['nest']['FS']['V_peak'] = 25.    # (E.M. Izhikevich 2007)
        dic['nest']['FS']['V_th']   = -50.   # (Tateno et al. 2004)
    
    
        #CTX-FSN
        dic['nest']['FS']['AMPA_1_Tau_decay'] = 12.   # CTX to FSN ampa
        dic['nest']['FS']['AMPA_1_E_rev']    =  0.   # n.d. set as for  CTX to MSN
         
        # From FSN
        dic['nest']['FS']['GABAA_1_E_rev']    = -74.     # n.d.; set as for MSNs
        dic['nest']['FS']['GABAA_1_Tau_decay']= GetNest('FS_FS_gaba','tau_psc')
          
        # From GPe
        dic['nest']['FS']['GABAA_2_Tau_decay'] =  12.*5  
        dic['nest']['FS']['GABAA_2_E_rev']    = -74.  # n.d. set as for MSNs
          
        dic['nest']['FS']['beta_E_L'] = 0.078
        dic['nest']['FS']['tata_dop'] = DepNetw('calc_tata_dop')
        
        dic['nest']['FS']['beta_I_GABAA_1'] = -0.83 #0.8 # From FSN
        dic['nest']['FS']['beta_I_GABAA_2'] = 0.0#-0.83 #0.8 # From GPe A
    
        
        dic['nest']['FS_low']  = deepcopy(dic['nest']['FS'])
        dic['nest']['FS_high'] = deepcopy(dic['nest']['FS'])
        dic['nest']['FS_high']['GABAA_1_E_rev']  = -64.     # n.d. set as for MSNs
        dic['nest']['FS_high']['GABAA_2_E_rev']  = -64.     # n.d. set as for MSNs
          

        # STN
        # ===
       
        dic['nest']['ST']={}
        dic['nest']['ST']['type_id'] = 'my_aeif_cond_exp'
        
        dic['nest']['ST']['tau_w']    = 333.0 # I-V relation, spike frequency adaptation
        dic['nest']['ST']['a_1']      =  0.3    # I-V relation
        dic['nest']['ST']['a_2']      =  0.0      # I-V relation
        dic['nest']['ST']['b']        =  0.05    #0.1 #0.1#200./5.                                                     
        dic['nest']['ST']['C_m']      = 60.0    # t_m/R_in
        dic['nest']['ST']['Delta_T']  = 16.2                      
        dic['nest']['ST']['g_L']      = 10.0
        dic['nest']['ST']['E_L']      =-80.2                                                               
        dic['nest']['ST']['I_e']      =  6.0
        dic['nest']['ST']['V_peak']   = 15.0                                                                
        dic['nest']['ST']['V_reset']  =-70.0    # I-V relation
        dic['nest']['ST']['V_a']      =-70.0 # I-V relation
        dic['nest']['ST']['V_th']     =-64.0                                                               
        
        dic['nest']['ST']['V_reset_slope1']     = -10. # Slope u<0 
        dic['nest']['ST']['V_reset_slope2']     = .0 #  Slope u>=0
        dic['nest']['ST']['V_reset_max_slope1'] = -60. # Max v restet u<0  
        dic['nest']['ST']['V_reset_max_slope2'] = dic['nest']['ST']['V_reset'] # Max v restet u>=0  
       
       
        # CTX-STN
        dic['nest']['ST']['AMPA_1_Tau_decay'] = 4.0  # (Baufreton et al. 2005)
        dic['nest']['ST']['AMPA_1_E_rev']     = 0.   # (Baufreton et al. 2009)
        
        dic['nest']['ST']['NMDA_1_Tau_decay'] = 160.   # n.d. estimated 1:2 AMPA:NMDA
        dic['nest']['ST']['NMDA_1_E_rev']     =   0.   # n.d.; set as  E_ampa
        dic['nest']['ST']['NMDA_1_Vact']      = -20.0
        dic['nest']['ST']['NMDA_1_Sact']      =  16.0
        
        #GPE-STN
        dic['nest']['ST']['GABAA_1_Tau_decay'] =   8.   # (Baufreton et al. 2009)
        dic['nest']['ST']['GABAA_1_E_rev']     = -84.0  # (Baufreton et al. 2009)
    
        dic['nest']['ST']['beta_I_AMPA_1']  = -0.45 #0.4 # From Cortex
        dic['nest']['ST']['beta_I_NMDA_1']  = -0.45 #0.4 # From Cortex
        dic['nest']['ST']['beta_I_GABAA_1'] = -0.24 #0.4 # From GPe I 
        
        dic['nest']['ST']['tata_dop'] = DepNetw('calc_tata_dop')
    
        
        # GPE
        # ===
    
        dic['nest']['GP']={}
        dic['nest']['GP']['type_id'] = 'my_aeif_cond_exp'
    
        dic['nest']['GP']['a_1']       =  2.5    # I-V relation # I-V relation
        dic['nest']['GP']['a_2']       =  dic['nest']['GP']['a_1'] 
        dic['nest']['GP']['b']       = 70.   # I-F relation
        dic['nest']['GP']['C_m']     = 40.  # t_m/R_in
        dic['nest']['GP']['Delta_T'] =  1.7                      
        dic['nest']['GP']['g_L']     =   1.
        dic['nest']['GP']['E_L']     = -55.1  # v_t    = -56.4                                                               #
        dic['nest']['GP']['I_e']     =  0.
        dic['nest']['GP']['tau_w']   = 20.  # I-V relation, spike frequency adaptation
        dic['nest']['GP']['V_peak']  =  15.  # Cooper and standford
        dic['nest']['GP']['V_reset'] = -60.  # I-V relation
        dic['nest']['GP']['V_th']    = -54.7
        dic['nest']['GP']['V_a']     = dic['nest']['GP']['E_L']
    
        
        # STN-GPe
        dic['nest']['GP']['AMPA_1_Tau_decay'] = 12.   # (Hanson & Dieter Jaeger 2002)
        dic['nest']['GP']['AMPA_1_E_rev']     = 0.    # n.d.; same as CTX to STN
        
        dic['nest']['GP']['NMDA_1_Tau_decay'] = 100.  # n.d.; estimated
        dic['nest']['GP']['NMDA_1_E_rev']     = 0.    # n.d.; same as CTX to STN
        dic['nest']['GP']['NMDA_1_Vact']      = -20.0
        dic['nest']['GP']['NMDA_1_Sact']      =  16.0
            
        #EXT-GPe
        dic['nest']['GP']['AMPA_2_Tau_decay'] = 5.0
        dic['nest']['GP']['AMPA_2_E_rev']     = 0.0
        
        # GPe-GPe
        dic['nest']['GP']['GABAA_2_Tau_decay']  = 5.  # (Sims et al. 2008)
        dic['nest']['GP']['GABAA_2_E_rev']     = -65.  # n.d same as for MSN (Rav-Acha 2005)       
        
    
        dic['nest']['GP']['beta_E_L'] = 0.181
        dic['nest']['GP']['beta_V_a'] = 0.181
        dic['nest']['GP']['beta_I_AMPA_1']  =-0.45 # 0.4 # From STN
        dic['nest']['GP']['beta_I_GABAA_1'] = 0.0 #0.8 # From From MSNs
        dic['nest']['GP']['beta_I_GABAA_2'] = -0.83 #0.8 # From GPe A
    
        dic['nest']['GP']['tata_dop'] = DepNetw('calc_tata_dop')
        
        dic['nest']['GA']  = deepcopy(dic['nest']['GP'])
#         dic['nest']['GA']['a_1']  = 0.5
        dic['nest']['GI']  = deepcopy(dic['nest']['GP'])
        
        #MSN D2-GPe
        dic['nest']['GI']['GABAA_1_E_rev']     = -65.  # (Rav-Acha et al. 2005)
        dic['nest']['GI']['GABAA_1_Tau_decay']=GetNest('M2_GI_gaba', 'tau_psc')     # (Shen et al. 2008)    

    
        # SNR
        # ===
     
        dic['nest']['SN']={}
        dic['nest']['SN']['type_id'] = 'my_aeif_cond_exp'
        
        dic['nest']['SN']['tau_w']   = 20.  # I-V relation, spike frequency adaptation
        dic['nest']['SN']['a_1']     =  3.      # I-V relation
        dic['nest']['SN']['a_2']     =  dic['nest']['SN']['a_1']      # I-V relation
        dic['nest']['SN']['b']       = 200.   # I-F relation
        dic['nest']['SN']['C_m']     =  80.    # t_m/R_in
        dic['nest']['SN']['Delta_T'] =  1.8                      
        dic['nest']['SN']['g_L']     =   3.
        dic['nest']['SN']['E_L']     = -55.8    #
        dic['nest']['SN']['I_e']     = 15.0 
        dic['nest']['SN']['V_peak']  =  20.                                                               # 
        dic['nest']['SN']['V_reset'] = -65.    # I-V relation
        dic['nest']['SN']['V_th']    = -55.2    # 
        dic['nest']['SN']['V_a']     = dic['nest']['SN']['E_L']     # I-V relation
        
        #STN-SNr
        dic['nest']['SN']['AMPA_1_Tau_decay'] =  12.   # n.d.; set as for STN to GPE
        dic['nest']['SN']['AMPA_1_E_rev']     =   0.   # n.d. same as CTX to STN
    
        # EXT-SNr
        dic['nest']['SN']['AMPA_2_Tau_decay'] = 5.0
        dic['nest']['SN']['AMPA_2_E_rev']     = 0.
        
        # MSN D1-SNr
        dic['nest']['SN']['GABAA_1_E_rev']    =-80.     # (Connelly et al. 2010)
        dic['nest']['SN']['GABAA_1_Tau_decay']=GetNest('M1_SN_gaba', 'tau_psc')      # (Connelly et al. 2010)
        dic['nest']['SN']['beta_I_GABAA_1'] = 0.56 #0.8 # From MSN D1
       
        # GPe-SNr
        dic['nest']['SN']['GABAA_2_E_rev']    =-72.     # (Connelly et al. 2010)
        dic['nest']['SN']['GABAA_2_Tau_decay']=GetNest('GI_SN_gaba', 'tau_psc')
    
        dic['nest']['SN']['beta_E_L'] = 0.0896
        dic['nest']['SN']['beta_V_a'] = 0.0896
     
        dic['nest']['SN']['tata_dop'] = DepNetw('calc_tata_dop')
        
        
        
        # ========================
        # Default node parameters 
        # ========================
        
        dic['node']={}
    
        
        # Model inputs
        inputs={'C1': { 'target':'M1', 'rate':560.}, #530.-20.0},
                'C2': { 'target':'M2', 'rate':740.}, #690.-20.0}, 
                'CF': { 'target':'FS', 'rate':950.}, #624.},
                'CS': { 'target':'ST', 'rate':200.0}, #160.},#295 
                'EA': { 'target':'GA', 'rate':200.},
                'EI': { 'target':'GI', 'rate':1100.0},#1130.},
                'ES': { 'target':'SN', 'rate':1800.}}#295 
        
        for key, val in inputs.items():         
            d=self._get_defaults_node_input(key, val['target'])
            inputs[key]=misc.dict_update(d,inputs[key])

                
        dic['node']=misc.dict_merge(dic['node'], inputs)
        
           
        network={'M1':{'model':'M1_low', 'I_vitro':0.0, 'I_vivo':0.0, },
                 'M2':{'model':'M2_low', 'I_vitro':0.0, 'I_vivo':0.0, },
                 'FS':{'model':'FS_low', 'I_vitro':0.0, 'I_vivo':0.0, },
                 'ST':{'model':'ST',     'I_vitro':6.0, 'I_vivo':6.0,  },
                 'GA':{'model':'GA',     'I_vitro':5.0, 'I_vivo':-3.6, }, #23, -8
                 'GI':{'model':'GI',     'I_vitro':5.0, 'I_vivo':4.5,  }, #51, 56
                 'SN':{'model':'SN',     'I_vitro':15.0,'I_vivo':19.2, }}
        

        GA_prop=GetNetw('GA_prop')
        GP_tr=GetNetw('GP_rate')
        GA_tr=GetNode('GA', 'rate')
        network['M1'].update({'rate':0.1, 'rate_in_vitro':0.0})
        network['M2'].update({'rate':0.1, 'rate_in_vitro':0.0})
        network['FS'].update({'rate':15.0, 'rate_in_vitro':0.0})
        network['ST'].update({'rate':10.0, 'rate_in_vitro':10.0})
        network['GA'].update({'rate':5.0, 'rate_in_vitro':4.0})
        network['GI'].update({'rate':(GP_tr-GA_prop*GA_tr)/(1-GA_prop),
                              'rate_in_vitro':15.0})
        network['SN'].update({'rate':30., 'rate_in_vitro':15.0})
        
                 
        # Randomization of C_m and V_m
        for key in network.keys():     
            #model=GetNode(key, 'model')      
            d=self._get_defaults_node_network(key)     
            network[key]=misc.dict_update(d,network[key])

        d={ 'M1':{'n_sets':GetNetw('n_actions')},
             'M2':{'n_sets':GetNetw('n_actions')},
             'GI':{'n_sets':GetNetw('n_actions')},
             'SN':{'n_sets':GetNetw('n_actions')}}
        network=misc.dict_update(network, d)     

        dic['node']=misc.dict_update(dic['node'], network)   

        
        # ========================
        # Default conn parameters 
        # ========================
        
        
        # Input
        conns={'C1_M1_ampa':{},
               'C1_M1_nmda':{},
               'C2_M2_ampa':{},
               'C2_M2_nmda':{},
               'CF_FS_ampa':{},
               'CS_ST_ampa':{},
               'CS_ST_nmda':{},
               'EA_GA_ampa':{},
               'EI_GI_ampa':{},
               'ES_SN_ampa':{}}
        
        for k in conns.keys():
            conns[k].update({'fan_in0':1,  'rule':'1-1' })
        
        dic['conn']={}
        # Number of incomming connections from a nucleus to another nucleus.
    
        #conns.update(d)
        # Network      
        GP_fi=GetNetw('GP_fan_in')
        GA_pfi=GetNetw('GP_fan_in_prop_GA')
        
        M1_M1=int(round(2800*0.13))/GetNetw('sub_sampling', 'M1')
        M1_M2=int(round(2800*0.03))/GetNetw('sub_sampling', 'M1')
        M2_M1=int(round(2800*0.14))/GetNetw('sub_sampling', 'M2')
        M2_M2=int(round(2800*0.18))/GetNetw('sub_sampling', 'M2')
        
        FS_M1=int(round(60*0.27))
        FS_M2=int(round(60*0.18))
        
        GA_XX=GP_fi*GA_pfi
        GI_XX=GP_fi*(1-GA_pfi)
        
        M1_SN=500*1/GetNetw('sub_sampling', 'M1')
        M2_GI=500*1/GetNetw('sub_sampling', 'M2')
        M2_GA=M2_GI*0.1
        d={'M1_SN_gaba':{'fan_in0': M1_SN, 'rule':'set-set' },
           'M2_GI_gaba':{'fan_in0': M2_GI, 'rule':'set-set' },
           'M2_GA_gaba':{'fan_in0': M2_GA, 'rule':'all-all'},
                   
           'M1_M1_gaba':{'fan_in0': M1_M1, 'rule':'all-all' },
           'M1_M2_gaba':{'fan_in0': M1_M2, 'rule':'all-all' },                     
           'M2_M1_gaba':{'fan_in0': M2_M1, 'rule':'all-all' },
           'M2_M2_gaba':{'fan_in0': M2_M2, 'rule':'all-all' },                     
          
           'FS_M1_gaba':{'fan_in0': FS_M1, 'rule':'all-all' },
           'FS_M2_gaba':{'fan_in0': FS_M2, 'rule':'all-all' },                       
           'FS_FS_gaba':{'fan_in0': 9, 'rule':'all-all' },
         
           'ST_GA_ampa':{'fan_in0': 30, 'rule':'all-all' },
           'ST_GI_ampa':{'fan_in0': 30, 'rule':'all-all' },
           'ST_SN_ampa':{'fan_in0': 30, 'rule':'all-all' },
          
           'GA_FS_gaba':{'fan_in0': 10, 'rule':'all-all' },
           'GA_M1_gaba':{'fan_in0': 10, 'rule':'all-all' },
           'GA_M2_gaba':{'fan_in0': 10, 'rule':'all-all' },
           'GA_GA_gaba':{'fan_in0': GA_XX,'rule':'all-all' }, 
           'GA_GI_gaba':{'fan_in0': GA_XX,'rule':'all-all' },

           'GI_FS_gaba':{'fan_in0': 10, 'rule':'all-all' },
           'GI_M1_gaba':{'fan_in0': 10, 'rule':'all-all' },
           'GI_M2_gaba':{'fan_in0': 10, 'rule':'all-all' },
           'GI_GI_gaba':{'fan_in0': GI_XX,'rule':'all-all' },
           'GI_GA_gaba':{'fan_in0': GI_XX,'rule':'all-all' },
            
           'GI_ST_gaba':{'fan_in0': 30, 'rule':'all-all' },
           'GI_SN_gaba':{'fan_in0': 32, 'rule':'all-all' }}  
        
        conns.update(d)
#         pp(d)
        
        # Add extend to conn
        for k in sorted(conns.keys()): 
            source=k.split('_')[0]

            # Cortical input do not randomally change CTX input

            if dic['node'][source]['type']=='input':
                kwargs={'conn_type':'constant'}
            if dic['node'][source]['type']=='network':
                kwargs={'conn_type':'uniform'}
 
            #Get Defaults
            d=self._get_defaults_conn(k, k, **kwargs)
            
            # Set beta for FS_M2
            if k=='FS_M2_gaba':
                d['beta_fan_in']=-0.9 #0.8
            # Set beta for MS_MS
            if k in ['M1_M1_gaba', 'M1_M2_gaba','M2_M1_gaba','M2_M2_gaba']:
                d['beta_fan_in']=0.56

            e=None
            if k in ['M1_M1_gaba', 'M1_M2_gaba', 'M2_M1_gaba', 'M2_M2_gaba']:#pre[0:2] in ['M1', 'M2'] and post[0:2] in ['M1', 'M2']:
                e=2800./(DepNode('M1', 'calc_n')+DepNode('M2', 'calc_n'))#None
                e=min(e, 0.5)
            elif k in ['FS_M1','FS_M2', 'FS_FS']:#pre[0:2] =='FS' and post[0:2] in ['M1', 'M2', 'FS']:
                e=560./(DepNode('M1', 'calc_n')+DepNode('M2', 'calc_n'))
                e=min(e, 0.5)
            if e!=None:
                d['mask']=[-e, e]   
        
            #Default they are leasioned
            if k in ['M2_GA_gaba', 'GI_FS_gaba', 
                     'GI_M1_gaba', 'GI_M2_gaba']:
                d['lesion']=True
            conns[k]=misc.dict_update(d,conns[k])
            
        dic['conn']=misc.dict_update(dic['conn'], conns)            
         
        return dic                      
 
class Inhibition(Par_base, InhibitionPar_base, Par_base_mixin):
    pass      
      

class Inhibition_striatum_base(object):

    def _get_par_constant(self):
        
        
        dic_other=self.other.get_par_constant()
        
        C1={'amplitudes':[1.],
            'duration':[200],
            'repetitions':1}
         
        C2={'amplitudes':[1.],
            'duration':[200],
            'repetitions':1}
        
        CF={'amplitudes':[1.],
            'duration':[200],
            'repetitions':1}         
        CS={'amplitudes':[1.],
            'duration':[200],
            'repetitions':1}   
        
        dic={'netw':{'input':{}}}
        dic['netw']['input']={'C1':{'type':'burst2','params':C1},
                              'C2':{'type':'burst2','params':C2},
                              'CF':{'type':'burst2','params':CF},
                              'CS':{'type':'burst2','params':CS}}
        
        dic = misc.dict_update(dic_other, dic)
        return dic


class Inhibition_striatum(Par_base, Inhibition_striatum_base, Par_base_mixin): 
    pass 

def setup_burst3(dic_other, max_n_set_pre,  inps, dic={}):

    l = []
    for inp in inps:
        l.append([inp, dic_other['node'][inp]['rate']])
    
    
    for inp, rate in l:
        d={'type':'burst3',
           'params':{
                     'repetitions':2},
                     }
        params_sets={}
        d['params']['n_set_pre'] = dic_other['node'][inp]['n_sets']
        for i in range(max_n_set_pre):
            dd = {str(i):{'active':False, 
                    'amplitudes':[1.0], 
                    'durations':[100.0], 
                    'proportion_connected':1, 
                    'rate':rate}}
            params_sets.update(dd)
        
        d['params'].update({'params_sets':params_sets})
        dic = misc.dict_update(dic, {'netw':{'input':{inp:d}}})
    
    return dic

def setup_burst3_wo(dic_other, max_n_set_pre,  inps, dic={}):

    
    for inp in inps:
        d={'type':'burst3_with_oscillations',
           'params':{
                     'repetitions':2},
                     }
        params_sets={}
        d['params']['n_set_pre'] = dic_other['node'][inp]['n_sets']
        for i in range(max_n_set_pre):
            dd = {str(i):{'active':False, 
                    'amplitudes':[1.0], 
                    'durations':[100.0], 
                    'proportion_connected':1}}
            params_sets.update(dd)
        
        d['params'].update({'params_sets':params_sets})
        dic = misc.dict_update(dic, {'netw':{'input':{inp:d}}})
    
    return dic

class Compete_base(object):

    def _get_par_constant(self):

        dic={'netw':{'input':{}},
             'nest':{},
             'node':{}}
        
        dic_other=self.other.get_par_constant()
        max_n_set_pre=80
        inps=['C1', 'C2', 'CF', 'CS']
        
        dic = setup_burst3(dic_other, max_n_set_pre, inps, dic)
    
        for key in inps: 
            new_name=key+'d'             
            dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
                                   'rates':[0.],
                                   'timings':[1.]}   
            dic['node'][key]={'model':new_name}
        
        dic = misc.dict_update(dic_other, dic)
        

        return dic

class Compete_with_oscillations_base(object):

    def _get_par_constant(self):

        dic={'netw':{'input':{}},
             'nest':{},
             'node':{}}
        
        dic_other=self.other.get_par_constant()
        max_n_set_pre=2
        inps=['C1', 'C2', 'CF', 'CS']
        
        dic = setup_burst3(dic_other, max_n_set_pre, inps, dic)
    
        for key in inps: 
            new_name=key+'d'             
            dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
                                   'rates':[0.],
                                   'timings':[1.]}   
            dic['node'][key]={'model':new_name}
        
#         dic = misc.dict_update(dic_other, dic)
        

#         dic={'netw':{'input':{}},
#              'nest':{},
#              'node':{}}
        
        d={'type':'burst3_with_oscillations', 
             'params':{
                       'p_amplitude_upp':0.1,
                       'p_amplitude_down':-0.1,
                       'p_amplitude0':1.0,
                       'freq': 20.,
                       'freq_min':None,
                       'freq_max':None,
                       'period':'constant',
                       }} 
        
        for key in ['C1', 'C2', 'CF', 'CS']: 
            dic['netw']['input'][key]['params'].update(d['params'])
            dic['netw']['input'][key]['type']='burst3_with_oscillations'
              

#             dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
#                                    'rates':[0.],
#                                    'timings':[1.]
#                                    }   
#             dic['node'][key]={'model':new_name
#                               }
#         
        dic = misc.dict_update(dic_other, dic)

        return dic

class Compete_with_oscillations(Par_base, Compete_with_oscillations_base, Par_base_mixin): 
    pass 



class MSN_cluster_compete(Par_base, Compete_base, Par_base_mixin): 
    pass 


class Go_NoGo_compete(Par_base, Compete_base, Par_base_mixin): 
    pass 


class FSN_effect_base(object):

    def _get_par_constant(self):
        
        
        dic_other=self.other.get_par_constant()
        max_n_set_pre=80
        inps=['CF']
        
        dic = setup_burst3(dic_other, max_n_set_pre, inps)

        dic = misc.dict_update(dic_other, dic)
        return dic


class FSN_effect(Par_base, FSN_effect_base, Par_base_mixin): 
    pass 
              
class Slow_wave_base(object):

    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        #self._dic_con['node']['M1']['n']
        
        
        dic={'netw':{'input':{}}}
        
        d={'type':'oscillation', 
           'paramsc':{'p_amplitude_mod':0.8,
                     'freq': 1.,
                     'freq_min':None,
                     'freq_max':None,
                     'period':'constant'}} 
        for key in ['C1', 'C2', 'CF', 'CS']: 
            dic['netw']['input'][key]=d      
        
        dic = misc.dict_update(dic_other, dic)
        return dic


class Slow_wave(Par_base, Slow_wave_base, Par_base_mixin): 
    pass   


class Slow_wave2_base(object):

    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
              
        #self._dic_con['node']['M1']['n']
        
#         dic['nest']['CF']['type_id']='poisson_generator'
        dic={'netw':{'input':{}},
             'nest':{},
             'node':{}}
        
        d={'type':'oscillation2', 
             'params':{
                       'p_amplitude_upp':0.1,
                       'p_amplitude_down':-0.1,
                       'p_amplitude0':1.0,
                       'freq': 1.,
                       'freq_min':None,
                       'freq_max':None,
                       'period':'constant',
                       }} 
        
        
        for key in ['C1', 'C2', 'CF', 'CS']: 
            dic['netw']['input'][key]=d  
            new_name=key+'d' 
            dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
                                   'rates':[0.],
                                   'timings':[1.]}   
            dic['node'][key]={'model':new_name}
        
        dic = misc.dict_update(dic_other, dic)
        return dic


class Slow_wave2(Par_base, Slow_wave2_base, Par_base_mixin): 
    pass 

class Slow_wave2_EI_EA_base(object):

    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
              
        #self._dic_con['node']['M1']['n']
        
#         dic['nest']['CF']['type_id']='poisson_generator'
        dic={'netw':{'input':{}},
             'nest':{},
             'node':{}}
        
        d={'type':'oscillation2', 
             'params':{'p_amplitude_upp':0.1,
                       'p_amplitude_down':-0.1,
                       'p_amplitude0':1.0,
                       'freq': 1.,
                       'freq_min':None,
                       'freq_max':None,
                       'period':'constant'}} 
        
        
        for key in ['C1', 'C2', 'CF', 'CS', 'EA','EI']: 
            dic['netw']['input'][key]=d  
            new_name=key+'d' 
            dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
                                   'rates':[0.],
                                   'timings':[1.]}   
            dic['node'][key]={'model':new_name}
        
        dic = misc.dict_update(dic_other, dic)
        return dic


class Slow_wave2_EI_EA(Par_base, Slow_wave2_EI_EA_base, Par_base_mixin): 
    pass 

class Beta_base(object):
    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        #self._dic_con['node']['M1']['n']
        
        
        dic={'netw':{'input':{}},
             'nest':{},
             'node':{}}
        
        d={'type':'oscillation2', 
             'params':{'p_amplitude_upp':0.1,
                       'p_amplitude_down':-0.1,
                       'p_amplitude0':1.0,
                       'freq': 20.,
                       'freq_min':None,
                       'freq_max':None,
                       'period':'constant'}} 
        
        
        for key in ['C1', 'C2', 'CF', 'CS']: 
            dic['netw']['input'][key]=d  
            new_name=key+'d' 
            dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
                                   'rates':[0.],
                                   'timings':[1.]}   
            dic['node'][key]={'model':new_name}
        
        dic = misc.dict_update(dic_other, dic)
        return dic


class Beta(Par_base, Beta_base, Par_base_mixin): 
    pass 
class Beta_EI_EA_base(object):
    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        #self._dic_con['node']['M1']['n']
        
        
        dic={'netw':{'input':{}},
             'nest':{},
             'node':{}}
        
        d={'type':'oscillation2', 
             'params':{'p_amplitude_upp':0.1,
                       'p_amplitude_down':-0.1,
                       'p_amplitude0':1.0,
                       'freq': 20.,
                       'freq_min':None,
                       'freq_max':None,
                       'period':'constant'}} 
        
    
        for key in ['C1', 'C2', 'CF', 'CS', 'EA', 'EI']: 
            dic['netw']['input'][key]=d  
            new_name=key+'d' 
            dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
                                   'rates':[0.],
                                   'timings':[1.]}   
            dic['node'][key]={'model':new_name}
        
        dic = misc.dict_update(dic_other, dic)
        return dic


class Beta_EI_EA(Par_base, Beta_EI_EA_base, Par_base_mixin): 
    pass 

class Beta_striatum_base(object):
    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        #self._dic_con['node']['M1']['n']
        
        
        dic={'netw':{'input':{}},
             'nest':{},
             'node':{}}
        
        d={'type':'oscillation2', 
             'params':{'p_amplitude_upp':0.1,
                       'p_amplitude_down':-0.1,
                       'p_amplitude0':1.0,
                       'freq': 20.,
                       'freq_min':None,
                       'freq_max':None,
                       'period':'constant'}} 
        
        
        for key in ['C1', 'C2', 'CF']: 
            dic['netw']['input'][key]=d  
            new_name=key+'d' 
            dic['nest'][new_name]={'type_id':'poisson_generator_dynamic',
                                   'rates':[0.],
                                   'timings':[1.]}   
            dic['node'][key]={'model':new_name}
        
        dic = misc.dict_update(dic_other, dic)
        return dic


class Beta_striatum(Par_base, Beta_striatum_base, Par_base_mixin): 
    pass 


class ThalamusPar_base(object):
    
    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        dic={}
        
        
        # ========================
        # Default simu parameters 
        # ========================
        
        #@TODO change save path
        dic['simu']={}

        dp=dr.HOME_DATA+'/'
        dco= dp + 'conn/'       
        dcl= dp + self.__class__.__name__+'/'
        df=  dp + 'fig/'
        dn=  dp +  self.__class__.__name__+'/nest/'
        
                 
        dic['simu']['path_data']=dp
        dic['simu']['path_conn']=dco
        dic['simu']['path_class']=dcl
        dic['simu']['path_figure']=df
        dic['simu']['path_nest']=dn
        
        # ========================
        # Default netw parameters 
        # ========================
        
        dic['netw']={}    
        dic['netw']['n_nuclei']={'PF':30000}
        
        d={'type':'constant', 'params':{}}
        dic['netw']['input']={}
        for key in ['CP']: 
            dic['netw']['input'][key]=d             
        
        # ========================        
        # Default nest parameters 
        # ========================
        # Defining default parameters
        
        # EXT-CTX 
        dic['nest']={}
        dic['nest']['CP_PF_ampa']={}
        dic['nest']['CP_PF_ampa']['weight']   = 0.25    # n.d. set as for CTX to MSN   
        dic['nest']['CP_PF_ampa']['delay']    = 12.    # n.d. set as for CTX to MSN   
        dic['nest']['CP_PF_ampa']['type_id'] = 'static_synapse'   
        dic['nest']['CP_PF_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']            
        
        
        # PF/CM-MSN_D1 
        dic['nest']['PF_M1_ampa']={}
        dic['nest']['PF_M1_ampa']['weight']  = 0.5   # n.d. set as for CTX to MSN   
        dic['nest']['PF_M1_ampa']['delay']   = 10.0   # n.d. set as for CTX to MSN   
        dic['nest']['PF_M1_ampa']['type_id'] = 'static_synapse'   
        dic['nest']['PF_M1_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']        


        # PF/CM-MSN_D2
        dic['nest']['PF_M2_ampa'] = deepcopy(dic['nest']['PF_M1_ampa'])   
        
        
        # SNR-PF/CM 
        dic['nest']['SN_PF_gaba']={}
        dic['nest']['SN_PF_gaba']['weight']   = 1.    # n.d. set as for CTX to MSN   
        dic['nest']['SN_PF_gaba']['delay']    = 5.    # n.d. set as for CTX to MSN   
        dic['nest']['SN_PF_gaba']['type_id'] = 'static_synapse'   
        dic['nest']['SN_PF_gaba']['receptor_type'] = self.rec['izh']['GABAA_1']      
    
        
        # =============        
        # Neuron Models
        # =============

        #CTX-Pf/Cm
        dic['nest']['CP']={}
        dic['nest']['CP']['type_id']='poisson_generator'
        dic['nest']['CP']['rate']=0.0
    
        # Pf/Cm 
        # ===
        dic['nest']['PF']={}    
        dic['nest']['PF']['type_id'] = 'izhik_cond_exp'
    
        dic['nest']['PF']['a']      =  0.03      # (E.M. Izhikevich 2007)
        dic['nest']['PF']['b_1']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['PF']['b_2']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['PF']['c']      = -50.       # (E.M. Izhikevich 2007)
        dic['nest']['PF']['C_m']    =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['PF']['d']      =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['PF']['E_L']    = -60.0      # (E.M. Izhikevich 2007)
        dic['nest']['PF']['k']      =   0.7      # (E.M. Izhikevich 2007)
        dic['nest']['PF']['V_peak'] =  35.       # (E.M. Izhikevich 2007)
        dic['nest']['PF']['V_b']    = dic['nest']['PF']['E_L']    # (E.M. Izhikevich 2007)
        dic['nest']['PF']['V_th']   = -40.        # (E.M. Izhikevich 2007)
        dic['nest']['PF']['V_m']    =  -50.    
    
        dic['nest']['PF']['AMPA_1_Tau_decay'] = 12.  # Same as MSN
        dic['nest']['PF']['AMPA_1_E_rev']     =  0.  # Same as MSN
        
        dic['nest']['PF']['NMDA_1_Tau_decay'] = 160.# Same as MSN
        dic['nest']['PF']['NMDA_1_E_rev']     =  dic['nest']['PF']['AMPA_1_E_rev']    
        dic['nest']['PF']['NMDA_1_Vact']      = -20.0
        dic['nest']['PF']['NMDA_1_Sact']      =  16.0

        dic['nest']['PF']['GABAA_1_Tau_decay'] = 8.  # Same as MSN
        dic['nest']['PF']['GABAA_1_E_rev']     =  0.  # Same as MSN
        dic['nest']['PF']['tata_dop']      =  DepNetw('calc_tata_dop')
        
        # ========================
        # Default node parameters 
        # ========================
           
        dic['node']={}
        inputs={'CP': {'target':'PF',  'rate':300}} 
        
        for key, val in inputs.items():  
            d = self._get_defaults_node_input(key, val)
            inputs[key]=misc.dict_update(d, inputs[key])
        
        
        dic['node']=misc.dict_merge(dic['node'], inputs)             
        
        network={'PF':{}}

        # Randomization of C_m and V_m
        for key in network.keys():     
            model=GetNode(key, 'model')
            d = self._get_defaults_node_network(key, model)             
            network[key]=misc.dict_update(d, network[key]) 
            
        dic['node']=misc.dict_update(dic['node'], network) 
        
                
        # ========================
        # Default conn parameters 
        # ========================         
        conns={'EC_CO_ampa':{ 'syn':'EC_CO_ampa' }}
        for key in conns.keys():
            conns[key].update({'fan_in0':1,  'rule':'1-1' })
            
        d={}
        conns={}
        #d['conn']={}  
        d['CP_PF_ampa']={'fan_in0': 1,   'rule':'1-1'}             
        d['SN_PF_gaba']={'fan_in0': 20,  'rule':'all-all'}   
        conns=misc.dict_update(conns, d)      
        
        for k in sorted(conns.keys()): 
                        
            syn=GetConn(k, 'syn')
            d=self._get_defaults_conn(k, syn)
            # Cortical input do not randomally change CTX input
            if k=='CP_PF_ampa':
                d['delay']  = {'type':'constant', 
                              'params':GetNest(syn,'delay')}
                d['weight'] = {'type':'constant', 
                              'params':GetNest(syn,'weight')}

            conns=misc.dict_update({k:d}, conns)             

            
            
        dic['conn']={}
        dic['conn']=misc.dict_merge(dic['conn'], conns)       

        dic = misc.dict_update(dic_other, dic)
        return dic               

class Thalamus(Par_base, ThalamusPar_base, Par_base_mixin): 
    pass   


class Bcpnn_h0_base(object):

    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        dic={}
        
        # ========================
        # Default simu parameters 
        # ========================
        
        dic['simu']={}
        
        dc=(dr.HOME+'/results/papers/bcpnn/'
            +'network/conn')        
        dp=(dr.HOME+'/results/papers/bcpnn/'
            +'network/'+self.__class__.__name__)
        df=(dr.HOME+'/projects/papers/bcpnn/'
           +'figures/'+self.__class__.__name__)
        dn=(dr.HOME+'/results/papers/bcpnn/'
            +'network/'+self.__class__.__name__+'/nest')     
        
                 
        dic['simu']['path_conn']=dc
        dic['simu']['path_data']=dp
        dic['simu']['path_figure']=df
        dic['simu']['path_nest']=dn

        # ========================
        # Default netw parameters 
        # ========================
        
        dic['netw']={}
        dic['netw']['FF_curve']={'input':'EC',
                                 'output':'CO'}
  
        dic['netw']['sub_sampling']={'M1':50.0*4,'M2':50.0*4, 'CO':300.0}
        
        dic['netw']['n_states']=10
        dic['netw']['n_actions']=5
        
        
        
        
        dic['netw']['n_nuclei']={'CO':17000000}
        
        dic['netw']['input']={}    
        dic['netw']['input']['EC']={'type':'bcpnn', 
                                    'params':{'time':100.0, 
                                              'n_set_pre':10, 
                                              'p_amplitude':3, 
                                              'n_MSN_stim':20}}
        # ========================
        # Default nest parameters 
        # ========================
        # Defining default parameters
        
        # EXT-CTX 
        dic['nest']={}
        dic['nest']['EC_CO_ampa']={}
        dic['nest']['EC_CO_ampa']['weight']   = 0.25    # n.d. set as for CTX to MSN   
        dic['nest']['EC_CO_ampa']['delay']    = 12.    # n.d. set as for CTX to MSN   
        dic['nest']['EC_CO_ampa']['type_id'] = 'static_synapse'   
        dic['nest']['EC_CO_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']      
    
        
        # CTX-FSN 
        dic['nest']['CO_FS_ampa']={}
        dic['nest']['CO_FS_ampa']['weight']  = 252.5/GetConn('CO_FS_ampa', 'fan_in0')  # n.d. set as for CTX to MSN   
        dic['nest']['CO_FS_ampa']['delay']   = GetNest('CF_FS_ampa', 'delay')    # n.d. set as for CTX to MSN   
        dic['nest']['CO_FS_ampa']['type_id'] = 'static_synapse'   
        dic['nest']['CO_FS_ampa']['receptor_type'] = self.rec['izh']['AMPA_1']        
        
        
        # CTX-MSN D1
        dic['nest']['CO_M1_ampa']={}
        dic['nest']['CO_M1_ampa']['weight']   =  159./GetConn('CO_M1_ampa', 'fan_in0')    # constrained by Ellender 2011
        dic['nest']['CO_M1_ampa']['delay']    = GetNest('C1_M1_ampa', 'delay')       # Mallet 2005
        dic['nest']['CO_M1_ampa']['type_id'] = 'static_synapse'
        dic['nest']['CO_M1_ampa']['receptor_type'] = self.rec['aeif'] ['AMPA_1']
        
        dic['nest']['CO_M1_nmda'] = deepcopy(dic['nest']['CO_M1_ampa'])
        dic['nest']['CO_M1_nmda']['weight'] =  6.042/GetConn('CO_M1_nmda', 'fan_in0')   # (Humphries, Wood, and Gurney 2009)
        dic['nest']['CO_M1_nmda']['receptor_type'] = self.rec['aeif'] [ 'NMDA_1' ]
        
        
        # CTX-MSN D2
        dic['nest']['CO_M2_ampa'] = deepcopy(dic['nest']['CO_M1_ampa'])
        dic['nest']['CO_M2_ampa']['weight'] =  237.636/GetConn('CO_M1_ampa', 'fan_in0')    # constrained by Ellender 2011
        
        dic['nest']['CO_M2_nmda'] = deepcopy(dic['nest']['CO_M1_nmda'])
        dic['nest']['CO_M2_nmda']['weight'] =  237.636/GetConn('CO_M1_nmda', 'fan_in0')   # (Humphries, Wood, and Gurney 2009) 
    

        # CTX-STN
        dic['nest']['CO_ST_ampa']={}
        dic['nest']['CO_ST_ampa']['weight']  =  22./GetConn('CO_ST_ampa', 'fan_in0')
        dic['nest']['CO_ST_ampa']['delay']   = GetNest('CS_ST_ampa', 'delay')  # Fujimoto and Kita 1993
        dic['nest']['CO_ST_ampa']['type_id'] = 'static_synapse'  
        dic['nest']['CO_ST_ampa']['receptor_type'] = self.rec['aeif'] [ 'AMPA_1' ]  
        
        dic['nest']['CO_ST_nmda'] = deepcopy(dic['nest']['CO_ST_ampa'])
        dic['nest']['CO_ST_nmda']['weight'] =  .55/GetConn('CO_ST_nmda', 'fan_in0')   # n.d.; same ratio ampa/nmda as MSN
        dic['nest']['CO_ST_nmda']['receptor_type'] = self.rec['aeif'] [ 'NMDA_1' ]  
        
        # ============        
        # Input Models
        # ============ 
        
        
        #EXT-CTX
        dic['nest']['EC']={}
        dic['nest']['EC']['type_id']='poisson_generator' 
        dic['nest']['EC']['rate']=0.0
        
        
        # =============        
        # Neuron Models
        # =============
    
        # CTX
        # ===
        dic['nest']['CO']={}    
        dic['nest']['CO']['type_id'] = 'izhik_cond_exp'
    
        dic['nest']['CO']['a']      =  0.03      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['b_1']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['CO']['b_2']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['CO']['c']      = -50.       # (E.M. Izhikevich 2007)
        dic['nest']['CO']['C_m']    =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['d']      =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['E_L']    = -60.0      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['k']      =   0.7      # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_peak'] =  35.       # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_b']    = dic['nest']['CO']['E_L']    # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_th']   = -40.        # (E.M. Izhikevich 2007)
        dic['nest']['CO']['V_m']    =  -50.    
    
        dic['nest']['CO']['AMPA_1_Tau_decay'] = 12.  # Same as MSN
        dic['nest']['CO']['AMPA_1_E_rev']     =  0.  # Same as MSN
        
        dic['nest']['CO']['NMDA_1_Tau_decay'] = 160.# Same as MSN
        dic['nest']['CO']['NMDA_1_E_rev']     =  dic['nest']['CO']['AMPA_1_E_rev']    
        dic['nest']['CO']['NMDA_1_Vact']      = -20.0
        dic['nest']['CO']['NMDA_1_Sact']      =  16.0
        
        dic['nest']['CO']['tata_dop']      =  DepNetw('calc_tata_dop')
        
        
        # ========================
        # Default node parameters 
        # ========================
   

        dic['node']={}
        inputs={'EC': {'target':'CO',  'rate':300}} 
        
        for key, val in inputs.items():  
            d=self._get_defaults_node_input(key, val)

            inputs=misc.dict_update({key:d}, inputs)
            
        dic['node']=misc.dict_merge(dic['node'], inputs) 

        network={'CO':{}}
        network.update({ 'M1':{'n_sets':GetNetw('n_actions')},
                        'M2':{'n_sets':GetNetw('n_actions')},
                        'GI':{'n_sets':GetNetw('n_actions')},
                        'SN':{'n_sets':GetNetw('n_actions')}})  
        
        # Randomization of C_m and V_m
        for key in network.keys():     
            model=GetNode(key, 'model')
            d=self._get_defaults_node_network(key, model)
    
            network=misc.dict_update({key:d}, network)            
        
        dic['node']=misc.dict_update(dic['node'], network)   
        
        # ========================
        # Default conn parameters 
        # ========================
        

        conns={}
        d={'EC_CO_ampa':{ 'fan_in0': 1, 'rule':'1-1'},
           'CO_M1_ampa':{'fan_in0': 20,  'rule':'all_set-all_set'},
           'CO_M1_nmda':{'fan_in0': 20, 'rule':'all_set-all_set' },
           'CO_M2_ampa':{'fan_in0': 20, 'rule':'all_set-all_set' },
           'CO_M2_nmda':{'fan_in0': 20, 'rule':'all_set-all_set' },
           'CO_FS_ampa':{'fan_in0': 20, 'rule':'all-all' },
           'CO_ST_ampa':{'fan_in0': 20, 'rule':'all-all' },
           'CO_ST_nmda':{'fan_in0': 20, 'rule':'all-all' },
           'GI_SN_gaba':{'fan_in0': 20, 'rule':'set-set'}}
        d=misc.dict_update(conns, d)


        # Add extend to conn
        p1='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h0/CO_M1.pkl'
        p2='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h0/CO_M1.pkl'
        p3='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h0/CO_M2.pkl'
        p4='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h0/CO_M2.pkl'
        path={'CO_M1_ampa':p1}
        path['CO_M1_nmda']=p2
        path['CO_M2_ampa']=p3
        path['CO_M2_nmda']=p4
        
        for k in sorted(conns.keys()): 
                        
            syn=GetConn(k, 'syn')
            d=self._get_defaults_conn(k, syn)
            # Cortical input do not randomally change CTX input
            if k=='EC_CO_ampa':
                d['delay']  = {'type':'constant', 
                               'params':GetNest(syn,'delay')}
                d['weight'] = {'type':'constant', 
                               'params':GetNest(syn,'weight')}

            if k in ['CO_M1_ampa', 'CO_M1_nmda','CO_M2_ampa','CO_M2_nmda']:
                d['weight'] =  {'type':'learned', 
                                'params':GetNest(syn,'weight'), 
                                'path':path[k]}
            
            conns=misc.dict_update({k:d}, conns)    
                
        dic['conn']={}
        dic['conn']=misc.dict_merge(dic['conn'], conns)  



        netw_inp_removed=['C1', 'C2', 'CF', 'CS']
        for model in netw_inp_removed: 
            del dic_other['netw']['input'][model]
        # Remove models for other dic
        conn_rem=['C1_M1_ampa', 'C1_M1_nmda',
                  'C2_M2_ampa', 'C2_M2_nmda',
                  'CF_FS_ampa', 'CS_ST_ampa',
                  'CS_ST_nmda']
        for model in conn_rem: del dic_other['conn'][model]
        
        nest_models_rem=[]
        for model in nest_models_rem: del dic_other['nest'][model]
        
        
        node_rem=['CF', 'C1', 'C2', 'CS']
        for model in node_rem: del dic_other['node'][model]   
               

        dic = misc.dict_update(dic_other, dic)
        return dic

class Bcpnn_h0(Par_base, Bcpnn_h0_base, Par_base_mixin_bcpnn):
    pass      
      
class Bcpnn_h1_base(object):
    

    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        dic={}
        dic['netw']={}
        dic['netw']['n_nuclei']={'F1':dic_other['netw']['n_nuclei']['FS']/2,
                                 'F2':dic_other['netw']['n_nuclei']['FS']/2}
                  
        dic['nest']={'CO_FS_ampa':{'weight': 
                                   252.5/GetConn('CO_F1_ampa', 'fan_in0')}} 
        
           
        dic['node']={}
        dic['node']['CO'] = deepcopy(dic_other['node']['CO'])    
        dic['node']['GA'] = deepcopy(dic_other['node']['GA'])  
                
        for key in ['F1', 'F2']:
            model=GetNode(key, 'model')
            dic['node'][key] = deepcopy(dic_other['node']['FS'])    
            dic['node'][key]['n']=DepNode(key,'calc_n')
            dic['node'][key]['n_sets']=GetNetw('n_actions')
            dic['node'][key]['nest_params']={'I_e':GetNode(key, 'I_vivo')}
            dic['node'][key]['rand']= self.build_setup_node_rand(model)  
            dic['node'][key]['sets']=DepNode(key, 'calc_sets')
      
        conns={}
        u=dic_other['conn']
        d={'CO_M1_ampa':{'fan_in0': u['CO_M1_ampa']['fan_in0'],  
                         'rule':    u['CO_M1_ampa']['rule'],},
           'CO_M1_nmda':{'fan_in0': u['CO_M1_nmda']['fan_in0'],  
                         'rule':    u['CO_M1_nmda']['rule'],},
           'CO_M2_ampa':{'fan_in0': u['CO_M2_ampa']['fan_in0'],  
                         'rule':    u['CO_M2_ampa']['rule'],},
           'CO_M2_nmda':{'fan_in0': u['CO_M2_nmda']['fan_in0'],  
                         'rule':    u['CO_M2_nmda']['rule'],},
           'CO_F1_ampa':{'fan_in0': 20, 
                         'syn':'CO_FS_ampa',
                         'rule':'all_set-all_set' },
           'CO_F2_ampa':{'fan_in0': 20,        
                         'syn':'CO_FS_ampa',
                         'rule':'all_set-all_set' }}
        
        conns=misc.dict_update(conns, d)
        
        
        d={}
        pairs=[['F1_M1_gaba', 'FS_M1_gaba'],
               ['F2_M2_gaba', 'FS_M2_gaba'],
               ['GA_F1_gaba', 'GA_FS_gaba'],
               ['GA_F2_gaba', 'GA_FS_gaba'],
               ['F1_F1_gaba', 'FS_FS_gaba'],
               ['F1_F2_gaba', 'FS_FS_gaba'],
               ['F2_F1_gaba', 'FS_FS_gaba'],
               ['F2_F2_gaba', 'FS_FS_gaba']]
        for key in ['fan_in0', 'syn', 'rule']:
            for c1, c2 in pairs:
                if not c1 in d:
                    d[c1]={}
                d[c1][key]=dic_other['conn'][c2][key]
        conns=misc.dict_update(conns, d)
  
        p1='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M1.pkl'
        p2='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M1.pkl'
        p3='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M2.pkl'
        p4='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_M2.pkl'
        p5='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_F1.pkl'
        p6='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-h1/CO_F2.pkl'
        path={'CO_M1_ampa':p1}
        path['CO_M1_nmda']=p2
        path['CO_M2_ampa']=p3
        path['CO_M2_nmda']=p4
        path['CO_F1_ampa']=p5
        path['CO_F2_ampa']=p6
        
        for k in sorted(conns.keys()): 
            source, _=k.split('_')[0:2]

            syn=GetConn(k, 'syn')
            d=self._get_defaults_conn(k, syn)
            
            if k in ['F1_M2_gaba', 'F2_M2_gaba']:
                conns[k]['beta_fan_in']=0.8
            
            # Cortical input do not randomally change CTX input
            if dic['node'][source]['type']=='input':
                d['delay']  = {'type':'constant', 
                               'params':GetNest(syn,'delay')}
                d['weight'] = {'type':'constant', 
                               'params':GetNest(syn,'weight')}

            if k in ['CO_M1_ampa', 'CO_M1_nmda','CO_M2_ampa','CO_M2_nmda',
                     'CO_F1_ampa','CO_F2_ampa']:
                d['weight'] =  {'type':'learned', 
                                'params':GetNest(syn,'weight'), 
                                'path':path[k]}     

            conns=misc.dict_update({k:d}, conns)              
           
                             
        dic['conn']={}
        dic['conn']=misc.dict_merge(dic['conn'], conns)
        
        node_rem=['FS']
        for model in node_rem: del dic_other['node'][model]
  
        del dic_other['netw']['n_nuclei']['FS']
        
        rem=['CO_FS_ampa','FS_M1_gaba', 'FS_M2_gaba', 'GA_FS_gaba', 'FS_FS_gaba']
        for model in rem:  
            del dic_other['conn'][model]  

        dic = misc.dict_update(dic_other, dic)
        return dic
            

class Bcpnn_h1(Par_base, Bcpnn_h1_base, Par_base_mixin_bcpnn):
    pass  


class Bcpnn_learning_base(object):
    
    def _get_par_constant(self):
        dic_other=self.other.get_par_constant()
        
        dic={}
        
        # ========================        
        # Default nest parameters 
        # ========================
        # Defining default parameters
                        
        # GPh 
        # ===
        #copy GP model
        
        # Hab 
        # ===
        dic['nest']['HA']={}    
        dic['nest']['HA']['type_id'] = 'izhik_cond_exp'
    
        dic['nest']['HA']['a']      =  0.03      # (E.M. Izhikevich 2007)
        dic['nest']['HA']['b_1']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['HA']['b_2']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['HA']['c']      = -50.       # (E.M. Izhikevich 2007)
        dic['nest']['HA']['C_m']    =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['HA']['d']      =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['HA']['E_L']    = -60.0      # (E.M. Izhikevich 2007)
        dic['nest']['HA']['k']      =   0.7      # (E.M. Izhikevich 2007)
        dic['nest']['HA']['V_peak'] =  35.       # (E.M. Izhikevich 2007)
        dic['nest']['HA']['V_b']    = dic['nest']['HA']['E_L']    # (E.M. Izhikevich 2007)
        dic['nest']['HA']['V_th']   = -40.        # (E.M. Izhikevich 2007)
        dic['nest']['HA']['V_m']    =  -50.    
    
    
        # GPh-Hab
        dic['nest']['HA']['AMPA_1_Tau_decay'] = 12.  # Same as MSN
        dic['nest']['HA']['AMPA_1_E_rev']     =  0.  # Same as MSN
        
        dic['nest']['HA']['NMDA_1_Tau_decay'] = 160.# Same as MSN
        dic['nest']['HA']['NMDA_1_E_rev']     =  dic['nest']['HA']['AMPA_1_E_rev']    
        dic['nest']['HA']['NMDA_1_Vact']      = -20.0
        dic['nest']['HA']['NMDA_1_Sact']      =  16.0
        
        
        # RMTg 
        # ====
        dic['nest']['RM']={}    
        dic['nest']['RM']['type_id'] = 'izhik_cond_exp'
    
        dic['nest']['RM']['a']      =  0.03      # (E.M. Izhikevich 2007)
        dic['nest']['RM']['b_1']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['RM']['b_2']    = -2.        # (E.M. Izhikevich 2007)
        dic['nest']['RM']['c']      = -50.       # (E.M. Izhikevich 2007)
        dic['nest']['RM']['C_m']    =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['RM']['d']      =  100.      # (E.M. Izhikevich 2007)
        dic['nest']['RM']['E_L']    = -60.0      # (E.M. Izhikevich 2007)
        dic['nest']['RM']['k']      =   0.7      # (E.M. Izhikevich 2007)
        dic['nest']['RM']['V_peak'] =  35.       # (E.M. Izhikevich 2007)
        dic['nest']['RM']['V_b']    = dic['nest']['RM']['E_L']    # (E.M. Izhikevich 2007)
        dic['nest']['RM']['V_th']   = -40.        # (E.M. Izhikevich 2007)
        dic['nest']['RM']['V_m']    =  -50.    
    

        # Hab-RMtg    
        dic['nest']['RM']['AMPA_1_Tau_decay'] = 12.  # Same as MSN
        dic['nest']['RM']['AMPA_1_E_rev']     =  0.  # Same as MSN
        
#        dic['nest']['RM']['NMDA_1_Tau_decay'] = 160.# Same as MSN
#        dic['nest']['RM']['NMDA_1_E_rev']     =  dic['nest']['RM']['AMPA_1_E_rev']    
#        dic['nest']['RM']['NMDA_1_Vact']      = -20.0
#        dic['nest']['RM']['NMDA_1_Sact']      =  16.0



        # SNc 
        # ===
        dic['nest']['SC']={}
        dic['nest']['SC']['type_id'] = 'my_aeif_cond_exp'
        
        dic['nest']['SC']['tau_w']   = 20.  # I-V relation, spike frequency adaptation
        dic['nest']['SC']['a_1']     =  3.      # I-V relation
        dic['nest']['SC']['a_2']     =  dic['nest']['SC']['a_1']      # I-V relation
        dic['nest']['SC']['b']       = 200.   # I-F relation
        dic['nest']['SC']['C_m']     =  80.    # t_m/R_in
        dic['nest']['SC']['Delta_T'] =  1.8                      
        dic['nest']['SC']['g_L']     =   3.
        dic['nest']['SC']['E_L']     = -55.8    #
        dic['nest']['SC']['I_e']     = 15.0 
        dic['nest']['SC']['V_peak']  =  20.                                                               # 
        dic['nest']['SC']['V_reset'] = -65.    # I-V relation
        dic['nest']['SC']['V_th']    = -55.2    # 
        dic['nest']['SC']['V_a']     = dic['nest']['SC']['E_L']     # I-V relation
        
        #STN-SCr
        dic['nest']['SC']['AMPA_1_Tau_decay'] =  12.   # n.d.; set as for STN to GPE
        dic['nest']['SC']['AMPA_1_E_rev']     =   0.   # n.d. same as CTX to STN
    
        # EXT-SCr
        dic['nest']['SC']['AMPA_2_Tau_decay'] = 5.0
        dic['nest']['SC']['AMPA_2_E_rev']     = 0.
        
        # Striosomes MSN D1-SNc
        dic['nest']['SC']['GABAA_1_E_rev']    =-80.     # (Connelly et al. 2010)
        dic['nest']['SC']['GABAA_1_Tau_decay']=GetNest('M1_SC_gaba', 'tau_psc')      # (Connelly et al. 2010)
 
        # RMTg-SNc
        dic['nest']['SC']['GABAA_2_E_rev']    =-72.     # (Connelly et al. 2010)
        dic['nest']['SC']['GABAA_2_Tau_decay']=GetNest('GI_SC_gaba', 'tau_psc')
    
    
        # Dopamine cells to Striatal connections     
        dic['nest']['SC']['tata_dop'] = DepNetw('calc_tata_dop')
        
                
        # ========================
        # Default node parameters 
        # ========================
        
                
        # ========================
        # Default conn parameters 
        # ========================
                

class Bcpnn_learning(Par_base, Bcpnn_learning_base, Par_base_mixin_bcpnn):
    pass      
def compute_conn_delay_params(delay, delay_type):
    
   
    if delay_type=='constant':
        delay  = delay
        
    if delay_type=='uniform':
        pr=0.5
        delay =  {'min':(1-pr)*delay,  
                        'max':(1+pr)*delay}
    return delay

def calc_fan_in(beta, dop, fan_in0):
    fan_in0=int(round(fan_in0))
    return round(fan_in0*(1+dop*beta))

def compute_conn_mask_dist_restric(n_dentritic_volume, n_total):   
     
    e=min(n_dentritic_volume/float(n_total), 0.5)
    return [-e, e]

def compute_conn_weight_params(weight, weight_type):
    if weight_type in ['constant', 'learned']:
        weight = weight      
                  
    if weight_type=='uniform':
        pr=0.5
        weight = {'min':(1-pr)*weight, 
                  'max':(1+pr)*weight}    
    
    
    return weight

def calc_n(d_nuclei_sizes, d_sub_sampl, name, netw_size, obj=None):
    nodes, sizes=[],[]
    for k,v in d_nuclei_sizes.items():
        if isinstance(v, Call):
            v=v.do(obj)
        if k in d_sub_sampl.keys():
            v=v/d_sub_sampl[k]
            
        nodes.append(k)
        sizes.append(v)
        
    proportions=numpy.array(sizes)/numpy.sum(sizes)    
    node_sizes=proportions*netw_size
    node_sizes=numpy.round(node_sizes,0)
    node_sizes=numpy.array(node_sizes, dtype=int)
    
    ind=numpy.argsort(node_sizes)
    for i in ind:
        if numpy.sum(node_sizes)<netw_size:
            node_sizes[i]+=1    
        elif numpy.sum(node_sizes)>netw_size:
            node_sizes[i]-=1
        else: break
        
        
    
    return dict(zip(nodes, node_sizes))[name]

def get_oscillation_time_rates(params, stop, testing, ru, rd):
    if params['period'] == 'constant':
        step = 1000 / 2 / params['freq']
        cycles = int(stop / (2 * step))
        rates = [rd, ru] * cycles
        times = list(numpy.arange(0, 2. * cycles * step, step))
    
    if params['period'] == 'uniform':
        f_min =params['freq_min']
        f_max =params['freq_max']
        step = 1000 / 2 / ((params['freq_min'] + params['freq_max']) / 2)
        cycles = int(stop / (2 * step))
        rates = [rd, ru] * cycles
        if not testing:
            periods = numpy.random.uniform(f_min, f_max, cycles)
        else:
            numpy.random.seed(0)
            periods = numpy.random.uniform(f_min, f_max, cycles)
            periods = numpy.round(periods, 2)
        periods/=2/1000.0
        times = []
        accum = 0
        for i in periods:
            times.extend([accum + 0, accum + i])
            accum += 2 * i

        times=numpy.array(times)
        rates=numpy.array(rates)
        rates=list(rates[times<stop])
        times=list(times[times<stop])
    return rates, times

def calc_spike_setup(n, params, rate ,start, stop, typ, testing=False):
        
    setup=[]

    n=int(n)

    if typ=='constant':           
        rates=[ rate]
        idx=range(n)
        setup+=[{'rates':rates, 
                 'times':[1.], 
                 'idx':idx, 
                 't_stop':stop}]    
        
    if typ=='bcpnn': 
        tt=params['time']
        p=params['p_amplitude']
        
        for i in range(params['n_set_pre']):
            idx=list(range(i,n, 
                            params['n_set_pre']))
            rate=rate
            setup+=[{'rates':[ rate, rate*p, rate], 
                     'times':[1., 
                              start+i*tt, 
                              start+(i+1)*tt],
                     'idx':idx,
                     't_stop':stop}]

    if typ=='burst': 
        t=numpy.array(params['duration'])
        r=params['rates']     
        rep=params['repetitions']
        
        idx=range(n)
        times=numpy.array([[0]+list(numpy.cumsum(t)[0:-1])
                          +i*numpy.sum(t) for i in range(rep) ])
        times=list(times.ravel())
        
        rates=r*rep
        setup+=[{'rates':rates, 
                 'times':times, 
                 'idx':idx,
                 't_stop':stop}]
        
    if typ=='burst2': 
        
    
        t=numpy.array(params['duration'])
        r=rate
        amp=params['amplitudes']   
        rep=params['repetitions']
        
        idx=range(n)
        times=numpy.array([[0]+list(numpy.cumsum(t)[0:-1])
                          +i*numpy.sum(t) for i in range(rep) ])
        times=list(times.ravel())
        
        rates=list(r*numpy.array(amp))*rep
        setup+=[{'rates':rates, 
                 'times':times, 
                 'idx':idx,
                 't_stop':stop}] 
 
 
    if typ=='burst3': 
        # TODO: make sure that when proportion connected is 
        # less than zero that the other proportion recieves inp
        
        n_set_pre=params['n_set_pre']
        params_sets=params['params_sets']
        rep=params['repetitions']
        
        for k in range(n_set_pre):
#             print k
            d=params_sets[str(k)]
            
            if not d['active']:
                continue
            
            proportion_connected=d['proportion_connected']
            
            m=int(n*proportion_connected)
            idx=list(range(k,m, n_set_pre))
            idx0=list(range(k+m,n, n_set_pre))
           
            r=d['rate']   
            amp=d['amplitudes']   
            
            t=numpy.array(d['durations'])        
            times=numpy.array([[0]+list(numpy.cumsum(t)[0:-1])
                          +i*numpy.sum(t) for i in range(rep) ])
            times=list(times.ravel())
        
            rates=list(r*numpy.array(amp))*rep
            setup+=[{'rates':rates, 
                     'times':times, 
                     'idx':idx,
                     't_stop':stop}] 
            
            if  not idx0:
                continue
            
            rates0=list(r*numpy.ones(len(amp)))*rep
            setup+=[{'rates':rates0, 
                     'times':times, 
                     'idx':idx0,
                     't_stop':stop}] 


    if typ=='burst3_oscillations': 
        # TODO: make sure that when proportion connected is 
        # less than zero that the other proportion recieves inp
        
        n_set_pre=params['n_set_pre']
        params_sets=params['params_sets']
        rep=params['repetitions']
        
        for k in range(n_set_pre):
#             print k
            d=params_sets[str(k)]
            
            if not d['active']:
                continue
            
            proportion_connected=d['proportion_connected']
            
            m=int(n*proportion_connected)
            idx=list(range(k,m, n_set_pre))
            amp=d['amplitudes']   
            
            t=numpy.array(d['durations'])        
            times=numpy.array([[0]+list(numpy.cumsum(t)[0:-1])
                          +i*numpy.sum(t) for i in range(rep) ])
            times=list(times.ravel())
        
            # remove baseline from rates with -1
            rates=list(rate*(numpy.array(amp)-1))*rep
            setup+=[{'rates':rates, 
                     'times':times, 
                     'idx':idx,
                     't_stop':stop}] 


        ru=rate*float(params['p_amplitude0']+params['p_amplitude_upp'])
        rd=rate*max(0., float(params['p_amplitude0']+params['p_amplitude_down']))
        
        rates, times = get_oscillation_time_rates(params, 
                                                  stop, 
                                                  testing, 
                                                  ru, 
                                                  rd)
#         [ti in]
        idx=range(n)
       
        setup+=[{'rates':rates, 
                 'times':times, 
                 'idx':range(n), 
                 't_stop':stop}]
        
        
    if typ=='oscillation':
        ru=rate*(2-params['p_amplitude_mod'])
        rd=rate*params['p_amplitude_mod']

        
        rates, times = get_oscillation_time_rates(params, stop, testing, ru, rd)        
        
        idx=range(n)

        setup+=[{'rates':rates, 
                 'times':times, 
                 'idx':idx, 
                 't_stop':stop}]


    if typ=='oscillation2':

        ru=rate*float(params['p_amplitude0']+params['p_amplitude_upp'])
        rd=rate*max(0., float(params['p_amplitude0']+params['p_amplitude_down']))
        
        rates, times = get_oscillation_time_rates(params, 
                                                  stop, 
                                                  testing, 
                                                  ru, 
                                                  rd)
#         
#         step=1000/2/params['freq']
#         cycles=int(stop/(2*step))
#         rates=[rd, ru]*cycles
#         times=list(numpy.arange(0, 2.*cycles*step, step))
        
        idx=range(n)
       
        setup+=[{'rates':rates, 
                 'times':times, 
                 'idx':idx, 
                 't_stop':stop}]

    if typ=='oscillation3':
        ru=rate*(params['p_amplitude0']+params['p_amplitude_mod'])
        rd=rate*max(0, (params['p_amplitude0']-params['p_amplitude_mod']))
        
        setup+=[{'rate_first':rd,
                 'rate_second':ru,
                 'period_first':params['period_first'], 
                 'period_second':params['period_second'], 
                 'idx':idx, 
                 't_stop':stop}]


    if typ=='burst_compete':
        p=params['p_amplitude']
        start=params['start']
        d=params['duration']
        idx_sets=params['idx_sets']
        for i, _p in zip(idx_sets,p):
            idx=list(range(i,n, params['n_set_pre']))
            setup+=[{'rates':[ rate, rate*_p, rate], 
                     'times':[1., 
                              start, 
                              start+d],
                     'idx':idx,
                     't_stop':stop}]
            
    return setup


def calc_sets(n_sets, n):
    return [my_slice(s, n, n_sets) for s in range(n_sets)]
    
def dummy_args(flag, **kwargs):
    args=[]
    out=[]
        
    if flag=='compute_conn_delay_params':
        args.append([2.0, 'constant'])
        out.append(2.0)
        
        args.append([2.0, 'uniform'])
        out.append({'max': 3.0, 'min': 1.0} )
        
    if flag=='calc_fan_in':
        args.append([0.2, 0.8, 30])
        out.append(round(30*(1+0.2*0.8)))
        
    if flag=='compute_conn_weight_params':
        args.append([2.0, 'constant'])
        out.append( 2.0)

        args.append([2.0, 'learned'])
        out.append(2.0)
        
        args.append([2.0, 'uniform'])
        out.append({'max': 3.0, 'min': 1.0} )
    
    if flag=='calc_n':
        args.append([{'u1':1000.0, 'u2':200.0},{'u1':10.0},'u1', 16.0 ])
        args.append([{'u1':1000.0, 'u2':200.0},{'u1':10.0},'u2', 16.0 ])
        n_u1, n_u2=1000./10, 200.0
        n_tot=n_u1+200.0
        out.append(int(numpy.round(n_u1/n_tot*16,0)))
        out.append(int(numpy.round(n_u2/n_tot*16,0)))
        
        args.append([{'u1':1000.0, 'u2':2000.0},{},'u2', 10000 ])
        out.append(int(round(2./3*10000)))

        args.append([{'u1':1000.0, 'u2':2000.0},{},'u2', 40000 ])
        out.append(int(round(2./3*10000*4)))
                    
    if flag=='calc_spike_setup':
        #CONSTANT
        args.append([4, {}, 25.0, 100.0, 2000.0, 'constant',])
        out.append([{'rates': [25.0], 
                     't_stop': 2000.0, 
                     'idx': [0, 1, 2, 3], 
                     'times': [1.0]}])
        
        #OSCILLATION with constant period
        params={'p_amplitude_mod':0.9,
                'freq': 1.,
                'period':'constant'
                }
        args.append([4, params, 25.0,100.0, 2000.0, 'oscillation'])
        out.append([{'rates': [22.5, 27.500000000000004, 
                               22.5, 27.500000000000004], 
                     't_stop': 2000.0, 
                     'idx': [0, 1, 2, 3], 
                     'times': [    0.,   500.,  1000.,  1500.]}])


        #OSCILLATION with uniform
        params={'p_amplitude_mod':0.9,
                'freq_min': 0.5,
                'freq_max': 1.5,
                'period':'uniform'
                }
        args.append([4, params, 25.0, 100.0, 2000.0, 'oscillation', True])
        out.append([{'idx': [0, 1, 2, 3],
                    'rates': [22.5, 27.500000000000004, 22.5, 27.500000000000004],
                    't_stop': 2000.0,
                    'times': [0, 525.0, 1050.0, 1660.0]}])



        #OSCILLATION2
        params={'p_amplitude_upp':0.1,
                'p_amplitude_down':-0.1,
                'p_amplitude0':1.0,
                'freq': 1.,
                'period':'constant'}
        args.append([4, params, 25.0,100.0, 2000.0, 'oscillation2'])
        out.append([{'rates': [22.5, 27.500000000000004, 
                               22.5, 27.500000000000004], 
                     't_stop': 2000.0, 
                     'idx': [0, 1, 2, 3], 
                     'times': [    0.,   500.,  1000.,  1500.]}])


        
        #BCPNN
        params={'time':950.0, 
                 'n_set_pre':2, 
                 'p_amplitude':3, 
                 'n_MSN_stim':20}
        args.append([4, params, 25.0, 100.0, 2000.0, 'bcpnn'])
        out.append([{'rates': [25.0, 75.0, 25.0], 
             't_stop': 2000.0, 
             'idx': [0, 2], 
             'times': [1.0, 100.0, 1050.0]}, 
            {'rates': [25.0, 75.0, 25.0], 
             't_stop': 2000.0, 
             'idx': [1, 3], 
             'times': [1.0, 1050.0, 2000.0]}])
 
        #BURST
        params={'duration':[100.0, 500.0, 200.0],
                'rates':[10.0,0.0, 20.0],
                'repetitions':2}
        
        args.append([4, params, 25.0, 100.0, 2000.0, 'burst']) 
        out.append([{'rates': [10.0, 0.0, 20.0, 
                               10.0, 0.0, 20.0], 
                     't_stop': 2000.0, 
                     'idx': [0,1,2,3], 
                     'times': [0.0, 100.0, 600.0,  
                               800.0, 900.0, 1400]}]) 
    
        #BURST3
        params={'n_set_pre':3,
                'repetitions':2,
                'params_sets':{'0':{'active':True,
                                    'amplitudes':[3.0,],
                                  'durations':[100.0],
                                  'proportion_connected':1,
                                  'rate':10.0,
                                  },
                               '1':{'active':True,
                                    'amplitudes':[2.0,],
                                  'durations':[200.0],
                                  'proportion_connected':0.25,
                                  'rate':10.0,
                                  },
                               '2':{'active':True,
                                    'amplitudes':[1.0,1.5],
                                  'durations':[100.0, 100.0],
                                  'proportion_connected':0.5,
                                  'rate':10.0,
                                  }}}
        args.append([12, params, 25.0, 100.0, 1500.0, 'burst3'])                       
        out.append( [{'idx': [0, 3, 6, 9],
                      'rates': [30.0, 30.0],
                      't_stop': 1500.0,
                      'times': [0.0, 100.0]},
                     {'idx': [1], 
                      'rates': [20.0, 20.0],
                      't_stop': 1500.0, 
                      'times': [0.0, 200.0]},
                     {'idx': [4, 7, 10], 
                      'rates': [10.0, 10.0], 
                      't_stop': 1500.0,
                      'times': [0.0, 200.0]},
                     {'idx': [2, 5], 
                       'rates': [10.0, 15.0, 10.0, 15.0],
                       't_stop': 1500.0,
                       'times': [0.0, 100.0, 200.0, 300.0]},
                    {'idx': [8, 11],
                     'rates': [10.0, 10.0, 10.0, 10.0],
                     't_stop': 1500.0,
                     'times': [0.0, 100.0, 200.0, 300.0]}]) 


        #BURST3_OSCILLATIONS
        params={'n_set_pre':3,
                'repetitions':2,
                'params_sets':{'0':{'active':True,
                                    'amplitudes':[3.0,],
                                  'durations':[100.0],
                                  'proportion_connected':1,
                                  },
                               '1':{'active':True,
                                    'amplitudes':[2.0,],
                                  'durations':[200.0],
                                  'proportion_connected':0.25,
    
                                  },
                               '2':{'active':True,
                                    'amplitudes':[1.0,1.5],
                                  'durations':[100.0, 100.0],
                                  'proportion_connected':0.5,
           
                                  }},
                'p_amplitude_upp':0.1,
                'p_amplitude_down':-0.1,
                'p_amplitude0':1.0,
                'freq': 1.,
                'period':'constant'}
        args.append([12, params, 10.0, 100.0, 2000.0, 'burst3_oscillations'])                       
        out.append( [{'idx': [0, 3, 6, 9],
                      'rates': [20.0, 20.0],
                      't_stop': 2000.0,
                      'times': [0.0, 100.0]},
                     {'idx': [1], 
                      'rates': [10.0, 10.0],
                      't_stop': 2000.0, 
                      'times': [0.0, 200.0]},
                     {'idx': [2, 5], 
                       'rates': [0.0, 5.0, 0.0, 5.0],
                       't_stop': 2000.0,
                       'times': [0.0, 100.0, 200.0, 300.0]},
                     
                     {'rates': [9, 11, 
                                9, 11], 
                     't_stop': 2000.0, 
                     'idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                     'times': [    0.,   500.,  1000.,  1500.]}]) 

        #BURST COMPETE
        params={'duration':100.0,
                'idx_sets':[0, 2],
                'n_set_pre':3,
                'p_amplitude':[2.0, 3.0],
                'start':1000.0,}
        args.append([6, params, 25.0, 100.0, 1500.0, 'burst_compete'])
        out.append([{'idx':[0,3],
                     'rates':[25., 50.,25.],
                     't_stop':1500.0,
                     'times':[1,1000.,1100]},
                    {'idx':[2,5],
                     'rates':[25., 75.,25.],
                     't_stop':1500.0,
                     'times':[1,1000.,1100]}]) 
        
    if flag=='calc_sets':
        args.append([3,9]) 
        out=[[my_slice(s, 9, 3) for s in range(3)]]       
        
        
    return args, out
def dummy_unittest_small(inp='i1', net='n1', n=10, **kwargs):

        dic={}
        size=kwargs.get('size', 10)
        # ========================
        # Default node parameters 
        # ========================  
        d1={'fan_in_distribution':'constant',
            'n':n,
            'spike_setup': [{'rates': [10.0], 
                             't_stop': 3000.0, 
                             'idx': range(n), 
                             'times': [1.0]}],                   
            'sets':[my_slice(s, n, 1) for s in range(1)],
            }
        
        ra={'C_m': {'active':True,
                     'gaussian':{'sigma':0.2*200.0, 'my':200.0}},
             'V_th':{'active':True,
                     'gaussian':{'sigma':1.0, 'my':-50.0, 
                               }},
             'V_m': {'active':True,
                     'uniform': {'min': -70.,  'max':-50. }},
             'I_e':{'active':True,
                     'gaussian':{'sigma':0.0, 
                                  'my':0.0, 
#                                   'cut':True, 
#                                   'cut_at':3
                               }}}
        
        d2={'fan_in_distribution':'constant',
            'n':n,
            'mm':{'params':{'interval':0.5, 
                           'start':0.0, 
                           'stop':numpy.Inf,
                           'to_file':True, 
                           'to_memory':False,
                           'record_from':['V_m']}},
            'nest_params':{'I_e':0.0},
            'rand':ra, 
            'sd':{'params':{'start':0.0, 'stop':numpy.Inf,
                            'to_file':True, 'to_memory':False}}, 
            'sets':[my_slice(s, n, 3) for s in range(3)], 
            }
        
        dic['node']= {inp:d1, net:d2}
        
        # ========================
        # Default conn parameters 
        # ========================
        
        
        k={
           'beta_fan_in':str(0.0),
           'fan_in':str(1.0),
           'rule':'1-1',
           'size':str(size),
           'source':inp,
#            'source_edge_wrap':str(True),
           'source_n':str(n),
           'source_n_sets':str(1),
           'target':net,
#            'target_edge_wrap':str(True),
           'target_n':str(n),
           'target_n_sets':str(3),}
        
        d1={'delay': { 'params':1.0},
            'fan_in':1.0,
            'netw_size':size,
            'save':{'active':False,
                    'overwrite':False, 
                    'path':dr.HOME+('/results/unittest/conn/'
                                 +format_connectionn_save_path(**k))}, 
            'tata_dop':0.0,
            'local_num_threads':4,
            'weight': {'params':10.0},
            }

        dic['conn']={inp+'_'+net:d1}   
        
        return dic  

def dummy_unittest_extend(flag=False):
#     if not flag:
    n1=9
    n2=1
    dic=dummy_unittest_small(n=n1, size=10)
#     else:
#         n2=10
#         dic={}
        
    dic=misc.dict_update(dic, dummy_unittest_small('i2','n2', n2, size=10))
    dd=dic['node']['n2']['rand']
    dd['C_m']['gaussian']['my']=100.0
    dd['C_m']['gaussian']['sigma']=10.0   
 
 
    dic['node']['n2']['rate']=20.0
    dic['node']['i2']['n']=1
    dic['node']['i2']['sets']=[my_slice(0,1,1)]
    dic['node']['i2']['spike_setup']=[{'rates': [10.0], 
                                          't_stop': 3000.0, 
                                          'idx': [0], 
                                          'times': [1.0]}]           
    dic['conn']['i1_n1']['local_num_threads']=1
    
    path=('/afs/nada.kth.se/home/w/u1yxbcfw/results/unittest/'
          +'conn/10_i2-n1s1_n2-n1s3_bfi-0.0_r-1-1')
    
    dic['conn']['i2_n2']['save']['path']=path
    dic['conn']['i2_n2']['local_num_threads']=1

    
    
    d1={'delay': { 'params':1.0},
        'fan_in':10.0,
        'netw_size':10.0,
        'save':{'active':True,
                'path':(dr.HOME+'/results/unittest/'
                        +'conn/10_n1-n9s3_n2-n1s3_bfi-0.0_r-set-set')}, 
        'tata_dop':0.0,
        'local_num_threads':1.,
        'weight': {'params':10.0}, 
       }
    dic['conn'].update({'n1_n2':d1})
    return dic


def dummy_unittest_bcpnn_dopa(flag=True):
    n=1
    dic=dummy_unittest_small(n=n)
    
    
    dic['node']['m1']=deepcopy(dic['node']['n1'])
    dic['node']['m1']['n']=50
    dic['node']['m1']['sets']=[my_slice(s, 50, 1) for s in range(1)]
    
    dic['node']['n2']=deepcopy(dic['node']['n1']) 
    dic['node']['n2']['sets']=[my_slice(s, 1, 1) for s in range(1)]     
    
    dic['node']['n1']['sets']=[my_slice(s, 1, 1) for s in range(1)] 

    dic['node']['i2']=deepcopy(dic['node']['i1'])
    l=[{'rates': [10.0, 0.0, 10.0, 0.0], 
        't_stop': 3000.0, 
        'idx': [0], 
        'times': [0.0, 200.0, 800.0, 1000.0]}]
    dic['node']['i1']['spike_setup']=l

    
    l=[{'rates': [10.0, 0.0, 10.0, 0.0], 
        't_stop': 3000.0, 
        'idx': [0], 
        'times': [200., 400.0, 1000.0, 1200.0]}]
    dic['node']['i2']['spike_setup']=l


    dic['node']['i3']=deepcopy(dic['node']['i1'])
    l=[{'rates': [10.0, 20.0], 
        't_stop': 3000.0, 
        'idx':range(10), 
        'times': [0., 800.0]}]
    dic['node']['i3']['spike_setup']=l   
    dic['node']['i3']['n']=50
    dic['node']['i3']['sets']=[my_slice(s, 50, 1) for s in range(1)]
    

    d1={
        'fan_in':1.0,
        'netw_size':52,
        'save':{'active':True,
                'path':dr.HOME+'/results/unittest/conn/52_i1-n1s1_n1-n1s1_bfi-0.0_r-1-1'}, 
        'local_num_threads':4,
       }

    dic['conn']['i1_n1'].update(d1)    
    
    d1={'delay': { 'params':1.0},
        'fan_in':1.0,
        'netw_size':52,        
        'save':{'active':True,
                'path':dr.HOME+'/results/unittest/conn/52_n1-n1s1_n2-n1s1_bfi-0.0_r-1-1'}, 
        'tata_dop':0.0,
        'local_num_threads':4,
        'weight': {'params':0.0}, 
       }

    dic['conn'].update({'n1_n2':d1})
    

    d1={'delay': { 'params':1.0},
        'fan_in':1.0,
        'netw_size':52,
        'save':{'active':True,
                'path':dr.HOME+'/results/unittest/conn/52_n1-n1s1_n2-n1s1_bfi-0.0_r-1-1'}, 
        'tata_dop':0.0,
        'local_num_threads':4,
        'weight': {'params':0.0}, 
       }
    
    dic['conn'].update({'n1_n2_dopa':d1})
    

    d1={'delay': { 'params':1.0},
        'fan_in':1.0,
        'netw_size':52,
        'save':{'active':True,
                'path':dr.HOME+'/results/unittest/conn/52_i2-n1s1_n2-n1s1_bfi-0.0_r-1-1'}, 
        'tata_dop':0.0,
        'local_num_threads':4,
        'weight': {'params':10.0}, 
       }
    
    dic['conn'].update({'i2_n2':d1})
    
    d={'delay': { 'params':1.0},
        'fan_in':1,
        'netw_size':52,
        'save':{'active':True,
                'path':dr.HOME+'/results/unittest/conn/52_m1-n50s1_v1-n1s1_bfi-0.0_r-all-all'}, 
        'tata_dop':0.0,
        'local_num_threads':4,
        'weight': {'params':1.0}, 
       }    
    dic['conn'].update({'m1_v1':d})
    
 
    d={'delay': { 'params':1.0},
        'fan_in':1.0,
        'netw_size':52,
        'save':{'active':True,
                'path':dr.HOME+'/results/unittest/conn/52_i3-n50s1_m1-n50s1_bfi-0.0_r-1-1'}, 
        'tata_dop':0.0,
        'local_num_threads':4,
        'weight': {'params':10.0}, 
       }    
    dic['conn'].update({'i3_m1':d})
    
    return dic

def dummy_unittest_bcpnn(flag=True):
    n=1
    dic=dummy_unittest_bcpnn_dopa()
    
#         del dic_other['netw']['input']['i3']
    del dic['node']['i3']
    del dic['node']['m1']
    del dic['conn']['i3_m1']
    del dic['conn']['m1_v1']   
    
    for c in dic['conn'].values():
        c['netw_size']=2.
    
    return dic

def dummy_perturbations_lists(flag, conn):
    args=[]
    if flag in ['unittest', 
              'inhibition', 'thalamus', 'slow_wave','beta',
              'bcpnn_h0', 'bcpnn_h1', 'burst_compete']:
        args.append(Perturbation_list({'netw':{'size': 0.5}}, '*',
                                      **{'name':'label1'}))
        d={'conn':{conn:{'fan_in':0.5}}}
        args.append(Perturbation_list(d, '*', **{'name':'label1'}))
        
    return args

def format_connectionn_save_path(**kwargs):
    
    size=kwargs.get('size','X') 
    source=kwargs.get('source','X') 
#     source_edge_wrap=kwargs.get('source_edge_wrap', 'X')
           
    source_n=kwargs.get('source_n','X') 
    source_n_sets=kwargs.get('source_n_sets','X')  
    target=kwargs.get('target','X') 
#     target_edge_wrap=kwargs.get('target_edge_wrap','X')
    target_n=kwargs.get('target_n','X') 
    target_n_sets=kwargs.get('target_n_sets','X')

      
    beta_fan_in=kwargs.get('beta_fan_in','X') 
    rule=kwargs.get('rule','X') 
    fan_in=kwargs.get('fan_in','X')

     
    s=(size+'_'+source+'-n'+source_n+'s'+source_n_sets+'_'
           +target+'-n'+target_n+'s'+target_n_sets+'_bfi-'
           +beta_fan_in+'_r-'+rule+'_fi'+fan_in
#            +'_ew-src'+source_edge_wrap
#            +'_ew-trg'+target_edge_wrap
           )
    

    return s

def iter_conn_connected_to(d, target):
    for key, val in d.items():
        s, t, tp=key.split('_')
        if t==target: 
            yield key, val, s, tp  

def iter_node_connected_to(d, target):
    l=[]
    for key, val in d.items():
        s, t, tp=key.split('_')
        if t!=target:
            continue 
        if s in l:
            continue
        yield key, val, s, tp
        l.append(s) 

           
def print_dic_comparison(d1,d2, flag='keys'):
    ind1=numpy.argsort(misc.dict_reduce(d1, {}).keys())
    ind2=numpy.argsort(misc.dict_reduce(d2, {}).keys())
    a=misc.dict_reduce(d1, {}).values()
    b=misc.dict_reduce(d2, {}).values()
    c=misc.dict_reduce(d1, {}).keys()
    d=misc.dict_reduce(d2, {}).keys()
    for i,j in zip(ind1, ind2):


        if flag=='all':
            print 1, c[i], a[i]
            print 2, d[j], b[j]
        
        if flag=='keys':
            if c[i]!=d[j]:
                print 1, c[i], a[i]
                print 2, d[j], b[j]
        if flag=='values':
            
            if a[i]!=b[j]:
                print 'Depen data', c[i], a[i]
                print 'Dummy data', d[j], b[j]


    
import unittest
class TestModuleFuncions(unittest.TestCase):

    def setUp(self):
        self.m=misc.import_module('toolbox.network.default_params')

    def test_compute(self):
        
        for method in [
                       'calc_fan_in',
                       'calc_n',
                       'calc_sets',
                       'calc_spike_setup',
                       ]:
            call=getattr(self.m,method)
            for a, o in zip(*dummy_args(method)):
#                 pp(call(*a))
                print a, method
                self.assertEqual(call(*a),o)


class TestCall(unittest.TestCase):
    def test(self):
        return 4

    def test2(self):
        return 0.8


    def test3(self):
        return -4

    def test4(self):
        return 4
    
    def setUp(self):
        self.m=misc.import_module('toolbox.network.default_params')
        Call.set_obj(self)

        self.c=Call('test') 
        self.d=Call('test')
        self.e=Call('test2') 
        self.f=Call('test3')        
        self.g=Call('test4')
        self.h=Call('test4', dtype=str)    
#    def test_do(self):
#        self.assertEqual(self.c.do(),4)
        
    def test_add(self):
        self.assertEquals(self.c.do()+4, (self.c+4).do(), 4+4)
        self.assertEquals((self.c+self.d).do(), (self.c+4).do(), 4+4)
                
    def test_div(self):
        self.assertEquals(self.c.do()/4, (self.c/4).do(), 4/4)
        
    def test_mul(self):
        self.assertEquals(self.c.do()*4, (self.c*4).do(), 4*4)
        self.assertEquals(self.c.do()*self.d.do(), (self.c*4).do(), 4*4)     
           
    def test_sub(self):
        self.assertEquals(self.c.do()-4, (self.c-4).do(), 4-4)
        self.assertEquals(self.f.do()-4, (self.f-4).do(), -4-4)
        
    def test_complicated_left(self):
        self.assertEqual(self.c.do()*(1-0.5), (self.c*(1-0.5)).do())  
    
    def test_complicated_right(self):      
        self.assertEqual(self.c.do()*(1-0.5), ((1-0.5)*self.c).do())  
              
    def test_complicated_2(self):      
        
        self.assertEquals(3000.0*(1-self.e.do()), (3000.0*(1-self.e)).do() , 
                         3000.0*0.2) 
    def test_complicated_3(self):   
        self.assertEquals(((self.c-self.e*self.c)/(1-self.e)).do(),
                          (4-0.8*4)/(1-0.8))    

    def test_addStr(self):
        self.assertEqual((self.h+self.h).do(),'44')


class TestCallSubClassesWithPar_base(unittest.TestCase):
    def add(self, *args):
        return self.b.add(*args)
    
    def _get(self, *args):
        return self.b._get(*args)
    
    def _dep(self, *args):
        return self.b._dep(*args)

    def _get_par_constant(self):
        return {'conn':{'i1_n1':{'test':1,
                                 'test2':2}},
                'netw':{'test':1,
                        'test2':DepNest('n1','calc_test2')},
                
                'nest':{'n1':{'test':1,
                              'test2':{'1':DepNetw('calc_test2'),
                                       '2':DepNode('n1','calc_test2')}},
                        'n2':{'test2':{'1':DepNetw('calc_test2'),
                                       '2':DepConn('i1_n1','calc_test2')}}},
                'node':{'n1':{'test':1,
                              'test2':{'1':GetNetw('test'),
                                       '2':GetNode('n1','test'),
                                       '3':GetNest('n1','test')}}},}
   
    def calc_test2(self, name ):
        return int(name==name)
     
    def setUp(self):
        self.m=misc.import_module('toolbox.network.default_params')
        self.c=[]
        self.cd=[]
        Par_base._get_par_constant=self._get_par_constant
        Par_base.calc_test2=self.calc_test2
        self.b=Par_base_dummy()
        self.c.append(GetNetw('test'))
        self.c.append(GetNode('n1','test')) 
        self.c.append(GetNest('n1','test')) 
        
        self.cd.append(DepConn('i1_n1','calc_test2'))
        self.cd.append(DepNetw('calc_test2'))
        self.cd.append(DepNode('n1','calc_test2')) 
        self.cd.append(DepNest('n1','calc_test2'))
        
    def test_do_get(self):
        for e in self.c:
            v=e.do(self)
            self.assertEqual(v,1)  

    def test_do_dep(self):
        for e in self.cd:
            v=e.do(self)
            self.assertEqual(v,1)  
        
    def test_add(self):
        d1, d2, d3={}, {}, {}
        self.add(d1,'nest', 'test2')
        self.add(d2,'node', 'test2')
        self.assertDictEqual(d1, {'nest': {'n1': {'test2': {'1':1,
                                                            '2':1}},
                                           'n2': {'test2': {'1':1,
                                                            '2':1}}}})     
        self.assertDictEqual(d2, {'node': {'n1': {'test2': {'1': 1, 
                                                            '2': 1, 
                                                            '3': 1}}}})    

    def tearDown(self):
        del Par_base._get_par_constant
        del Par_base.calc_test2


class TestPerturbations(unittest.TestCase):
    def setUp(self):
        
        pert=[]
        d1={'netw':{'size': 0.5}}
        pert.append(Perturbation_list(d1, '*',  **{'name':'label1'}))
        d2={'conn':{'i1_n1':{'fan_in':0.5}}}
        pert.append(Perturbation_list(d2, '*', **{'name':'label2'}))
        d3={'simu':{'start_rec':100}}
        pert.append(Perturbation_list(d3, '=', **{'name':'label3'}))
        self.pert_list=pert
        
        self.per=Perturbation_list(d1,'*',d2,'*',d3,'=',
                                   **{'name':'label1_'+'label2_'+'label3'})
        
        
    
    def test_sum(self):

        p=sum(self.pert_list)
        self.assertEqual(p, self.per)
        
        s=("label1-label2-label3\n"+
        "{'conn': {'i1_n1': {'fan_in': '*[0.5]'}},\n"
        " 'netw': {'size': '*[0.5]'},\n"
        " 'simu': {'start_rec': '=[100]'}}")

        self.assertEqual(s, p.info())

   
class TestMixinDummy(object):
    def test_dic_dep(self):
        d1=self.par._get_par_dependable()
        d2=self.par.dic_dep
        d3=self.dummy(flag=self.par.get_unittest())
        self.assertDictEqual(d1, d2)
        
        #print_dic_comparison(d1,d3, 'all')
        print_dic_comparison(d1, d3, 'values')      
        self.assertDictEqual(d1, d3)

class TestMixinPar_base(object):
    def test__get(self):
        first=self.par._get(*['node', self.test_node_model,'type'])
        second=self.par.dic_con['node'][self.test_node_model]['type']
        self.assertEqual(first, second)
      
    def test__get_list_add(self):
        l=['node', self.test_node_model]
        first=self.par._get(l+['type'])
        second=self.par.dic_con['node'][self.test_node_model]['type']
        self.assertEqual(first, second)
      
            
    def test_call_with__get(self):
        c1=self.C(*['_get','node', self.test_node_model,'type'])
        c2=self.par.dic_con['node'][self.test_node_model]['type']
        self.assertEqual(c1.do(), c2)
      
      
             
    def test__get_with_call(self):        
        c1=self.C(*['_get', 'node', self.test_node_model, 'model'])
        c2=self.par.dic_con['node'][self.test_node_model]['model']
        first=self.par._get(*['nest', c1,'C_m'])
        second=self.par.dic_con['nest'][c2]['C_m']
        self.assertEqual(first,second)
                 
    def test__get_functions(self):
             
        keys=dir(self.par)
        for key in keys:
            if key[0:3]=='_get':
                #print key
                args=[]  
                call=getattr(self.par, key)
                call(*args)

    def test_get_funtions(self):
        keys=dir(self.par)
        for key in keys:
            if key[0:3]=='get':
                #print key
                call=getattr(self.par, key)
                call(*[])
            
  
    def test_dic_integrity(self):
        d1=self.par._get_par_constant()
        d2=self.par._get_par_dependable()
          
        s='' 
        for keys, val in misc.dict_iter(d1):
            if not isinstance(val, Call):
                continue
  
            if not misc.dict_haskey(d2, keys):
                s+='dic_dep is missing key {}\n'.format(keys)
                continue
               
            val2=misc.dict_recursive_get(d2, keys)
            if not val.do()==val2:
                s+='dic_con:{} != dic_dep:{} at {}'.format(val,val2, keys)
         
        if s:
            print s
               
        self.assertFalse(s)        

           
    def test_change_val(self):
        sim_stop=self.par['simu']['sim_stop']
        self.par['simu']['sim_stop']=sim_stop*2
        self.par.dic_rep['simu']={'stop_rec':1000.0}
        self.assertFalse(sim_stop==self.par['simu']['sim_stop'])
   
         
    def test_input_par_added(self):
        '''
        Test that par is updated with par_rep for both constant
        and dependable parameters.  
        '''
        dic_rep={'netw':{'size':20000.0}, # constant value    
                 'node':{self.test_node_model:{'n':100*3}}} # dependable value
             
        self.par.update_dic_rep(dic_rep)
        dic_post=self.par.dic
        dic_post_dep=self.par.dic_dep
        dic_post_con=self.par.dic_con
        dic_netw=self.par['netw'] #Test that get item works
        dic_node=self.par['node'] 
             
        l1=([dic_rep['netw']['size']]*3
            +[dic_rep['node'][self.test_node_model]['n']]*2)
        l2=[dic_post['netw']['size'], 
            dic_post_con['netw']['size'], 
            dic_netw['size'], 
            dic_post['node'][self.test_node_model]['n'],
            dic_node[self.test_node_model]['n'],]
        self.assertListEqual(l1, l2)
             
        # Assert 'n' is not present in dependable dict any more
        if self.test_node_model in dic_post_dep['node'].keys():
            d=dic_post_dep['node'][self.test_node_model]
            self.assertFalse('n' in d.keys())
     


    def test_apply_perturbations_only_once(self):   
        

        a=self.par['simu']['sim_time']
#         print a
        p=Perturbation_list({'simu':{'sim_time': 0.5}}, '*',
                                       **{'name':'label1'})
        
        p2=Perturbation_list({'simu':{'sim_time': 0.5}}, '*',
                                       **{'name':'label1'})
        p.append(p2)
#         print p
        self.par.update_perturbations(p)
        b=self.par['simu']['sim_time'] 
        self.par.set_path_nest('')
#         print b
        
        c=self.par['simu']['sim_time']
        
#         print c
        self.assertEqual(0.5**2*a, b)
        self.assertEqual(0.5**2*a, c)
 
    def test_perturbations_double_mult_same(self):   
        

        a=self.par['simu']['sim_time']
#         print a
        p=Perturbation_list({'simu':{'sim_time': 0.5}}, '*',
                                       **{'name':'label1'})
        
        
        self.par.update_perturbations(p)
        b=self.par['simu']['sim_time'] 
        self.par.set_path_nest('')
#         print b
        
        c=self.par['simu']['sim_time']
        
#         print c
        self.assertEqual(0.5*a, b)
        self.assertEqual(0.5*a, c)
        
             
    def test_pertubations(self):
        l=self.pert
            
        l1=[]
        l2=[]
        for pl in l:
            self.par.set_perturbation_list(Perturbation_list())    
            for p in pl:
                l1.append(misc.dict_recursive_get(self.par.dic, p.keys)*p.val[0])
                
            self.par.set_perturbation_list(pl)    
            for p in pl:
                l2.append(misc.dict_recursive_get(self.par.dic, p.keys))
     
        self.assertListEqual(l1, l2)            
    
    
    def test_pertubations_append(self):
        p1, p2=self.pert
        
        p1.append(p2)
        l1=[]
        l2=[]
        for p in p1:
            l1.append(misc.dict_recursive_get(self.par.dic, p.keys)*p.val[0])      
            
        self.par.set_perturbation_list(p1)
        for p in p1:
            l2.append(misc.dict_recursive_get(self.par.dic, p.keys))
            
        self.assertListEqual(l1, l2)         



    
    def test_change_con_effect_dep(self):
        mul=4
        dic=deepcopy(self.par.dic)
        dic_rep={'netw':{'size':dic['netw']['size']*mul}} # constant value  
        v1=self.par.dic['node'][self.test_node_model]['n']
        self.par.update_dic_rep(dic_rep)
        v2=self.par.dic['node'][self.test_node_model]['n']
        self.assertAlmostEqual(v1*mul,v2,delta=mul+1)
   
    def test_change_con_effect_dep_initiation(self):
        mul=8
        d={'netw': {'size': self.par.dic['netw']['size']}}
        self.kwargs['dic_rep']=d 
           
        par= self.the_class(**self.kwargs)
        dic=deepcopy(par.dic)
        dic_rep={'netw':{'size':dic['netw']['size']*mul}} # constant value    
        v1=self.par.dic['node'][self.test_node_model]['n']
        self.par.update_dic_rep(dic_rep)
        v2=self.par.dic['node'][self.test_node_model]['n']
        self.assertAlmostEqual(v1*mul,v2,delta=mul+1)       
   
    def test_nest_params_exist(self):
        d_nest=self.par.dic['nest']  
        d1, d2={}, {}
        
        for dn in d_nest.values():
            if 'type_id' in dn.keys():
                df=nest.GetDefaults(dn['type_id'])
                del dn['type_id']                
                for key, val in dn.items():
                    if key not in df.keys():
                        d1[key]=val
        self.assertDictEqual(d1, d2)


    
    def test_node_conns_keys_integrity(self):
        for key, val in self.par.dic_con['node'].items():
                
            self.assertTrue('type' in val.keys())
            if val['type']=='network':
                self.assertEqual(set(val.keys()), 
                                 set(self.par.keys_node_network))
            if val['type']=='input':
                self.assertEqual(set(val.keys()), 
                                 set(self.par.keys_node_input))
        
        for val in self.par.dic_con['conn'].values():
            self.assertEqual(set(val.keys()), set(self.par.keys_conn))
    
    def test_nodes_network_sum_size(self):
        s=0.0
        for name in self.par._get('node').keys():
#             print name
            tp=self.par.dic['node'][name]['type']
            if tp=='network':
                s+=self.par.dic['node'][name]['n']
        self.assertAlmostEqual(s, self.par.dic['netw']['size'], 7)
    
    def test_conn_keys_integrity(self):
        keys=[]
        for val in self.par.dic['conn'].values():
            keys.extend(val.keys())
        unique_keys=set(keys)
        counts=[]
        for key in unique_keys:
            counts.append(keys.count(key))
                
    def test_model_copy(self):
        keys=(self.par['nest'].keys()+
              [val['model'] for val in self.par['node'].values()]+
              [val['syn'] for val in self.par['conn'].values()],
              )    
        
        for model in keys[0]:
            params=self.par._get_nest_setup(model)
            if 'type_id' in params.keys():
                self.CopyModel( params, model )
                        
    def test_str_method(self):
        s=str(self.par)
        self.assertTrue(isinstance(s, str))
    
    def test_input_params_integrity(self):
        l1=self.par['netw']['input'].keys()
        l2=[]   
        for key, val in self.par['node'].items():
#             print key
            if val['type'] == 'input':
                l2.append(key)
                    
        self.assertListEqual(sorted(l1), sorted(l2))    
      
    
    def test_update_dic_rep(self):
        dic_rep={'node':{self.test_node_model:{'n':100*3}}} 
        self.par.update_dic_rep(dic_rep)
            
        dic_rep={'node':{'dummy':{'n':100*3}}} 
        self.assertRaises(LookupError, self.par.update_dic_rep, dic_rep)
            
           
   
   
class TestSetup_mixin(object):
    def _setUp(self):
        from toolbox import my_nest
        self.CopyModel= my_nest.MyCopyModel
        self.par=self.the_class(**self.kwargs)
        self.C=Call
        self.C.set_obj(self.par)
        
class TestUnittestPar_base(unittest.TestCase):
    def setUp(self):
        self.dummy=dummy_unittest_small
        self.kwargs={}
        self.the_class=Unittest
        self.pert=dummy_perturbations_lists('unittest', 'i1_n1')
        self.test_node_model='n1'
        self._setUp()
        
class TestUnittest(TestUnittestPar_base, TestMixinPar_base,TestMixinDummy, TestSetup_mixin ):
    pass
 
class TestUnittestExtendPar_base(unittest.TestCase):
    def setUp(self):
        self.dummy=dummy_unittest_extend
        self.kwargs={'other':Unittest(),
                     'unittest':True}
        self.the_class=Unittest_extend
        self.pert=dummy_perturbations_lists('unittest', 'i2_n2')
        self.test_node_model='n2'
        self._setUp()
        
class TestUnittestExtend(TestUnittestExtendPar_base, TestMixinPar_base,TestMixinDummy,
                         TestSetup_mixin ):
    pass                    

class TestUnittestBcpnnDopaPar_base(unittest.TestCase):
    def setUp(self):
        self.dummy=dummy_unittest_bcpnn_dopa        
        self.kwargs={'other':Unittest(),
                     'unittest':True}
        self.the_class=Unittest_bcpnn_dopa
        self.pert=dummy_perturbations_lists('unittest', 'i1_n1')
        self.test_node_model='n1'
        self._setUp()
        
class TestUnittestBcpnnDopa(TestUnittestBcpnnDopaPar_base, TestMixinPar_base,TestMixinDummy,
                         TestSetup_mixin ):
    pass  

class TestUnittestStdpPar_base(unittest.TestCase):
    def setUp(self):
        self.dummy=dummy_unittest_bcpnn        
        self.kwargs={'other':Unittest_bcpnn_dopa(**{'other':Unittest()}),
                     'unittest':True}
        self.the_class=Unittest_stdp
        self.pert=dummy_perturbations_lists('unittest', 'i1_n1')
        self.test_node_model='n1'
        self._setUp()
        
class TestUnittestStdp(TestUnittestStdpPar_base, TestMixinPar_base,TestMixinDummy,
                         TestSetup_mixin ):
    pass         
        
class TestUnittestBcpnnPar_base(unittest.TestCase):
    def setUp(self):
        self.dummy=dummy_unittest_bcpnn_dopa
        self.kwargs={'other':Unittest_bcpnn_dopa(**{'other':Unittest()}),
                     'unittest':True}
        self.the_class=Unittest_bcpnn
        self.pert=dummy_perturbations_lists('unittest', 'i1_n1')
        self.test_node_model='n1'
        self._setUp()
        
class TestUnittestBcpnn(TestUnittestBcpnnPar_base, TestMixinPar_base,TestMixinDummy,
                         TestSetup_mixin ):
    pass  

class TestInhibitionPar_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={}
        self.the_class=Inhibition
        self.pert=dummy_perturbations_lists('inhibition', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()
        
class TestInhibition(TestInhibitionPar_base, TestMixinPar_base, TestSetup_mixin ):
    pass  
        
class TestSingleUnitPar_base(unittest.TestCase):     
    def setUp(self):
        self.kwargs={'dic_rep':{'netw':{'single_unit':'FS'}}, 
                     'other':Inhibition()}
        self.the_class=Single_unit
        self.pert=dummy_perturbations_lists('inhibition', 'CFp_FS_ampa')
        self.test_node_model='FS'
        self._setUp()       
        
class TestSingleUnit(TestSingleUnitPar_base, TestMixinPar_base, TestSetup_mixin):
    pass

class TestSlowWavePar_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Slow_wave
        self.pert=dummy_perturbations_lists('slow_wave', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestSlowwave(TestSlowWavePar_base, TestMixinPar_base, TestSetup_mixin):
    pass


class TestSlowWave2Par_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={ 'other':Inhibition(),
                     'unittest':False}
        self.the_class=Slow_wave2
        self.pert=dummy_perturbations_lists('slow_wave', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestSlowwave2(TestSlowWave2Par_base, TestMixinPar_base, TestSetup_mixin):
    pass

class TestSlowWave2_EI_EA_Par_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Slow_wave2_EI_EA
        self.pert=dummy_perturbations_lists('slow_wave', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestSlowwave2_EI_EA(TestSlowWave2_EI_EA_Par_base, TestMixinPar_base, TestSetup_mixin):
    pass

class TestBetaPar_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Beta
        self.pert=dummy_perturbations_lists('beta', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestBeta(TestBetaPar_base, TestMixinPar_base, TestSetup_mixin):
    pass

class TestCompeteWithOscillationsPar_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Compete_with_oscillations
        self.pert=dummy_perturbations_lists('beta', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestTestCompeteWithOscillations(TestCompeteWithOscillationsPar_base, 
           TestMixinPar_base, TestSetup_mixin):
    pass

class TestBeta_EI_EA_Par_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Beta_EI_EA
        self.pert=dummy_perturbations_lists('beta', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestBeta_EI_EA(TestBeta_EI_EA_Par_base, TestMixinPar_base, TestSetup_mixin):
    pass

class TestInhibitionStriatumPar_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Inhibition_striatum
        self.pert=dummy_perturbations_lists('slow_wave', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestInhibitionStriatum(TestInhibitionStriatumPar_base, TestMixinPar_base, TestSetup_mixin):
    pass



class TestMSNClusterCompetePar_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=MSN_cluster_compete
        self.pert=dummy_perturbations_lists('slow_wave', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestMSNClusterCompetePar(TestMSNClusterCompetePar_base, TestMixinPar_base, TestSetup_mixin):
    pass


class TestGo_NoGo_compete_base(unittest.TestCase):
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Go_NoGo_compete
        self.pert=dummy_perturbations_lists('burst_compete', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()

class TestGo_NoGo_compete(TestGo_NoGo_compete_base, TestMixinPar_base, TestSetup_mixin):
    pass

class TestThalamusPar_base(unittest.TestCase):     
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Thalamus
        self.pert=dummy_perturbations_lists('thalamus', 'C1_M1_ampa')
        self.test_node_model='M1'
        self._setUp()      

class TestThalamus(TestThalamusPar_base,  TestMixinPar_base, TestSetup_mixin):
    pass
        
class TestBcpnnH0Par_base(unittest.TestCase):     
    def setUp(self):
        self.kwargs={'other':Inhibition(),
                     'unittest':False}
        self.the_class=Bcpnn_h0
        self.pert=dummy_perturbations_lists('bcpnn_h0', 'RM_M1_ampa')
        self.test_node_model='M1'
        self._setUp()      

class TestBcpnnH0(TestBcpnnH0Par_base,  TestMixinPar_base, TestSetup_mixin):
    pass

class TestBcpnnH1Par_base(unittest.TestCase):     
    def setUp(self):
        self.kwargs={'other':Bcpnn_h0(**{'other':Inhibition()}),
                     'unittest':False}
        self.the_class=Bcpnn_h1
        self.pert=dummy_perturbations_lists('bcpnn_h1', 'CO_M1_ampa')
        self.test_node_model='M1'
        self._setUp()      

class TestBcpnnH1(TestBcpnnH1Par_base,  TestMixinPar_base, TestSetup_mixin):
    pass


if __name__ == '__main__':

    test_fun_par=[
                    'test_apply_perturbations_only_once',
                    'test_call_with__get',
                    'test_change_val', 
                    'test_change_con_effect_dep',
                    'test_change_con_effect_dep_initiation',
                    'test_conn_keys_integrity', 
                    'test_dic_integrity',
                    'test__get', 
                    'test__get_list_add', 
                    'test__get_with_call',
                    'test__get_functions', 
                    'test_get_funtions', 
                    'test_input_par_added', 
                    'test_input_params_integrity', 
                    'test_model_copy', 
                    'test_nest_params_exist',
                    'test_node_conns_keys_integrity',
                    'test_nodes_network_sum_size',
                    'test_pertubations',
                    'test_pertubations_append', 
                    'test_str_method',
                    'test_update_dic_rep',
                     ]
    
    
    d={
        TestModuleFuncions:[
                            'test_compute'
                            ],
           
         TestPerturbations:[
                            'test_sum'
                            ],
        TestCall:[
                  'test_add',
                  'test_addStr',
                  'test_complicated_left',
                  'test_complicated_right',
                  'test_complicated_2',
                  'test_complicated_3',
                  'test_div',
                  'test_mul',
                  'test_sub',
                   ],
        TestCallSubClassesWithPar_base:[
                                        'test_add',                      
                                        'test_do_get',
                                        'test_do_dep',
                                        ],
        TestUnittest:([
                    'test_dic_dep',
                      ]
                    +test_fun_par
                      ),
#                         TestUnittestExtend,
#                         TestUnittestBcpnn,   
#                         TestUnittestBcpnnDopa,   
#                         TestUnittestStdp, 
#                         TestSingleUnit,
#                         TestInhibitionStriatum
#                         TestInhibition,
#                         TestMSNClusterCompetePar
#                         TestThalamus,
#                         TestSlowwave,
#                         TestSlowwave2,
#                         TestSlowwave2_EI_EA:test_fun_par,
                        TestBeta:test_fun_par,
#                         TestTestCompeteWithOscillations:test_fun_par,
#                         TestBeta_EI_EA:test_fun_par,  
#                         TestGo_NoGo_compete:test_fun_par,
#                         TestBcpnnH0,
#                         TestBcpnnH1,

                         }
    
    
    suite = unittest.TestSuite()
    for test_class, val in  d.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=1).run(suite)    
    
#     suites_list = []
#     for test_class in test_classes_to_run:
#         suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
#         suites_list.append(suite)
# 
#     big_suite = unittest.TestSuite(suites_list)
#     unittest.TextTestRunner(verbosity=1).run(big_suite)
#     #suite = unittest.TestLoAader().loadTestsFromTestCase(TestUnittest)
#     #suite = unittest.TestLoader().loadTestsFromTestCase(TestInhibition)
#     #suite = unittest.TestLoader().loadTestsFromTestCase(TestBcpnn_h1)
#     #suite = unittest.TestLoader().loadTestsFromTestCase(TestSingle_unit)
#     #unittest.TextTestRunner(verbosity=2).run(suite)
#     #unittest.main()
#     
# 
# if __name__ == '__main__':
# 
#     d={
#         TestUnittest:[
#                     'test_1_build',
#                     'test_2_connect',
#                     'test_3_run',
#                     'test_4_reset',
#                     'test_5_simulation_loop', 
#                     'test_6_simulation_loop_x3_update_par_rep',
#                     'test_7_simulation_loop_x3_replace_pertubations',
#                     'test_90_IV_curve',   
#                     'test_90_IF_curve',
#                     'test_91_FF_curve', 
#                     'test_92_voltage_trace',
#                     'test_93_optimization',
#                       ],
#         TestUnittestExtend:[
#                            ],
#        TestUnittestBcpnnDopa:[
#                               ],
#        TestUnittestStdp:[
#                          ],
#        TestUnittestBcpnn:[],
#        TestSingle_unit:[],
#        TestNetwork_dic:[],
# 
# 
#        }
#     test_classes_to_run=d
#     suite = unittest.TestSuite()
#     for test_class, val in  test_classes_to_run.items():
#         for test in val:
#             suite.addTest(test_class(test))
# 
#     unittest.TextTestRunner(verbosity=2).run(suite)
#         