'''
Created on Feb 17, 2015

@author: mikael
'''
import cPickle
import numpy
import psycopg2
import subprocess
import time
import unittest
import warnings

from toolbox import data_to_disk
from toolbox.network import default_params
from toolbox.misc import my_slice
from toolbox import misc
from toolbox import my_socket


import pprint
from psycopg2.errorcodes import DATATYPE_MISMATCH
pp=pprint.pprint

#ssh port forwarding
#-p 23 bcause it is ssh port 23 that we should connect ot
PORT_FORWARD_CMD=['ssh',
                  '-p',
                  '23',
                  '-L',
                  '1234:localhost:5432',
                  '-N',
                  'mikael@mikaellindahl.se']

'''
ssh -p 23 -L 1234:localhost:5432 -N mikael@mikaellindahl.se
psql -U mikael -d test -h localhost -p 1234
'''

'''
From http://stackoverflow.com/questions/13668735/
how-to-unpickle-binary-data-stored-in-postgresql-by-psycopg2-module-in-python
'''
def cast_pickle(data, cur):
    if data is None: return None
    return cPickle.loads(str(psycopg2.BINARY(data, cur)))

psycopg2.extensions.register_type(
    psycopg2.extensions.new_type(
        psycopg2.BINARY.values, 'BINARY_PICKLE', cast_pickle))

class Database(object):
    def __init__(self):
        self.computer = my_socket.determine_computer()
        self.conn=None
        self.cur=None
        self.spf=None #po4t forwarding object
    
    def __del__(self):
        self.close()
    
    def connect(self):
        if self.computer=='supermicro':
            self.conn, _=connect_from_inside_network()
            self.cur=self.conn.cursor()
            
        if self.computer=='mikaellaptop':
            self.conn, self.spf=connect_from_outside_network()
            self.cur=self.conn.cursor()

        if self.computer=='milner':
            self.conn, self.spf=connect_from_outside_network()
            self.cur=self.conn.cursor()

    def check_if_table_exist(self, table_name):
        #An alternative using EXISTS is better in that it doesn't 
        #require that all rows be retrieved, but merely that at 
        #least one such row exists:

        s="SELECT EXISTS (SELECT * FROM information_schema.tables WHERE table_name={})"
        s=s.format("'"+table_name+"'")
        self.cur.execute(s)
        return self.cur.fetchone()[0]

    
    def close(self):
        if self.cur:self.cur.close()
        if self.conn:self.conn.close()
        self.kill_spf()
        
    def commit(self):
        '''Commite changes to table'''
        self.conn.commit()

    def create_table(self, table_name, keys, datatypes, **kwargs):
        
        if self.check_if_table_exist(table_name):
            warnings.warn('Table already exist')
            return 
            
        s='CREATE TABLE {} (id serial PRIMARY KEY'.format(table_name)
        s+=', added timestamp DEFAULT CURRENT_TIMESTAMP'

    
        for key, dt in zip(keys, datatypes):
            s+=', {} {}'.format(key, dt)
        
        s+=');'
        
        self.execute(s)

    def insert(self, table_name, keys, values):
        s0= "INSERT INTO {} (".format(table_name)
        s1=  ', '.join([k.lower() for k in keys])
        s2=') VALUES ('
        
        # Strings need to have '' around them
        values=["'"+str(e)+"'"  if type(e)==str  else e for e in values  ]
      
        #arrays can be  inserted as ARRAY[1,1,2] or {1,1,2}
        s3= ', '.join([str(e) if type(e)!=list  else 'ARRAY'+str(e) for e in values])
        s3=s3[:]+')'
        
        s=s0+s1+s2+s3

        self.execute(s)

    def create_table_from_dic(self, d, table_name):
        s=create_table_string(table_name, dummy_data_small())
        self.execute(s)
        self.commit()

    def execute(self, *args):
        '''Execute SQL command'''
        
        self.cur.execute(*args)
        
    def fetchall(self):
        '''To fetch results from query'''
        return self.cur.fetchall()

    def fetchone(self):
        '''To fetch results from query'''
        return self.cur.fetchone()
    
    def get_column_names(self, table):
        s=query_table_names(table)
        self.execute(s) 
        l=self.fetchall()
        l,_= zip(*l)
        
        exclude=['tableoid', 'cmax', 'xmax', 'cmin', 'xmin', 'ctid']
        l=[e for e in l if not e in exclude]
        return l

    def kill_spf(self):
        if self.spf:
            self.spf.kill()

class SSH_port_forwarding():
    '''
    setup a tunnel ssh port forwarding from the client computer
    to the computer running postgresql. Then connect ot postgresql 
    true the port.   
    '''
    
    def __init__(self, **kwargs):
        default='ssh -p 23 -L 1234:localhost:5432 -N mikael@mikaellindahl.se'
        cmd=kwargs.get('cmd',default)
  
        self.cmd=cmd.split(' ')
        self.pid=None
        
    def do(self):
        
        p=subprocess.Popen(self.cmd)
        time.sleep(1)
        self.pid=p.pid
        
    def kill(self):
        if self.pid:
            subprocess.Popen(['kill', str(self.pid)])
        
    def __del__(self):
        #Need to import other complaine subprocess is empty
        import subprocess
        if self.pid:
            subprocess.Popen(['kill', str(self.pid)])


def alter_table_string(table, column, dataype):
    
    s='ALTER TABLE {} ADD {} {}'
    s=s.format(table, column, dataype)
    return s

def connect(**kwargs):
    return psycopg2.connect(**kwargs )
    

def connect_from_inside_network():
    '''
    inside network connecti directly
    '''
    filename='/home/mikael/.postgresql_los'
    s=data_to_disk.text_load(filename)
    s=s[:-1]
    
    kwargs={'database':'test',
            'user':'mikael',
            'password':s,
            'host':'192.168.1.14',
            'sslmode':'require'}
    return connect(**kwargs), None

def connect_from_outside_network():
    filename='/home/mikael/.postgresql_los'
    s=data_to_disk.text_load(filename)
    s=s[:-1]

    
    spf=SSH_port_forwarding()
    spf.do()

    kwargs={'database':'test',
            'user':'mikael',
            'password':s,
            'host':'localhost',
            'port':'1234',
            'sslmode':'require'}
    return connect(**kwargs), spf

def keys_and_dataypes_from_dic(d):
    d_red={}
    misc.dict_reduce(d, d_red, deliminator='_')
    
    keys, values=zip(*d_red.items())
    
    datatypes=[]
    extract_datatype(values, datatypes) 
    
    return keys, datatypes

def create_table_string(name,d):
    
    keys, datatypes=keys_and_dataypes_from_dic(d)
    
    s='CREATE TABLE {} (id serial PRIMARY KEY'.format(name)
    s+=', {} {}'.format('timestamp', 'timestamp')
    s+=', {} {}'.format('computer', 'varchar')
    
    for key, dt in zip(keys, datatypes):
        s+=', {} {}'.format(key, dt)
    
    s+=');'
            
    return s
#     cur.execute(s)


def drop_table_string(name):
    return 'DROP TABLE {};'.format(name)
    
    
def _extract_datatype(val):

    if type(val) == int:
        out= 'integer'
    elif type(val) == float:
        out= 'float'
    elif type(val) in str:
        out= 'varchar'
    elif type(val) == bool:
        out= 'boolean'
    else:
        out= 'varchar'
    return out

    
def extract_datatypes(values):
    datatypes=[]
    for val in values:       
        dt=_extract_datatype(val, 0)
        datatypes.append(dt)
    return datatypes


def is_port_used():
    '''
    "netstat -ln | grep ':1234 ' | grep 'LISTEN'"
    
    | means pip and can be accomplished by feeding stdout 
    in to another subprocess stdin 
    '''
    s0=['netstat', '-ln']
    s1=['grep', ":1234 "]
    s2=['grep', "LISTEN"]
    
#         p = subprocess.Popen(args, **kwargs)

    kwargs={
            'stdout':subprocess.PIPE}
    p0 = subprocess.Popen(s0, **kwargs)

    kwargs={'stdin':p0.stdout,
            'stdout':subprocess.PIPE}
    p1 = subprocess.Popen(s1, **kwargs)
    p0.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.

    kwargs={'stdin':p1.stdout,
            'stdout':subprocess.PIPE}
    p2 = subprocess.Popen(s2, **kwargs)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.

    out, err = p2.communicate()
    if out!='':
        return True
    else: 
        return False


def extract_data_from_dic():
 
    d_red={}
    misc.dict_reduce(d, d_red, deliminator='_')
    
    # Sort on keys
    items=sorted(d_red.items(), key=lambda x:x[0])   
    keys, values=zip(*items)    

    # Convert tpyes that are not
    values=[v if type(v) in [int, str, list, bool, float, numpy.ndarray] else str(v) for v in values]
 
    return keys, values



def insert_table_string(name, d, columns, if_not_exist_append=False):
#     columns=[c.lower() for c in columns]
    datatypes=[]
#     l=[pair for pair in items if pair[0].lower() not in columns]
    
    extract_datatype(values_new, datatypes)
 
    for key, dt in zip(keys_new, datatypes):
        s_list.append(alter_table_string(name, key, dt))
    
    keys, values=zip(*items)
    
    
    s0= "INSERT INTO {} (".format(name)
    s1=  ', '.join([k.lower() for k in keys])
    s2=') VALUES ('
    
    # Strings need to have '' around them
    values=["'"+str(e)+"'"  if type(e)==str  else e for e in values  ]
  
    #arrays can be  inserted as ARRAY[1,1,2] or {1,1,2}
    s3= ', '.join([str(e) if type(e)!=list  else   'ARRAY'+str(e) for e in values  ])
    s3=s3[1:]+')'
    s=s0+s1+s2+s3

    s_list.append(s)
    return s_list
        
        
#         cur.execute("INSERT INTO {} (num, data) VALUES (%s, %s)",
# ...      (100, "abc'def")) 
# 
#     
    

#You can create a customized typecaster to automatically convert pickled values to Python:




    
def query_table_names(table):
#     s=("SELECT *FROM information_schema.columns"+
# #      "WHERE table_schema = '{schema}'",
#      " WHERE table_name   = '{}';")
    s=("SELECT attname , typname FROM pg_attribute "
    +", pg_type WHERE typrelid=attrelid AND typname = {};")
    s=s.format("'"+table+"'")
    return s  
    

def dummy_data_small(beta=0.0):
    d={'conn': {'C1_M1_ampa': {'beta_fan_in': beta,
                                 'class_surface': 'Conn',
                                 'delay': {'params': 12.0, 'type': 'constant'},
                                 'lesion': False,
                                 'mask': [-0.5, 0.5],
                                 'netw_size': 10000.0,},},

        'netw': {'FF_curve': {'input': 'C1', 'output': 'M1'},
                  'GA_prop': 0.2,
                  'GI_prop': 0.8,
                  },
        
        'node':{ 'EA': {'class_population': 'MyPoissonInput'},  
                 'GA': {'I_vitro': 5.0,},},
        'nest':{ 'GA_M2_gaba': {'delay': 1.7},
                'GI': {'AMPA_1_E_rev': 0.0,
                       'type_id': 'my_aeif_cond_exp'},},
        'simu': {'do_reset': False,
                  'local_num_threads': 1,
                  'stop_rec': 2000.0}}
    return d

def dummy_data_large():
    d={'conn': {'C1_M1_ampa': {'beta_fan_in': 0.0,
                                 'class_surface': 'Conn',
                                 'delay': {'params': 12.0, 'type': 'constant'},
                                 'fan_in': 1.0,
                                 'fan_in0': 1,
                                 'lesion': False,
                                 'local_num_threads': 1,
                                 'mask': [-0.5, 0.5],
                                 'netw_size': 10000.0,
                                 'rule': '1-1',
                                 'save': {'active': True,
                                          'overwrite': False,
                                          'path': '/home/mikael/results/papers/inhibition/network/supermicro/conn/10000.0_C1-n4718s1_M1-n4718s1_bfi-0.0_r-1-1_fi1.0'},
                                 'source': 'C1',
                                 'syn': 'C1_M1_ampa',
                                 'target': 'M1',
                                 'tata_dop': 0.0,
                                 'weight': {'params': 0.5, 'type': 'constant'}},},
        'netw': {'FF_curve': {'input': 'C1', 'output': 'M1'},
                  'GA_prop': 0.2,
                  'GI_prop': 0.8,
                  'GP_fan_in': 30,
                  'GP_fan_in_prop_GA': 0.058823529411764705,
                  'GP_rate': 30.0,
                  'V_th_sigma': 1.0,
                  'attr_popu': ['class_population',
                                'model',
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
                                'type'],
                  'attr_surf': ['class_surface',
                                'extent',
                                'edge_wrap',
                                'fan_in_distribution',
                                'lesion',
                                'model',
                                'n',
                                'n_sets',
                                'sets',
                                'type'],
                  'fan_in_distribution': 'constant',
                  'input': {'C1': {'params': {}, 'type': 'constant'},
                            'C2': {'params': {}, 'type': 'constant'},
                            'CF': {'params': {}, 'type': 'constant'},
                            'CS': {'params': {}, 'type': 'constant'},
                            'EA': {'params': {}, 'type': 'constant'},
                            'EI': {'params': {}, 'type': 'constant'},
                            'ES': {'params': {}, 'type': 'constant'}},
                  'n_actions': 1,
                  'n_nuclei': {'FS': 55820.0,
                               'GA': 9192.0,
                               'GI': 36768.0,
                               'M1': 1186175.0,
                               'M2': 1186175.0,
                               'SN': 26320.0,
                               'ST': 13560.0},
                  'optimization': {'f': ['M1'], 'x': ['node.C1.rate'], 'x0': [700.0]},
                  'rand_nodes': {'C_m': True, 'V_m': True, 'V_th': True},
                  'size': 10000.0,
                  'sub_sampling': {'M1': 1.0, 'M2': 1.0},
                  'tata_dop': 0.8,
                  'tata_dop0': 0.8},
        
        'node':{ 'EA': {'class_population': 'MyPoissonInput',
                 'class_surface': 'Surface',
                 'edge_wrap': True,
                 'extent': [-0.5, 0.5],
                 'fan_in_distribution': 'constant',
                 'lesion': False,
                 'model': 'EA',
                 'n': 37,
                 'n_sets': 1,
                 'nest_params': {},
                 'rate': 200.0,
                 'sets': [my_slice(0,37,1)],
                 'spike_setup': [{'idx':range(37),
                                  'rates': [200.0],
                                  't_stop': 2000.0,
                                  'times': [1.0]}],
                 'target': 'GA',
                 'type': 'input'},
                 
              'GA': {'I_vitro': 5.0,
                     'I_vivo': -3.6,
                     'class_population': 'MyNetworkNode',
                     'class_surface': 'Surface',
                     'edge_wrap': True,
                     'extent': [-0.5, 0.5],
                     'fan_in_distribution': 'constant',
                     'lesion': False,
                     'mm': {'active': False,
                            'params': {'interval': 0.5,
                                       'record_from': ['V_m'],
                                       'start': 1000.0,
                                       'stop': 2000.0,
                                       'to_file': True,
                                       'to_memory': False}},
                     'model': 'GA',
                     'n': 37,
                     'n_sets': 1,
                     'nest_params': {'I_e': -3.6},
                     'rand': {'C_m': {'active': True,
                                      'gaussian': {'my': 40.0, 'sigma': 4.0}},
                              'V_m': {'active': True,
                                      'uniform': {'max': -54.7, 'min': -74.7}},
                              'V_th': {'active': True,
                                       'gaussian': {'cut': True,
                                                    'cut_at': 3.0,
                                                    'my': -54.7,
                                                    'sigma': 1.0}}},
                     'rate': 5.0,
                     'rate_in_vitro': 4.0,
                     'sd': {'active': True,
                            'params': {'start': 1000.0,
                                       'stop': 2000.0,
                                       'to_file': False,
                                       'to_memory': True}},
                     'sets': [my_slice(0,37,1)],
                     'type': 'network'},},
        'nest':{ 'GA_M2_gaba': {'delay': 1.7,
                         'receptor_type': 5,
                         'type_id': 'static_synapse',
                         'weight': 0.4},
          'GI': {'AMPA_1_E_rev': 0.0,
                 'AMPA_1_Tau_decay': 12.0,
                 'AMPA_2_E_rev': 0.0,
                 'AMPA_2_Tau_decay': 5.0,
                 'C_m': 40.0,
                 'Delta_T': 1.7,
                 'E_L': -55.1,
                 'GABAA_1_E_rev': -65.0,
                 'GABAA_1_Tau_decay': 6.0,
                 'GABAA_2_E_rev': -65.0,
                 'GABAA_2_Tau_decay': 5.0,
                 'I_e': 0.0,
                 'NMDA_1_E_rev': 0.0,
                 'NMDA_1_Sact': 16.0,
                 'NMDA_1_Tau_decay': 100.0,
                 'NMDA_1_Vact': -20.0,
                 'V_a': -55.1,
                 'V_peak': 15.0,
                 'V_reset': -60.0,
                 'V_th': -54.7,
                 'a_1': 2.5,
                 'a_2': 2.5,
                 'b': 70.0,
                 'beta_E_L': 0.181,
                 'beta_I_AMPA_1': -0.45,
                 'beta_I_GABAA_2': -0.83,
                 'beta_V_a': 0.181,
                 'g_L': 1.0,
                 'tata_dop': 0.0,
                 'tau_w': 20.0,
                 'type_id': 'my_aeif_cond_exp'},},
        'simu': {'do_reset': False,
                  'local_num_threads': 1,
                  'mm_params': {'interval': 0.5,
                                'record_from': ['V_m'],
                                'to_file': True,
                                'to_memory': False},
                  'path_class': '/home/mikael/results/papers/inhibition/network/supermicro/Inhibition/',
                  'path_conn': '/home/mikael/results/papers/inhibition/network/supermicro/conn/',
                  'path_data': '/home/mikael/results/papers/inhibition/network/supermicro/',
                  'path_figure': '/home/mikael/results/papers/inhibition/network/supermicro/fig/',
                  'path_nest': '/home/mikael/results/papers/inhibition/network/supermicro/Inhibition/nest/',
                  'print_time': True,
                  'save_conn': {'active': True, 'overwrite': False},
                  'sd_params': {'to_file': False, 'to_memory': True},
                  'sim_stop': 2000.0,
                  'sim_time': 2000.0,
                  'start_rec': 1000.0,
                  'stop_rec': 2000.0}}

    return d

class Test_database(unittest.TestCase):
    def setUp(self):
        self.db=Database()
        self.db.connect()
           
    def tearDown(self):
        self.db.close()       
        
    def test_check_if_table_exist(self):
                 
        b=self.db.check_if_table_exist('table_to_check_for_existans')
        self.assertTrue(b)
        b=self.db.check_if_table_exist('none')
        self.assertFalse(b)
        
    def test_create_table(self):
        table_name='unittest_create_table'
        keys=['a','b','c']
        datatypes=['integer', 'varchar', 'float[]']
        self.db.create_table(table_name, keys, datatypes)
        
        s="SELECT EXISTS (SELECT * FROM information_schema.tables WHERE table_name={})"
        s=s.format("'"+table_name+"'")
        self.db.cur.execute(s)
        self.assertTrue(self.db.cur.fetchone()[0])
        
        s='DROP TABLE '+table_name
        self.db.cur.execute(s)

        s="SELECT EXISTS (SELECT * FROM information_schema.tables WHERE table_name={})"
        s=s.format("'"+table_name+"'")
        self.db.cur.execute(s)
        self.assertFalse(self.db.cur.fetchone()[0])
        
    def test_insert(self):
        # CREATE TABLE
        
        # "CREATE TABLE unittets_insert_table (id serial PRIMARY KEY, a integer, b varchar, c float, d boolean, e integer[], f varchar[], g float[])" 

        
        table_name='unittest_insert_table'
#         keys=['a', 'b','c','d','e', 'f', 'g']
#         values=['integer', 'varchar', 'float','boolean',
#                 'integer[]','varchar[]', 'float[]']
#         values=['integer', 'varchar', 'float','boolean',
#                 'integer[]','varchar[]', 'float[]']                
#         self.db.insert(table_name, keys, values)

        table_name='unittest_insert_table'
        keys=['a', 'b']
        values=[1, 'hej']
                
        self.db.insert(table_name, keys, values)
    
    def test_cPickle_binary(self):
        table_name='unittest_cPickle_binary'
        
        keys=['binary_pickle']
        datatypes=['bytea']
        self.db.create_table(table_name, keys, datatypes)
        
        obj = {'a': 10}
        data = cPickle.dumps(obj, -1)
        self.db.insert(table_name, ['binary_pickle'], [psycopg2.Binary(data)])
        # import psycopg2
        
        
        
#         cnn = psycopg2.connect('')
#         cur = cnn.cursor()
        self.db.execute("select %s::bytea", [psycopg2.Binary(data)])
        print self.db.fetchone()
        # ({'a': 10},)

        

        
class ModuleFunctions(unittest.TestCase):     
    def setUp(self):
        self.db=Database()
        
    def tearDown(self):
        self.db.kill_spf()

    def test_create_table_string(self):
        s=create_table_string('unittest_small',dummy_data_small())
   
    def test_create_table(self): 


        with self.db:                    
            s=drop_table_string('unittest_small')
            
            print self.db.execute(s) 
        
        
            s=create_table_string('unittest_small', dummy_data_small())
            self.db.execute(s)
            self.db.commit()
        
#         cur.execute("INSERT INTO test2 (num, data) VALUES (%s, %s)",(100, "abc'def"))


        

    def test_query_table_names(self):
        with self.db:                    
            s=query_table_names('unittest_small')
            
            self.db.execute(s) 
            l=self.db.fetchall()
            l0,_= zip(*l)
        l1=('tableoid', 'cmax', 'xmax', 'cmin', 'xmin', 'ctid', 'id', 
            'timestamp', 'computer', 'nest_gi_ampa_1_e_rev', 
            'simu_stop_rec', 'netw_ga_prop', 'simu_local_num_threads', 
            'conn_c1_m1_ampa_delay_type', 'nest_gi_type_id', 
            'netw_gi_prop', 'conn_c1_m1_ampa_netw_size', 
            'conn_c1_m1_ampa_mask', 'conn_c1_m1_ampa_delay_params', 
            'conn_c1_m1_ampa_beta_fan_in', 'simu_do_reset', 
            'netw_ff_curve_input', 'nest_ga_m2_gaba_delay', 
            'netw_ff_curve_output', 'node_ga_i_vitro', 
            'conn_c1_m1_ampa_class_surface', 'node_ea_class_population',
            'conn_c1_m1_ampa_lesion')
        self.assertListEqual(sorted(l0), sorted(l1))
            
        print 'hej'

    def test_drop_table(self):
        
        s0="SELECT relname FROM pg_class WHERE relname = 'unittest_small';"
    
        with self.db:                    
            s=drop_table_string('unittest_small')
            
            print self.db.execute(s0) 
            self.db.commit()
            
            self.db.execute(s)
            
            self.db.commit()
            print self.db.execute(s0) 
            
            s=drop_table_string('unittest_small')

    def test_insert_table_string(self):
        table='unittest_small'
        with self.db:
            columns=self.db.get_column_names(table)
            
        d=dummy_data_small()
        l0=insert_table_string(table,d,columns)
        l1=['INSERT INTO unittest_small (conn_c1_m1_ampa_beta_fan_in,'
            +' conn_c1_m1_ampa_class_surface, conn_c1_m1_ampa_delay_params, '
            +'conn_c1_m1_ampa_delay_type, conn_c1_m1_ampa_lesion, '
            +'conn_c1_m1_ampa_mask, conn_c1_m1_ampa_netw_size, '
            +'nest_ga_m2_gaba_delay, nest_gi_ampa_1_e_rev, nest_gi_type_id, '
            +'netw_ff_curve_input, netw_ff_curve_output, netw_ga_prop, '
            +'netw_gi_prop, node_ea_class_population, node_ga_i_vitro, '
            +'simu_do_reset, simu_local_num_threads, simu_stop_rec) VALUES '
            +'(.0, Conn, 12.0, constant, False, ARRAY[-0.5, 0.5], 10000.0, 1.7, 0.0,'
            +' my_aeif_cond_exp, C1, M1, 0.2, 0.8, MyPoissonInput, 5.0, False, '
            +'1, 2000.0)']
        
        self.assertListEqual(l0, l1)
        
        l1=['INSERT INTO unittest_small (conn_c1_m1_ampa_beta_fan_in,'
            +' conn_c1_m1_ampa_class_surface, conn_c1_m1_ampa_delay_params, '
            +'conn_c1_m1_ampa_delay_type, conn_c1_m1_ampa_lesion, '
            +'conn_c1_m1_ampa_mask, conn_c1_m1_ampa_netw_size, '
            +'dummy_dummy1, dummy_dummy2, '
            +'nest_ga_m2_gaba_delay, nest_gi_ampa_1_e_rev, nest_gi_type_id, '
            +'netw_ff_curve_input, netw_ff_curve_output, netw_ga_prop, '
            +'netw_gi_prop, node_ea_class_population, node_ga_i_vitro, '
            +'simu_do_reset, simu_local_num_threads, simu_stop_rec) VALUES '
            +'(.0, Conn, 12.0, constant, False, ARRAY[-0.5, 0.5], 10000.0, '
            +'1.0, hej, '
            +'1.7, 0.0,'
            +' my_aeif_cond_exp, C1, M1, 0.2, 0.8, MyPoissonInput, 5.0, False, '
            +'1, 2000.0)']
        
        d.update({'dummy':{'dummy1':1.,
                           'dummy2':'hej'}})
        l0=insert_table_string(table,d,columns, if_not_exist_append=True)
        l1=(['ALTER TABLE unittest_small ADD dummy_dummy1 float',
             'ALTER TABLE unittest_small ADD dummy_dummy2 varchar']
            +l1)

        self.assertListEqual(l0, l1)
        
    def test_insert(self):
        table='unittest_small'
        with self.db:
            columns=self.db.get_column_names(table)
            
        d=dummy_data_small()
        l0=insert_table_string(table,d,columns)
        
        with self.db:
            self.db.execute(l0[-1])
            self.db.commit()

    def test_create_table_large(self):
        
        with self.db:                    
            s=create_table_string('unittest_small', dummy_data_small())
            self.db.execute(s)
            self.db.commit()


    def test_is_port_used(self):
        
        self.assertFalse(is_port_used())
        spf=SSH_port_forwarding()
        spf.do()
#         s='ssh -p 23 -L 1234:localhost:5432 -N mikael@mikaellindahl.se'
#         p=subprocess.Popen(s.split(' '))
        print spf.pid
        self.assertTrue(is_port_used())
        
        
    def test_extract_datatypes(self):
        values=[1,1.2,'a',True,[1, 2], [1., 2.], ['a','b'], [True, False], numpy.array([1,2]),
                [[1,2],[1,2]], [[1,2],[1]], numpy.array([[1,2],[1,2]])]
        print extract_datatypes(values)
if __name__ == '__main__':
    d={
       Test_database:[
#                     'test_create',
#                     'test__enter__',
#                       'test_check_if_table_exist',  
#                       'test_create_table',
#                         'test_insert',
                    'test_cPickle_binary',
                      ],
       
        ModuleFunctions:[
#                         'test_create_table_string',
#                         'test_drop_table',
#                         'test_create_table',
#                         'test_query_table_names',
#                         'test_insert_table_string',
#                         'test_insert',
#                         'test_is_port_used',
                          'test_extract_datatypes',
                         
                         
                    ],

       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)

# conn=psycopg2.connect("dname=test")