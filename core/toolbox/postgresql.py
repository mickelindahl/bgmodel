'''
Created on Feb 17, 2015

@author: mikael
'''
import psycopg2
import unittest

from toolbox import data_to_disk
from toolbox.network import default_params
from toolbox.misc import my_slice
from toolbox import misc
from toolbox import my_socket


import pprint
pp=pprint.pprint


class Database(object):
    def __init__(self):
        self.computer = my_socket.determine_computer()
        self.conn=None
        self.cur=None
        
    
    def __enter__(self):
        if self.computer=='supermicro':
            self.conn=connect_from_supermicro()
            self.cur=self.conn.cursor()
        if self.computer=='milner':
            self.conn=connect_from_milner()
            self.cur=self.conn.cursor()
        return self
      
    def __exit__(self, type, value, traceback):
        self.cur.close()
        self.conn.close()
        
    def execute(self, s):
        self.cur.execute(s)
        
    def commit(self):
        self.conn.commit()
        
    def fetchall(self):
        return self.cur.fetchall()
    
    def get_column_names(self, table):
        s=query_table_names(table)
        self.execute(s) 
        l=self.fetchall()
        l,_= zip(*l)
        
        exclude=['tableoid', 'cmax', 'xmax', 'cmin', 'xmin', 'ctid']
        l=[e for e in l if not e in exclude]
        return l
        
        
def connect(**kwargs):
    return psycopg2.connect(**kwargs )
    

def connect_from_supermicro():
    filename='/home/mikael/.postgresql_los'
    s=data_to_disk.text_load(filename)
    s=s[:-1]
    
    kwargs={'database':'test',
            'user':'mikael',
            'password':s,
            'host':'192.168.1.14',
            'sslmode':'require'}
    return connect(**kwargs)

def connect_from_milner():
    filename='/home/mikael/.postgresql_los'
    s=data_to_disk.text_load(filename)
    s=s[:-1]
    
    kwargs={'database':'test',
            'user':'mikael',
            'password':s,
            'host':'192.168.1.14',
            'sslmode':'require'}
    return connect(**kwargs)

def create_insert_string(d):
    d_red={}
    misc.dict_reduce(d, d_red)
    

def extract_datatype(values, datatypes):
    for i, val in enumerate(values):
        if type(val) == int:
            datatypes.append('integer')
        elif type(val) == float:
            datatypes.append('float')
        elif type(val) == str:
            datatypes.append('varchar')
        elif type(val) == bool:
            datatypes.append('boolean')
        elif type(val) == list:
            if type(val[0]) == int:
                datatypes.append('integer[]')
            elif type(val[0]) == float:
                datatypes.append('float[]')
            elif type(val[0]) == str:
                datatypes.append('varchar[]')
        else:
            datatypes.append('varchar')
            values[i] = 'None'

def create_table_string(name,d):
    d_red={}
    misc.dict_reduce(d, d_red, deliminator='_')
    
    keys, values=zip(*d_red.items())
    
    datatypes=[]
    extract_datatype(values, datatypes) 
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
    
def alter_table_string(table, column, dataype):
    
    s='ALTER TABLE {} ADD {} {}'
    s=s.format(table, column, dataype)
    return s
    
def insert_table_string(name, d, columns, if_not_exist_append=False):
#     columns=[c.lower() for c in columns]
    
    s_list=[]
    d_red={}
    misc.dict_reduce(d, d_red, deliminator='_')
    items=sorted(d_red.items(), key=lambda x:x[0])

    l=[pair for pair in items if pair[0].lower() not in columns]
   
    if l and if_not_exist_append:
        keys_new, values_new=zip(*l)    
        datatypes=[]
        
        extract_datatype(values_new, datatypes)
        for key, dt in zip(keys_new, datatypes):
            s_list.append(alter_table_string(name, key, dt))
    
    keys, values=zip(*items)
    s0= "INSERT INTO {} (".format(name)
    s1=  ', '.join([k.lower() for k in keys])
    s2=') VALUES ('
    values=["'"+str(e)+"'"  if type(e)==str  else e for e in values  ]
  
    s3= ', '.join([str(e) if type(e)!=list  else   'ARRAY'+str(e) for e in values  ])
    s3=s3[1:]+')'
    s=s0+s1+s2+s3

    s_list.append(s)
    return s_list
        
        
        
        
    

#         cur.execute("INSERT INTO {} (num, data) VALUES (%s, %s)",
# ...      (100, "abc'def")) 
# 
#     
    
    
    
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

class ModuleFunctions(unittest.TestCase):     
    def setUp(self):
        
        pass

    def test_connect_supermicro(self):
        filename='/home/mikael/.postgresql_los'
        s=data_to_disk.text_load(filename)
        s=s[:-1]
        conn=connect(database='test',
                     user='mikael',
                     password=s,
                     host='192.168.1.14',
                     sslmode='require'
#                      pot='5432'
                     )
        self.assertTrue(conn)
        conn.close()

    def test_create_table_string(self):
        s=create_table_string('unittest_small',dummy_data_small())
   
    def test_create_table(self): 


        with Database() as db:                    
            s=create_table_string('unittest_small', dummy_data_small())
            db.execute(s)
            db.commit()

#         cur.execute("INSERT INTO test2 (num, data) VALUES (%s, %s)",(100, "abc'def"))

    def test_query_table_names(self):
        with Database() as db:                    
            s=query_table_names('unittest_small')
            
            db.execute(s) 
            l=db.fetchall()
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
    
        with Database() as db:                    
            s=drop_table_string('unittest_small')
            
            print db.execute(s0) 
            db.commit()
            
            db.execute(s)
            
            db.commit()
            print db.execute(s0) 
            
            s=drop_table_string('unittest_small')

    def test_insert_table_string(self):
        table='unittest_small'
        with Database() as db:
            columns=db.get_column_names(table)
            
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
        with Database() as db:
            columns=db.get_column_names(table)
            
        d=dummy_data_small()
        l0=insert_table_string(table,d,columns)
        
        with Database() as db:
            db.execute(l0[-1])
            db.commit()
        
        
if __name__ == '__main__':
    d={
        ModuleFunctions:[
#                          'test_connect_supermicro',
#                          'test_create_table_string',
                        'test_drop_table',
                        'test_create_table',
#                          'test_query_table_names',
                        'test_insert_table_string',
                        'test_insert',
                         
                         
                    ],

       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)

# conn=psycopg2.connect("dname=test")